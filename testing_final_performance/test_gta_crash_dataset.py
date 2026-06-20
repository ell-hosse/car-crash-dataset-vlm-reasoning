"""Test VLA + RAG on GTA_Crash_Dataset (six inference variants).

Same models and six variants as visualize_correct_rag_injection.py
(no-RAG, Original/broken, Fix1-sep, Fix2-blend, Fix3-adaptive, Fix4-uniform)
but uses the GTA_Crash_Dataset instead of CoVLA preprocessed frames.

Ground-truth trajectories are derived from consecutive frame world-positions
transformed into the ego vehicle's local BEV coordinate frame (GTA V uses
Z-up, so the ground plane is XY).

Usage:
    python -m testing_final_performance.test_gta_crash_dataset \\
        --num-clips 10 [--crash-only] [--save-video] \\
        [--policy-source crash|abstract] [--seed 0]
"""
from __future__ import annotations
import argparse, json, random, sys, textwrap, time
from pathlib import Path

import cv2, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_repo_root(start: Path) -> Path:
    for d in (start, *start.parents):
        if (d / "covla_vla").is_dir() and (d / "crash_policies.jsonl").exists():
            return d
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compute_dist"))

from covla_vla.config import DATA, REALTIME
from covla_vla.dataset import preprocess_image, state_to_vec, denormalize_traj
from covla_vla.infer_realtime import load_model, project_traj
from testing_final_performance.visualize_vla_performance_on_CoVLA_before_rag import ade_fde
from clip_retrieval import (build_or_load_policy_index, ClipEmbedder,
                             pool_clip_video_embedding, build_vlm_rag_prompt,
                             DEFAULT_CLIP_MODEL, DEFAULT_POLICIES, DEFAULT_INDEX)

OUT_DIR = Path(__file__).resolve().parent / "viz_gta"

GTA_PARTITION_NAMES = [
    "GTACrash_accident_part1",
    "GTACrash_accident_part2",
    "GTACrash_accident_part3",
    "GTACrash_nonaccident_part1",
    "GTACrash_nonaccident_part2",
]

# BGR colours for OpenCV overlay / matplotlib colours for BEV
COL_GT    = (0,   0,   255)
COL_NORAG = (0,   255, 0  )
COL_ORIG  = (255, 0,   255)
COL_SEP   = (255, 255, 0  )
COL_BLEND = (0,   165, 255)
COL_FIX3  = (128, 0,   128)
COL_FIX4  = (255, 215, 0  )

MPL_COLS = {"GT": "red", "no-RAG": "green",
            "Original": "magenta", "Fix1-sep": "cyan",
            "Fix2-blend": "orange", "Fix3-adaptive": "purple",
            "Fix4-uniform": "gold"}


# ===========================================================================
#  GTA data loading
# ===========================================================================

def _load_label(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gta_state_dict(label: dict) -> dict:
    """Map GTA JSON annotation to CoVLA state_to_vec-compatible keys.

    GTA does not expose steering, brake, gas, or blinkers, so those default
    to 0.  Speed units in the GTA labels are assumed to be m/s (same as vEgo).
    """
    return {
        "vEgo":             float(label.get("speed", 0.0)),
        "aEgo":             float(label.get("acceleration", 0.0)),
        "steeringAngleDeg": 0.0,
        "brake":            0.0,
        "gas":              0.0,
        "leftBlinker":      0.0,
        "rightBlinker":     0.0,
    }


def gta_gt_traj(frames: list[dict], idx: int,
                traj_step: int, num_waypoints: int) -> np.ndarray | None:
    """Build a GT trajectory in the ego-local BEV frame from world positions.

    GTA V uses Z-up, so the ground plane is XY.
    Returns (num_waypoints, 2) in world-space metres, or None if not enough
    future frames remain.
    """
    if idx + traj_step * num_waypoints >= len(frames):
        return None
    anchor = frames[idx]
    pos0 = np.array(anchor["position"][:2], dtype=np.float64)
    fwd  = np.array(anchor["forwardV"][:2],  dtype=np.float64)
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        return None
    fwd  /= norm
    right = np.array([-fwd[1], fwd[0]])  # 90° CCW in XY = left lateral

    waypoints = []
    for k in range(1, num_waypoints + 1):
        future = frames[idx + traj_step * k]
        delta  = np.array(future["position"][:2], dtype=np.float64) - pos0
        waypoints.append([float(np.dot(delta, fwd)),    # forward
                          float(np.dot(delta, right))])  # lateral
    return np.array(waypoints, dtype=np.float32)


def load_partition_frames(gta_root: Path, part_name: str) -> list[dict]:
    """Load all (image_path, label) pairs from one partition, sorted numerically.

    Dataset layout: <gta_root>/images/<part_name>/*.jpg
                    <gta_root>/labels/<part_name>/*.json
    """
    img_dir = gta_root / "images" / part_name
    lbl_dir = gta_root / "labels" / part_name
    if not img_dir.exists() or not lbl_dir.exists():
        return []
    frames = []
    for img_path in sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem)):
        lbl_path = lbl_dir / f"{img_path.stem}.json"
        if not lbl_path.exists():
            continue
        label = _load_label(lbl_path)
        label["_image_path"] = img_path
        label["_is_crash"]   = "nonaccident" not in part_name
        label["_part_name"]  = part_name
        frames.append(label)
    return frames


def split_into_clips(frames: list[dict], boundary_m: float) -> list[list[dict]]:
    """Split a sorted frame list into clips at large position jumps."""
    if not frames:
        return []
    clips, current = [], [frames[0]]
    for prev, cur in zip(frames, frames[1:]):
        p0 = np.array(prev["position"][:2])
        p1 = np.array(cur["position"][:2])
        if np.linalg.norm(p1 - p0) > boundary_m:
            if len(current) > 1:
                clips.append(current)
            current = [cur]
        else:
            current.append(cur)
    if len(current) > 1:
        clips.append(current)
    return clips


def build_gta_samples(clip_frames: list[dict], clip_id: str,
                      traj_step: int, num_waypoints: int) -> list[dict]:
    """Build CoVLA-compatible sample dicts for each usable frame in a clip."""
    samples = []
    for idx, frame in enumerate(clip_frames):
        gt = gta_gt_traj(clip_frames, idx, traj_step, num_waypoints)
        if gt is None:
            continue
        samples.append({
            "clip_id":    clip_id,
            "frame_idx":  idx,
            "image_path": frame["_image_path"],
            "state":      gta_state_dict(frame),
            "gt_traj":    gt,
            "is_crash":   frame["_is_crash"],
            "part_name":  frame["_part_name"],
        })
    return samples


# ===========================================================================
#  SmolVLM2 captioner
# ===========================================================================

class VLMCaptioner:
    def __init__(self, device, cfg=REALTIME):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.cfg, self.device = cfg, device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(cfg.captioner_model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.captioner_model, torch_dtype=dtype).to(device).eval()

    @torch.no_grad()
    def caption(self, bgr: np.ndarray, prompt: str) -> str:
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device, dtype=self.model.dtype)
        out = self.model.generate(
            **inputs, max_new_tokens=self.cfg.caption_max_new_tokens, do_sample=False)
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


# ===========================================================================
#  CLIP scene retriever (GTA-aware: reads images from absolute paths)
# ===========================================================================

class GtaSceneRetriever:
    def __init__(self, clip_emb, matcher, top_k, caption_interval_frames):
        self.clip    = clip_emb
        self.matcher = matcher
        self.top_k   = top_k
        self.cad     = max(1, caption_interval_frames)
        self._cache: dict = {}

    def _frame_emb(self, s: dict):
        key = (s["clip_id"], s["frame_idx"])
        if key not in self._cache:
            bgr = cv2.imread(str(s["image_path"]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._cache[key] = self.clip.embed_image_arrays([rgb])[0]
        return self._cache[key]

    def hits_for(self, clip_samples: list[dict], j: int):
        lo   = max(0, j - self.cad + 1)
        embs = np.stack([self._frame_emb(clip_samples[k]) for k in range(lo, j + 1)])
        return self.matcher.retrieve(pool_clip_video_embedding(embs), top_k=self.top_k)


# ===========================================================================
#  Text embedding helpers (verbatim from visualize_correct_rag_injection.py)
# ===========================================================================

def _enc(model, tokenizer, text, device):
    tok = tokenizer([text], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    return model.encode_text(tok["input_ids"], tok["attention_mask"])   # (1,1,d)


def make_sep_embed(model, tok, caps, pols, device):
    caps_e = torch.cat([_enc(model, tok, c, device) for c in caps], dim=0)
    pols_e = torch.cat([_enc(model, tok, p, device) for p in pols], dim=0)
    return torch.cat([caps_e, pols_e], dim=1)   # (B,2,d)


def make_blend_embed(model, tok, caps, pols, device, alpha=0.25):
    blended = [(1-alpha)*_enc(model,tok,c,device) + alpha*_enc(model,tok,p,device)
               for c, p in zip(caps, pols)]
    return torch.cat(blended, dim=0)   # (B,1,d)


def make_fix3_embed(model, tok, caps, hits_batch, device, temperature=0.1):
    results = []
    for cap, hits in zip(caps, hits_batch):
        cap_e = _enc(model, tok, cap, device)
        if not hits:
            results.append(cap_e)
            continue
        scores  = np.array([h["score"] for h in hits], dtype=np.float32)
        exp_s   = np.exp((scores - scores.max()) / temperature)
        weights = exp_s / exp_s.sum()
        pol_embs = torch.cat(
            [_enc(model, tok,
                  f"{h.get('latent_risk','')}. {h.get('mitigation','')}".strip(". "),
                  device) for h in hits], dim=0)
        w_t      = torch.tensor(weights, dtype=pol_embs.dtype,
                                device=device).view(-1, 1, 1)
        pol_pool = (w_t * pol_embs).sum(dim=0, keepdim=True)
        gate     = float(scores.max())
        results.append((1.0 - gate) * cap_e + gate * pol_pool)
    return torch.cat(results, dim=0)


def make_fix4_embed(model, tok, caps, hits_batch, device):
    results = []
    for cap, hits in zip(caps, hits_batch):
        cap_e = _enc(model, tok, cap, device)
        if not hits:
            results.append(cap_e)
            continue
        scores   = np.array([h["score"] for h in hits], dtype=np.float32)
        w        = float(scores.max())
        pol_embs = torch.cat(
            [_enc(model, tok,
                  f"{h.get('latent_risk','')}. {h.get('mitigation','')}".strip(". "),
                  device) for h in hits], dim=0)
        pol_mean = pol_embs.mean(dim=0, keepdim=True)
        blended  = (1.0 - w) * cap_e + w * pol_mean
        blended  = blended / (blended.norm(dim=-1, keepdim=True) + 1e-12)
        results.append(blended)
    return torch.cat(results, dim=0)


def hits_to_policy_text(hits):
    parts = [f"{h.get('latent_risk','')}. {h.get('mitigation','')}".strip(". ")
             for h in hits if h.get("latent_risk") or h.get("mitigation")]
    return " | ".join(parts) if parts else "no policy retrieved"


# ===========================================================================
#  Six-variant inference
# ===========================================================================

@torch.no_grad()
def predict_all(model, tokenizer, samples, caps_plain, caps_orig,
                policy_texts, hits_list, device, alpha=0.25, batch_size=32,
                num_waypoints=None):
    """Returns (p_norag, p_orig, p_sep, p_blend, p_fix3, p_fix4) each (N,T,2) metres.

    num_waypoints: if set, truncates model output (always 20) to match shorter GT.
    """
    preds = {k: [] for k in ("norag", "orig", "sep", "blend", "fix3", "fix4")}

    for i in range(0, len(samples), batch_size):
        sl   = slice(i, i + batch_size)
        samp = samples[sl]
        cp   = caps_plain[sl]
        co   = caps_orig[sl]
        pl   = policy_texts[sl]

        imgs   = torch.stack([preprocess_image(
            cv2.imread(str(s["image_path"]))) for s in samp]).to(device)
        states = torch.stack([state_to_vec(s["state"]) for s in samp]).to(device)

        tok = tokenizer(list(cp), padding=True, truncation=True,
                        max_length=77, return_tensors="pt").to(device)
        preds["norag"].append(
            model(imgs, states, tok["input_ids"], tok["attention_mask"])
            .float().cpu().numpy())

        tok2 = tokenizer(list(co), padding=True, truncation=True,
                         max_length=77, return_tensors="pt").to(device)
        preds["orig"].append(
            model(imgs, states, tok2["input_ids"], tok2["attention_mask"])
            .float().cpu().numpy())

        preds["sep"].append(
            model(imgs, states,
                  text_embed=make_sep_embed(model, tokenizer, list(cp), list(pl), device))
            .float().cpu().numpy())

        preds["blend"].append(
            model(imgs, states,
                  text_embed=make_blend_embed(model, tokenizer, list(cp), list(pl),
                                             device, alpha=alpha))
            .float().cpu().numpy())

        hits_chunk = hits_list[i:i + batch_size]
        preds["fix3"].append(
            model(imgs, states,
                  text_embed=make_fix3_embed(model, tokenizer, list(cp),
                                            hits_chunk, device))
            .float().cpu().numpy())

        preds["fix4"].append(
            model(imgs, states,
                  text_embed=make_fix4_embed(model, tokenizer, list(cp),
                                            hits_chunk, device))
            .float().cpu().numpy())

    results = tuple(denormalize_traj(np.concatenate(preds[k], axis=0))
                    for k in ("norag", "orig", "sep", "blend", "fix3", "fix4"))
    if num_waypoints is not None:
        results = tuple(r[:, :num_waypoints, :] for r in results)
    return results


# ===========================================================================
#  Visualisation
# ===========================================================================

def draw_overlay(sample, gt, p_no, p_or, p_se, p_bl, p_f3, p_f4):
    bgr = cv2.imread(str(sample["image_path"]))
    for traj, col in ((gt, COL_GT), (p_no, COL_NORAG), (p_or, COL_ORIG),
                      (p_se, COL_SEP), (p_bl, COL_BLEND), (p_f3, COL_FIX3),
                      (p_f4, COL_FIX4)):
        pts = project_traj(traj, bgr.shape)
        if len(pts) >= 2:
            cv2.polylines(bgr, [pts], False, col, 2)
        for p in pts:
            cv2.circle(bgr, tuple(p), 3, col, -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_bev(ax, gt, p_no, p_or, p_se, p_bl, p_f3, p_f4, title=""):
    for arr, lbl in ((gt, "GT"), (p_no, "no-RAG"), (p_or, "Original"),
                     (p_se, "Fix1-sep"), (p_bl, "Fix2-blend"),
                     (p_f3, "Fix3-adaptive"), (p_f4, "Fix4-uniform")):
        ax.plot(-arr[:,1], arr[:,0], "o-", color=MPL_COLS[lbl],
                ms=2, lw=1.3, label=lbl)
    ax.scatter([0], [0], marker="^", s=50, color="black", zorder=5, label="ego")
    ax.set_xlabel("lateral (m)"); ax.set_ylabel("forward (m)")
    ax.set_title(title, fontsize=7); ax.axis("equal")
    ax.grid(alpha=0.3); ax.legend(fontsize=5)


def _wrap(label, text, width=70, max_lines=3):
    lines = textwrap.wrap(text, width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]; lines[-1] += " ..."
    return label + ("\n" + " "*len(label)).join(lines)


def render_video(samples, p_no, p_or, p_se, p_bl, p_f3, p_f4,
                 caps_plain, caps_orig, policy_texts, hits_list,
                 policy_source_label, out_path, fps=4):
    writer = None
    for k, s in enumerate(samples):
        gt = s["gt_traj"]
        an, fn   = ade_fde(p_no[k], gt)
        ao, fo   = ade_fde(p_or[k], gt)
        as_, fs  = ade_fde(p_se[k], gt)
        ab, fb   = ade_fde(p_bl[k], gt)
        af3, ff3 = ade_fde(p_f3[k], gt)
        af4, ff4 = ade_fde(p_f4[k], gt)
        top_score = hits_list[k][0]["score"] if hits_list[k] else 0.0
        crash_tag = "CRASH" if s["is_crash"] else "BENIGN"

        fig = plt.figure(figsize=(15, 9))
        gs  = fig.add_gridspec(3, 2, height_ratios=[2.8, 0.9, 0.9],
                               width_ratios=[1.7, 1])
        ax_img = fig.add_subplot(gs[0, 0])
        ax_bev = fig.add_subplot(gs[0, 1])
        ax_t1  = fig.add_subplot(gs[1, :])
        ax_t2  = fig.add_subplot(gs[2, :])

        ax_img.imshow(draw_overlay(s, gt, p_no[k], p_or[k], p_se[k],
                                   p_bl[k], p_f3[k], p_f4[k]))
        ax_img.set_title(
            f"{s['clip_id']}  frame {s['frame_idx']}  [{crash_tag}]  "
            f"policies: {policy_source_label}  top-score={top_score:.3f}\n"
            f"GT=red  no-RAG=green  Orig=magenta  "
            f"Fix1=cyan  Fix2=orange  Fix3=purple  Fix4=gold",
            fontsize=7)
        ax_img.axis("off")

        plot_bev(ax_bev, gt, p_no[k], p_or[k], p_se[k], p_bl[k], p_f3[k], p_f4[k],
                 f"ADE  no-RAG {an:.2f}  Orig {ao:.2f}  Sep {as_:.2f}  "
                 f"Blend {ab:.2f}  F3 {af3:.2f}  F4 {af4:.2f} m\n"
                 f"FDE  no-RAG {fn:.2f}  Orig {fo:.2f}  Sep {fs:.2f}  "
                 f"Blend {fb:.2f}  F3 {ff3:.2f}  F4 {ff4:.2f} m")

        ids = ", ".join(f"{h['clip_id']}({h['score']:.2f})" for h in hits_list[k])
        ax_t1.axis("off")
        ax_t1.text(0, 1,
            _wrap("Caption (plain):       ", caps_plain[k]) + "\n" +
            _wrap("Caption (orig/prompt): ", caps_orig[k]),
            transform=ax_t1.transAxes, fontsize=7, family="monospace",
            va="top", ha="left")
        ax_t2.axis("off")
        ax_t2.text(0, 1,
            _wrap("Policy text (sep/blend):", policy_texts[k]) + "\n" +
            f"Retrieved:              {ids}",
            transform=ax_t2.transAxes, fontsize=7, family="monospace",
            va="top", ha="left")

        fig.tight_layout()
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        if writer is None:
            h_px, w_px = rgb.shape[:2]
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_px, h_px))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if writer:
        writer.release()


# ===========================================================================
#  Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark VLA + RAG on GTA_Crash_Dataset.")
    ap.add_argument("--ckpt",             default=str(REPO_ROOT / "rag" / "covla_vla_best.pt"))
    ap.add_argument("--gta-root",         default=str(REPO_ROOT / "GTA_Crash_Dataset"))
    ap.add_argument("--policy-source",    default="crash", choices=["crash", "abstract"])
    ap.add_argument("--num-clips",        type=int,   default=10)
    ap.add_argument("--seed",             type=int,   default=0)
    ap.add_argument("--top-k",            type=int,   default=5)
    ap.add_argument("--alpha",            type=float, default=0.25,
                    help="Fix2 blend weight (0=caption only, 1=policy only)")
    ap.add_argument("--max-samples",      type=int,   default=None,
                    help="Truncate each clip to this many frames")
    ap.add_argument("--fps",              type=int,   default=4)
    ap.add_argument("--clip-model",       default=DEFAULT_CLIP_MODEL)
    ap.add_argument("--traj-step",        type=int,   default=1,
                    help="Frames between consecutive GT waypoints")
    ap.add_argument("--traj-horizon",     type=int,   default=15,
                    help="Lookahead frame count; num_waypoints = horizon // step "
                         "(GTA clips are ~20 frames so keep this < clip length)")
    ap.add_argument("--clip-boundary",    type=float, default=50.0,
                    help="Position jump (m) that marks a new clip")
    ap.add_argument("--caption-interval", type=int,   default=10,
                    help="Caption refresh interval in frames")
    ap.add_argument("--save-video",       action="store_true",
                    help="Render MP4 visualisation per clip")
    ap.add_argument("--crash-only",       action="store_true",
                    help="Use only accident partitions")
    args = ap.parse_args()

    num_waypoints = args.traj_horizon // args.traj_step

    if args.policy_source == "abstract":
        pol_path  = REPO_ROOT / "abstract_patterns.jsonl"
        idx_path  = REPO_ROOT / "compute_dist" / "clip_abstract_index.npz"
        pol_label = "abstract_patterns"
    else:
        pol_path  = REPO_ROOT / "crash_policies.jsonl"
        idx_path  = Path(DEFAULT_INDEX)
        pol_label = "crash_policies"

    gta_root = Path(args.gta_root)
    rng      = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device         : {device}")
    print(f"gta root       : {gta_root}")
    print(f"policy source  : {pol_label}")
    print(f"alpha (Fix2)   : {args.alpha}")
    print(f"num_waypoints  : {num_waypoints}  "
          f"(traj_step={args.traj_step}, traj_horizon={args.traj_horizon})")

    model, tokenizer = load_model(args.ckpt, device)
    clip_emb = ClipEmbedder(args.clip_model, device=device)
    matcher  = build_or_load_policy_index(pol_path, idx_path,
                                          args.clip_model, embedder=clip_emb)
    retr     = GtaSceneRetriever(clip_emb, matcher, args.top_k,
                                 args.caption_interval)
    vlm      = VLMCaptioner(device)
    base_prompt = REALTIME.caption_prompt

    print(f"policy index   : {len(matcher)} entries  |  "
          f"captioner = {REALTIME.captioner_model}\n")

    # Build clip list from selected partitions
    partition_names = [n for n in GTA_PARTITION_NAMES
                       if not args.crash_only or "accident" in n]
    all_clips: list[tuple[str, list[dict]]] = []

    for part_name in partition_names:
        if not (gta_root / "images" / part_name).exists():
            print(f"  [warn] partition not found: {gta_root / 'images' / part_name}")
            continue
        print(f"  scanning {part_name} ...", end=" ", flush=True)
        frames = load_partition_frames(gta_root, part_name)
        clips  = split_into_clips(frames, args.clip_boundary)
        print(f"{len(frames)} frames → {len(clips)} clips")
        for ci, clip_frames in enumerate(clips):
            clip_id = f"{part_name}_clip{ci:04d}"
            samples = build_gta_samples(clip_frames, clip_id,
                                        args.traj_step, num_waypoints)
            if samples:
                all_clips.append((clip_id, samples))

    print(f"\ntotal usable clips : {len(all_clips)}")
    chosen = rng.sample(all_clips, min(args.num_clips, len(all_clips)))
    print(f"selected clips     : {len(chosen)}\n")

    all_metrics: dict  = {}
    crash_records: list = []
    benign_records: list = []

    for ci, (clip_id, samples) in enumerate(chosen):
        if args.max_samples:
            samples = samples[:args.max_samples]
        is_crash  = samples[0]["is_crash"]
        crash_tag = "CRASH" if is_crash else "BENIGN"
        print(f"[{ci}] {clip_id} [{crash_tag}]: {len(samples)} frames")

        caps_plain, caps_orig, policy_texts, hits_list = [], [], [], []
        cur_plain = cur_orig = cur_pol = None
        cur_hits  = None
        t_cap: list[float] = []

        for j, s in enumerate(samples):
            if (j % retr.cad == 0) or cur_hits is None:
                bgr      = cv2.imread(str(s["image_path"]))
                cur_hits = retr.hits_for(samples, j)

                t0        = time.time()
                cur_plain = vlm.caption(bgr, base_prompt)
                cur_orig  = vlm.caption(bgr, build_vlm_rag_prompt(base_prompt, cur_hits))
                t_cap.append(time.time() - t0)

                cur_pol = hits_to_policy_text(cur_hits)

            caps_plain.append(cur_plain)
            caps_orig.append(cur_orig)
            policy_texts.append(cur_pol)
            hits_list.append(cur_hits)
            if (j + 1) % 10 == 0:
                print(f"    captioned {j + 1}/{len(samples)} frames")

        caps_plain_arr   = np.array(caps_plain,   dtype=object)
        caps_orig_arr    = np.array(caps_orig,    dtype=object)
        policy_texts_arr = np.array(policy_texts, dtype=object)

        p_no, p_or, p_se, p_bl, p_f3, p_f4 = predict_all(
            model, tokenizer, samples,
            caps_plain_arr, caps_orig_arr, policy_texts_arr, hits_list,
            device, alpha=args.alpha, num_waypoints=num_waypoints)

        def _ade(preds):
            return float(np.mean([ade_fde(preds[k], s["gt_traj"])[0]
                                  for k, s in enumerate(samples)]))

        an, ao, as_, ab, af3, af4 = (_ade(p_no), _ade(p_or), _ade(p_se),
                                      _ade(p_bl), _ade(p_f3), _ade(p_f4))
        print(f"    [{crash_tag}] mean ADE  "
              f"no-RAG {an:.3f}  |  Orig {ao:.3f}  |  Fix1-sep {as_:.3f}  |  "
              f"Fix2 {ab:.3f}  |  Fix3 {af3:.3f}  |  Fix4 {af4:.3f} m  "
              f"(caption {np.mean(t_cap)*1e3:.0f} ms/refresh ×2)")

        if args.save_video:
            out = OUT_DIR / f"gta_{ci:02d}_{clip_id}_{pol_label}.mp4"
            render_video(samples, p_no, p_or, p_se, p_bl, p_f3, p_f4,
                         caps_plain, caps_orig, policy_texts, hits_list,
                         pol_label, out, fps=args.fps)
            print(f"    wrote {out.name}")

        clip_records: list = []
        for k, s in enumerate(samples):
            g = s["gt_traj"]
            r_no, _ = ade_fde(p_no[k], g)
            r_or, _ = ade_fde(p_or[k], g)
            r_se, _ = ade_fde(p_se[k], g)
            r_bl, _ = ade_fde(p_bl[k], g)
            r_f3, _ = ade_fde(p_f3[k], g)
            r_f4, _ = ade_fde(p_f4[k], g)
            rec = {
                "frame_idx":         s["frame_idx"],
                "is_crash":          s["is_crash"],
                "caption_plain":     caps_plain[k],
                "caption_orig":      caps_orig[k],
                "policy_text":       policy_texts[k],
                "retrieved":         [{"clip_id": h["clip_id"], "score": h["score"]}
                                      for h in hits_list[k]],
                "top_score":         hits_list[k][0]["score"] if hits_list[k] else 0.0,
                "ade_no_rag":        r_no,
                "ade_original":      r_or,
                "ade_fix1_sep":      r_se,
                "ade_fix2_blend":    r_bl,
                "ade_fix3_adaptive": r_f3,
                "ade_fix4_uniform":  r_f4,
            }
            clip_records.append(rec)
            (crash_records if is_crash else benign_records).append(rec)

        all_metrics[clip_id] = clip_records

    metrics_path = OUT_DIR / f"metrics_gta_{pol_label}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {metrics_path.name}  →  {OUT_DIR}")

    def _agg(records, key):
        if not records:
            return float("nan")
        return float(np.mean([r[key] for r in records]))

    def _print_table(label, records):
        if not records:
            print(f"  ({label}: no samples)")
            return
        n = len(records)
        print(f"\n=== aggregate ADE  [{pol_label}]  {label}  (n={n}) ===")
        print(f"  no-RAG (baseline)          : {_agg(records, 'ade_no_rag'):.3f} m")
        print(f"  Original (bug: cap+policy) : {_agg(records, 'ade_original'):.3f} m")
        print(f"  Fix1 – separate embeddings : {_agg(records, 'ade_fix1_sep'):.3f} m")
        print(f"  Fix2 – fixed blend α=0.25  : {_agg(records, 'ade_fix2_blend'):.3f} m")
        print(f"  Fix3 – score-adaptive pool : {_agg(records, 'ade_fix3_adaptive'):.3f} m")
        print(f"  Fix4 – uniform avg + L2    : {_agg(records, 'ade_fix4_uniform'):.3f} m")

    all_records = crash_records + benign_records
    _print_table("ALL",    all_records)
    _print_table("CRASH",  crash_records)
    _print_table("BENIGN", benign_records)


if __name__ == "__main__":
    main()
