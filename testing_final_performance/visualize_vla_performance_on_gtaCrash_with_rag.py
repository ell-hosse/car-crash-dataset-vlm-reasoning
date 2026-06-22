"""Visualize VLA performance WITH vs WITHOUT RAG on the GTA_Crash_Dataset (video).

Per clip this renders an MP4 that steps through *every* frame of the original
clip (~20 frames). Each frame shows:
  * the camera image with three trajectories overlaid:
      - GT          (red)    real future ego path in the ego-local BEV frame
      - no-RAG      (green)  VLA conditioned on a PLAIN caption
      - with-RAG    (magenta) VLA conditioned on a RAG caption whose embedding is
                    fused with the closest retrieved crash policy
  * a BEV plot of the same three trajectories with ADE/FDE
  * the two captions for that frame (plain vs RAG) and the retrieved policies.

RAG is applied in TWO places, both keyed on the CLIP-retrieved crash policies:
  1. Captioning: the CLOSEST policy (top-1) fills the {trigger}/{latent_risk}/
     {mitigation} slots of RAG_CAPTION_PROMPT, so the captioner writes a
     risk-aware, action-first caption.
  2. Embedding: that RAG caption is encoded and score-gated-fused with the
     closest policy embedding, following test_gta_crash_dataset.py's adaptive
     fix reduced to the single closest policy:
         fused = (1 - s)*caption_emb + s*policy_emb,   s = top-1 similarity score
     (policy text = "latent_risk. mitigation"). The fused embedding conditions
     the with-RAG prediction.

The captioner is selectable with --captioner:
     smolvlm2   -> HuggingFaceTB/SmolVLM2-256M-Video-Instruct  (tiny, fast)
     qwen2.5-vl -> Qwen/Qwen2.5-VL-3B-Instruct                 (free, stronger)
     qwen2-vl   -> Qwen/Qwen2-VL-2B-Instruct
--max-pixels caps the per-frame visual-token count for the Qwen captioners.

Ground truth is NOT fabricated: it is the car's real future world-positions
projected into the ego frame. Near the end of a clip fewer future frames exist,
so the GT horizon simply shrinks (and the very last frame, with no future, has
no GT); predictions and captions are still drawn for those frames.

Usage (from repo root):
    python -m testing_final_performance.visualize_vla_performance_on_gtaCrash_with_rag \
        --num-clips 5 [--captioner qwen2.5-vl] [--policy-source crash|abstract] [--seed 0]
"""
from __future__ import annotations
import argparse, json, random, sys, textwrap, time
from pathlib import Path

import cv2, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compute_dist"))

from covla_vla.config import REALTIME                                         # noqa: E402
from covla_vla.dataset import preprocess_image, state_to_vec, denormalize_traj  # noqa: E402
from covla_vla.infer_realtime import load_model, project_traj                 # noqa: E402
from clip_retrieval import (build_or_load_policy_index, ClipEmbedder,         # noqa: E402
                            pool_clip_video_embedding,
                            DEFAULT_CLIP_MODEL, DEFAULT_INDEX)

OUT_DIR = Path(__file__).resolve().parent / "viz_gta_with_rag"
DEFAULT_GTA_ROOT = r"D:\GTA_carCrashDS\GTA_Crash_Dataset"

# Short --captioner choices -> HuggingFace model ids. All are free/open weights.
CAPTIONER_IDS = {
    "smolvlm2":   REALTIME.captioner_model,          # SmolVLM2-256M-Video-Instruct
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct",     # smallest free Qwen2.5-VL
    "qwen2-vl":   "Qwen/Qwen2-VL-2B-Instruct",       # smallest free Qwen2-VL
}

# RAG captioner prompt: the CLOSEST retrieved crash policy is injected into the
# {trigger}/{latent_risk}/{mitigation} slots so the captioner writes a
# risk-aware, action-first caption.
RAG_CAPTION_PROMPT = (
    "You are the perception-and-safety module of a self-driving car. Look "
    "carefully at this front-camera frame.\n\n"
    "A retrieval system flagged a crash pattern that may match this scene "
    "(matched by visual similarity to past incidents — trust your own eyes "
    "over it):\n"
    "  • Hazard: {trigger}\n"
    "  • Why dangerous: {latent_risk}\n"
    "  • Safe response: {mitigation}\n\n"
    "Write ONE caption, under 45 words, in this exact order:\n"
    "1) The single most safety-critical thing visible right now (lead vehicle, "
    "pedestrian, cyclist, signal, or road/weather condition).\n"
    "2) If that hazard — or the flagged one — is visible or plausibly "
    "developing, state that THE EGO VEHICLE slows down, increases following "
    "distance, and is ready to brake or yield as appropriate. Describe the ego "
    "actually doing it.\n"
    "3) Brief remaining context only if space allows.\n\n"
    "Rules: describe only what you can see; never invent objects. But when a "
    "hazard is uncertain, choose the more cautious description and the slower "
    "action. Put the hazard and the cautious maneuver FIRST so they are never "
    "cut off."
)


def build_custom_rag_prompt(hits, base_prompt):
    """Fill RAG_CAPTION_PROMPT with the CLOSEST retrieved policy (top-1 hit).

    Falls back to the plain base prompt when nothing is retrieved.
    """
    if not hits:
        return base_prompt
    h = hits[0]
    return (RAG_CAPTION_PROMPT
            .replace("{trigger}",     str(h.get("trigger", "")).strip())
            .replace("{latent_risk}", str(h.get("latent_risk", "")).strip())
            .replace("{mitigation}",  str(h.get("mitigation", "")).strip()))


def closest_policy_text(hits):
    """'latent_risk. mitigation' for the closest policy (top-1), or None."""
    if not hits:
        return None
    h = hits[0]
    txt = f"{h.get('latent_risk','')}. {h.get('mitigation','')}".strip(". ")
    return txt or None


GTA_PARTITION_NAMES = [
    "GTACrash_accident_part1",
    "GTACrash_accident_part2",
    "GTACrash_accident_part3",
    "GTACrash_nonaccident_part1",
    "GTACrash_nonaccident_part2",
]

# BGR for OpenCV overlay
COL_GT   = (0,   0,   255)   # red
COL_NORAG = (0,  255, 0)     # green
COL_RAG  = (255, 0,   255)   # magenta

MPL_COLS = {"GT": "red", "no-RAG": "green", "with-RAG": "magenta"}


# ===========================================================================
#  GTA data loading
# ===========================================================================

def _load_label(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gta_state_dict(label: dict) -> dict:
    """Map a GTA JSON annotation to CoVLA state_to_vec-compatible keys."""
    return {
        "vEgo":             float(label.get("speed", 0.0)),
        "aEgo":             float(label.get("acceleration", 0.0)),
        "steeringAngleDeg": 0.0,
        "brake":            0.0,
        "gas":              0.0,
        "leftBlinker":      0.0,
        "rightBlinker":     0.0,
    }


def gta_gt_traj(frames: list[dict], idx: int, traj_step: int,
                num_waypoints: int, min_waypoints: int = 2) -> np.ndarray | None:
    """Real future ego path in the ego-local BEV frame (forward, lateral) metres.

    Uses up to num_waypoints future waypoints, but shrinks gracefully near the
    end of a clip: as long as at least min_waypoints future frames exist the GT
    is returned at that shorter length. Returns None only when fewer than
    min_waypoints future frames remain (e.g. the final frame).
    """
    avail = (len(frames) - 1 - idx) // traj_step
    n = min(num_waypoints, avail)
    if n < min_waypoints:
        return None
    pos0 = np.array(frames[idx]["position"][:2], dtype=np.float64)
    fwd  = np.array(frames[idx]["forwardV"][:2],  dtype=np.float64)
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        return None
    fwd  /= norm
    right = np.array([-fwd[1], fwd[0]])  # 90 deg CCW = left lateral
    waypoints = []
    for k in range(1, n + 1):
        delta = np.array(frames[idx + traj_step * k]["position"][:2],
                         dtype=np.float64) - pos0
        waypoints.append([float(np.dot(delta, fwd)), float(np.dot(delta, right))])
    return np.array(waypoints, dtype=np.float32)


def load_partition_frames(gta_root: Path, part_name: str,
                          scan_limit: int | None = None) -> list[dict]:
    """Load (image_path, label) pairs from one partition by walking the
    zero-padded sequential index, stopping at scan_limit so we never enumerate
    the whole ~52k-frame directory just to grab a few clips."""
    img_dir = gta_root / "images" / part_name
    lbl_dir = gta_root / "labels" / part_name
    if not img_dir.exists() or not lbl_dir.exists():
        return []
    frames = []
    idx, misses = 0, 0
    while scan_limit is None or len(frames) < scan_limit:
        img_path = None
        for width in (6, 5, 0):
            stem = f"{idx:0{width}d}" if width else str(idx)
            cand = img_dir / f"{stem}.jpg"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            misses += 1
            if misses > 50:
                break
            idx += 1
            continue
        misses = 0
        lbl_path = lbl_dir / f"{img_path.stem}.json"
        if lbl_path.exists():
            label = _load_label(lbl_path)
            label["_image_path"] = img_path
            label["_is_crash"]   = "nonaccident" not in part_name
            label["_part_name"]  = part_name
            frames.append(label)
        idx += 1
    return frames


def split_into_clips(frames: list[dict], boundary_m: float) -> list[list[dict]]:
    """Split a sorted frame list into clips at large position jumps."""
    if not frames:
        return []
    clips, current = [], [frames[0]]
    for prev, cur in zip(frames, frames[1:]):
        p0 = np.array(prev["position"][:2]); p1 = np.array(cur["position"][:2])
        if np.linalg.norm(p1 - p0) > boundary_m:
            if len(current) > 1:
                clips.append(current)
            current = [cur]
        else:
            current.append(cur)
    if len(current) > 1:
        clips.append(current)
    return clips


def build_gta_samples(clip_frames: list[dict], clip_id: str, traj_step: int,
                      num_waypoints: int, min_waypoints: int) -> list[dict]:
    """Build a sample dict for EVERY frame in the clip (so the video covers the
    whole clip). gt_traj may be None for the final frame(s) with no future."""
    samples = []
    for idx, frame in enumerate(clip_frames):
        gt = gta_gt_traj(clip_frames, idx, traj_step, num_waypoints, min_waypoints)
        samples.append({
            "clip_id":    clip_id,
            "frame_idx":  idx,
            "image_path": frame["_image_path"],
            "state":      gta_state_dict(frame),
            "gt_traj":    gt,                       # may be None near clip end
            "is_crash":   frame["_is_crash"],
            "part_name":  frame["_part_name"],
        })
    return samples


# ===========================================================================
#  VLM captioner (SmolVLM2 or Qwen2/2.5-VL)
# ===========================================================================

class VLMCaptioner:
    """Image captioner that works with both SmolVLM2 and Qwen2/2.5-VL.

    For Qwen, max_pixels caps how many visual tokens a frame produces (Qwen uses
    dynamic-resolution encoding, so a high-res frame can be hundreds-thousands of
    tokens); SmolVLM2 ignores it.
    """
    def __init__(self, device, model_id=None, cfg=REALTIME,
                 max_pixels=None, max_new_tokens=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.cfg, self.device = cfg, device
        self.model_id = model_id or cfg.captioner_model
        self.max_new_tokens = max_new_tokens or cfg.caption_max_new_tokens
        self.is_qwen = "qwen" in self.model_id.lower()
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        proc_kwargs = {}
        if self.is_qwen and max_pixels:
            proc_kwargs.update(min_pixels=256 * 28 * 28, max_pixels=int(max_pixels))
        self.processor = AutoProcessor.from_pretrained(self.model_id, **proc_kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=dtype).to(device).eval()

    @torch.no_grad()
    def caption(self, bgr: np.ndarray, prompt: str) -> str:
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
        if self.is_qwen:
            text = self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=[text], images=[pil], return_tensors="pt")
        else:
            inputs = self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


# ===========================================================================
#  CLIP scene retriever (reads images from absolute GTA paths)
# ===========================================================================

class GtaSceneRetriever:
    def __init__(self, clip_emb, matcher, top_k, window_frames):
        self.clip    = clip_emb
        self.matcher = matcher
        self.top_k   = top_k
        self.win     = max(1, window_frames)
        self._cache: dict = {}

    def _frame_emb(self, s: dict):
        key = (s["clip_id"], s["frame_idx"])
        if key not in self._cache:
            bgr = cv2.imread(str(s["image_path"]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._cache[key] = self.clip.embed_image_arrays([rgb])[0]
        return self._cache[key]

    def hits_for(self, clip_samples: list[dict], j: int):
        lo   = max(0, j - self.win + 1)
        embs = np.stack([self._frame_emb(clip_samples[k]) for k in range(lo, j + 1)])
        return self.matcher.retrieve(pool_clip_video_embedding(embs), top_k=self.top_k)


def hits_to_policy_text(hits):
    parts = [f"{h.get('latent_risk','')}. {h.get('mitigation','')}".strip(". ")
             for h in hits if h.get("latent_risk") or h.get("mitigation")]
    return " | ".join(parts) if parts else "no policy retrieved"


# ===========================================================================
#  Inference (no-RAG caption vs with-RAG fused embedding) + metrics
# ===========================================================================

def ade_fde(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """ADE / FDE in metres over the overlapping horizon of pred and gt."""
    L = min(len(pred), len(gt))
    d = np.linalg.norm(pred[:L] - gt[:L], axis=-1)
    return float(d.mean()), float(d[-1])


@torch.no_grad()
def predict_caption_variant(model, tokenizer, samples, captions, device,
                            batch_size=32, num_waypoints=None) -> np.ndarray:
    """No-RAG baseline: VLA conditioned on a plain caption (tokenized). (N,T,2) m."""
    preds = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        caps  = list(captions[i:i + batch_size])
        imgs  = torch.stack([preprocess_image(
            cv2.imread(str(s["image_path"]))) for s in chunk]).to(device)
        states = torch.stack([state_to_vec(s["state"]) for s in chunk]).to(device)
        tok = tokenizer(caps, padding=True, truncation=True,
                        max_length=77, return_tensors="pt").to(device)
        pred = model(imgs, states, tok["input_ids"], tok["attention_mask"])
        preds.append(pred.float().cpu().numpy())
    out = denormalize_traj(np.concatenate(preds, axis=0))
    if num_waypoints is not None:
        out = out[:, :num_waypoints, :]
    return out


def _enc(model, tokenizer, text, device):
    """Pooled caption/policy embedding -> (1, 1, d)."""
    tok = tokenizer([text], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    return model.encode_text(tok["input_ids"], tok["attention_mask"])


def make_rag_fused_embed(model, tokenizer, rag_caps, hits_batch, device):
    """Score-gated fusion of the RAG-caption embedding with the closest policy.

    Mirrors test_gta_crash_dataset.py's adaptive fix, reduced to the single
    closest (top-1) policy:  fused = (1 - s)*cap_e + s*pol_e,  where s is the
    top-1 retrieval similarity score (gate in [0, 1]). When nothing is
    retrieved the plain RAG-caption embedding is used. Returns (B, 1, d).
    """
    results = []
    for cap, hits in zip(rag_caps, hits_batch):
        cap_e = _enc(model, tokenizer, cap, device)
        ptxt  = closest_policy_text(hits)
        if ptxt is None:
            results.append(cap_e)
            continue
        pol_e = _enc(model, tokenizer, ptxt, device)
        gate  = float(np.clip(hits[0]["score"], 0.0, 1.0))   # top-1 score gate
        results.append((1.0 - gate) * cap_e + gate * pol_e)
    return torch.cat(results, dim=0)


@torch.no_grad()
def predict_rag_fused(model, tokenizer, samples, rag_caps, hits_list, device,
                      batch_size=32, num_waypoints=None) -> np.ndarray:
    """With-RAG prediction: VLA conditioned on the fused (RAG caption +
    closest policy) text embedding. (N, T, 2) metres."""
    preds = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        caps  = list(rag_caps[i:i + batch_size])
        hits  = hits_list[i:i + batch_size]
        imgs  = torch.stack([preprocess_image(
            cv2.imread(str(s["image_path"]))) for s in chunk]).to(device)
        states = torch.stack([state_to_vec(s["state"]) for s in chunk]).to(device)
        fused = make_rag_fused_embed(model, tokenizer, caps, hits, device)
        pred  = model(imgs, states, text_embed=fused)
        preds.append(pred.float().cpu().numpy())
    out = denormalize_traj(np.concatenate(preds, axis=0))
    if num_waypoints is not None:
        out = out[:, :num_waypoints, :]
    return out


# ===========================================================================
#  Visualisation
# ===========================================================================

def draw_overlay(sample, gt, p_no, p_rag):
    bgr = cv2.imread(str(sample["image_path"]))
    layers = [(p_no, COL_NORAG), (p_rag, COL_RAG)]
    if gt is not None:
        layers = [(gt, COL_GT)] + layers
    for traj, col in layers:
        pts = project_traj(traj, bgr.shape)
        if len(pts) >= 2:
            cv2.polylines(bgr, [pts], False, col, 2)
        for p in pts:
            cv2.circle(bgr, tuple(p), 3, col, -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_bev(ax, gt, p_no, p_rag, title=""):
    series = [(p_no, "no-RAG"), (p_rag, "with-RAG")]
    if gt is not None:
        series = [(gt, "GT")] + series
    for arr, lbl in series:
        ax.plot(-arr[:, 1], arr[:, 0], "o-", color=MPL_COLS[lbl],
                ms=3, lw=1.5, label=lbl)
    ax.scatter([0], [0], marker="^", s=70, color="black", zorder=5, label="ego")
    ax.set_xlabel("lateral (m, right +)"); ax.set_ylabel("forward (m)")
    ax.set_title(title, fontsize=8); ax.axis("equal")
    ax.grid(alpha=0.3); ax.legend(fontsize=6)


def _wrap(label, text, width=80, max_lines=3):
    lines = textwrap.wrap(text or "", width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]; lines[-1] += " ..."
    return label + ("\n" + " " * len(label)).join(lines)


def render_clip_video(samples, p_no, p_rag, caps_plain, caps_rag,
                      policy_texts, hits_list, policy_label, captioner_label,
                      out_path, fps=4):
    """One MP4 covering every frame of the clip; each frame shows both
    trajectories, both captions and the retrieved policies."""
    writer = None
    for k, s in enumerate(samples):
        gt = s["gt_traj"]
        if gt is not None:
            an, fn = ade_fde(p_no[k], gt)
            ar, fr = ade_fde(p_rag[k], gt)
            metric_line = (f"ADE  no-RAG {an:.2f}  with-RAG {ar:.2f} m   "
                           f"FDE  no-RAG {fn:.2f}  with-RAG {fr:.2f} m")
        else:
            metric_line = "ADE/FDE  n/a (no future frames for GT)"
        top_score = hits_list[k][0]["score"] if hits_list[k] else 0.0
        crash_tag = "CRASH" if s["is_crash"] else "BENIGN"

        fig = plt.figure(figsize=(15, 9))
        gs  = fig.add_gridspec(3, 2, height_ratios=[2.8, 0.9, 0.9],
                               width_ratios=[1.7, 1])
        ax_img = fig.add_subplot(gs[0, 0])
        ax_bev = fig.add_subplot(gs[0, 1])
        ax_t1  = fig.add_subplot(gs[1, :])
        ax_t2  = fig.add_subplot(gs[2, :])

        ax_img.imshow(draw_overlay(s, gt, p_no[k], p_rag[k]))
        ax_img.set_title(
            f"{s['clip_id']}  frame {s['frame_idx']}/{len(samples)-1}  "
            f"[{crash_tag}]  policies: {policy_label}  captioner: {captioner_label}  "
            f"top-score={top_score:.3f}\n"
            f"GT=red   no-RAG=green   with-RAG=magenta", fontsize=8)
        ax_img.axis("off")

        plot_bev(ax_bev, gt, p_no[k], p_rag[k], metric_line)

        ids = ", ".join(f"{h['clip_id']}({h['score']:.2f})" for h in hits_list[k])
        ax_t1.axis("off")
        ax_t1.text(0, 1,
            _wrap("Caption (no-RAG): ", caps_plain[k]) + "\n" +
            _wrap("Caption (with-RAG): ", caps_rag[k]),
            transform=ax_t1.transAxes, fontsize=7.5, family="monospace",
            va="top", ha="left")
        ax_t2.axis("off")
        ax_t2.text(0, 1,
            _wrap("Retrieved policies: ", policy_texts[k]) + "\n" +
            f"Retrieved ids:      {ids}",
            transform=ax_t2.transAxes, fontsize=7.5, family="monospace",
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
        description="Render with-RAG vs no-RAG trajectory videos on GTA crash clips.")
    ap.add_argument("--ckpt",             default=str(REPO_ROOT / "covla_vla_best.pt"))
    ap.add_argument("--gta-root",         default=DEFAULT_GTA_ROOT)
    ap.add_argument("--policy-source",    default="crash", choices=["crash", "abstract"])
    ap.add_argument("--captioner",        default="smolvlm2",
                    choices=list(CAPTIONER_IDS.keys()),
                    help="Which VLM captions the frames")
    ap.add_argument("--captioner-model",  default=None,
                    help="Override the exact HuggingFace captioner id")
    ap.add_argument("--max-pixels",       type=int, default=512 * 512,
                    help="Cap on visual tokens per frame for Qwen captioners")
    ap.add_argument("--num-clips",        type=int,   default=5)
    ap.add_argument("--seed",             type=int,   default=0)
    ap.add_argument("--top-k",            type=int,   default=5)
    ap.add_argument("--fps",              type=int,   default=4)
    ap.add_argument("--clip-model",       default=DEFAULT_CLIP_MODEL)
    ap.add_argument("--traj-step",        type=int,   default=1,
                    help="Frames between consecutive GT waypoints")
    ap.add_argument("--traj-horizon",     type=int,   default=15,
                    help="Max lookahead frames; num_waypoints = horizon // step")
    ap.add_argument("--min-waypoints",    type=int,   default=2,
                    help="Min future waypoints needed to draw GT for a frame")
    ap.add_argument("--clip-boundary",    type=float, default=50.0,
                    help="Position jump (m) that marks a new clip")
    ap.add_argument("--retr-window",      type=int,   default=5,
                    help="Frames pooled by CLIP for each retrieval")
    ap.add_argument("--scan-limit",       type=int,   default=1500,
                    help="Max frames read per partition (each has ~52k); 0 = all")
    ap.add_argument("--include-benign",   action="store_true",
                    help="Also sample nonaccident partitions (default: crash only)")
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

    captioner_id = args.captioner_model or CAPTIONER_IDS[args.captioner]
    cap_tag = args.captioner

    gta_root = Path(args.gta_root)
    rng      = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device         : {device}")
    print(f"gta root       : {gta_root}")
    print(f"policy source  : {pol_label}")
    print(f"captioner      : {captioner_id}")
    print(f"num_waypoints  : {num_waypoints} (step={args.traj_step}, "
          f"horizon={args.traj_horizon})")

    model, tokenizer = load_model(args.ckpt, device)
    clip_emb = ClipEmbedder(args.clip_model, device=device)
    matcher  = build_or_load_policy_index(pol_path, idx_path,
                                          args.clip_model, embedder=clip_emb)
    retr     = GtaSceneRetriever(clip_emb, matcher, args.top_k, args.retr_window)
    vlm      = VLMCaptioner(device, model_id=captioner_id, max_pixels=args.max_pixels)
    base_prompt = REALTIME.caption_prompt
    print(f"policy index   : {len(matcher)} entries\n")

    # ---- gather a small pool of clips, stopping early
    partition_names = [n for n in GTA_PARTITION_NAMES
                       if args.include_benign or "nonaccident" not in n]
    scan_limit = None if args.scan_limit in (0, None) else args.scan_limit
    target_pool = max(args.num_clips * 3, args.num_clips)
    all_clips: list[tuple[str, list[dict]]] = []
    for part_name in partition_names:
        if len(all_clips) >= target_pool:
            break
        if not (gta_root / "images" / part_name).exists():
            print(f"  [warn] partition not found: {gta_root / 'images' / part_name}")
            continue
        print(f"  scanning {part_name} ...", end=" ", flush=True)
        frames = load_partition_frames(gta_root, part_name, scan_limit)
        clips  = split_into_clips(frames, args.clip_boundary)
        print(f"{len(frames)} frames -> {len(clips)} clips")
        for ci, clip_frames in enumerate(clips):
            clip_id = f"{part_name}_clip{ci:04d}"
            samples = build_gta_samples(clip_frames, clip_id, args.traj_step,
                                        num_waypoints, args.min_waypoints)
            if samples:
                all_clips.append((clip_id, samples))

    print(f"\ntotal usable clips : {len(all_clips)}")
    if not all_clips:
        print("no usable clips found - check --gta-root and partition layout.")
        return
    chosen = rng.sample(all_clips, min(args.num_clips, len(all_clips)))
    print(f"selected clips     : {len(chosen)}\n")

    all_metrics: dict = {}

    for ci, (clip_id, samples) in enumerate(chosen):
        crash_tag = "CRASH" if samples[0]["is_crash"] else "BENIGN"
        print(f"[{ci}] {clip_id} [{crash_tag}]: {len(samples)} frames "
              f"(captioning every frame x2 with {cap_tag})")

        # ---- per-frame captions: plain (no-RAG) and RAG-steered (with-RAG)
        caps_plain, caps_rag, policy_texts, hits_list = [], [], [], []
        t_cap: list[float] = []
        for j, s in enumerate(samples):
            bgr  = cv2.imread(str(s["image_path"]))
            hits = retr.hits_for(samples, j)
            t0 = time.time()
            cap_plain = vlm.caption(bgr, base_prompt)
            cap_rag   = vlm.caption(bgr, build_custom_rag_prompt(hits, base_prompt))
            t_cap.append(time.time() - t0)
            caps_plain.append(cap_plain)
            caps_rag.append(cap_rag)
            policy_texts.append(hits_to_policy_text(hits))
            hits_list.append(hits)
            if (j + 1) % 10 == 0:
                print(f"    captioned {j + 1}/{len(samples)} frames")

        caps_plain_arr = np.array(caps_plain, dtype=object)
        caps_rag_arr   = np.array(caps_rag,   dtype=object)

        # ---- predictions
        #  no-RAG  : plain caption -> VLA
        #  with-RAG: RAG caption (custom prompt + closest policy) whose embedding
        #            is score-gated-fused with the closest policy embedding -> VLA
        p_no  = predict_caption_variant(model, tokenizer, samples, caps_plain_arr,
                                        device, num_waypoints=num_waypoints)
        p_rag = predict_rag_fused(model, tokenizer, samples, caps_rag_arr,
                                  hits_list, device, num_waypoints=num_waypoints)

        # ---- aggregate ADE over frames that have GT
        ades_no  = [ade_fde(p_no[k], s["gt_traj"])[0]
                    for k, s in enumerate(samples) if s["gt_traj"] is not None]
        ades_rag = [ade_fde(p_rag[k], s["gt_traj"])[0]
                    for k, s in enumerate(samples) if s["gt_traj"] is not None]
        mean_no  = float(np.mean(ades_no))  if ades_no  else float("nan")
        mean_rag = float(np.mean(ades_rag)) if ades_rag else float("nan")
        print(f"    mean ADE  no-RAG {mean_no:.3f}  |  with-RAG {mean_rag:.3f} m  "
              f"({len(ades_no)} frames w/ GT, caption {np.mean(t_cap)*1e3:.0f} ms/frame x2)")

        # ---- render the video
        out = OUT_DIR / f"gta_{ci:02d}_{clip_id}_{pol_label}_{cap_tag}.mp4"
        render_clip_video(samples, p_no, p_rag, caps_plain, caps_rag,
                          policy_texts, hits_list, pol_label, cap_tag, out, fps=args.fps)
        print(f"    wrote {out.name}")

        all_metrics[clip_id] = [{
            "frame_idx":   s["frame_idx"],
            "is_crash":    s["is_crash"],
            "has_gt":      s["gt_traj"] is not None,
            "caption_no_rag":   caps_plain[k],
            "caption_with_rag": caps_rag[k],
            "policy_text":      policy_texts[k],
            "retrieved":   [{"clip_id": h["clip_id"], "score": h["score"]}
                            for h in hits_list[k]],
            "ade_no_rag":  (ade_fde(p_no[k], s["gt_traj"])[0]
                            if s["gt_traj"] is not None else None),
            "ade_with_rag": (ade_fde(p_rag[k], s["gt_traj"])[0]
                             if s["gt_traj"] is not None else None),
        } for k, s in enumerate(samples)]

    metrics_path = OUT_DIR / f"metrics_gta_with_rag_{pol_label}_{cap_tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    all_recs = [r for recs in all_metrics.values() for r in recs if r["has_gt"]]
    if all_recs:
        print(f"\n=== aggregate ADE  [{pol_label} | {cap_tag}]  "
              f"(n={len(all_recs)} frames w/ GT) ===")
        print(f"  no-RAG   : {np.mean([r['ade_no_rag']   for r in all_recs]):.3f} m")
        print(f"  with-RAG : {np.mean([r['ade_with_rag'] for r in all_recs]):.3f} m")
    print(f"\nall outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
