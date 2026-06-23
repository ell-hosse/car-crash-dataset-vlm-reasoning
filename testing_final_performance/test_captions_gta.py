"""
test_captions_gta.py
--------------------
Same four RAG-strategy comparison as test_captions_using_policies_inside_the_prompt.py
but runs on GTA_Crash_Dataset instead of preprocessed CoVLA samples.

  GT            = red
  no-RAG        = yellow   plain caption → VLA
  policy-prompt = green    policy injected into captioner prompt → caption → VLA
  Fix3-adaptive = purple   plain caption + score-adaptive embedding blend → VLA
  Fix4-uniform  = gold     plain caption + uniform top-k avg + L2 blend → VLA

GTA dataset layout:
  <gta-root>/images/<partition>/*.jpg
  <gta-root>/labels/<partition>/*.json

GTA fields not exposed (steeringAngleDeg, brake, gas, blinkers) are filled with 0.

Usage (from repo root):
    python -m testing_final_performance.test_captions_gta \\
        [--num-clips 10] [--crash-only] [--seed 0] \\
        [--traj-horizon 15] [--caption-interval 10]
"""
import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compute_dist"))

from covla_vla.config import REALTIME                                       # noqa: E402
from covla_vla.dataset import preprocess_image, state_to_vec, denormalize_traj  # noqa: E402
from covla_vla.infer_realtime import load_model, project_traj              # noqa: E402
from clip_retrieval import (build_or_load_policy_index, ClipEmbedder,      # noqa: E402
                            DEFAULT_CLIP_MODEL, DEFAULT_POLICIES, DEFAULT_INDEX)

OUT_DIR = Path(__file__).resolve().parent / "viz_gta_policy_in_prompt"

GTA_PARTITION_NAMES = [
    "GTACrash_accident_part1",
    "GTACrash_accident_part2",
    "GTACrash_accident_part3",
    "GTACrash_nonaccident_part1",
    "GTACrash_nonaccident_part2",
]


# ===========================================================================
#  GTA data loading
# ===========================================================================

def _load_label(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gta_state_dict(label: dict) -> dict:
    """Map GTA JSON annotation to CoVLA state_to_vec-compatible keys.

    Fields not exposed by GTA (steering, brake, gas, blinkers) default to 0.
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


def gta_gt_traj(frames: list, idx: int,
                traj_step: int, num_waypoints: int):
    """Build GT trajectory in ego-local BEV frame from world positions.

    GTA V uses Z-up, so the ground plane is XY.
    Returns (num_waypoints, 2) float32 or None if not enough future frames.
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
    right = np.array([-fwd[1], fwd[0]])

    waypoints = []
    for k in range(1, num_waypoints + 1):
        future = frames[idx + traj_step * k]
        delta  = np.array(future["position"][:2], dtype=np.float64) - pos0
        waypoints.append([float(np.dot(delta, fwd)),
                          float(np.dot(delta, right))])
    return np.array(waypoints, dtype=np.float32)


def load_partition_frames(gta_root: Path, part_name: str) -> list:
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


def split_into_clips(frames: list, boundary_m: float) -> list:
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


def build_gta_samples(clip_frames: list, clip_id: str,
                      traj_step: int, num_waypoints: int) -> list:
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
#  Policy-conditioned captioner prompt
# ===========================================================================

BASE_PROMPT = REALTIME.caption_prompt


def build_policy_prompt(hit) -> str:
    return (
        "You are the perception-and-safety module of a self-driving car."
        " Look carefully at this front-camera frame.\n\n"
        "A retrieval system flagged a crash pattern that may match this scene "
        "(matched by visual similarity to past incidents -- trust your own eyes "
        "over it):\n"
        f"  - Hazard: {hit['trigger']}\n"
        f"  - Why dangerous: {hit['latent_risk']}\n"
        f"  - Safe response: {hit['mitigation']}\n\n"
        "Write ONE caption, under 45 words, in this exact order:\n"
        "1) The single most safety-critical thing visible right now (lead "
        "vehicle, pedestrian, cyclist, signal, or road/weather condition).\n"
        "2) If that hazard -- or the flagged one -- is visible or plausibly "
        "developing, state that THE EGO VEHICLE slows down, increases following "
        "distance, and is ready to brake or yield as appropriate. Describe the "
        "ego actually doing it.\n"
        "3) Brief remaining context only if space allows.\n\n"
        "Rules: describe only what you can see; never invent objects. But when a "
        "hazard is uncertain, choose the more cautious description and the "
        "slower action. Put the hazard and the cautious maneuver FIRST so they "
        "are never cut off."
    )


# ===========================================================================
#  SmolVLM2 captioner
# ===========================================================================

def make_prompt_captioner(device):
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    class PromptVLM:
        def __init__(self):
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(REALTIME.captioner_model)
            self.model = AutoModelForImageTextToText.from_pretrained(
                REALTIME.captioner_model, torch_dtype=dtype).to(device).eval()

        @torch.no_grad()
        def caption(self, bgr, prompt) -> str:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt}]}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(
                    device, dtype=self.model.dtype)
            out = self.model.generate(
                **inputs, max_new_tokens=REALTIME.caption_max_new_tokens,
                do_sample=False)
            text = self.processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0]
            return text.strip()

    return PromptVLM()


# ===========================================================================
#  Fix3 / Fix4 embedding helpers
# ===========================================================================

def _enc(model, tokenizer, text, device):
    tok = tokenizer([text], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    return model.encode_text(tok["input_ids"], tok["attention_mask"])   # (1,1,d)


def _softmax_weights(scores: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    score_range = scores.max() - scores.min()
    normed = (scores - scores.min()) / (score_range + 1e-8)
    exp_s  = np.exp(normed / temperature)
    return exp_s / exp_s.sum()


def _fix3_embed(model, tokenizer, caption, hits, device, temperature=0.1):
    cap_e = _enc(model, tokenizer, caption, device)
    if not hits:
        return cap_e
    scores  = np.array([h["score"] for h in hits], dtype=np.float32)
    weights = _softmax_weights(scores, temperature)
    pol_embs = torch.cat([
        _enc(model, tokenizer,
             f"{h.get('latent_risk','')}. {h.get('mitigation','')}".strip(". "),
             device) for h in hits], dim=0)
    w_t      = torch.tensor(weights, dtype=pol_embs.dtype,
                            device=device).view(-1, 1, 1)
    pol_pool = (w_t * pol_embs).sum(dim=0, keepdim=True)
    gate     = float(scores.max())
    return (1.0 - gate) * cap_e + gate * pol_pool


def _fix4_embed(model, tokenizer, caption, hits, device):
    cap_e = _enc(model, tokenizer, caption, device)
    if not hits:
        return cap_e
    scores   = np.array([h["score"] for h in hits], dtype=np.float32)
    w        = float(scores.max())
    pol_embs = torch.cat([
        _enc(model, tokenizer,
             f"{h.get('latent_risk','')}. {h.get('mitigation','')}".strip(". "),
             device) for h in hits], dim=0)
    pol_mean = pol_embs.mean(dim=0, keepdim=True)
    blended  = (1.0 - w) * cap_e + w * pol_mean
    return blended / (blended.norm(dim=-1, keepdim=True) + 1e-12)


# ===========================================================================
#  VLA inference + metrics
# ===========================================================================

@torch.no_grad()
def predict_traj(model, tokenizer, device, bgr, state_vec, caption):
    img = preprocess_image(bgr).unsqueeze(0).to(device)
    st  = state_vec.unsqueeze(0).to(device)
    tok = tokenizer([caption], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    pred = model(img, st, input_ids=tok["input_ids"],
                 attention_mask=tok["attention_mask"])
    return denormalize_traj(pred[0].float().cpu().numpy())


@torch.no_grad()
def predict_traj_embed(model, device, bgr, state_vec, text_embed):
    img = preprocess_image(bgr).unsqueeze(0).to(device)
    st  = state_vec.unsqueeze(0).to(device)
    pred = model(img, st, text_embed=text_embed)
    return denormalize_traj(pred[0].float().cpu().numpy())


def ade_fde(pred, gt):
    d = np.linalg.norm(pred - gt, axis=-1)
    return float(d.mean()), float(d[-1])


# ===========================================================================
#  Drawing / visualisation
# ===========================================================================

def _wrap(text, width):
    return "\n".join(textwrap.wrap(text or "", width=width)) or "(none)"


def draw_trajs(bgr, gt, pred_base, pred_rag, pred_f3, pred_f4):
    """GT=red, no-RAG=yellow, policy-prompt=green, Fix3=purple, Fix4=gold."""
    out = bgr.copy()
    for traj, color in (
            (gt,        (0,   0,   255)),
            (pred_base, (0,   255, 255)),
            (pred_rag,  (0,   200, 0  )),
            (pred_f3,   (128, 0,   128)),
            (pred_f4,   (0,   215, 255)),
    ):
        if traj is None:
            continue
        pts = project_traj(traj, out.shape)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, color, 2)
        for p in pts:
            cv2.circle(out, tuple(p), 3, color, -1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _fix3_weights(hits, temperature=0.1):
    scores  = np.array([h["score"] for h in hits], dtype=np.float32)
    weights = _softmax_weights(scores, temperature)
    return scores, weights, float(scores.max())


def compose_frame(bgr, gt, pred_base, pred_rag, pred_f3, pred_f4,
                  hits, caption,
                  ade_b, ade_rag, ade_f3, ade_f4,
                  clip_id="", is_crash=False):
    hit  = hits[0] if hits else None
    dist = hit["dist"] if hit else 0.0
    crash_tag = "CRASH" if is_crash else "BENIGN"

    rgb = draw_trajs(bgr, gt, pred_base, pred_rag, pred_f3, pred_f4)
    fig = plt.figure(figsize=(14, 9.5))
    gs  = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.4],
                           width_ratios=[1.0, 2.2], hspace=0.10, wspace=0.04)
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(rgb)
    ax_img.axis("off")
    title = (f"[{crash_tag}] {clip_id}   "
             "GT=red  |  no-RAG=yellow  |  policy-prompt=green  |  "
             "Fix3-adaptive=purple  |  Fix4-uniform=gold")
    if hit is not None:
        title += (f"\n[policy {hit['clip_id']} dist={dist:.3f}]   "
                  f"ADE  no-RAG {ade_b:.2f}  prompt {ade_rag:.2f}  "
                  f"Fix3 {ade_f3:.2f}  Fix4 {ade_f4:.2f} m")
    ax_img.set_title(title, fontsize=9.5, color="#333", fontweight="bold")

    if hits:
        scores, weights, gate = _fix3_weights(hits)
        lines = [f"Fix3 weights  (gate={gate:.3f})"]
        for i, (h, s, w) in enumerate(zip(hits, scores, weights)):
            lines.append(f"  [{i+1}] {h['clip_id']}  score={s:.3f}  w={w:.3f}")
        ax_img.text(0.01, 0.03, "\n".join(lines),
                    transform=ax_img.transAxes,
                    fontsize=7.5, family="monospace", color="white",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black",
                              alpha=0.55, ec="none"))

    ax_bev = fig.add_subplot(gs[1, 0])
    for arr, lbl, col in (
            (gt,        "GT",            "red"),
            (pred_base, "no-RAG",        "gold"),
            (pred_rag,  "policy-prompt", "green"),
            (pred_f3,   "Fix3-adaptive", "purple"),
            (pred_f4,   "Fix4-uniform",  "darkorange"),
    ):
        if arr is not None:
            ax_bev.plot(-arr[:, 1], arr[:, 0], "o-", color=col,
                        ms=3, lw=1.5, label=lbl)
    ax_bev.scatter([0], [0], marker="^", s=70, color="black", zorder=5, label="ego")
    ax_bev.set_xlabel("lateral (m, right +)", fontsize=8)
    ax_bev.set_ylabel("forward (m)", fontsize=8)
    ax_bev.set_title("BEV", fontsize=9)
    ax_bev.axis("equal")
    ax_bev.grid(alpha=0.3)
    ax_bev.legend(fontsize=7)

    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")
    pol_block = _wrap(
        f"When {hit['trigger']}  ->  risk: {hit['latent_risk']}  "
        f"mitigation: {hit['mitigation']}", 90) if hit else "(none)"
    ax_txt.text(0.0, 1.0,
                f"TOP-1 POLICY (injected into prompt / used by Fix3+Fix4):\n{pol_block}\n\n"
                f"POLICY-CONDITIONED CAPTION (used by green):\n{_wrap(caption, 90)}",
                fontsize=9, va="top", ha="left", family="monospace",
                transform=ax_txt.transAxes)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.05)
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return out


# ===========================================================================
#  Per-clip processing
# ===========================================================================

def process_clip(clip_id, samples, clip_emb, matcher, captioner,
                 model, tokenizer, device, args, num_waypoints):
    is_crash  = samples[0]["is_crash"]
    group     = "crash" if is_crash else "benign"
    out_group = OUT_DIR / group
    out_group.mkdir(parents=True, exist_ok=True)
    out_path  = out_group / f"{clip_id}.mp4"

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"  [{group}] {clip_id}: {len(samples)} frames, "
          f"refresh every {args.caption_interval} frames")

    cur_hits         = None
    cur_hit          = None
    cur_caption      = None
    cur_base_caption = None
    writer  = None
    records = []
    ades_b, ades_rag, ades_f3, ades_f4, ades_f3h, ades_f4h = [], [], [], [], [], []
    n = 0

    for j, s in enumerate(samples):
        bgr = cv2.imread(str(s["image_path"]))
        if bgr is None:
            print(f"    WARNING missing frame {s['image_path']}, skipping")
            continue
        # Flip horizontally so the VLA (trained on left-hand CoVLA traffic) sees
        # a mirrored scene that matches its training distribution.
        bgr_vla   = cv2.flip(bgr, 1)
        rgb       = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt        = s["gt_traj"][:num_waypoints]
        state_vec = state_to_vec(s["state"])

        if (j % args.caption_interval == 0) or (cur_hits is None):
            img_emb          = clip_emb.embed_image_arrays([rgb])[0]
            cur_hits         = matcher.retrieve(img_emb, top_k=args.top_k)
            cur_hit          = cur_hits[0]
            cur_caption      = captioner.caption(bgr_vla, build_policy_prompt(cur_hit))
            cur_base_caption = captioner.caption(bgr_vla, BASE_PROMPT)

        pred_b   = predict_traj(model, tokenizer, device, bgr_vla, state_vec,
                                cur_base_caption)
        pred_rag = predict_traj(model, tokenizer, device, bgr_vla, state_vec,
                                cur_caption)
        with torch.no_grad():
            pred_f3 = predict_traj_embed(
                model, device, bgr_vla, state_vec,
                _fix3_embed(model, tokenizer, cur_base_caption, cur_hits, device))
            pred_f4 = predict_traj_embed(
                model, device, bgr_vla, state_vec,
                _fix4_embed(model, tokenizer, cur_base_caption, cur_hits, device))
            # Hybrid: use policy-conditioned caption as the base for Fix3/Fix4
            pred_f3h = predict_traj_embed(
                model, device, bgr_vla, state_vec,
                _fix3_embed(model, tokenizer, cur_caption, cur_hits, device))
            pred_f4h = predict_traj_embed(
                model, device, bgr_vla, state_vec,
                _fix4_embed(model, tokenizer, cur_caption, cur_hits, device))

        # Trim predicted trajectories to match GT length, then negate lateral
        # to convert from the flipped coordinate frame back to GTA's frame.
        pred_b    = pred_b[:num_waypoints];    pred_b[:, 1]    *= -1
        pred_rag  = pred_rag[:num_waypoints];  pred_rag[:, 1]  *= -1
        pred_f3   = pred_f3[:num_waypoints];   pred_f3[:, 1]   *= -1
        pred_f4   = pred_f4[:num_waypoints];   pred_f4[:, 1]   *= -1
        pred_f3h  = pred_f3h[:num_waypoints];  pred_f3h[:, 1]  *= -1
        pred_f4h  = pred_f4h[:num_waypoints];  pred_f4h[:, 1]  *= -1

        ade_b,    _ = ade_fde(pred_b,    gt)
        ade_rag,  _ = ade_fde(pred_rag,  gt)
        ade_f3,   _ = ade_fde(pred_f3,   gt)
        ade_f4,   _ = ade_fde(pred_f4,   gt)
        ade_f3h,  _ = ade_fde(pred_f3h,  gt)
        ade_f4h,  _ = ade_fde(pred_f4h,  gt)
        ades_b.append(ade_b);    ades_rag.append(ade_rag)
        ades_f3.append(ade_f3);  ades_f4.append(ade_f4)
        ades_f3h.append(ade_f3h); ades_f4h.append(ade_f4h)

        if j % args.caption_interval == 0:
            print(f"    f{s['frame_idx']:>4} | policy[{cur_hit['clip_id']}] "
                  f"dist={cur_hit['dist']:.3f} | ADE  no-RAG {ade_b:.2f}  "
                  f"prompt {ade_rag:.2f}  F3 {ade_f3:.2f}  F4 {ade_f4:.2f}  "
                  f"F3h {ade_f3h:.2f}  F4h {ade_f4h:.2f}")
            records.append({
                "clip_id": clip_id, "group": group, "is_crash": is_crash,
                "frame_idx": s["frame_idx"],
                "frame_to_policy_dist":  cur_hit["dist"],
                "frame_to_policy_sim":   cur_hit["score"],
                "injected_policy": {
                    "clip_id":      cur_hit["clip_id"],
                    "trigger":      cur_hit["trigger"],
                    "latent_risk":  cur_hit["latent_risk"],
                    "mitigation":   cur_hit["mitigation"],
                },
                "policy_caption":    cur_caption,
                "base_caption":      cur_base_caption,
                "ade_no_rag":           ade_b,
                "ade_policy_prompt":    ade_rag,
                "ade_fix3_adaptive":    ade_f3,
                "ade_fix4_uniform":     ade_f4,
                "ade_fix3_hybrid":      ade_f3h,
                "ade_fix4_hybrid":      ade_f4h,
            })

        frame_rgb = compose_frame(
            bgr, gt, pred_b, pred_rag, pred_f3, pred_f4,
            cur_hits, cur_caption,
            ade_b, ade_rag, ade_f3, ade_f4,
            clip_id=clip_id, is_crash=is_crash)
        if writer is None:
            h, w = frame_rgb.shape[:2]
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps_out, (w, h))
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        n += 1

    if writer:
        writer.release()
    _m = lambda a: round(float(np.mean(a)), 3) if a else None
    print(f"    -> wrote {out_path.name} ({n} frames) | mean ADE  "
          f"no-RAG {_m(ades_b)}  prompt {_m(ades_rag)}  "
          f"F3 {_m(ades_f3)}  F4 {_m(ades_f4)}  "
          f"F3h {_m(ades_f3h)}  F4h {_m(ades_f4h)} m")
    return {
        "clip_id": clip_id, "group": group, "is_crash": is_crash,
        "n_frames": n,
        "mean_ade_no_rag":        _m(ades_b),
        "mean_ade_policy_prompt": _m(ades_rag),
        "mean_ade_fix3_adaptive": _m(ades_f3),
        "mean_ade_fix4_uniform":  _m(ades_f4),
        "mean_ade_fix3_hybrid":   _m(ades_f3h),
        "mean_ade_fix4_hybrid":   _m(ades_f4h),
        "records": records,
    }


# ===========================================================================
#  Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt",             default=str(REPO_ROOT / "rag" / "covla_vla_best.pt"))
    ap.add_argument("--gta-root",         default=str(REPO_ROOT / "GTA_Crash_Dataset"))
    ap.add_argument("--policies",         default=str(DEFAULT_POLICIES))
    ap.add_argument("--index",            default=str(DEFAULT_INDEX))
    ap.add_argument("--clip-model",       default=DEFAULT_CLIP_MODEL)
    ap.add_argument("--rebuild-index",    action="store_true")
    ap.add_argument("--num-clips",        type=int,   default=10)
    ap.add_argument("--seed",             type=int,   default=0)
    ap.add_argument("--crash-only",       action="store_true",
                    help="Only use accident partitions")
    ap.add_argument("--top-k",            type=int,   default=3)
    ap.add_argument("--traj-step",        type=int,   default=1)
    ap.add_argument("--traj-horizon",     type=int,   default=15,
                    help="Lookahead frame count; num_waypoints = horizon // step")
    ap.add_argument("--clip-boundary",    type=float, default=50.0,
                    help="Position jump (m) that marks a new GTA clip")
    ap.add_argument("--caption-interval", type=int,   default=10,
                    help="Frames between caption + policy retrieval refreshes")
    ap.add_argument("--max-samples",      type=int,   default=None,
                    help="Truncate each clip to this many frames")
    ap.add_argument("--fps-out",          type=float, default=4.0)
    args = ap.parse_args()

    num_waypoints = args.traj_horizon // args.traj_step
    gta_root      = Path(args.gta_root)
    rng           = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device         : {device}")
    print(f"gta root       : {gta_root}")
    print(f"num_waypoints  : {num_waypoints}  "
          f"(traj_step={args.traj_step}, traj_horizon={args.traj_horizon})")

    clip_emb = ClipEmbedder(args.clip_model, device=device)
    matcher  = build_or_load_policy_index(
        Path(args.policies), Path(args.index), args.clip_model,
        embedder=clip_emb, rebuild=args.rebuild_index)
    print(f"policy matcher : {len(matcher)} policies")
    captioner        = make_prompt_captioner(device)
    model, tokenizer = load_model(args.ckpt, device)

    # Build clip list
    partition_names = [n for n in GTA_PARTITION_NAMES
                       if not args.crash_only or "accident" in n]
    all_clips: list = []
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for clip_id, samples in chosen:
        print(f"\nclip {clip_id}")
        summaries.append(
            process_clip(clip_id, samples, clip_emb, matcher, captioner,
                         model, tokenizer, device, args, num_waypoints))

    out_json = OUT_DIR / "policy_in_prompt_summary_gta.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "num_clips":         len(chosen),
            "top_k":             args.top_k,
            "caption_interval":  args.caption_interval,
            "traj_horizon":      args.traj_horizon,
            "num_waypoints":     num_waypoints,
            "retrieval_route":   "frame_image_to_policy",
            "trajectories": {
                "GT":            "red",
                "no_RAG":        "yellow",
                "policy_prompt": "green",
                "Fix3_adaptive": "purple",
                "Fix4_uniform":  "gold",
            },
            "data_source": "GTA_Crash_Dataset",
            "clips": summaries,
        }, f, indent=2)

    all_records = [r for sv in summaries for r in sv.get("records", [])]
    crash_records  = [r for r in all_records if r["is_crash"]]
    benign_records = [r for r in all_records if not r["is_crash"]]

    def _agg(recs, key):
        vals = [r[key] for r in recs if r.get(key) is not None]
        if not vals:
            return None, None
        a = np.array(vals, dtype=np.float64)
        return round(float(a.mean()), 3), round(float(a.std()), 3)

    def _print_table(label, recs):
        if not recs:
            print(f"  ({label}: no samples)")
            return
        print(f"\n=== aggregate ADE  [{label}]  (n={len(recs)}) ===")
        print(f"  {'strategy':<28}  {'mean':>6}  {'std':>6}")
        print(f"  {'-'*28}  {'-'*6}  {'-'*6}")
        for name, key in [
            ("no-RAG (baseline)",     "ade_no_rag"),
            ("policy-prompt",         "ade_policy_prompt"),
            ("Fix3-adaptive",         "ade_fix3_adaptive"),
            ("Fix4-uniform",          "ade_fix4_uniform"),
            ("Fix3-hybrid",           "ade_fix3_hybrid"),
            ("Fix4-hybrid",           "ade_fix4_hybrid"),
        ]:
            mean, std = _agg(recs, key)
            print(f"  {name:<28}  {mean:>6.3f}  {std:>6.3f} m")

    _print_table("ALL",    all_records)
    _print_table("CRASH",  crash_records)
    _print_table("BENIGN", benign_records)
    print(f"\nall outputs in {OUT_DIR}\nsummary -> {out_json}")


if __name__ == "__main__":
    main()
