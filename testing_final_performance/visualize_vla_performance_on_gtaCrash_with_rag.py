"""Visualize VLA performance with SIX RAG strategies on GTA_Crash_Dataset (video).

Per clip this renders an MP4 stepping through every frame. Each frame shows:
  * the camera image with trajectories overlaid:
      - GT             (red)       real future ego path in the ego-local BEV frame
      - no-RAG         (yellow)    VLA conditioned on a plain caption
      - policy-prompt  (green)     VLA conditioned on a policy-injected caption
      - Fix3-adaptive  (purple)    plain caption + softmax-weighted top-k policy
                                   embedding blend → VLA
      - Fix4-uniform   (gold)      plain caption + uniform top-k avg + L2-norm
                                   embedding blend → VLA
      - Fix3-hybrid    (cyan)      policy-prompt caption + Fix3 embedding blend
      - Fix4-hybrid    (orange)    policy-prompt caption + Fix4 embedding blend
  * a BEV plot of the same trajectories with ADE metrics
  * both captions (plain and policy-conditioned) and retrieved policy text

Six RAG strategies:
  1. no-RAG         plain caption tokenised → VLA
  2. policy-prompt  top-1 policy fills {trigger}/{latent_risk}/{mitigation} in the
                    RAG_CAPTION_PROMPT; that caption is tokenised → VLA
  3. Fix3-adaptive  plain caption embedding score-gated-fused with softmax-weighted
                    top-k policy embeddings
  4. Fix4-uniform   plain caption embedding blended with uniform-mean top-k policy
                    embeddings then L2-normalised
  5. Fix3-hybrid    same as Fix3 but uses the policy-prompt caption as the text base
  6. Fix4-hybrid    same as Fix4 but uses the policy-prompt caption as the text base

The captioner is selectable with --captioner:
     smolvlm2   -> HuggingFaceTB/SmolVLM2-256M-Video-Instruct  (tiny, fast)
     qwen2.5-vl -> Qwen/Qwen2.5-VL-3B-Instruct                 (free, stronger)
     qwen2-vl   -> Qwen/Qwen2-VL-2B-Instruct
--max-pixels caps the per-frame visual-token count for the Qwen captioners.

Ground truth is NOT fabricated: it is the car's real future world-positions
projected into the ego frame. Near the end of a clip fewer future frames exist,
so the GT horizon simply shrinks (and the very last frame, with no future, has
no GT); predictions and captions are still drawn for those frames.

Frames are horizontally flipped before being passed to the VLA to match the
CoVLA training distribution (left-hand traffic). The lateral component of all
predicted waypoints is negated afterwards to restore GTA's coordinate frame.

Usage (from repo root):
    python -m testing_final_performance.visualize_vla_performance_on_gtaCrash_with_rag \\
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

# RAG captioner prompt: the top-1 retrieved crash policy is injected into the
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
    """Fill RAG_CAPTION_PROMPT with the closest retrieved policy (top-1 hit).

    Falls back to the plain base prompt when nothing is retrieved.
    """
    if not hits:
        return base_prompt
    h = hits[0]
    return (RAG_CAPTION_PROMPT
            .replace("{trigger}",     str(h.get("trigger", "")).strip())
            .replace("{latent_risk}", str(h.get("latent_risk", "")).strip())
            .replace("{mitigation}",  str(h.get("mitigation", "")).strip()))


GTA_PARTITION_NAMES = [
    "GTACrash_accident_part1",
    "GTACrash_accident_part2",
    "GTACrash_accident_part3",
    "GTACrash_nonaccident_part1",
    "GTACrash_nonaccident_part2",
]

# BGR colours for OpenCV overlays
COL_GT      = (0,   0,   255)   # red
COL_NORAG   = (0,   255, 255)   # yellow
COL_PROMPT  = (0,   200, 0  )   # green
COL_FIX3    = (128, 0,   128)   # purple
COL_FIX4    = (0,   215, 255)   # gold
COL_FIX3H   = (255, 200, 0  )   # cyan
COL_FIX4H   = (0,   128, 255)   # orange

# Matplotlib colours
MPL_COLS = {
    "GT":            "red",
    "no-RAG":        "gold",
    "policy-prompt": "green",
    "Fix3-adaptive": "purple",
    "Fix4-uniform":  "darkorange",
    "Fix3-hybrid":   "cyan",
    "Fix4-hybrid":   "orange",
}


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
#  Embedding helpers: Fix3 (softmax-weighted) and Fix4 (uniform + L2)
# ===========================================================================

def _enc(model, tokenizer, text, device):
    """Pooled caption/policy embedding -> (1, 1, d)."""
    tok = tokenizer([text], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    return model.encode_text(tok["input_ids"], tok["attention_mask"])


def _softmax_weights(scores: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    score_range = scores.max() - scores.min()
    normed = (scores - scores.min()) / (score_range + 1e-8)
    exp_s  = np.exp(normed / temperature)
    return exp_s / exp_s.sum()


def _fix3_embed(model, tokenizer, caption, hits, device, temperature=0.1):
    """Score-adaptive softmax-weighted top-k policy embedding fusion.

    fused = (1 - gate) * cap_e + gate * softmax_weighted_pol_pool
    where gate = max retrieval score.
    """
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
    """Uniform-mean top-k policy embedding blend, L2-normalised.

    blended = (1 - w) * cap_e + w * mean(pol_embs),  then L2-normed
    where w = max retrieval score.
    """
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

def ade_fde(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """ADE / FDE in metres over the overlapping horizon of pred and gt."""
    L = min(len(pred), len(gt))
    d = np.linalg.norm(pred[:L] - gt[:L], axis=-1)
    return float(d.mean()), float(d[-1])


@torch.no_grad()
def predict_traj(model, tokenizer, device, bgr_vla: np.ndarray,
                 state_vec, caption: str, num_waypoints: int) -> np.ndarray:
    """Tokenised-caption path (no-RAG or policy-prompt). Returns (T, 2) m."""
    img = preprocess_image(bgr_vla).unsqueeze(0).to(device)
    st  = state_vec.unsqueeze(0).to(device)
    tok = tokenizer([caption], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    pred = model(img, st, input_ids=tok["input_ids"],
                 attention_mask=tok["attention_mask"])
    out = denormalize_traj(pred[0].float().cpu().numpy())
    return out[:num_waypoints]


@torch.no_grad()
def predict_traj_embed(model, device, bgr_vla: np.ndarray,
                       state_vec, text_embed, num_waypoints: int) -> np.ndarray:
    """Pre-computed embedding path (Fix3, Fix4). Returns (T, 2) m."""
    img = preprocess_image(bgr_vla).unsqueeze(0).to(device)
    st  = state_vec.unsqueeze(0).to(device)
    pred = model(img, st, text_embed=text_embed)
    out = denormalize_traj(pred[0].float().cpu().numpy())
    return out[:num_waypoints]


# ===========================================================================
#  Visualisation
# ===========================================================================

def draw_overlay(bgr_orig, gt, p_norag, p_prompt, p_f3, p_f4, p_f3h, p_f4h):
    """Camera frame (original, unflipped) with all trajectory overlays."""
    out = bgr_orig.copy()
    layers = [
        (p_norag,  COL_NORAG),
        (p_prompt, COL_PROMPT),
        (p_f3,     COL_FIX3),
        (p_f4,     COL_FIX4),
        (p_f3h,    COL_FIX3H),
        (p_f4h,    COL_FIX4H),
    ]
    if gt is not None:
        layers = [(gt, COL_GT)] + layers
    for traj, col in layers:
        if traj is None:
            continue
        pts = project_traj(traj, out.shape)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, col, 2)
        for p in pts:
            cv2.circle(out, tuple(p), 3, col, -1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def plot_bev(ax, gt, p_norag, p_prompt, p_f3, p_f4, p_f3h, p_f4h, title=""):
    series = [
        (p_norag,  "no-RAG"),
        (p_prompt, "policy-prompt"),
        (p_f3,     "Fix3-adaptive"),
        (p_f4,     "Fix4-uniform"),
        (p_f3h,    "Fix3-hybrid"),
        (p_f4h,    "Fix4-hybrid"),
    ]
    if gt is not None:
        series = [(gt, "GT")] + series
    for arr, lbl in series:
        if arr is None:
            continue
        ax.plot(-arr[:, 1], arr[:, 0], "o-", color=MPL_COLS[lbl],
                ms=3, lw=1.5, label=lbl)
    ax.scatter([0], [0], marker="^", s=70, color="black", zorder=5, label="ego")
    ax.set_xlabel("lateral (m, right +)"); ax.set_ylabel("forward (m)")
    ax.set_title(title, fontsize=7); ax.axis("equal")
    ax.grid(alpha=0.3); ax.legend(fontsize=6)


def _wrap(label, text, width=80, max_lines=3):
    lines = textwrap.wrap(text or "", width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]; lines[-1] += " ..."
    return label + ("\n" + " " * len(label)).join(lines)


def render_clip_video(samples, trajs, caps_plain, caps_rag,
                      policy_texts, hits_list, policy_label, captioner_label,
                      out_path, fps=4):
    """One MP4 covering every frame of the clip.

    trajs is a dict with keys: no_rag, prompt, f3, f4, f3h, f4h —
    each a list of (T,2) arrays.
    """
    writer = None
    for k, s in enumerate(samples):
        gt       = s["gt_traj"]
        p_norag  = trajs["no_rag"][k]
        p_prompt = trajs["prompt"][k]
        p_f3     = trajs["f3"][k]
        p_f4     = trajs["f4"][k]
        p_f3h    = trajs["f3h"][k]
        p_f4h    = trajs["f4h"][k]

        if gt is not None:
            an, _  = ade_fde(p_norag,  gt)
            ar, _  = ade_fde(p_prompt, gt)
            a3, _  = ade_fde(p_f3,     gt)
            a4, _  = ade_fde(p_f4,     gt)
            a3h, _ = ade_fde(p_f3h,    gt)
            a4h, _ = ade_fde(p_f4h,    gt)
            metric_line = (
                f"ADE  no-RAG {an:.2f}  prompt {ar:.2f}  "
                f"Fix3 {a3:.2f}  Fix4 {a4:.2f}  "
                f"Fix3h {a3h:.2f}  Fix4h {a4h:.2f} m"
            )
        else:
            metric_line = "ADE/FDE  n/a (no future frames for GT)"

        top_score = hits_list[k][0]["score"] if hits_list[k] else 0.0
        crash_tag = "CRASH" if s["is_crash"] else "BENIGN"

        fig = plt.figure(figsize=(15, 10))
        gs  = fig.add_gridspec(4, 2, height_ratios=[2.8, 0.7, 0.7, 0.7],
                               width_ratios=[1.7, 1])
        ax_img = fig.add_subplot(gs[0, 0])
        ax_bev = fig.add_subplot(gs[0, 1])
        ax_t1  = fig.add_subplot(gs[1, :])
        ax_t2  = fig.add_subplot(gs[2, :])
        ax_t3  = fig.add_subplot(gs[3, :])

        bgr_orig = cv2.imread(str(s["image_path"]))
        ax_img.imshow(draw_overlay(bgr_orig, gt, p_norag, p_prompt,
                                   p_f3, p_f4, p_f3h, p_f4h))
        ax_img.set_title(
            f"{s['clip_id']}  frame {s['frame_idx']}/{len(samples)-1}  "
            f"[{crash_tag}]  policies: {policy_label}  captioner: {captioner_label}  "
            f"top-score={top_score:.3f}\n"
            "GT=red  no-RAG=yellow  policy-prompt=green  "
            "Fix3=purple  Fix4=gold  Fix3h=cyan  Fix4h=orange", fontsize=7.5)
        ax_img.axis("off")

        plot_bev(ax_bev, gt, p_norag, p_prompt, p_f3, p_f4, p_f3h, p_f4h,
                 metric_line)

        ids = ", ".join(f"{h['clip_id']}({h['score']:.2f})" for h in hits_list[k])
        ax_t1.axis("off")
        ax_t1.text(0, 1,
            _wrap("Caption (no-RAG): ",         caps_plain[k]) + "\n" +
            _wrap("Caption (policy-prompt): ",   caps_rag[k]),
            transform=ax_t1.transAxes, fontsize=7.5, family="monospace",
            va="top", ha="left")
        ax_t2.axis("off")
        ax_t2.text(0, 1,
            _wrap("Retrieved policies: ", policy_texts[k]),
            transform=ax_t2.transAxes, fontsize=7.5, family="monospace",
            va="top", ha="left")
        ax_t3.axis("off")
        ax_t3.text(0, 1,
            f"Retrieved ids: {ids}",
            transform=ax_t3.transAxes, fontsize=7.5, family="monospace",
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
        description="Render four-strategy RAG trajectory videos on GTA crash clips.")
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
    ap.add_argument("--caption-interval", type=int,   default=1,
                    help="Frames between caption + retrieval refreshes (1 = every frame)")
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
        print(f"[{ci}] {clip_id} [{crash_tag}]: {len(samples)} frames  "
              f"(caption every {args.caption_interval} frame(s) x2)")

        # ---- per-frame captions and retrieval hits
        caps_plain, caps_rag, policy_texts, hits_list = [], [], [], []
        cur_cap_plain: str | None = None
        cur_cap_rag:   str | None = None
        t_cap: list[float] = []

        for j, s in enumerate(samples):
            bgr     = cv2.imread(str(s["image_path"]))
            bgr_vla = cv2.flip(bgr, 1)          # flip to match CoVLA training distribution
            hits    = retr.hits_for(samples, j)

            if (j % args.caption_interval == 0) or (cur_cap_plain is None):
                t0 = time.time()
                cur_cap_plain = vlm.caption(bgr_vla, base_prompt)
                cur_cap_rag   = vlm.caption(bgr_vla,
                                            build_custom_rag_prompt(hits, base_prompt))
                t_cap.append(time.time() - t0)

            caps_plain.append(cur_cap_plain)
            caps_rag.append(cur_cap_rag)
            policy_texts.append(hits_to_policy_text(hits))
            hits_list.append(hits)

            if (j + 1) % 10 == 0:
                print(f"    captioned {j + 1}/{len(samples)} frames")

        # ---- per-frame predictions — six strategies
        trajs: dict[str, list] = {
            "no_rag": [], "prompt": [], "f3": [], "f4": [], "f3h": [], "f4h": []
        }

        for k, s in enumerate(samples):
            bgr     = cv2.imread(str(s["image_path"]))
            bgr_vla = cv2.flip(bgr, 1)
            sv      = state_to_vec(s["state"])
            hits    = hits_list[k]
            cap_plain = caps_plain[k]
            cap_rag   = caps_rag[k]

            with torch.no_grad():
                p_norag  = predict_traj(model, tokenizer, device, bgr_vla, sv,
                                        cap_plain, num_waypoints)
                p_prompt = predict_traj(model, tokenizer, device, bgr_vla, sv,
                                        cap_rag, num_waypoints)
                p_f3     = predict_traj_embed(
                    model, device, bgr_vla, sv,
                    _fix3_embed(model, tokenizer, cap_plain, hits, device),
                    num_waypoints)
                p_f4     = predict_traj_embed(
                    model, device, bgr_vla, sv,
                    _fix4_embed(model, tokenizer, cap_plain, hits, device),
                    num_waypoints)
                # Hybrid: use policy-prompt caption as the text base for Fix3/Fix4
                p_f3h    = predict_traj_embed(
                    model, device, bgr_vla, sv,
                    _fix3_embed(model, tokenizer, cap_rag, hits, device),
                    num_waypoints)
                p_f4h    = predict_traj_embed(
                    model, device, bgr_vla, sv,
                    _fix4_embed(model, tokenizer, cap_rag, hits, device),
                    num_waypoints)

            # Negate lateral to undo the horizontal flip (restore GTA frame)
            for p in (p_norag, p_prompt, p_f3, p_f4, p_f3h, p_f4h):
                p[:, 1] *= -1

            trajs["no_rag"].append(p_norag)
            trajs["prompt"].append(p_prompt)
            trajs["f3"].append(p_f3)
            trajs["f4"].append(p_f4)
            trajs["f3h"].append(p_f3h)
            trajs["f4h"].append(p_f4h)

        # ---- aggregate ADE over frames that have GT
        def _mean_ade(preds):
            vals = [ade_fde(preds[k], s["gt_traj"])[0]
                    for k, s in enumerate(samples) if s["gt_traj"] is not None]
            return float(np.mean(vals)) if vals else float("nan")

        n_gt    = sum(1 for s in samples if s["gt_traj"] is not None)
        cap_ms  = np.mean(t_cap) * 1e3 if t_cap else 0.0
        print(f"    mean ADE  "
              f"no-RAG {_mean_ade(trajs['no_rag']):.3f}  "
              f"prompt {_mean_ade(trajs['prompt']):.3f}  "
              f"Fix3 {_mean_ade(trajs['f3']):.3f}  "
              f"Fix4 {_mean_ade(trajs['f4']):.3f}  "
              f"Fix3h {_mean_ade(trajs['f3h']):.3f}  "
              f"Fix4h {_mean_ade(trajs['f4h']):.3f} m  "
              f"({n_gt} frames w/ GT, {cap_ms:.0f} ms/caption x2)")

        # ---- render the video
        out = OUT_DIR / f"gta_{ci:02d}_{clip_id}_{pol_label}_{cap_tag}.mp4"
        render_clip_video(samples, trajs, caps_plain, caps_rag,
                          policy_texts, hits_list, pol_label, cap_tag, out,
                          fps=args.fps)
        print(f"    wrote {out.name}")

        all_metrics[clip_id] = [{
            "frame_idx":             s["frame_idx"],
            "is_crash":              s["is_crash"],
            "has_gt":                s["gt_traj"] is not None,
            "caption_no_rag":        caps_plain[k],
            "caption_policy_prompt": caps_rag[k],
            "policy_text":           policy_texts[k],
            "retrieved": [{"clip_id": h["clip_id"], "score": h["score"]}
                          for h in hits_list[k]],
            "ade_no_rag":          (ade_fde(trajs["no_rag"][k], s["gt_traj"])[0]
                                    if s["gt_traj"] is not None else None),
            "ade_policy_prompt":   (ade_fde(trajs["prompt"][k], s["gt_traj"])[0]
                                    if s["gt_traj"] is not None else None),
            "ade_fix3_adaptive":   (ade_fde(trajs["f3"][k], s["gt_traj"])[0]
                                    if s["gt_traj"] is not None else None),
            "ade_fix4_uniform":    (ade_fde(trajs["f4"][k], s["gt_traj"])[0]
                                    if s["gt_traj"] is not None else None),
            "ade_fix3_hybrid":     (ade_fde(trajs["f3h"][k], s["gt_traj"])[0]
                                    if s["gt_traj"] is not None else None),
            "ade_fix4_hybrid":     (ade_fde(trajs["f4h"][k], s["gt_traj"])[0]
                                    if s["gt_traj"] is not None else None),
        } for k, s in enumerate(samples)]

    metrics_path = OUT_DIR / f"metrics_gta_with_rag_{pol_label}_{cap_tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    all_recs = [r for recs in all_metrics.values() for r in recs if r["has_gt"]]
    if all_recs:
        def _agg(key):
            vals = [r[key] for r in all_recs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        print(f"\n=== aggregate ADE  [{pol_label} | {cap_tag}]  "
              f"(n={len(all_recs)} frames w/ GT) ===")
        for name, key in [
            ("no-RAG",        "ade_no_rag"),
            ("policy-prompt", "ade_policy_prompt"),
            ("Fix3-adaptive", "ade_fix3_adaptive"),
            ("Fix4-uniform",  "ade_fix4_uniform"),
            ("Fix3-hybrid",   "ade_fix3_hybrid"),
            ("Fix4-hybrid",   "ade_fix4_hybrid"),
        ]:
            print(f"  {name:<18}: {_agg(key):.3f} m")
    print(f"\nall outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
