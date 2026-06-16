"""Visualize VLA trajectory performance AFTER RAG on the CoVLA test split.

REAL-TIME pipeline, NO ground-truth captions anywhere. For each frame:

  1. CLIP retrieval : the last X frames (X matched to the caption cadence) are
                      CLIP-embedded and AVERAGED into one scene vector, matched
                      against the ~2k pre-embedded crash policies -> top-k.
  2. Captioning     : SmolVLM2 generates the scene caption from the frame.
                        * WITHOUT RAG : plain prompt (VLM only)
                        * WITH    RAG : the same prompt CONDITIONED on the top-k
                                        retrieved policies (RAG beside the VLM)
  3. Trajectory     : DINOv2 vision over the frame + the generated caption ->
                      3 s trajectory (run for each caption variant).

Captions refresh every REALTIME.caption_interval_s and are reused until the next
refresh - exactly like the async real-time captioner, where the caption lags
while the trajectory model runs every frame.

Outputs (in testing_final_performance/viz_covla_after_rag/):
  * sample_<k>_<video>_<frame>.mp4   - 10 videos (one per random test sample).
      Each frame shows the camera image with THREE trajectories
        red   = ground-truth
        green = predicted from the VLM-only caption (no RAG)
        blue  = predicted from the VLM+RAG caption
      a BEV plot (ADE no-RAG vs RAG), and BOTH captions printed on screen so you
      can see what the VLM wrote with and without RAG.
  * captions_after_rag.json          - per-frame captions (no-RAG / with-RAG),
      retrieved policy ids/scores and ADE/FDE, for inspection.

Usage (machine with D:/hf data + checkpoint + a GPU for SmolVLM2):
    python -m testing_final_performance.visualize_vla_performance_on_CoVLA_after_rag \
        [--ckpt covla_vla_best.pt] [--num-videos 10] [--seed 0]
        [--top-k 5] [--max-samples N] [--fps 4]
"""
import argparse
import json
import random
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_repo_root(start: Path) -> Path:
    for d in (start, *start.parents):
        if (d / "covla_vla").is_dir() and (d / "crash_policies.jsonl").exists():
            return d
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compute_dist"))

from covla_vla.config import DATA, REALTIME, PREPROCESSED_ROOT           # noqa: E402
from covla_vla.dataset import preprocess_image, state_to_vec, denormalize_traj  # noqa: E402
from covla_vla.infer_realtime import load_model, project_traj           # noqa: E402

# reuse the geometry helpers (NOT captions) from the before-RAG script
from testing_final_performance.visualize_vla_performance_on_CoVLA_before_rag import (  # noqa: E402
    gt_traj, ade_fde)

from clip_retrieval import (build_or_load_policy_index, ClipEmbedder,    # noqa: E402
                            pool_clip_video_embedding, build_vlm_rag_prompt,
                            DEFAULT_CLIP_MODEL, DEFAULT_POLICIES, DEFAULT_INDEX)

OUT_DIR = Path(__file__).resolve().parent / "viz_covla_after_rag"

COL_GT = (0, 0, 255)       # red    (BGR)
COL_NORAG = (0, 255, 0)    # green
COL_RAG = (255, 128, 0)    # blue   (BGR)


# ============================================================ SmolVLM2 captioner
class VLMCaptioner:
    """Synchronous, prompt-controllable SmolVLM2 captioner (same model the
    real-time loop uses). Synchronous so the with/with-out-RAG prompts produce
    deterministic, directly comparable captions for the video."""

    def __init__(self, device, cfg=REALTIME):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.cfg = cfg
        self.device = device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(cfg.captioner_model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.captioner_model, torch_dtype=dtype).to(device).eval()

    @torch.no_grad()
    def caption(self, bgr, prompt: str) -> str:
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(
                self.device, dtype=self.model.dtype)
        out = self.model.generate(
            **inputs, max_new_tokens=self.cfg.caption_max_new_tokens,
            do_sample=False)
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return text.strip()


# ============================================================ CLIP retrieval
class SceneRetriever:
    """Top-k crash policies for a scene = avg CLIP over the last X frames
    (X matched to the caption cadence) matched against the policy index."""

    def __init__(self, clip, matcher, top_k):
        self.clip = clip
        self.matcher = matcher
        self.top_k = top_k
        self.cad = max(1, int(round(REALTIME.caption_interval_s * DATA.sample_hz)))
        self._emb_cache = {}

    def _frame_emb(self, s):
        key = (s["video_id"], s["frame_idx"])
        if key not in self._emb_cache:
            bgr = cv2.imread(str(PREPROCESSED_ROOT / s["image"]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._emb_cache[key] = self.clip.embed_image_arrays([rgb])[0]
        return self._emb_cache[key]

    def hits_for(self, video_samples, j, window=None):
        window = window or self.cad
        lo = max(0, j - window + 1)
        embs = np.stack([self._frame_emb(video_samples[k])
                         for k in range(lo, j + 1)])
        scene = pool_clip_video_embedding(embs)   # mean-pool + L2-normalize
        return self.matcher.retrieve(scene, top_k=self.top_k)


# ============================================================ VLA inference
@torch.no_grad()
def predict_with_captions(model, tokenizer, samples, captions, device,
                          batch_size=32):
    """VLA trajectories for (frame, generated caption) pairs."""
    preds = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        cap_chunk = captions[i:i + batch_size]
        imgs, states = [], []
        for s in chunk:
            bgr = cv2.imread(str(PREPROCESSED_ROOT / s["image"]))
            if bgr is None:
                raise IOError(f"missing frame {s['image']}")
            imgs.append(preprocess_image(bgr))
            states.append(state_to_vec(s["state"]))
        tok = tokenizer(list(cap_chunk), padding=True, truncation=True,
                        max_length=77, return_tensors="pt").to(device)
        pred = model(torch.stack(imgs).to(device), torch.stack(states).to(device),
                     tok["input_ids"], tok["attention_mask"])
        preds.append(pred.float().cpu().numpy())
    return denormalize_traj(np.concatenate(preds, axis=0))


# ============================================================ drawing
def draw_image_overlay(sample, gt, p_norag, p_rag):
    bgr = cv2.imread(str(PREPROCESSED_ROOT / sample["image"]))
    for traj, color in ((gt, COL_GT), (p_norag, COL_NORAG), (p_rag, COL_RAG)):
        pts = project_traj(traj, bgr.shape)
        if len(pts) >= 2:
            cv2.polylines(bgr, [pts], False, color, 2)
        for p in pts:
            cv2.circle(bgr, tuple(p), 3, color, -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_bev(ax, gt, p_norag, p_rag, title=""):
    ax.plot(-gt[:, 1], gt[:, 0], "o-", color="red", ms=3, lw=1.5, label="GT")
    ax.plot(-p_norag[:, 1], p_norag[:, 0], "o-", color="green", ms=3, lw=1.5,
            label="VLM only")
    ax.plot(-p_rag[:, 1], p_rag[:, 0], "o-", color="blue", ms=3, lw=1.5,
            label="VLM + RAG")
    ax.scatter([0], [0], marker="^", s=80, color="black", zorder=5, label="ego")
    ax.set_xlabel("lateral (m, right +)")
    ax.set_ylabel("forward (m)")
    ax.set_title(title, fontsize=9)
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def _wrap(label, text, width=70, max_lines=4):
    lines = textwrap.wrap(text, width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1] + " ..."
    body = ("\n" + " " * (len(label))).join(lines)
    return f"{label}{body}"


def render_video(samples, preds_norag, preds_rag, caps_norag, caps_rag,
                 hits_list, out_path, fps=4):
    """Real-time-style mp4 for ONE test video: frame + BEV + both captions."""
    writer = None
    for k, s in enumerate(samples):
        gt = gt_traj(s)
        a_n, f_n = ade_fde(preds_norag[k], gt)
        a_r, f_r = ade_fde(preds_rag[k], gt)

        fig = plt.figure(figsize=(13, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.4],
                              width_ratios=[1.7, 1])
        ax_img = fig.add_subplot(gs[0, 0])
        ax_bev = fig.add_subplot(gs[0, 1])
        ax_txt = fig.add_subplot(gs[1, :])

        ax_img.imshow(draw_image_overlay(s, gt, preds_norag[k], preds_rag[k]))
        ax_img.set_title(
            f"{s['video_id']}  frame {s['frame_idx']}  "
            f"t={s['frame_idx'] / DATA.video_fps:.1f}s   "
            f"(GT=red, VLM=green, VLM+RAG=blue)", fontsize=9)
        ax_img.axis("off")
        plot_bev(ax_bev, gt, preds_norag[k], preds_rag[k],
                 f"ADE  no-RAG {a_n:.2f} m  |  RAG {a_r:.2f} m\n"
                 f"FDE  no-RAG {f_n:.2f} m  |  RAG {f_r:.2f} m")

        ids = ", ".join(f"{h['clip_id']}({h['score']:.2f})"
                        for h in hits_list[k])
        ax_txt.axis("off")
        ax_txt.text(
            0.0, 1.0,
            _wrap("VLM caption (no RAG):  ", caps_norag[k]) + "\n\n" +
            _wrap("VLM + RAG caption:     ", caps_rag[k]) + "\n\n" +
            f"retrieved policies:    {ids}",
            transform=ax_txt.transAxes, fontsize=8, family="monospace",
            va="top", ha="left", wrap=True)

        fig.tight_layout()
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        if writer is None:
            h, w = rgb.shape[:2]
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if writer:
        writer.release()


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "covla_vla_best.pt"))
    ap.add_argument("--num-videos", type=int, default=10,
                    help="number of random test samples/videos to render")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap samples per video (SmolVLM2 is slow on CPU)")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--policies", default=str(DEFAULT_POLICIES))
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"repo root: {REPO_ROOT}")

    # --- models / index ---
    model, tokenizer = load_model(args.ckpt, device)
    clip = ClipEmbedder(args.clip_model, device=device)
    matcher = build_or_load_policy_index(
        Path(args.policies), Path(args.index), args.clip_model, embedder=clip)
    retr = SceneRetriever(clip, matcher, args.top_k)
    vlm = VLMCaptioner(device)
    base_prompt = REALTIME.caption_prompt
    print(f"policy matcher: {len(matcher)} policies | "
          f"clip avg window = {retr.cad} frames (caption cadence) | "
          f"captioner = {REALTIME.captioner_model}")

    # --- test split grouped + ordered by video ---
    index_path = PREPROCESSED_ROOT / "index" / "test.jsonl"
    test = [json.loads(l) for l in open(index_path, encoding="utf-8")]
    for s in test:
        s["image"] = s["image"].replace("\\", "/")
    by_video = defaultdict(list)
    for s in test:
        by_video[s["video_id"]].append(s)
    for v in by_video.values():
        v.sort(key=lambda s: s["frame_idx"])
    print(f"test split: {len(test)} samples / {len(by_video)} videos")

    # --- pick N random test samples (each from a distinct video) ---
    vids = rng.sample(sorted(by_video), min(args.num_videos, len(by_video)))
    refresh_every = retr.cad
    all_caps = {}

    for vi, vid in enumerate(vids):
        samples = by_video[vid]
        if args.max_samples:
            samples = samples[:args.max_samples]
        pick = rng.choice(samples)            # the "random test sample"
        print(f"\n[{vi}] video {vid}: {len(samples)} samples @ "
              f"{DATA.sample_hz} Hz  (caption refresh every {refresh_every})")

        # ---- real-time loop: VLM captions (no-RAG / with-RAG), cadence-locked
        caps_norag, caps_rag, hits_list = [], [], []
        cur_norag = cur_rag = None
        cur_hits = None
        t_cap = []
        for j, s in enumerate(samples):
            if (j % refresh_every == 0) or (cur_hits is None):
                bgr = cv2.imread(str(PREPROCESSED_ROOT / s["image"]))
                cur_hits = retr.hits_for(samples, j)            # avg last X frames
                t0 = time.time()
                cur_norag = vlm.caption(bgr, base_prompt)        # VLM only
                cur_rag = vlm.caption(                           # VLM + RAG
                    bgr, build_vlm_rag_prompt(base_prompt, cur_hits))
                t_cap.append(time.time() - t0)
            caps_norag.append(cur_norag)
            caps_rag.append(cur_rag)
            hits_list.append(cur_hits)
            if (j + 1) % 10 == 0:
                print(f"    captioned {j + 1}/{len(samples)} frames")

        # ---- DINO/VLA trajectories for each caption variant ----
        preds_norag = predict_with_captions(model, tokenizer, samples,
                                            caps_norag, device)
        preds_rag = predict_with_captions(model, tokenizer, samples,
                                          caps_rag, device)
        ade_n = np.mean([ade_fde(preds_norag[k], gt_traj(s))[0]
                         for k, s in enumerate(samples)])
        ade_r = np.mean([ade_fde(preds_rag[k], gt_traj(s))[0]
                         for k, s in enumerate(samples)])
        print(f"    video mean ADE: no-RAG {ade_n:.2f} m -> RAG {ade_r:.2f} m | "
              f"caption {np.mean(t_cap) * 1e3:.0f} ms/refresh x2")

        # ---- render the real-time video ----
        out = OUT_DIR / f"sample_{vi:02d}_{vid}_{pick['frame_idx']}.mp4"
        render_video(samples, preds_norag, preds_rag, caps_norag, caps_rag,
                     hits_list, out, fps=args.fps)
        print(f"    wrote {out.name}")

        all_caps[vid] = [{
            "frame_idx": s["frame_idx"],
            "caption_no_rag": caps_norag[k],
            "caption_with_rag": caps_rag[k],
            "retrieved": [{"clip_id": h["clip_id"], "score": h["score"]}
                          for h in hits_list[k]],
            "ade_no_rag": ade_fde(preds_norag[k], gt_traj(s))[0],
            "ade_with_rag": ade_fde(preds_rag[k], gt_traj(s))[0],
        } for k, s in enumerate(samples)]

    cap_path = OUT_DIR / "captions_after_rag.json"
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(all_caps, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(vids)} videos + {cap_path.name} in {OUT_DIR}")


if __name__ == "__main__":
    main()
