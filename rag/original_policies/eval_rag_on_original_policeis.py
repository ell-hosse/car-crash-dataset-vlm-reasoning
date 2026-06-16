"""
eval_rag_on_original_policeis.py
--------------------------------
FINAL evaluation of the CoVLA trajectory-VLA on the test split, comparing the
model WITHOUT RAG against the model WITH CLIP-retrieved crash policies.

Pipeline (per scene, mirrors the real-time loop):
  1. Caption        : the scene caption comes either from the GT rich_caption
                      (fast, isolates the RAG effect) or live from the VLM
                      captioner (SmolVLM2, fully realistic).  --caption-source
  2. CLIP retrieval : the last X camera frames are embedded with CLIP, mean-
                      pooled into one scene vector, and matched against ALL
                      ~2k crash policies (pre-embedded once into
                      compute_dist/clip_policy_index.npz).  Top-5 closest
                      policies are returned.
  3. Augment        : the top-5 policies are folded into the caption.
  4. VLA            : DINOv2 vision + (CLIP) text(caption) + ego-state ->
                      3 s trajectory.  Scored with ADE/FDE vs GT.

Matching X to the caption cadence
---------------------------------
The captioner refreshes every REALTIME.caption_interval_s (default 1.0 s) and
the VLA/retrieval reuse that caption until the next refresh - exactly like the
real-time loop, where the caption lags.  We therefore:
  * recompute the caption + retrieval every `refresh_every` test samples,
        refresh_every = round(caption_interval_s * sample_hz)        (= 2)
  * pool CLIP over the last `clip_window` frames, i.e. the frames seen during
    one caption interval,
        clip_window  = round(caption_interval_s * sample_hz)         (= 2 preproc frames)
                     ~ caption_interval_s * video_fps = 20 raw frames.
Both are derived from config and overridable with --refresh-every / --clip-window.

Outputs
-------
  rag/original_policies/eval_rag_on_original_policies_results.json
        per-sample records (captions, retrieved policy ids/scores, ADE/FDE
        before & after RAG) + an aggregate summary block with timing.

Paths are resolved relative to the auto-detected repo root, so this script runs
correctly from any working directory (and whether it lives at the repo root or
here under rag/original_policies/).

Usage (machine with the D:/hf data + checkpoint):
    python rag/original_policies/eval_rag_on_original_policeis.py
    python rag/original_policies/eval_rag_on_original_policeis.py --caption-source vlm --max-frames 200
    python rag/original_policies/eval_rag_on_original_policeis.py --top-k 5 --augment mitigations
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


def _find_repo_root(start: Path) -> Path:
    """Locate the repo root by walking up from this file until we find the
    `covla_vla` package next to `crash_policies.jsonl`. This keeps every path
    below correct whether the script sits at the repo root or under
    rag/original_policies/."""
    for d in (start, *start.parents):
        if (d / "covla_vla").is_dir() and (d / "crash_policies.jsonl").exists():
            return d
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compute_dist"))

# --- CoVLA config / data, with the same D:/hf -> local fallback as the rag evals
import covla_vla.config as covla_config
import covla_vla.dataset as covla_dataset_mod

if not covla_config.PREPROCESSED_ROOT.exists():
    _local = REPO_ROOT / "covla_preprocessed"
    if _local.exists():
        covla_config.PREPROCESSED_ROOT = _local
        covla_dataset_mod.PREPROCESSED_ROOT = _local

from covla_vla.config import DATA, REALTIME                              # noqa: E402
from covla_vla.dataset import (preprocess_image, state_to_vec,           # noqa: E402
                               denormalize_traj)
from covla_vla.infer_realtime import load_model                          # noqa: E402

from clip_retrieval import (build_or_load_policy_index, ClipEmbedder,    # noqa: E402
                            pool_clip_video_embedding, augment_caption,
                            DEFAULT_CLIP_MODEL, DEFAULT_POLICIES, DEFAULT_INDEX)

PREPROCESSED_ROOT = covla_config.PREPROCESSED_ROOT
OUT_PATH = REPO_ROOT / "rag" / "original_policies" / \
    "eval_rag_on_original_policies_results.json"


def _require(path, what: str) -> Path:
    """Fail fast with a clear message if a referenced file/dir is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


# ---------------------------------------------------------------- metrics
def ade_fde(pred: np.ndarray, gt: np.ndarray):
    d = np.linalg.norm(pred - gt, axis=-1)
    return float(d.mean()), float(d[-1])


# ---------------------------------------------------------------- VLA forward
@torch.no_grad()
def predict_traj(model, tokenizer, device, rgb_or_bgr, state_vec, caption):
    """Single-sample VLA forward. `rgb_or_bgr` is a BGR frame (cv2)."""
    img = preprocess_image(rgb_or_bgr).unsqueeze(0).to(device)
    st = state_vec.unsqueeze(0).to(device)
    tok = tokenizer([caption], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    pred = model(img, st, input_ids=tok["input_ids"],
                 attention_mask=tok["attention_mask"])
    return denormalize_traj(pred[0].float().cpu().numpy())


# ---------------------------------------------------------------- optional VLM
def make_vlm_captioner(device):
    """Lazily build the SmolVLM2 captioner (same model as the real-time loop).

    Mirrors rag/abstract_policies/eval_rag_online.py: the installed
    transformers wants a PIL image (not a raw numpy array) in the chat
    template, so we wrap the frame before the parent's _generate logic."""
    from PIL import Image
    from covla_vla.captioner import AsyncCaptioner

    class PILFrameCaptioner(AsyncCaptioner):
        @torch.no_grad()
        def _generate(self, bgr):
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": self.cfg.caption_prompt}]}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(
                    self.device, dtype=self.model.dtype)
            out = self.model.generate(
                **inputs, max_new_tokens=self.cfg.caption_max_new_tokens,
                do_sample=False)
            text = self.processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0]
            return text.strip()

    return PILFrameCaptioner(device, cfg=REALTIME).start()


def vlm_caption(captioner, bgr, timeout_s=3.0, fallback=""):
    base = captioner.caption_version
    captioner.submit_frame(bgr)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if captioner.caption_version > base:
            return captioner.caption
        time.sleep(0.03)
    return fallback


# ---------------------------------------------------------------- main eval
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "covla_vla_best.pt"))
    ap.add_argument("--policies", default=str(DEFAULT_POLICIES))
    ap.add_argument("--index", default=str(DEFAULT_INDEX),
                    help="cached CLIP policy index (.npz)")
    ap.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--top-k", type=int, default=5,
                    help="number of closest policies used as RAG context")
    ap.add_argument("--augment", default="mitigations",
                    choices=["mitigations", "risks", "triplet"],
                    help="how the retrieved policies are folded into the caption")
    ap.add_argument("--caption-source", default="gt", choices=["gt", "vlm"],
                    help="gt = GT rich_caption (fast, full split); "
                         "vlm = live SmolVLM2 caption (realistic, slow)")
    ap.add_argument("--clip-window", type=int, default=None,
                    help="# preprocessed frames pooled for CLIP retrieval "
                         "(default: derived from caption cadence)")
    ap.add_argument("--refresh-every", type=int, default=None,
                    help="recompute caption+retrieval every N samples "
                         "(default: derived from caption cadence)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap total samples evaluated (for --caption-source vlm)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"repo root        : {REPO_ROOT}")
    print(f"preprocessed root: {PREPROCESSED_ROOT}")

    # --- validate every referenced file up-front (clear error if mis-addressed)
    test_index = PREPROCESSED_ROOT / "index" / "test.jsonl"
    _require(args.ckpt, "checkpoint (--ckpt)")
    _require(args.policies, "policies file (--policies)")
    _require(PREPROCESSED_ROOT, "preprocessed data root (covla_vla.config.PREPROCESSED_ROOT)")
    _require(test_index, "preprocessed test index (run covla_vla.preprocess first)")
    # the CLIP policy index (.npz) is rebuilt automatically if missing/stale.

    # cadence derived from config (X frames matched to caption frequency)
    cad = max(1, int(round(REALTIME.caption_interval_s * DATA.sample_hz)))
    clip_window = args.clip_window or cad
    refresh_every = args.refresh_every or cad
    raw_equiv = int(round(REALTIME.caption_interval_s * DATA.video_fps))
    print(f"caption cadence: every {REALTIME.caption_interval_s:.1f}s "
          f"-> refresh_every={refresh_every} samples, "
          f"clip_window={clip_window} preproc frames "
          f"(~{raw_equiv} raw frames @ {DATA.video_fps:.0f} fps)")

    # --- models / index ---
    model, tokenizer = load_model(args.ckpt, device)
    clip = ClipEmbedder(args.clip_model, device=device)
    matcher = build_or_load_policy_index(
        Path(args.policies), Path(args.index), args.clip_model,
        embedder=clip, rebuild=args.rebuild_index)
    print(f"policy matcher ready with {len(matcher)} policies")
    captioner = make_vlm_captioner(device) if args.caption_source == "vlm" else None

    # --- test split, grouped + ordered by video (mirrors before-RAG viz) ---
    test = [json.loads(l) for l in open(test_index, encoding="utf-8")]
    for s in test:
        s["image"] = s["image"].replace("\\", "/")
    by_video = defaultdict(list)
    for s in test:
        by_video[s["video_id"]].append(s)
    for v in by_video.values():
        v.sort(key=lambda s: s["frame_idx"])
    print(f"test split: {len(test)} samples / {len(by_video)} videos")

    # timing accumulators (seconds)
    t_clip_img, t_retrieve, t_caption, t_fwd_base, t_fwd_rag = [], [], [], [], []
    ade_base, fde_base, ade_rag, fde_rag = [], [], [], []
    results = []
    n_seen = 0

    try:
        for vid, samples in by_video.items():
            # per-frame CLIP image embeddings for this video (each frame once)
            frame_embs = [None] * len(samples)
            cur_hits, cur_caption_aug = None, None

            for j, s in enumerate(samples):
                if args.max_frames and n_seen >= args.max_frames:
                    break
                bgr = cv2.imread(str(PREPROCESSED_ROOT / s["image"]))
                if bgr is None:
                    print(f"  WARNING missing frame {s['image']}, skipping")
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # CLIP-embed this frame once (timed)
                t0 = time.time()
                frame_embs[j] = clip.embed_image_arrays([rgb])[0]
                t_clip_img.append(time.time() - t0)

                state_vec = state_to_vec(s["state"])
                gt = np.asarray(s["traj"], dtype=np.float32)

                # base caption (no RAG)
                if args.caption_source == "vlm":
                    tc = time.time()
                    base_caption = vlm_caption(captioner, bgr,
                                               fallback=s["caption"])
                    t_caption.append(time.time() - tc)
                else:
                    base_caption = s["caption"]

                # refresh retrieval + augmented caption on cadence
                if (j % refresh_every == 0) or (cur_hits is None):
                    lo = max(0, j - clip_window + 1)
                    window = np.stack([frame_embs[k] for k in range(lo, j + 1)
                                       if frame_embs[k] is not None])
                    tr = time.time()
                    scene_emb = pool_clip_video_embedding(window)
                    cur_hits = matcher.retrieve(scene_emb, top_k=args.top_k)
                    t_retrieve.append(time.time() - tr)
                    cur_caption_aug = augment_caption(base_caption, cur_hits,
                                                      style=args.augment)
                elif args.caption_source == "vlm":
                    # caption refreshed but reuse the cadence-locked policies
                    cur_caption_aug = augment_caption(base_caption, cur_hits,
                                                      style=args.augment)

                # VLA forward: before-RAG (base caption)
                tb = time.time()
                pred_base = predict_traj(model, tokenizer, device, bgr,
                                         state_vec, base_caption)
                t_fwd_base.append(time.time() - tb)
                a_b, f_b = ade_fde(pred_base, gt)
                ade_base.append(a_b); fde_base.append(f_b)

                # VLA forward: after-RAG (augmented caption)
                tr2 = time.time()
                pred_rag = predict_traj(model, tokenizer, device, bgr,
                                        state_vec, cur_caption_aug)
                t_fwd_rag.append(time.time() - tr2)
                a_r, f_r = ade_fde(pred_rag, gt)
                ade_rag.append(a_r); fde_rag.append(f_r)

                results.append({
                    "video_id": vid, "frame_idx": s["frame_idx"],
                    "base_caption": base_caption,
                    "augmented_caption": cur_caption_aug,
                    "retrieved": [{"clip_id": h["clip_id"], "score": h["score"],
                                   "dist": h["dist"]} for h in cur_hits],
                    "ade_before": a_b, "fde_before": f_b,
                    "ade_after": a_r, "fde_after": f_r,
                })
                n_seen += 1
                if n_seen % 100 == 0:
                    print(f"  {n_seen} samples | "
                          f"ADE base {np.mean(ade_base):.3f} -> "
                          f"RAG {np.mean(ade_rag):.3f} m")
            if args.max_frames and n_seen >= args.max_frames:
                break
    finally:
        if captioner is not None:
            captioner.stop()

    # ------------------------------------------------------------ summary
    def ms(a):
        return float(np.mean(a) * 1e3) if a else None

    summary = {
        "n_samples": n_seen,
        "caption_source": args.caption_source,
        "top_k": args.top_k,
        "augment_style": args.augment,
        "clip_window_frames": clip_window,
        "refresh_every_samples": refresh_every,
        "caption_interval_s": REALTIME.caption_interval_s,
        "ade_before_rag_m": float(np.mean(ade_base)) if ade_base else None,
        "fde_before_rag_m": float(np.mean(fde_base)) if fde_base else None,
        "ade_after_rag_m": float(np.mean(ade_rag)) if ade_rag else None,
        "fde_after_rag_m": float(np.mean(fde_rag)) if fde_rag else None,
        "timing_ms": {
            "clip_image_embed_per_frame": ms(t_clip_img),
            "policy_retrieval_per_refresh": ms(t_retrieve),
            "vlm_caption_per_frame": ms(t_caption),
            "vla_forward_before_rag": ms(t_fwd_base),
            "vla_forward_after_rag": ms(t_fwd_rag),
        },
        "n_retrievals": len(t_retrieve),
    }

    print("\n=== FINAL RAG EVALUATION (CoVLA test) ===")
    print(f"samples evaluated : {n_seen}  (caption={args.caption_source}, "
          f"top-{args.top_k} policies, augment={args.augment})")
    print(f"ADE  before RAG   : {summary['ade_before_rag_m']:.3f} m")
    print(f"ADE  after  RAG   : {summary['ade_after_rag_m']:.3f} m")
    print(f"FDE  before RAG   : {summary['fde_before_rag_m']:.3f} m")
    print(f"FDE  after  RAG   : {summary['fde_after_rag_m']:.3f} m")
    print("--- timing (mean) ---")
    print(f"CLIP image embed / frame      : {ms(t_clip_img):.2f} ms")
    print(f"policy retrieval / refresh    : {ms(t_retrieve):.3f} ms  "
          f"(over {len(t_retrieve)} refreshes)")
    if t_caption:
        print(f"VLM caption / frame           : {ms(t_caption):.1f} ms")
    print(f"VLA forward (before RAG)      : {ms(t_fwd_base):.2f} ms")
    print(f"VLA forward (after  RAG)      : {ms(t_fwd_rag):.2f} ms")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_sample": results}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
