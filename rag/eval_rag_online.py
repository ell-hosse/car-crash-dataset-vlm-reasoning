"""
eval_rag_online.py
------------------
Two evaluations on the preprocessed CoVLA test split:

  TEST 1 - Baseline VLA (ADE/FDE) over the full test split using GT captions.
  TEST 2 - RAG retrieval quality: live SmolVLM2 captions on preprocessed JPEG
           frames queried against distilled_index.npz (top-1).

Run from the project root:
    python rag/eval_rag_online.py
    python rag/eval_rag_online.py --ckpt covla_vla/runs/covla_vla_best.pt \
        --distilled-index distilled_index.npz --max-frames 200 --batch-size 32

Per-frame results are saved to rag/eval_rag_online_results.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))  # retriever.py lives here

import covla_vla.config as covla_config
import covla_vla.dataset as covla_dataset_mod

# config.py points PREPROCESSED_ROOT at the Windows data drive (D:/hf). On a
# machine where that path does not exist, fall back to the local copy without
# modifying config.py (dataset.py resolves PREPROCESSED_ROOT at call time).
if not covla_config.PREPROCESSED_ROOT.exists():
    _local = PROJECT_ROOT / "covla_preprocessed"
    if _local.exists():
        covla_config.PREPROCESSED_ROOT = _local
        covla_dataset_mod.PREPROCESSED_ROOT = _local

from covla_vla.config import PREPROCESSED_ROOT, REALTIME  # noqa: E402
from covla_vla.captioner import AsyncCaptioner  # noqa: E402
from covla_vla.dataset import CoVLADataset, denormalize_traj, make_collate  # noqa: E402
from covla_vla.model import ade_fde, build_model_and_tokenizer  # noqa: E402
from retriever import PolicyRetriever  # noqa: E402

PREPROCESSED_ROOT = covla_config.PREPROCESSED_ROOT  # post-fallback value

CAPTION_TIMEOUT_S = 3.0  # max wait for a live caption before GT fallback


def load_model(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, tokenizer = build_model_and_tokenizer()
    # Checkpoints trained under a transformers version that flattened
    # CLIPTextModel store the frozen text tower as "text.encoder..."; the
    # version here nests it as "text.text_model.encoder...". Remap if needed.
    expected = model.state_dict().keys()
    state = {
        ("text.text_model." + k[len("text."):]
         if k.startswith("text.") and k not in expected else k): v
        for k, v in ck["model"].items()
    }
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"loaded {ckpt_path} (epoch {ck.get('epoch')}, "
          f"best val ADE {ck.get('best_ade', float('nan')):.3f} m)")
    return model, tokenizer


def load_test_dataset() -> CoVLADataset:
    ds = CoVLADataset("test")
    # index.jsonl was written on Windows; normalize path separators in memory
    for s in ds.samples:
        s["image"] = s["image"].replace("\\", "/")
    return ds


# ---------------------------------------------------------------------------
# TEST 1 - baseline VLA with GT captions
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_test1(model, tokenizer, dataset, device, batch_size: int):
    """Full test split forward pass; returns per-sample ADE/FDE lists
    aligned with dataset.samples order."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=make_collate(tokenizer))

    per_ade, per_fde = [], []
    t0 = time.time()
    for n_batch, batch in enumerate(loader, 1):
        pred = model(
            batch["image"].to(device),
            batch["state"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device))
        pred_m = torch.from_numpy(denormalize_traj(pred.float().cpu().numpy()))
        gt_m = torch.from_numpy(denormalize_traj(batch["traj"].float().numpy()))
        for i in range(pred_m.shape[0]):
            a, f = ade_fde(pred_m[i:i + 1], gt_m[i:i + 1])
            per_ade.append(a)
            per_fde.append(f)
        if n_batch % 50 == 0:
            print(f"  [test1] {len(per_ade)}/{len(dataset)} samples | "
                  f"ADE={np.mean(per_ade):.3f}m FDE={np.mean(per_fde):.3f}m | "
                  f"{time.time() - t0:.0f}s elapsed")
    return per_ade, per_fde


# ---------------------------------------------------------------------------
# TEST 2 - RAG retrieval on live SmolVLM2 captions
# ---------------------------------------------------------------------------
class PILFrameCaptioner(AsyncCaptioner):
    """AsyncCaptioner hands a raw numpy array to apply_chat_template, which
    the transformers version installed here rejects (wants PIL/url/path).
    Only the image wrapping differs from the parent implementation."""

    @torch.no_grad()
    def _generate(self, bgr):
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": self.cfg.caption_prompt},
            ],
        }]
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


def wait_for_caption(captioner: AsyncCaptioner, base_version: int,
                     timeout: float = CAPTION_TIMEOUT_S):
    """Poll until the captioner publishes a caption newer than base_version.
    Returns the live caption, or None on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if captioner.caption_version > base_version:
            return captioner.caption
        time.sleep(0.05)
    return None


def run_test2(dataset, retriever, device, max_frames: int,
              per_ade, per_fde) -> list[dict]:
    n = min(max_frames, len(dataset))
    print(f"  [test2] captioning + retrieving on {n} frames "
          f"(timeout {CAPTION_TIMEOUT_S:.0f}s/frame, captioner "
          f"{REALTIME.captioner_model})")
    captioner = PILFrameCaptioner(device, cfg=REALTIME).start()
    results = []
    try:
        for idx in range(n):
            s = dataset.samples[idx]
            frame_path = PREPROCESSED_ROOT / s["image"]
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                print(f"  [test2] WARNING: missing frame {frame_path}, skipping")
                continue

            base_version = captioner.caption_version
            captioner.submit_frame(bgr)
            live_caption = wait_for_caption(captioner, base_version)
            if live_caption is None:
                live_caption = s["caption"]  # GT fallback on timeout

            hit = retriever.retrieve(live_caption, top_k=1)[0]
            results.append({
                "sample_idx": idx,
                "video_id": s["video_id"],
                "frame_idx": s["frame_idx"],
                "gt_caption": s["caption"],
                "live_caption": live_caption,
                "retrieved_pattern_id": hit["pattern_id"],
                "retrieved_pattern_name": hit["pattern_name"],
                "retrieval_score": hit["score"],
                "ade": per_ade[idx],
                "fde": per_fde[idx],
            })
            if len(results) % 20 == 0:
                scores = [r["retrieval_score"] for r in results]
                print(f"  [test2] {len(results)}/{n} frames | "
                      f"mean score {np.mean(scores):.3f}")
    finally:
        captioner.stop()
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_test1_summary(per_ade, per_fde):
    print("\n=== TEST 1: BASELINE VLA ===")
    print(f"Frames evaluated : {len(per_ade)}")
    print(f"ADE              : {np.mean(per_ade):.3f} m")
    print(f"FDE              : {np.mean(per_fde):.3f} m")


def print_test2_summary(results):
    print("\n=== TEST 2: RAG RETRIEVAL ON LIVE CAPTIONS ===")
    n = len(results)
    print(f"Frames queried        : {n}")
    if n == 0:
        return
    scores = np.array([r["retrieval_score"] for r in results])
    print(f"Mean retrieval score  : {scores.mean():.3f} (std={scores.std():.3f})")

    by_pattern = {}
    for r in results:
        key = (r["retrieved_pattern_id"], r["retrieved_pattern_name"])
        by_pattern.setdefault(key, []).append(r["retrieval_score"])
    top5 = sorted(by_pattern.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    print("Top 5 retrieved patterns:")
    for (pid, name), ss in top5:
        print(f"  [{pid}] {name} — {len(ss)} times (mean score {np.mean(ss):.2f})")

    buckets = [
        ("> 0.80", scores > 0.80),
        ("0.70-0.80", (scores > 0.70) & (scores <= 0.80)),
        ("0.60-0.70", (scores > 0.60) & (scores <= 0.70)),
        ("< 0.60", scores <= 0.60),
    ]
    print("Score distribution:")
    for label, mask in buckets:
        c = int(mask.sum())
        print(f"  {label} : {c} frames ({100.0 * c / n:.0f}%)")


def main():
    ap = argparse.ArgumentParser(
        description="Baseline VLA ADE/FDE + RAG retrieval quality on CoVLA test")
    ap.add_argument("--ckpt", type=str,
                    default=str(PROJECT_ROOT / "rag" / "covla_vla_best.pt"),
                    help="checkpoint from covla_vla/train.py")
    ap.add_argument("--distilled-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "distilled_index.npz"),
                    help="distilled pattern index from distill_patterns.py")
    ap.add_argument("--max-frames", type=int, default=200,
                    help="max frames for Test 2 (captioner is slow)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Test 1 DataLoader batch size")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"preprocessed root: {PREPROCESSED_ROOT}")

    dataset = load_test_dataset()
    print(f"test split: {len(dataset)} samples")
    model, tokenizer = load_model(args.ckpt, device)

    # TEST 1 - full split, GT captions
    per_ade, per_fde = run_test1(model, tokenizer, dataset, device,
                                 args.batch_size)

    # TEST 2 - live captions + retrieval (limited to --max-frames)
    retriever = PolicyRetriever(args.distilled_index)
    print(f"distilled index: {args.distilled_index} ({len(retriever)} patterns)")
    results = run_test2(dataset, retriever, device, args.max_frames,
                        per_ade, per_fde)

    print_test1_summary(per_ade, per_fde)
    print_test2_summary(results)

    out_path = PROJECT_ROOT / "rag" / "eval_rag_online_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nper-frame results saved to {out_path}")


if __name__ == "__main__":
    main()
