"""
eval_two_tier_captioner.py
--------------------------
Tests the two-tier captioner idea: does a hazard-focused prompt and/or a
bigger tier-2 VLM produce captions that actually separate pre-crash frames
from benign driving in retrieval score?

Conditions on the SAME frames (150 CCD pre-crash from eval_ccd_retrieval.py,
200 benign CoVLA from eval_rag_online.py):

  A. 256M + generic prompt   (baseline, scores reused from earlier runs)
  B. 256M + hazard prompt    (isolates the prompt effect)
  C. 2.2B + hazard prompt    (adds the model-size effect = tier 2)

Run from the project root:
    python rag/eval_two_tier_captioner.py
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))

from covla_vla.config import REALTIME  # noqa: E402
from eval_ccd_retrieval import CCD_ROOT, auc_score, tpr_at_fpr  # noqa: E402
from eval_rag_online import PILFrameCaptioner, load_test_dataset  # noqa: E402
import covla_vla.config as covla_config  # noqa: E402
from retriever import PolicyRetriever  # noqa: E402

TIER2_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

HAZARD_PROMPT = (
    "You are the hazard-detection module of an autonomous vehicle. Look at "
    "this front-camera driving scene and describe the most dangerous thing "
    "happening. Be specific about what other road users are doing: any "
    "vehicle braking, stopped, turning across your path, drifting or cutting "
    "into your lane, or approaching head-on; any pedestrian, cyclist or "
    "motorcyclist on or near the road; the road surface (dry, wet, icy, "
    "snowy) and visibility. If the road ahead is clear and nothing is "
    "dangerous, say so. One short paragraph."
)

CONDITIONS = [
    ("B_small_hazard", REALTIME.captioner_model),
    ("C_large_hazard", TIER2_MODEL),
]


def collect_frames(ccd_results: str, benign_results: str, dataset):
    frames = []  # (kind, ident, path)
    for r in json.load(open(ccd_results, encoding="utf-8")):
        frames.append(("crash", f"ccd_{r['vidname']}",
                       CCD_ROOT / f"C_{int(r['vidname']):06d}_{int(r['frame']):02d}.jpg"))
    for r in json.load(open(benign_results, encoding="utf-8")):
        s = dataset.samples[r["sample_idx"]]
        frames.append(("benign", f"covla_{r['sample_idx']}",
                       covla_config.PREPROCESSED_ROOT / s["image"]))
    return frames


def main():
    ap = argparse.ArgumentParser(description="Two-tier hazard captioner ablation")
    ap.add_argument("--ccd-results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_ccd_retrieval_results.json"))
    ap.add_argument("--benign-results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_rag_online_results.json"))
    ap.add_argument("--distilled-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "distilled_index.npz"))
    ap.add_argument("--expanded-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "expanded_index.npz"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_test_dataset()
    frames = collect_frames(args.ccd_results, args.benign_results, dataset)
    n_crash = sum(k == "crash" for k, _, _ in frames)
    print(f"{len(frames)} frames ({n_crash} crash, {len(frames) - n_crash} benign)")

    retrievers = {"distilled": PolicyRetriever(args.distilled_index),
                  "expanded": PolicyRetriever(args.expanded_index)}

    results = {}
    for tag, model_id in CONDITIONS:
        print(f"\n--- condition {tag}: {model_id} ---")
        cfg = replace(REALTIME, captioner_model=model_id,
                      caption_prompt=HAZARD_PROMPT)
        captioner = PILFrameCaptioner(device, cfg=cfg)  # synchronous use
        rows, lats = [], []
        for n, (kind, ident, path) in enumerate(frames, 1):
            bgr = cv2.imread(str(path))
            if bgr is None:
                print(f"  WARNING: missing {path}, skipping")
                continue
            t0 = time.time()
            caption = captioner._generate(bgr)
            lats.append(time.time() - t0)
            row = {"kind": kind, "id": ident, "caption": caption}
            for iname, retr in retrievers.items():
                hit = retr.retrieve(caption, top_k=1)[0]
                row[f"{iname}_score"] = hit["score"]
                row[f"{iname}_pattern_id"] = hit["pattern_id"]
                row[f"{iname}_pattern_name"] = hit["pattern_name"]
            rows.append(row)
            if n % 50 == 0:
                print(f"  {n}/{len(frames)} captioned "
                      f"(mean latency {np.mean(lats):.2f}s)")
        results[tag] = {"rows": rows, "mean_latency_s": float(np.mean(lats))}
        del captioner.model, captioner
        torch.cuda.empty_cache()

    # condition A baseline scores from the earlier runs
    a_crash = json.load(open(args.ccd_results, encoding="utf-8"))
    a_benign_exp = json.load(open(
        PROJECT_ROOT / "rag" / "eval_expanded_retrieval_results.json",
        encoding="utf-8"))
    baseline = {
        "distilled": (np.array([r["old_score"] for r in a_crash]),
                      np.array([r["old_score"] for r in a_benign_exp])),
        "expanded": (np.array([r["new_score"] for r in a_crash]),
                     np.array([r["new_score"] for r in a_benign_exp])),
    }

    print("\n================ SEPARATION SUMMARY ================")
    for iname in retrievers:
        print(f"\n--- {iname} index ---")
        crash, benign = baseline[iname]
        print(f"{'A_small_generic':<16} AUC {auc_score(crash, benign):.3f} | "
              f"crash {crash.mean():.3f} vs benign {benign.mean():.3f}")
        for tag, _ in CONDITIONS:
            rows = results[tag]["rows"]
            crash = np.array([r[f"{iname}_score"] for r in rows if r["kind"] == "crash"])
            benign = np.array([r[f"{iname}_score"] for r in rows if r["kind"] == "benign"])
            tpr10, thr10 = tpr_at_fpr(crash, benign, 0.10)
            print(f"{tag:<16} AUC {auc_score(crash, benign):.3f} | "
                  f"crash {crash.mean():.3f} vs benign {benign.mean():.3f} | "
                  f"TPR@10%FP {tpr10:.0%} (thr {thr10:.3f})")

    print("\nCaption latency: " + " | ".join(
        f"{tag}: {results[tag]['mean_latency_s']:.2f}s" for tag, _ in CONDITIONS))

    print("\nExample tier-2 crash captions:")
    for r in [r for r in results["C_large_hazard"]["rows"] if r["kind"] == "crash"][:4]:
        print(f"  [{r['id']}] {r['caption'][:200]}")

    out_path = PROJECT_ROOT / "rag" / "eval_two_tier_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nfull results saved to {out_path}")


if __name__ == "__main__":
    main()
