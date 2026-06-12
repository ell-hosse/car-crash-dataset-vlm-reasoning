"""
eval_ccd_retrieval.py
---------------------
Tests the "helpful when needed" side of retrieval on the CCD crash dataset
(the same corpus the patterns were distilled from, so results are an upper
bound for unseen crashes — but the query path is new: live SmolVLM2 captions
of raw pre-crash frames, not Gemini policy text).

For each sampled clip, captions a frame safely before the labeled accident
onset, retrieves top-1 from the distilled and expanded indexes, and compares
the crash score distribution against the benign CoVLA caption scores saved
by eval_expanded_retrieval.py (separation / threshold viability).

Run from the project root:
    python rag/eval_ccd_retrieval.py --n-clips 150
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))

from eval_rag_online import PILFrameCaptioner  # noqa: E402
from retriever import PolicyRetriever  # noqa: E402

CCD_ROOT = Path.home() / (".cache/kagglehub/datasets/asefjamilajwad/"
                          "car-crash-dataset-ccd/versions/2/CrashBest")
PRE_CRASH_OFFSET = 8  # caption this many frames before the labeled onset


def first_accident_frame(df: pd.DataFrame) -> np.ndarray:
    fcols = [c for c in df.columns if c.startswith("frame_")]
    labels = df[fcols].to_numpy()
    return np.argmax(labels > 0, axis=1) + 1  # 1-based frame number


def auc_score(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(crash score > benign score), i.e. ROC AUC via rank comparison."""
    return float(np.mean(pos[:, None] > neg[None, :])
                 + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, fpr: float) -> tuple[float, float]:
    thr = float(np.quantile(neg, 1.0 - fpr))
    return float((pos >= thr).mean()), thr


def main():
    ap = argparse.ArgumentParser(description="CCD pre-crash retrieval separation test")
    ap.add_argument("--crash-table", type=str,
                    default=str(PROJECT_ROOT / "rag" / "Crash_Table.xls"))
    ap.add_argument("--n-clips", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--distilled-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "distilled_index.npz"))
    ap.add_argument("--expanded-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "expanded_index.npz"))
    ap.add_argument("--benign-results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_expanded_retrieval_results.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.crash_table)  # CSV content despite the .xls name
    df["first_acc"] = first_accident_frame(df)
    rng = np.random.default_rng(args.seed)
    picks = df.iloc[rng.choice(len(df), size=min(args.n_clips, len(df)),
                               replace=False)]
    print(f"sampled {len(picks)} of {len(df)} CCD clips "
          f"(egoinvolve=Yes on {(picks['egoinvolve'] == 'Yes').sum()})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captioner = PILFrameCaptioner(device)  # used synchronously, no thread
    old = PolicyRetriever(args.distilled_index)
    new = PolicyRetriever(args.expanded_index)

    rows = []
    for n, (_, r) in enumerate(picks.iterrows(), 1):
        vid = int(r["vidname"])
        frame_no = max(1, int(r["first_acc"]) - PRE_CRASH_OFFSET)
        path = CCD_ROOT / f"C_{vid:06d}_{frame_no:02d}.jpg"
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"  WARNING: missing {path}, skipping")
            continue
        caption = captioner._generate(bgr)
        oh = old.retrieve(caption, top_k=1)[0]
        nh = new.retrieve(caption, top_k=1)[0]
        rows.append({
            "vidname": vid,
            "frame": frame_no,
            "first_accident_frame": int(r["first_acc"]),
            "egoinvolve": str(r["egoinvolve"]),
            "timing": str(r["timing"]),
            "weather": str(r["weather"]),
            "caption": caption,
            "old_pattern_id": oh["pattern_id"], "old_score": oh["score"],
            "new_pattern_id": nh["pattern_id"],
            "new_pattern_name": nh["pattern_name"], "new_score": nh["score"],
            "new_matched_key": nh["matched_key"],
        })
        if n % 25 == 0:
            print(f"  {n}/{len(picks)} clips captioned")

    benign = json.load(open(args.benign_results, encoding="utf-8"))
    b_old = np.array([b["old_score"] for b in benign])
    b_new = np.array([b["new_score"] for b in benign])
    c_old = np.array([r["old_score"] for r in rows])
    c_new = np.array([r["new_score"] for r in rows])

    print(f"\ncrash frames captioned : {len(rows)}")
    print(f"benign reference frames: {len(benign)}")
    for name, c, b in [("DISTILLED INDEX", c_old, b_old),
                       ("EXPANDED INDEX", c_new, b_new)]:
        print(f"\n=== {name}: crash (pre-crash CCD) vs benign (CoVLA) ===")
        print(f"crash  : mean {c.mean():.3f} (std={c.std():.3f})  "
              f"p10 {np.percentile(c, 10):.3f}  p90 {np.percentile(c, 90):.3f}")
        print(f"benign : mean {b.mean():.3f} (std={b.std():.3f})  "
              f"p10 {np.percentile(b, 10):.3f}  p90 {np.percentile(b, 90):.3f}")
        print(f"AUC (crash ranked above benign): {auc_score(c, b):.3f}")
        for fpr in (0.05, 0.10, 0.20):
            tpr, thr = tpr_at_fpr(c, b, fpr)
            print(f"  at {fpr:.0%} benign FP rate (threshold {thr:.3f}): "
                  f"catches {tpr:.0%} of crash frames")

    ego = np.array([r["egoinvolve"] == "Yes" for r in rows])
    if 0 < ego.sum() < len(rows):
        print(f"\nexpanded-index crash scores by ego involvement: "
              f"ego-involved {c_new[ego].mean():.3f} (n={ego.sum()}) vs "
              f"other-vehicle {c_new[~ego].mean():.3f} (n={(~ego).sum()})")

    by_pattern = {}
    for r in rows:
        by_pattern.setdefault((r["new_pattern_id"], r["new_pattern_name"]),
                              []).append(r["new_score"])
    top5 = sorted(by_pattern.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    print("\nTop 5 patterns retrieved on crash frames (expanded index):")
    for (pid, pname), ss in top5:
        print(f"  [{pid}] {pname} — {len(ss)} times (mean score {np.mean(ss):.2f})")

    out_path = PROJECT_ROOT / "rag" / "eval_ccd_retrieval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nper-clip results saved to {out_path}")


if __name__ == "__main__":
    main()
