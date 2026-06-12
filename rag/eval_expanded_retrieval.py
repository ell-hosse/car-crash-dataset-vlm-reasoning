"""
eval_expanded_retrieval.py
--------------------------
Re-scores the live SmolVLM2 captions saved by eval_rag_online.py against the
expanded multi-key index (rag/expand_trigger_keys.py) and compares it to the
original distilled index, including a retrieval-threshold (abstention) sweep.

Run from the project root, after eval_rag_online.py and expand_trigger_keys.py:
    python rag/eval_expanded_retrieval.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retriever import PolicyRetriever  # noqa: E402

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def score_distribution(scores: np.ndarray) -> list[tuple[str, int]]:
    return [
        ("> 0.80", int((scores > 0.80).sum())),
        ("0.70-0.80", int(((scores > 0.70) & (scores <= 0.80)).sum())),
        ("0.60-0.70", int(((scores > 0.60) & (scores <= 0.70)).sum())),
        ("< 0.60", int((scores <= 0.60).sum())),
    ]


def print_block(name, scores, hits):
    n = len(scores)
    print(f"\n=== {name} ===")
    print(f"Mean retrieval score  : {scores.mean():.3f} (std={scores.std():.3f})")
    by_pattern = {}
    for h in hits:
        by_pattern.setdefault((h["pattern_id"], h["pattern_name"]), []).append(h["score"])
    top5 = sorted(by_pattern.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    print("Top 5 retrieved patterns:")
    for (pid, pname), ss in top5:
        print(f"  [{pid}] {pname} — {len(ss)} times (mean score {np.mean(ss):.2f})")
    print("Score distribution:")
    for label, c in score_distribution(scores):
        print(f"  {label} : {c} frames ({100.0 * c / n:.0f}%)")


def main():
    ap = argparse.ArgumentParser(
        description="Compare distilled vs expanded-key retrieval on saved live captions")
    ap.add_argument("--distilled-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "distilled_index.npz"))
    ap.add_argument("--expanded-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "expanded_index.npz"))
    ap.add_argument("--results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_rag_online_results.json"))
    args = ap.parse_args()

    records = json.load(open(args.results, encoding="utf-8"))
    captions = [r["live_caption"] for r in records]
    print(f"loaded {len(captions)} live captions from {args.results}")

    old = PolicyRetriever(args.distilled_index)
    new = PolicyRetriever(args.expanded_index)
    print(f"distilled index: {len(old)} patterns, {len(old.key_embeddings)} keys")
    print(f"expanded index : {len(new)} patterns, {len(new.key_embeddings)} keys")

    old_hits = [old.retrieve(c, top_k=1)[0] for c in captions]
    new_hits = [new.retrieve(c, top_k=1)[0] for c in captions]
    old_scores = np.array([h["score"] for h in old_hits])
    new_scores = np.array([h["score"] for h in new_hits])

    print_block("ORIGINAL DISTILLED INDEX (single trigger key)", old_scores, old_hits)
    print_block("EXPANDED INDEX (trigger + paraphrase keys, max-pooled)",
                new_scores, new_hits)

    same = sum(o["pattern_id"] == n_["pattern_id"]
               for o, n_ in zip(old_hits, new_hits))
    print(f"\nTop-1 pattern agreement old vs new: {same}/{len(records)} "
          f"({100.0 * same / len(records):.0f}%)")
    print(f"Score delta per frame: mean {np.mean(new_scores - old_scores):+.3f}, "
          f"improved on {int((new_scores > old_scores).sum())}/{len(records)} frames")

    print("\n=== THRESHOLD SWEEP (expanded index, abstention rate) ===")
    print(f"{'threshold':>10} | {'retrieved':>9} | {'abstained':>9} | mean score of retrieved")
    for t in THRESHOLDS:
        kept = new_scores[new_scores >= t]
        print(f"{t:>10.2f} | {len(kept):>9} | {len(records) - len(kept):>9} | "
              f"{kept.mean():.3f}" if len(kept) else
              f"{t:>10.2f} | {0:>9} | {len(records):>9} | -")

    out_rows = []
    for r, oh, nh in zip(records, old_hits, new_hits):
        out_rows.append({
            "sample_idx": r["sample_idx"],
            "live_caption": r["live_caption"],
            "old_pattern_id": oh["pattern_id"], "old_score": oh["score"],
            "new_pattern_id": nh["pattern_id"],
            "new_pattern_name": nh["pattern_name"],
            "new_score": nh["score"],
            "new_matched_key": nh["matched_key"],
        })
    out_path = PROJECT_ROOT / "rag" / "eval_expanded_retrieval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)
    print(f"\nper-frame results saved to {out_path}")


if __name__ == "__main__":
    main()
