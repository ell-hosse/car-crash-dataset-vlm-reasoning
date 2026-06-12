"""
eval_verdict_gate.py
--------------------
Verdict-first vision gate: the captioner must answer "SAFE" or
"HAZARD: <description>" — the verdict token is the gate (replacing
embedding-score thresholds) and the description is the retrieval query
when the verdict is HAZARD.

Tested at three model sizes on the same 350 frames as the earlier ablations
(150 CCD pre-crash, 200 benign CoVLA), since the target platform is a Jetson
and the smallest sufficient model wins:

    SmolVLM2-256M-Video-Instruct   (current tier-1)
    SmolVLM-500M-Instruct          (middle Jetson candidate)
    SmolVLM2-2.2B-Instruct         (quality ceiling)

Reports per model: verdict compliance, crash recall / benign FPR of the
HAZARD verdict, retrieval over hazard descriptions, and caption latency.

Run from the project root:
    python rag/eval_verdict_gate.py
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
from eval_rag_online import PILFrameCaptioner, load_test_dataset  # noqa: E402
from eval_two_tier_captioner import collect_frames  # noqa: E402
from retriever import PolicyRetriever  # noqa: E402

MODELS = [
    ("256M", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"),
    ("500M", "HuggingFaceTB/SmolVLM-500M-Instruct"),
    ("2.2B", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
]

VERDICT_PROMPT = (
    "Look at this front-camera driving scene. Decide if there is a concrete, "
    "immediate hazard: a vehicle braking hard, stopped in your path, cutting "
    "into your lane, turning across your path, or approaching head-on; a "
    "pedestrian, cyclist or animal on the road ahead; or a slippery or "
    "blocked road surface. Normal flowing traffic, parked cars, traffic "
    "lights, and vehicles staying in their own lanes are NOT hazards. "
    "Reply with exactly one line: SAFE if there is no such hazard, or "
    "HAZARD: <one sentence saying who or what is doing what, specifically>."
)

MAX_NEW_TOKENS = 48


def parse_verdict(text: str):
    """Returns (verdict, description). verdict in {SAFE, HAZARD, UNPARSEABLE}."""
    head = text.strip().lstrip("\"'*. ").upper()
    if head.startswith("SAFE"):
        return "SAFE", ""
    if head.startswith("HAZARD"):
        desc = text.strip()
        desc = desc[desc.index(":") + 1:].strip() if ":" in desc[:12] else desc
        return "HAZARD", desc
    return "UNPARSEABLE", text.strip()


def main():
    ap = argparse.ArgumentParser(description="Verdict-first vision gate ablation")
    ap.add_argument("--ccd-results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_ccd_retrieval_results.json"))
    ap.add_argument("--benign-results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_rag_online_results.json"))
    ap.add_argument("--distilled-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "distilled_index.npz"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_test_dataset()
    frames = collect_frames(args.ccd_results, args.benign_results, dataset)
    n_crash = sum(k == "crash" for k, _, _ in frames)
    print(f"{len(frames)} frames ({n_crash} crash, {len(frames) - n_crash} benign)")
    retriever = PolicyRetriever(args.distilled_index)

    results = {}
    for tag, model_id in MODELS:
        print(f"\n--- {tag}: {model_id} ---")
        cfg = replace(REALTIME, captioner_model=model_id,
                      caption_prompt=VERDICT_PROMPT,
                      caption_max_new_tokens=MAX_NEW_TOKENS)
        try:
            captioner = PILFrameCaptioner(device, cfg=cfg)  # synchronous use
        except Exception as e:  # noqa: BLE001
            print(f"  SKIPPING {tag}: failed to load ({e})")
            continue
        rows, lats = [], []
        for n, (kind, ident, path) in enumerate(frames, 1):
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            t0 = time.time()
            raw = captioner._generate(bgr)
            lats.append(time.time() - t0)
            verdict, desc = parse_verdict(raw)
            row = {"kind": kind, "id": ident, "raw": raw,
                   "verdict": verdict, "description": desc}
            if verdict == "HAZARD" and desc:
                hit = retriever.retrieve(desc, top_k=1)[0]
                row.update(score=hit["score"], pattern_id=hit["pattern_id"],
                           pattern_name=hit["pattern_name"])
            rows.append(row)
            if n % 100 == 0:
                print(f"  {n}/{len(frames)} (mean latency {np.mean(lats):.2f}s)")
        results[tag] = {"rows": rows, "mean_latency_s": float(np.mean(lats)),
                        "p95_latency_s": float(np.percentile(lats, 95))}
        del captioner.model, captioner
        torch.cuda.empty_cache()

    print("\n================ VERDICT GATE SUMMARY ================")
    print(f"{'model':<6} {'compliance':>10} {'crash recall':>13} "
          f"{'benign FPR':>11} {'lat mean':>9} {'lat p95':>8}")
    for tag, _ in MODELS:
        if tag not in results:
            continue
        rows = results[tag]["rows"]
        ok = [r for r in rows if r["verdict"] != "UNPARSEABLE"]
        crash = [r for r in rows if r["kind"] == "crash"]
        benign = [r for r in rows if r["kind"] == "benign"]
        recall = np.mean([r["verdict"] == "HAZARD" for r in crash])
        fpr = np.mean([r["verdict"] == "HAZARD" for r in benign])
        print(f"{tag:<6} {len(ok)}/{len(rows):>6} {recall:>12.0%} "
              f"{fpr:>10.0%} {results[tag]['mean_latency_s']:>8.2f}s "
              f"{results[tag]['p95_latency_s']:>7.2f}s")

    for tag, _ in MODELS:
        if tag not in results:
            continue
        rows = results[tag]["rows"]
        hz = [r for r in rows if r["kind"] == "crash" and r["verdict"] == "HAZARD"
              and "score" in r]
        if not hz:
            continue
        print(f"\n--- {tag}: retrieval over crash HAZARD descriptions "
              f"(n={len(hz)}, mean score {np.mean([r['score'] for r in hz]):.3f}) ---")
        by_pattern = {}
        for r in hz:
            by_pattern.setdefault((r["pattern_id"], r["pattern_name"]), []).append(r["score"])
        for (pid, pname), ss in sorted(by_pattern.items(),
                                       key=lambda kv: len(kv[1]), reverse=True)[:5]:
            print(f"  [{pid}] {pname} — {len(ss)}x (mean {np.mean(ss):.2f})")
        print("  examples:")
        for r in hz[:3]:
            print(f"    [{r['id']}] {r['description'][:150]}")

    out_path = PROJECT_ROOT / "rag" / "eval_verdict_gate_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nfull results saved to {out_path}")


if __name__ == "__main__":
    main()
