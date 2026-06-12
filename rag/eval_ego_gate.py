"""
eval_ego_gate.py
----------------
Ego-state-gated RAG ablation. RAG text injection only happens when the ego
state suggests a developing hazard (brake pressed or decel beyond a
threshold); otherwise the VLA runs on the live caption alone.

No new forward passes: per-frame ADE/FDE for both the plain and the
RAG-conditioned pass are already in rag/eval_rag_ablation_results.json, and
a gated run is a per-frame mix of the two. Run after eval_rag_ablation.py:

    python rag/eval_ego_gate.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))

from eval_rag_online import load_test_dataset  # noqa: E402

AEGO_THRESHOLDS = [-0.25, -0.5, -1.0, -1.5]  # m/s^2; gate fires at/below


def main():
    ap = argparse.ArgumentParser(description="Ego-state gated RAG ablation")
    ap.add_argument("--ablation-results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_rag_ablation_results.json"))
    args = ap.parse_args()

    recs = json.load(open(args.ablation_results, encoding="utf-8"))
    ds = load_test_dataset()
    states = [ds.samples[r["sample_idx"]]["state"] for r in recs]
    brake = np.array([float(s.get("brake", 0.0) or 0.0) for s in states])
    aego = np.array([float(s.get("aEgo", 0.0) or 0.0) for s in states])

    ade_plain = np.array([r["ade_no_rag"] for r in recs])
    fde_plain = np.array([r["fde_no_rag"] for r in recs])
    ade_rag = np.array([r["ade_rag"] for r in recs])
    fde_rag = np.array([r["fde_rag"] for r in recs])
    lat_plain = np.array([r["lat_no_rag_s"] for r in recs])
    lat_rag = np.array([r["lat_rag_s"] for r in recs])
    n = len(recs)

    print(f"frames: {n} | brake>0 on {(brake > 0).sum()} | "
          f"aEgo min {aego.min():.2f} m/s^2")
    print(f"\n{'variant':<34} {'fired':>7} {'ADE':>8} {'FDE':>8} {'lat ms':>8}")

    def row(name, gate):
        ade = np.where(gate, ade_rag, ade_plain)
        fde = np.where(gate, fde_rag, fde_plain)
        lat = np.where(gate, lat_rag, lat_plain)
        print(f"{name:<34} {int(gate.sum()):>4}/{n:<3} {ade.mean():>7.3f}m "
              f"{fde.mean():>7.3f}m {1e3 * lat.mean():>7.1f}")
        return ade, fde

    row("no RAG (never inject)", np.zeros(n, dtype=bool))
    row("always RAG", np.ones(n, dtype=bool))
    for t in AEGO_THRESHOLDS:
        gate = (brake > 0) | (aego <= t)
        ade, fde = row(f"gated: brake>0 or aEgo<={t}", gate)
        if 0 < gate.sum() < n:
            print(f"{'':<10}on fired frames: plain ADE "
                  f"{ade_plain[gate].mean():.3f}m -> RAG ADE "
                  f"{ade_rag[gate].mean():.3f}m")


if __name__ == "__main__":
    main()
