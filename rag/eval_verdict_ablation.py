"""
eval_verdict_ablation.py
------------------------
Full RAG ablation on the 200 benign CoVLA frames with the 2.2B verdict gate:
RAG text is injected only when the 2.2B verdict (from eval_verdict_gate.py)
says HAZARD, and the retrieval query is the verdict's hazard description
rather than the generic live caption.

SAFE frames reuse the plain per-frame ADE/FDE from eval_rag_ablation.py;
only HAZARD frames need new forward passes. The verdict itself runs in the
async captioner thread at deployment, so it never sits on the 10 Hz
trajectory path — its latency is reported separately.

Run from the project root, after eval_rag_ablation.py and eval_verdict_gate.py:
    python rag/eval_verdict_ablation.py --ckpt covla_vla_best.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))

from covla_vla.dataset import denormalize_traj  # noqa: E402
from covla_vla.model import ade_fde  # noqa: E402
from eval_rag_ablation import RAG_TEMPLATE, forward_with_caption  # noqa: E402
from eval_rag_online import load_model, load_test_dataset  # noqa: E402
from retriever import PolicyRetriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="RAG ablation gated by the 2.2B verdict")
    ap.add_argument("--ckpt", type=str,
                    default=str(PROJECT_ROOT / "rag" / "covla_vla_best.pt"))
    ap.add_argument("--distilled-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "distilled_index.npz"))
    ap.add_argument("--verdict-results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_verdict_gate_results.json"))
    ap.add_argument("--ablation-results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_rag_ablation_results.json"))
    ap.add_argument("--online-results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_rag_online_results.json"))
    args = ap.parse_args()

    # 2.2B verdicts for the benign CoVLA frames, keyed by sample_idx
    verdict_rows = json.load(open(args.verdict_results, encoding="utf-8"))["2.2B"]["rows"]
    verdicts = {int(r["id"].split("_")[1]): r for r in verdict_rows
                if r["kind"] == "benign"}

    abl = {r["sample_idx"]: r
           for r in json.load(open(args.ablation_results, encoding="utf-8"))}
    captions = {r["sample_idx"]: r["live_caption"]
                for r in json.load(open(args.online_results, encoding="utf-8"))}
    idxs = sorted(abl.keys())
    fired = [i for i in idxs if verdicts[i]["verdict"] == "HAZARD"]
    print(f"frames: {len(idxs)} | verdict gate fires on {len(fired)} "
          f"({100 * len(fired) / len(idxs):.0f}%)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_test_dataset()
    model, tokenizer = load_model(args.ckpt, device)
    retriever = PolicyRetriever(args.distilled_index)

    # New forwards only for fired frames: retrieve on the hazard description,
    # inject retrieved pattern text ahead of the live caption.
    rag_on_fired = {}
    for i in fired:
        item = dataset[i]
        img = item["image"].unsqueeze(0).to(device)
        st = item["state"].unsqueeze(0).to(device)
        gt_m = torch.from_numpy(
            denormalize_traj(item["traj"].unsqueeze(0).float().numpy()))
        desc = verdicts[i]["description"] or verdicts[i]["raw"]
        t0 = time.perf_counter()
        hit = retriever.retrieve(desc, top_k=1)[0]
        rag_caption = RAG_TEMPLATE.format(
            latent_risk=hit["latent_risk"], mitigation=hit["mitigation"],
            caption=captions[i])
        pred = forward_with_caption(model, tokenizer, img, st, rag_caption, device)
        lat = time.perf_counter() - t0
        pred_m = torch.from_numpy(denormalize_traj(pred.float().cpu().numpy()))
        a, f = ade_fde(pred_m, gt_m)
        rag_on_fired[i] = {"ade": a, "fde": f, "lat": lat,
                           "pattern_id": hit["pattern_id"],
                           "pattern_name": hit["pattern_name"],
                           "score": hit["score"], "description": desc}
    print(f"ran {len(rag_on_fired)} RAG forwards on fired frames")

    fired_set = set(fired)
    ade_p = np.array([abl[i]["ade_no_rag"] for i in idxs])
    fde_p = np.array([abl[i]["fde_no_rag"] for i in idxs])
    lat_p = np.array([abl[i]["lat_no_rag_s"] for i in idxs])
    ade_g = np.array([rag_on_fired[i]["ade"] if i in fired_set
                      else abl[i]["ade_no_rag"] for i in idxs])
    fde_g = np.array([rag_on_fired[i]["fde"] if i in fired_set
                      else abl[i]["fde_no_rag"] for i in idxs])
    lat_g = np.array([rag_on_fired[i]["lat"] if i in fired_set
                      else abl[i]["lat_no_rag_s"] for i in idxs])
    ade_a = np.array([abl[i]["ade_rag"] for i in idxs])
    fde_a = np.array([abl[i]["fde_rag"] for i in idxs])
    lat_a = np.array([abl[i]["lat_rag_s"] for i in idxs])

    print(f"\n{'variant':<36} {'fired':>8} {'ADE':>8} {'FDE':>8} {'traj lat':>9}")
    for name, ade, fde, lat, nf in [
            ("no RAG (never inject)", ade_p, fde_p, lat_p, 0),
            ("always RAG (caption retrieval)", ade_a, fde_a, lat_a, len(idxs)),
            ("2.2B verdict-gated RAG", ade_g, fde_g, lat_g, len(fired))]:
        print(f"{name:<36} {nf:>4}/{len(idxs):<3} {ade.mean():>7.3f}m "
              f"{fde.mean():>7.3f}m {1e3 * lat.mean():>7.1f}ms")

    if fired:
        fa = np.array([abl[i]["ade_no_rag"] for i in fired])
        fr = np.array([rag_on_fired[i]["ade"] for i in fired])
        print(f"\non the {len(fired)} fired frames: plain ADE {fa.mean():.3f}m "
              f"-> RAG ADE {fr.mean():.3f}m")
        print("fired-frame details:")
        for i in fired:
            r = rag_on_fired[i]
            print(f"  [{i:>4}] {abl[i]['ade_no_rag']:.2f}m -> {r['ade']:.2f}m | "
                  f"[{r['pattern_id']}] {r['pattern_name'][:42]} "
                  f"(score {r['score']:.2f}) | {r['description'][:80]}")

    out = {
        "n_frames": len(idxs), "n_fired": len(fired),
        "summary": {
            "no_rag": {"ade": float(ade_p.mean()), "fde": float(fde_p.mean())},
            "always_rag": {"ade": float(ade_a.mean()), "fde": float(fde_a.mean())},
            "verdict_gated": {"ade": float(ade_g.mean()), "fde": float(fde_g.mean())},
        },
        "fired_frames": {str(i): rag_on_fired[i] for i in fired},
    }
    out_path = PROJECT_ROOT / "rag" / "eval_verdict_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
