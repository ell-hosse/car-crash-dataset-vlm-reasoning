"""
eval_rag_ablation.py
--------------------
ADE/FDE + latency ablation on the frames evaluated by eval_rag_online.py:

  A. no-RAG : VLA conditioned on the live SmolVLM2 caption alone
  B. RAG    : VLA conditioned on the retrieved pattern (latent_risk +
              mitigation) prepended to the live caption

Live captions are reused from rag/eval_rag_online_results.json (no
re-captioning), so run eval_rag_online.py first. Latency is the full text
path per frame: (retrieval for B) + tokenize + text encode + fused forward.

Run from the project root:
    python rag/eval_rag_ablation.py --ckpt covla_vla_best.pt
"""

import argparse
import json
import time

import numpy as np
import torch

from eval_rag_online import (  # noqa: F401 (applies PREPROCESSED_ROOT fallback)
    PROJECT_ROOT, load_model, load_test_dataset)
from covla_vla.dataset import denormalize_traj
from covla_vla.model import ade_fde
from retriever import PolicyRetriever

# CLIP text tower truncates at 77 tokens: the retrieved text goes first so
# it survives truncation; the caption tail is what gets cut if anything.
RAG_TEMPLATE = "{latent_risk} {mitigation} {caption}"


@torch.no_grad()
def forward_with_caption(model, tokenizer, image, state, caption, device):
    tok = tokenizer([caption], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    pred = model(image, state, input_ids=tok["input_ids"],
                 attention_mask=tok["attention_mask"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return pred


def summarize(name, ades, fdes, lats):
    lats_ms = 1e3 * np.asarray(lats)
    print(f"\n=== {name} ===")
    print(f"Frames  : {len(ades)}")
    print(f"ADE     : {np.mean(ades):.3f} m")
    print(f"FDE     : {np.mean(fdes):.3f} m")
    print(f"Latency : {lats_ms.mean():.1f} ms mean | "
          f"{np.percentile(lats_ms, 95):.1f} ms p95 "
          f"(~{1e3 / lats_ms.mean():.0f} Hz)")


def main():
    ap = argparse.ArgumentParser(
        description="ADE/FDE + latency with vs without RAG text conditioning")
    ap.add_argument("--ckpt", type=str,
                    default=str(PROJECT_ROOT / "rag" / "covla_vla_best.pt"))
    ap.add_argument("--distilled-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "distilled_index.npz"))
    ap.add_argument("--results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_rag_online_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = json.load(open(args.results, encoding="utf-8"))
    print(f"loaded {len(records)} live-captioned frames from {args.results}")

    dataset = load_test_dataset()
    model, tokenizer = load_model(args.ckpt, device)
    retriever = PolicyRetriever(args.distilled_index)

    # warm up GPU + sentence-transformer before timing
    item = dataset[records[0]["sample_idx"]]
    img = item["image"].unsqueeze(0).to(device)
    st = item["state"].unsqueeze(0).to(device)
    for _ in range(3):
        forward_with_caption(model, tokenizer, img, st, "warmup", device)
        retriever.retrieve("warmup", top_k=1)

    res = {v: {"ade": [], "fde": [], "lat": []} for v in ("no_rag", "rag")}
    out_rows = []
    for n, r in enumerate(records, 1):
        item = dataset[r["sample_idx"]]
        img = item["image"].unsqueeze(0).to(device)
        st = item["state"].unsqueeze(0).to(device)
        gt_m = torch.from_numpy(
            denormalize_traj(item["traj"].unsqueeze(0).float().numpy()))
        caption = r["live_caption"]

        # A: live caption only
        t0 = time.perf_counter()
        pred = forward_with_caption(model, tokenizer, img, st, caption, device)
        lat_a = time.perf_counter() - t0
        pred_m = torch.from_numpy(denormalize_traj(pred.float().cpu().numpy()))
        ade_a, fde_a = ade_fde(pred_m, gt_m)

        # B: retrieval + retrieved pattern text prepended to the caption
        t0 = time.perf_counter()
        hit = retriever.retrieve(caption, top_k=1)[0]
        rag_caption = RAG_TEMPLATE.format(
            latent_risk=hit["latent_risk"], mitigation=hit["mitigation"],
            caption=caption)
        pred = forward_with_caption(model, tokenizer, img, st, rag_caption, device)
        lat_b = time.perf_counter() - t0
        pred_m = torch.from_numpy(denormalize_traj(pred.float().cpu().numpy()))
        ade_b, fde_b = ade_fde(pred_m, gt_m)

        for v, a, f, l in (("no_rag", ade_a, fde_a, lat_a),
                           ("rag", ade_b, fde_b, lat_b)):
            res[v]["ade"].append(a)
            res[v]["fde"].append(f)
            res[v]["lat"].append(l)
        out_rows.append({
            "sample_idx": r["sample_idx"],
            "ade_gt_caption": r["ade"], "fde_gt_caption": r["fde"],
            "ade_no_rag": ade_a, "fde_no_rag": fde_a, "lat_no_rag_s": lat_a,
            "ade_rag": ade_b, "fde_rag": fde_b, "lat_rag_s": lat_b,
            "retrieved_pattern_id": hit["pattern_id"],
        })
        if n % 50 == 0:
            print(f"  {n}/{len(records)} frames")

    # GT-caption reference on the same frames (from Test 1 of eval_rag_online)
    print("\n=== REFERENCE: GT CAPTIONS, NO RAG (same frames, from Test 1) ===")
    print(f"ADE     : {np.mean([r['ade'] for r in records]):.3f} m")
    print(f"FDE     : {np.mean([r['fde'] for r in records]):.3f} m")

    summarize("LIVE CAPTIONS, NO RAG", **{k: v for k, v in zip(
        ("ades", "fdes", "lats"), res["no_rag"].values())})
    summarize("LIVE CAPTIONS + RAG (retrieved risk+mitigation prepended)",
              **{k: v for k, v in zip(
                  ("ades", "fdes", "lats"), res["rag"].values())})

    out_path = PROJECT_ROOT / "rag" / "eval_rag_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2)
    print(f"\nper-frame results saved to {out_path}")


if __name__ == "__main__":
    main()
