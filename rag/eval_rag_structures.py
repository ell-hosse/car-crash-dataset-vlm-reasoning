"""
eval_rag_structures.py
----------------------
Always-on (ungated) RAG injection structure ablation on the 200 benign
CoVLA frames. The question: how much of the always-on harm (ADE 0.616m ->
1.288m with full-text prepend) is the injection *structure* rather than
the injection itself? On benign data the ceiling for any variant is the
plain baseline, so this measures harm, not benefit.

Variants (no retraining, retrieval always on, 256M live captions as query):
  prepend_full   : "{latent_risk} {mitigation} {caption}"  (original)
  prepend_mit    : "{mitigation} {caption}"                 (shorter text)
  blend_a25/a50  : CLIP-embedding interpolation (1-a)*caption + a*pattern
  two_token      : caption and pattern as two separate text tokens

Run from the project root, after eval_rag_online.py:
    python rag/eval_rag_structures.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))

from covla_vla.dataset import denormalize_traj  # noqa: E402
from covla_vla.model import ade_fde  # noqa: E402
from eval_rag_online import load_model, load_test_dataset  # noqa: E402
from retriever import PolicyRetriever  # noqa: E402


@torch.no_grad()
def encode(model, tokenizer, text, device):
    tok = tokenizer([text], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    return model.encode_text(tok["input_ids"], tok["attention_mask"])  # (1,1,d)


def main():
    ap = argparse.ArgumentParser(description="Ungated RAG injection structure ablation")
    ap.add_argument("--ckpt", type=str,
                    default=str(PROJECT_ROOT / "rag" / "covla_vla_best.pt"))
    ap.add_argument("--distilled-index", type=str,
                    default=str(PROJECT_ROOT / "rag" / "distilled_index.npz"))
    ap.add_argument("--online-results", type=str,
                    default=str(PROJECT_ROOT / "rag" / "eval_rag_online_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_test_dataset()
    model, tokenizer = load_model(args.ckpt, device)
    retriever = PolicyRetriever(args.distilled_index)
    records = json.load(open(args.online_results, encoding="utf-8"))

    variants = ["plain", "prepend_full", "prepend_mit",
                "blend_a25", "blend_a50", "two_token"]
    res = {v: {"ade": [], "fde": []} for v in variants}

    with torch.no_grad():
        for n, r in enumerate(records, 1):
            item = dataset[r["sample_idx"]]
            img = item["image"].unsqueeze(0).to(device)
            st = item["state"].unsqueeze(0).to(device)
            gt_m = torch.from_numpy(
                denormalize_traj(item["traj"].unsqueeze(0).float().numpy()))
            cap = r["live_caption"]
            hit = retriever.retrieve(cap, top_k=1)[0]
            pattern_text = f"{hit['latent_risk']} {hit['mitigation']}"

            cap_emb = encode(model, tokenizer, cap, device)
            pat_emb = encode(model, tokenizer, pattern_text, device)
            texts = {
                "plain": cap_emb,
                "prepend_full": encode(model, tokenizer,
                                       f"{pattern_text} {cap}", device),
                "prepend_mit": encode(model, tokenizer,
                                      f"{hit['mitigation']} {cap}", device),
                "blend_a25": 0.75 * cap_emb + 0.25 * pat_emb,
                "blend_a50": 0.50 * cap_emb + 0.50 * pat_emb,
                "two_token": torch.cat([cap_emb, pat_emb], dim=1),
            }
            for v, emb in texts.items():
                pred = model(img, st, text_embed=emb)
                pred_m = torch.from_numpy(
                    denormalize_traj(pred.float().cpu().numpy()))
                a, f = ade_fde(pred_m, gt_m)
                res[v]["ade"].append(a)
                res[v]["fde"].append(f)
            if n % 50 == 0:
                print(f"  {n}/{len(records)} frames")

    print(f"\n{'variant (always-on, no gate)':<30} {'ADE':>8} {'FDE':>8} "
          f"{'vs plain':>9}")
    base = np.mean(res["plain"]["ade"])
    for v in variants:
        ade = np.mean(res[v]["ade"])
        fde = np.mean(res[v]["fde"])
        print(f"{v:<30} {ade:>7.3f}m {fde:>7.3f}m {ade - base:>+8.3f}m")

    out_path = PROJECT_ROOT / "rag" / "eval_rag_structures_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({v: {"ade": float(np.mean(res[v]["ade"])),
                       "fde": float(np.mean(res[v]["fde"]))}
                   for v in variants}, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
