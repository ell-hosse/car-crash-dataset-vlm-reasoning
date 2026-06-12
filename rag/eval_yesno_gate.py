"""
eval_yesno_gate.py
------------------
Makes a tiny VLM usable as a hazard gate by reading the verdict off its
logits instead of its generations: ask a one-word yes/no question and
compare P(yes) vs P(no) at the first output position. Compliance is 100%
by construction, the cost is a single forward pass (no decoding), and
P(yes) is a continuous gate score.

Evaluated on the same 350 frames as eval_verdict_gate.py (150 CCD
pre-crash, 200 benign CoVLA). Also reports a two-stage probe: for the
top-gated crash frames, a second "describe the danger" call whose output
is used as the retrieval query.

Run from the project root:
    python rag/eval_yesno_gate.py
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
from eval_ccd_retrieval import auc_score, tpr_at_fpr  # noqa: E402
from eval_rag_online import PILFrameCaptioner, load_test_dataset  # noqa: E402
from eval_two_tier_captioner import collect_frames  # noqa: E402
from retriever import PolicyRetriever  # noqa: E402

MODELS = [
    ("256M", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"),
    ("500M", "HuggingFaceTB/SmolVLM-500M-Instruct"),
]

YESNO_PROMPT = (
    "Look at this front-camera driving scene. Is there an immediate danger "
    "of a collision, such as a vehicle braking hard, stopped ahead, cutting "
    "into the lane, or coming head-on, or a person on the road? "
    "Answer with one word: yes or no."
)

DESCRIBE_PROMPT = (
    "Look at this front-camera driving scene. Describe the most likely "
    "danger in one short sentence: which vehicle or person, and what they "
    "are doing."
)


def yes_no_token_ids(tokenizer):
    yes_ids, no_ids = [], []
    for variants, out in [(("yes", "Yes", "YES", " yes", " Yes"), yes_ids),
                          (("no", "No", "NO", " no", " No"), no_ids)]:
        for v in variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                out.append(ids[0])
    return sorted(set(yes_ids)), sorted(set(no_ids))


@torch.no_grad()
def p_yes(captioner, bgr, yes_ids, no_ids):
    """P(yes | yes-or-no) from the logits of the first generated position."""
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    messages = [{"role": "user",
                 "content": [{"type": "image", "image": pil},
                             {"type": "text", "text": captioner.cfg.caption_prompt}]}]
    inputs = captioner.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(
            captioner.device, dtype=captioner.model.dtype)
    logits = captioner.model(**inputs).logits[0, -1].float()
    y = torch.logsumexp(logits[yes_ids], dim=0)
    n = torch.logsumexp(logits[no_ids], dim=0)
    return float(torch.sigmoid(y - n))


def main():
    ap = argparse.ArgumentParser(description="Yes/no logit gate for tiny VLMs")
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
                      caption_prompt=YESNO_PROMPT)
        captioner = PILFrameCaptioner(device, cfg=cfg)
        yes_ids, no_ids = yes_no_token_ids(captioner.processor.tokenizer)
        print(f"  yes tokens {yes_ids} | no tokens {no_ids}")
        rows, lats = [], []
        for n, (kind, ident, path) in enumerate(frames, 1):
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            t0 = time.time()
            p = p_yes(captioner, bgr, yes_ids, no_ids)
            lats.append(time.time() - t0)
            rows.append({"kind": kind, "id": ident, "p_yes": p, "path": str(path)})
            if n % 100 == 0:
                print(f"  {n}/{len(frames)} (mean latency {np.mean(lats):.3f}s)")

        crash = np.array([r["p_yes"] for r in rows if r["kind"] == "crash"])
        benign = np.array([r["p_yes"] for r in rows if r["kind"] == "benign"])
        auc = auc_score(crash, benign)
        print(f"  AUC {auc:.3f} | crash P(yes) {crash.mean():.3f} vs "
              f"benign {benign.mean():.3f} | latency {np.mean(lats):.3f}s")
        for fpr in (0.05, 0.10, 0.20):
            tpr, thr = tpr_at_fpr(crash, benign, fpr)
            print(f"    at {fpr:.0%} benign FP (P(yes)>={thr:.3f}): "
                  f"catches {tpr:.0%} of crash frames")

        # two-stage probe: describe + retrieve on the most-gated crash frames
        cfg2 = replace(REALTIME, captioner_model=model_id,
                       caption_prompt=DESCRIBE_PROMPT, caption_max_new_tokens=48)
        captioner.cfg = cfg2
        gated = sorted([r for r in rows if r["kind"] == "crash"],
                       key=lambda r: -r["p_yes"])[:5]
        print("  stage-2 descriptions on top-gated crash frames:")
        for r in gated:
            desc = captioner._generate(cv2.imread(r["path"]))
            hit = retriever.retrieve(desc, top_k=1)[0]
            print(f"    [{r['id']}] p={r['p_yes']:.2f} -> [{hit['pattern_id']}] "
                  f"{hit['pattern_name'][:38]} ({hit['score']:.2f}) | {desc[:110]}")

        results[tag] = {"auc": auc, "mean_latency_s": float(np.mean(lats)),
                        "rows": rows}
        del captioner.model, captioner
        torch.cuda.empty_cache()

    out_path = PROJECT_ROOT / "rag" / "eval_yesno_gate_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nfull results saved to {out_path}")


if __name__ == "__main__":
    main()
