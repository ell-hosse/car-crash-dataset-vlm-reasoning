"""
expand_trigger_keys.py
----------------------
Improves retrieval recall by giving each distilled pattern multiple keys:
the canonical trigger plus N Gemini-generated paraphrases written in the
style of the live SmolVLM2 dashcam captions that serve as queries at
inference time. Retrieval then max-pools key scores per pattern.

Reads patterns straight from distilled_index.npz (no other inputs needed).

Run from the project root:
    python rag/expand_trigger_keys.py
    python rag/expand_trigger_keys.py --n-paraphrases 8 \
        --distilled-index distilled_index.npz --out rag/expanded_index.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuses the Gemini setup (key loading, retry, JSON extraction) and the
# embedding model name from the distillation pipeline.
import google.generativeai as genai  # noqa: E402
from distill_patterns import CALL_DELAY, GEMINI_MODEL, MODEL_NAME, call_gemini  # noqa: E402
import os  # noqa: E402

PARAPHRASE_TEMPLATE = """A small vision-language model watches a front dashcam and describes driving \
scenes in simple, concrete language like this example:
"The vehicle is driving on a busy street, passing by a traffic light, and \
passing by a black truck. The driver should be careful about the traffic \
lights and the road conditions."

Below is one abstract crash-risk pattern:
PATTERN NAME: {pattern_name}
TRIGGER: {trigger}
LATENT RISK: {latent_risk}

Write {n} different one-sentence scene descriptions, in the same simple \
dashcam-caption style as the example, each describing an observable scene \
where this pattern applies. Vary the road type, weather, vehicles and \
phrasing. Describe only what a camera can see (no abstract risk language).

Respond with ONLY a valid JSON object: {{"paraphrases": ["...", "..."]}}"""


def load_patterns(index_path: Path) -> list[dict]:
    data = np.load(index_path, allow_pickle=True)
    return [
        {
            "pattern_id": str(data["vidnames"][i]),
            "pattern_name": str(data["pattern_names"][i]),
            "trigger": str(data["triggers"][i]),
            "latent_risk": str(data["latent_risks"][i]),
            "mitigation": str(data["mitigations"][i]),
        }
        for i in range(len(data["triggers"]))
    ]


def main():
    ap = argparse.ArgumentParser(description="Expand pattern trigger keys with paraphrases")
    ap.add_argument("--distilled-index", type=Path,
                    default=PROJECT_ROOT / "rag" / "distilled_index.npz")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "rag" / "expanded_index.npz")
    ap.add_argument("--paraphrases-out", type=Path,
                    default=PROJECT_ROOT / "rag" / "trigger_paraphrases.jsonl")
    ap.add_argument("--n-paraphrases", type=int, default=8)
    ap.add_argument("--skip-generation", action="store_true",
                    help="reuse existing trigger_paraphrases.jsonl, only rebuild the index")
    args = ap.parse_args()

    patterns = load_patterns(args.distilled_index)
    print(f"loaded {len(patterns)} patterns from {args.distilled_index}")

    if args.skip_generation:
        rows = [json.loads(l) for l in
                args.paraphrases_out.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_id = {r["pattern_id"]: r["paraphrases"] for r in rows}
    else:
        api_key = os.getenv("GEM_KEY03")
        if not api_key:
            raise EnvironmentError("GEM_KEY03 not set (distill_patterns loads .env)")
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)

        by_id = {}
        with open(args.paraphrases_out, "w", encoding="utf-8") as f:
            for i, p in enumerate(tqdm(patterns, desc="Paraphrasing triggers")):
                prompt = PARAPHRASE_TEMPLATE.format(
                    pattern_name=p["pattern_name"], trigger=p["trigger"],
                    latent_risk=p["latent_risk"], n=args.n_paraphrases)
                result = call_gemini(prompt, gemini_model)
                paras = [s.strip() for s in result.get("paraphrases", []) if s.strip()]
                by_id[p["pattern_id"]] = paras
                f.write(json.dumps({
                    "pattern_id": p["pattern_id"],
                    "pattern_name": p["pattern_name"],
                    "trigger": p["trigger"],
                    "paraphrases": paras,
                }, ensure_ascii=False) + "\n")
                f.flush()
                if i < len(patterns) - 1:
                    time.sleep(CALL_DELAY)
        print(f"paraphrases saved to {args.paraphrases_out}")

    # ---- build expanded index: keys = [trigger] + paraphrases per pattern ----
    key_texts, key_pattern_idx = [], []
    for i, p in enumerate(patterns):
        for text in [p["trigger"], *by_id.get(p["pattern_id"], [])]:
            key_texts.append(text)
            key_pattern_idx.append(i)
    print(f"{len(key_texts)} keys for {len(patterns)} patterns "
          f"(mean {len(key_texts) / len(patterns):.1f} keys/pattern)")

    st_model = SentenceTransformer(MODEL_NAME)
    key_embeddings = st_model.encode(
        key_texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    value_texts = [f"{p['latent_risk']} {p['mitigation']}" for p in patterns]
    value_embeddings = st_model.encode(
        value_texts, batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True).astype(np.float32)

    np.savez(
        args.out,
        key_embeddings=key_embeddings,
        key_pattern_idx=np.array(key_pattern_idx, dtype=np.int64),
        key_texts=np.array(key_texts, dtype=object),
        value_embeddings=value_embeddings,
        triggers=np.array([p["trigger"] for p in patterns], dtype=object),
        latent_risks=np.array([p["latent_risk"] for p in patterns], dtype=object),
        mitigations=np.array([p["mitigation"] for p in patterns], dtype=object),
        vidnames=np.array([p["pattern_id"] for p in patterns], dtype=object),
        pattern_names=np.array([p["pattern_name"] for p in patterns], dtype=object),
        model_name=np.array([MODEL_NAME]),
    )
    print(f"expanded index saved to {args.out}")


if __name__ == "__main__":
    main()
