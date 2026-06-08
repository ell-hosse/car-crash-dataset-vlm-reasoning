"""
build_index.py
--------------
Embeds all structured triplet policies from crash_policies.jsonl using
sentence-transformers and saves a fast-lookup index to policy_index.npz.

Each policy produces TWO embeddings:
  - key_embedding:   encodes the "trigger" field only (used for retrieval)
  - value_embedding: encodes "latent_risk" + " " + "mitigation" (passed to LLM)

Usage:
    python build_index.py
    python build_index.py --policies crash_policies.jsonl --out policy_index.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_policies(jsonl_path: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load structured triplet policies from a JSONL file.

    Returns (triggers, latent_risks, mitigations, vidnames).
    Skips entries that have an "error" key or are missing any required field.
    """
    triggers, latent_risks, mitigations, vidnames = [], [], [], []
    required_keys = {"trigger", "latent_risk", "mitigation"}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Skip errored entries
            if record.get("error") is not None:
                continue

            # Skip entries missing any required field
            if not required_keys.issubset(record.keys()):
                continue

            trigger = record["trigger"].strip()
            latent_risk = record["latent_risk"].strip()
            mitigation = record["mitigation"].strip()

            if not trigger or not latent_risk or not mitigation:
                continue

            triggers.append(trigger)
            latent_risks.append(latent_risk)
            mitigations.append(mitigation)
            vidnames.append(record.get("clip_id", record.get("vidname", "unknown")))

    return triggers, latent_risks, mitigations, vidnames


def build_index(
    jsonl_path: Path,
    out_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> None:
    print(f"[build_index] Loading policies from: {jsonl_path}")
    triggers, latent_risks, mitigations, vidnames = load_policies(jsonl_path)
    print(f"[build_index] Found {len(triggers)} valid triplet policies.")

    print(f"[build_index] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Key embeddings — encode triggers only (used for retrieval)
    print("[build_index] Encoding key embeddings (triggers)...")
    key_embeddings = model.encode(
        triggers,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine sim = dot product after normalization
        convert_to_numpy=True,
    )

    # Value embeddings — encode latent_risk + mitigation (passed to LLM)
    print("[build_index] Encoding value embeddings (latent_risk + mitigation)...")
    value_texts = [f"{lr} {mt}" for lr, mt in zip(latent_risks, mitigations)]
    value_embeddings = model.encode(
        value_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    np.savez(
        out_path,
        key_embeddings=key_embeddings.astype(np.float32),
        value_embeddings=value_embeddings.astype(np.float32),
        triggers=np.array(triggers, dtype=object),
        latent_risks=np.array(latent_risks, dtype=object),
        mitigations=np.array(mitigations, dtype=object),
        vidnames=np.array(vidnames, dtype=object),
        model_name=np.array([model_name]),
    )
    print(f"[build_index] Index saved to: {out_path}  ({len(triggers)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Build policy vector index")
    parser.add_argument(
        "--policies",
        type=Path,
        default=Path("crash_policies.jsonl"),
        help="Path to crash_policies.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("policy_index.npz"),
        help="Output path for the index file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name",
    )
    args = parser.parse_args()

    if not args.policies.exists():
        raise FileNotFoundError(f"Policies file not found: {args.policies}")

    build_index(args.policies, args.out, model_name=args.model)


if __name__ == "__main__":
    main()
