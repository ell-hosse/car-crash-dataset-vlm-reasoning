"""
retriever.py
------------
Vector retrieval over the policy/pattern indexes produced by build_index.py
(policy_index.npz) and distill_patterns.py (distilled_index.npz).

Both index formats share the same npz layout:
  key_embeddings   (N, D) float32, L2-normalized trigger embeddings
  value_embeddings (N, D) float32, L2-normalized latent_risk+mitigation embeddings
  triggers / latent_risks / mitigations / vidnames  (N,) object arrays
  pattern_names    (N,) object array (distilled index only)
  model_name       (1,) sentence-transformer used to build the index

For the distilled index, `vidnames` holds the pattern ids (P000, P001, ...).

Usage:
    from retriever import PolicyRetriever
    r = PolicyRetriever("distilled_index.npz")
    hits = r.retrieve("the ego vehicle is approaching a stopped car", top_k=3)
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class PolicyRetriever:
    def __init__(self, index_path="policy_index.npz", model_name: str = None):
        data = np.load(index_path, allow_pickle=True)
        self.key_embeddings = data["key_embeddings"].astype(np.float32)
        self.value_embeddings = data["value_embeddings"].astype(np.float32)
        self.triggers = data["triggers"]
        self.latent_risks = data["latent_risks"]
        self.mitigations = data["mitigations"]
        self.vidnames = data["vidnames"]
        self.pattern_names = (
            data["pattern_names"] if "pattern_names" in data.files else None)
        # Expanded indexes (rag/expand_trigger_keys.py) carry several keys per
        # entry; key_pattern_idx maps each key row to its entry and scores are
        # max-pooled per entry. Plain indexes are one key per entry.
        if "key_pattern_idx" in data.files:
            self.key_pattern_idx = data["key_pattern_idx"].astype(np.int64)
            self.key_texts = data["key_texts"]
        else:
            self.key_pattern_idx = np.arange(len(self.triggers), dtype=np.int64)
            self.key_texts = self.triggers
        self.model_name = model_name or str(data["model_name"][0])
        self.model = SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True,
        )[0].astype(np.float32)

    def retrieve(self, query: str, top_k: int = 1,
                 threshold: float = None) -> list[dict]:
        """Return the top_k entries by cosine similarity between the query
        and the key embeddings (max-pooled per entry when an entry has
        multiple keys), best first. With a threshold, entries scoring below
        it are dropped — the list may then be empty (abstention)."""
        q = self.embed_query(query)
        key_scores = self.key_embeddings @ q  # embeddings are normalized
        n = len(self.triggers)
        scores = np.full(n, -np.inf, dtype=np.float32)
        best_key = np.zeros(n, dtype=np.int64)  # best-matching key, for inspection
        for k, p in enumerate(self.key_pattern_idx):
            if key_scores[k] > scores[p]:
                scores[p] = key_scores[k]
                best_key[p] = k

        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in order:
            i = int(i)
            if threshold is not None and scores[i] < threshold:
                break  # scores are sorted; everything after is lower
            results.append({
                "index": i,
                "score": float(scores[i]),
                "trigger": str(self.triggers[i]),
                "matched_key": str(self.key_texts[best_key[i]]),
                "latent_risk": str(self.latent_risks[i]),
                "mitigation": str(self.mitigations[i]),
                "pattern_id": str(self.vidnames[i]),
                "pattern_name": (str(self.pattern_names[i])
                                 if self.pattern_names is not None
                                 else str(self.vidnames[i])),
            })
        return results

    def __len__(self):
        return len(self.triggers)
