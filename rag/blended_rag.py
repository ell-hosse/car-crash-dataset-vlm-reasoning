"""
blended_rag.py
--------------
The project's main RAG configuration as a reusable component: ungated
embedding-blend injection (see eval_rag_structures.py for the ablation
that selected it).

Every caption is matched against the distilled crash-pattern index, and the
VLA's text token becomes a convex blend in CLIP embedding space:

    text_embed = (1 - alpha) * CLIP(caption) + alpha * CLIP(risk + mitigation)

No gate, no string concatenation, no truncation. alpha = 0.25 measured
harm-free on benign CoVLA (ADE 0.602m vs 0.616m plain) while improving the
hazard subset (ADE 0.490m -> 0.467m, FDE 1.324m -> 1.227m).

Usage with the realtime loop (model/tokenizer from covla_vla):

    from blended_rag import BlendedRAG
    rag = BlendedRAG(model, tokenizer, device)
    ...
    emb, hit = rag.text_embed(captioner.caption)   # cached per caption
    pred = model(image, state, text_embed=emb)
    # hit carries pattern_id / pattern_name / mitigation for display or logs
"""

import torch

from retriever import PolicyRetriever

DEFAULT_ALPHA = 0.25


class BlendedRAG:
    def __init__(self, model, tokenizer, device,
                 index_path="rag/distilled_index.npz",
                 alpha: float = DEFAULT_ALPHA):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.retriever = PolicyRetriever(index_path)
        self.alpha = alpha
        self._cached_caption = None
        self._cached = None  # (embed, hit)

    @torch.no_grad()
    def _encode(self, text: str) -> torch.Tensor:
        tok = self.tokenizer([text], padding=True, truncation=True,
                             max_length=77, return_tensors="pt").to(self.device)
        return self.model.encode_text(tok["input_ids"], tok["attention_mask"])

    @torch.no_grad()
    def text_embed(self, caption: str):
        """Blended text token (1, 1, d) + the retrieved pattern dict.
        Retrieval and encoding run once per distinct caption (the captioner
        updates ~1 Hz; the 10 Hz trajectory loop reuses the cache)."""
        if caption != self._cached_caption:
            hit = self.retriever.retrieve(caption, top_k=1)[0]
            cap_e = self._encode(caption)
            pat_e = self._encode(f"{hit['latent_risk']} {hit['mitigation']}")
            emb = (1.0 - self.alpha) * cap_e + self.alpha * pat_e
            self._cached_caption, self._cached = caption, (emb, hit)
        return self._cached
