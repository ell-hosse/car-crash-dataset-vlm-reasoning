"""
clip_retrieval.py
-----------------
Reusable CLIP image<->text retrieval over crash_policies.jsonl, shared by the
RAG evaluation and the after-RAG visualisation.

Everything lives in ONE shared CLIP space (openai/clip-vit-base-patch32):
  - policy side : each policy is embedded from its TEXT content. The three
                  fields (trigger / latent_risk / mitigation) are encoded
                  separately (each fits CLIP's 77-token window), averaged and
                  re-normalised -> one L2-normalised vector per policy.
                  All ~2k policies are embedded ONCE and cached to
                  clip_policy_index.npz (this is the "compute CLIP over the
                  whole crash_policies.jsonl to make it optimised" step).
  - scene side  : the last X camera frames are each turned into a CLIP image
                  embedding, mean-pooled and L2-normalised -> one vector.

Retrieval = exact cosine top-k (a single matrix-vector product + top-k
partition over the 2017x512 policy matrix; sub-millisecond on CPU).

This mirrors the index format already produced by
compute_dist/dist_carCrashDS_vs_policies.py so the existing
clip_policy_index.npz is reused as-is (and only rebuilt if crash_policies.jsonl
or the model name changed).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Defaults (paths are relative to the repo root, resolved from this file).
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_POLICIES = REPO_ROOT / "crash_policies.jsonl"
DEFAULT_INDEX = _THIS_DIR / "clip_policy_index.npz"

_REQUIRED_KEYS = {"trigger", "latent_risk", "mitigation"}


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------
def load_policies(jsonl_path: Path):
    """Load valid triplet policies. clip_id is kept only for reporting,
    never for ranking (ranking is purely CLIP similarity)."""
    triggers, latent_risks, mitigations, clip_ids = [], [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error") is not None:
                continue
            if not _REQUIRED_KEYS.issubset(rec.keys()):
                continue
            trig = rec["trigger"].strip()
            lr = rec["latent_risk"].strip()
            mit = rec["mitigation"].strip()
            if not (trig and lr and mit):
                continue
            triggers.append(trig)
            latent_risks.append(lr)
            mitigations.append(mit)
            clip_ids.append(rec.get("clip_id", rec.get("vidname", "unknown")))
    if not triggers:
        raise RuntimeError(f"No valid policies parsed from {jsonl_path}")
    return triggers, latent_risks, mitigations, clip_ids


# ---------------------------------------------------------------------------
# CLIP embedder (image + text into the shared space)
# ---------------------------------------------------------------------------
class ClipEmbedder:
    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    # -- images -------------------------------------------------------------
    def embed_image_arrays(self, rgb_frames, batch_size: int = 32) -> np.ndarray:
        """RGB uint8 HxWx3 arrays -> L2-normalised CLIP image embeddings,
        one row per frame."""
        torch = self.torch
        feats = []
        for i in range(0, len(rgb_frames), batch_size):
            batch = rgb_frames[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                f = self.model.get_image_features(**inputs)

            # --- FIX ADDED HERE ---
            # If 'f' is a Hugging Face object wrapper, unpack the raw hidden tensor
            if hasattr(f, "image_embeds"):
                f = f.image_embeds
            elif hasattr(f, "pooler_output"):
                f = f.pooler_output
            elif not isinstance(f, torch.Tensor):
                f = f[0]
            # ----------------------

            f = torch.nn.functional.normalize(f, dim=-1)
            feats.append(f.cpu().numpy().astype(np.float32))
        if not feats:
            raise ValueError("No frames could be embedded.")
        return np.concatenate(feats, axis=0)

    # -- text ---------------------------------------------------------------
    def embed_texts(self, texts, batch_size: int = 64) -> np.ndarray:
        """Strings -> L2-normalised CLIP text embeddings, one row per string."""
        torch = self.torch
        feats = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                f = self.model.get_text_features(**inputs)
            f = torch.nn.functional.normalize(f, dim=-1)
            feats.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(feats, axis=0)

    def policy_text_embeddings(self, triggers, latent_risks, mitigations
                               ) -> np.ndarray:
        """Embed each policy's content. Each of the three fields is encoded
        separately (each fits CLIP's 77-token window), then averaged and
        re-normalised so the whole policy is represented without truncation
        loss. Returns (N, D) L2-normalised."""
        n = len(triggers)
        fields = list(triggers) + list(latent_risks) + list(mitigations)
        flat = self.embed_texts(fields)                 # (3N, D)
        per_field = flat.reshape(3, n, -1)              # (3, N, D)
        pooled = per_field.mean(axis=0)                 # (N, D)
        pooled /= (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12)
        return pooled.astype(np.float32)


def pool_clip_video_embedding(frame_embs: np.ndarray) -> np.ndarray:
    """Mean-pool per-frame CLIP embeddings into one L2-normalised vector."""
    pooled = frame_embs.mean(axis=0)
    pooled = pooled / (np.linalg.norm(pooled) + 1e-12)
    return pooled.astype(np.float32)


# ---------------------------------------------------------------------------
# Policy matcher (the in-memory index used at query time)
# ---------------------------------------------------------------------------
class PolicyMatcher:
    """Holds the precomputed policy matrix and answers nearest-policy queries.
    Build once at startup, then call .retrieve(video_emb) per scene."""

    def __init__(self, embeddings, triggers, latent_risks, mitigations, clip_ids):
        self.embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self.triggers = np.asarray(triggers, dtype=object)
        self.latent_risks = np.asarray(latent_risks, dtype=object)
        self.mitigations = np.asarray(mitigations, dtype=object)
        self.clip_ids = np.asarray(clip_ids, dtype=object)

    def __len__(self):
        return len(self.triggers)

    def retrieve(self, video_emb: np.ndarray, top_k: int = 5) -> list[dict]:
        """Exact top-k by cosine similarity. video_emb must be L2-normalised;
        policy rows already are, so the dot product IS the cosine."""
        scores = self.embeddings @ video_emb.astype(np.float32)
        k = min(top_k, len(scores))
        # argpartition for the top-k, then sort just those k by score desc.
        part = np.argpartition(scores, -k)[-k:]
        order = part[np.argsort(scores[part])[::-1]]
        out = []
        for i in order:
            i = int(i)
            out.append({
                "index": i,
                "score": float(scores[i]),
                "dist": float(1.0 - scores[i]),
                "trigger": str(self.triggers[i]),
                "latent_risk": str(self.latent_risks[i]),
                "mitigation": str(self.mitigations[i]),
                "clip_id": str(self.clip_ids[i]),
            })
        return out


# ---------------------------------------------------------------------------
# Build-or-load the cached policy index
# ---------------------------------------------------------------------------
def build_or_load_policy_index(
    policies_path: Path = DEFAULT_POLICIES,
    index_path: Path = DEFAULT_INDEX,
    model_name: str = DEFAULT_CLIP_MODEL,
    embedder: "ClipEmbedder | None" = None,
    rebuild: bool = False,
    verbose: bool = True,
) -> PolicyMatcher:
    """Return a PolicyMatcher, loading the cached CLIP embeddings if they are
    valid for this (policies file, model) pair, otherwise (re)building them
    once and caching to index_path."""
    policies_path = Path(policies_path)
    index_path = Path(index_path)
    pol_mtime = policies_path.stat().st_mtime

    if index_path.exists() and not rebuild:
        data = np.load(index_path, allow_pickle=True)
        same_model = str(data["model_name"][0]) == model_name
        same_file = float(data["policies_mtime"][0]) == float(pol_mtime)
        if same_model and same_file:
            if verbose:
                print(f"[index] reusing cached CLIP policy index "
                      f"({len(data['triggers'])} policies) at {index_path}")
            return PolicyMatcher(
                data["embeddings"], data["triggers"], data["latent_risks"],
                data["mitigations"], data["clip_ids"])

    # (Re)build.
    triggers, latent_risks, mitigations, clip_ids = load_policies(policies_path)
    if verbose:
        print(f"[index] building CLIP policy index for {len(triggers)} policies...")
    embedder = embedder or ClipEmbedder(model_name)
    embeddings = embedder.policy_text_embeddings(triggers, latent_risks, mitigations)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        index_path,
        embeddings=embeddings.astype(np.float32),
        triggers=np.array(triggers, dtype=object),
        latent_risks=np.array(latent_risks, dtype=object),
        mitigations=np.array(mitigations, dtype=object),
        clip_ids=np.array(clip_ids, dtype=object),
        model_name=np.array([model_name]),
        policies_mtime=np.array([pol_mtime], dtype=np.float64),
    )
    if verbose:
        print(f"[index] saved cache to {index_path}")
    return PolicyMatcher(embeddings, triggers, latent_risks, mitigations, clip_ids)


# ---------------------------------------------------------------------------
# Caption augmentation: fold the retrieved policies into the scene caption.
# ---------------------------------------------------------------------------
def build_vlm_rag_prompt(base_prompt: str, hits: list[dict],
                         k: int = None) -> str:
    """Build the SmolVLM2 caption prompt CONDITIONED on the retrieved policies.

    This is the "RAG beside the VLM" path: the top-k crash policies (retrieved
    by CLIP from the last X frames) are handed to the captioner as background
    safety knowledge, so the VLM writes a risk-aware caption. The policies are
    NOT pasted into the caption itself - they only steer generation, and the VLM
    is told to use them only when they match what it actually sees.
    """
    use = hits[:k] if k else hits
    if not use:
        return base_prompt
    lines = "\n".join(
        f"- When {h['trigger']} the latent risk is: {h['latent_risk']} "
        f"Recommended action: {h['mitigation']}" for h in use)
    return (
        f"{base_prompt}\n\n"
        "Retrieved driving-safety knowledge from visually similar scenes "
        "(use ONLY if it matches what you actually see in this frame; ignore "
        "anything that does not apply):\n"
        f"{lines}")


def augment_caption(base_caption: str, hits: list[dict],
                    style: str = "mitigations") -> str:
    """Fold the top-k retrieved policies into the VLA caption.

    The VLA text tower (frozen CLIP) truncates to 77 tokens, so we keep the
    base scene description first (most informative) and append compact safety
    guidance from the retrieved policies; truncation then drops the tail.

      style="mitigations" : base + the k mitigation imperatives (default)
      style="risks"       : base + the k latent-risk phrases
      style="triplet"     : base + "risk -> mitigation" for the top hit only
    """
    base = base_caption.strip()
    if not hits:
        return base
    if style == "triplet":
        h = hits[0]
        return f"{base} Risk: {h['latent_risk']} Mitigation: {h['mitigation']}"
    if style == "risks":
        tail = " ".join(h["latent_risk"] for h in hits)
    else:  # mitigations
        tail = " ".join(h["mitigation"] for h in hits)
    return f"{base} Safety guidance: {tail}"
