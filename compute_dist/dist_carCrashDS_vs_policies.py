"""
dist_carCrashDS_vs_policies.py
------------------------------
Pick a RANDOM clip from the Car Crash Dataset (CCD) and find the policies in
crash_policies.jsonl that are closest to that clip in CLIP space.

Both sides are embedded with the SAME CLIP model, which maps images and text
into one shared space:

  video side : each frame of the clip -> CLIP image embedding, then mean-pooled
               and L2-normalized into a single clip-level vector.
  policy side: each policy's textual content (trigger + latent_risk + mitigation)
               -> CLIP text embeddings (one per field, mean-pooled to dodge the
               77-token limit), L2-normalized into one vector per policy.

Distance = 1 - cosine similarity. The clip_id / index of each policy is IGNORED:
ranking is purely by CLIP similarity, so a clip can match any policy.

Usage:
    python compute_dist/dist_carCrashDS_vs_policies.py
    python compute_dist/dist_carCrashDS_vs_policies.py --top-k 10 --seed 7
    python compute_dist/dist_carCrashDS_vs_policies.py --clip-id C_001262
    python compute_dist/dist_carCrashDS_vs_policies.py --ccd-root "D:/path/to/CrashBest"
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Same CLIP backbone the project already standardises on (see covla_vla/config.py).
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

# Where the CCD frames live. Override with --ccd-root if yours differs.
DEFAULT_CCD_ROOT = Path(r"D:\car_crash\CrashBest")

DEFAULT_POLICIES = PROJECT_ROOT / "crash_policies.jsonl"

# Frame filenames look like C_001262_15.jpg  ->  clip_id "C_001262", frame 15.
FRAME_RE = re.compile(r"^(?P<clip>.+)_(?P<frame>\d+)$")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def index_clips(ccd_root: Path) -> dict[str, list[Path]]:
    """Group every .jpg under ccd_root into {clip_id: [frame paths sorted]}."""
    if not ccd_root.exists():
        raise FileNotFoundError(
            f"CCD frames not found at: {ccd_root}\n"
            f"Pass the correct folder with --ccd-root."
        )

    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for img_path in ccd_root.rglob("*.jpg"):
        m = FRAME_RE.match(img_path.stem)
        if not m:
            continue
        groups[m.group("clip")].append((int(m.group("frame")), img_path))

    if not groups:
        raise RuntimeError(f"No parsable C_*_<frame>.jpg files under {ccd_root}")

    return {
        clip: [p for _, p in sorted(frames, key=lambda x: x[0])]
        for clip, frames in groups.items()
    }


def load_policies(jsonl_path: Path) -> list[dict]:
    """Load valid triplet policies. Keeps clip_id only for reporting, never for ranking."""
    required = {"trigger", "latent_risk", "mitigation"}
    policies = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error") is not None:
                continue
            if not required.issubset(rec.keys()):
                continue
            trig = rec["trigger"].strip()
            risk = rec["latent_risk"].strip()
            mit = rec["mitigation"].strip()
            if not (trig and risk and mit):
                continue
            policies.append(
                {
                    "clip_id": rec.get("clip_id", rec.get("vidname", "unknown")),
                    "trigger": trig,
                    "latent_risk": risk,
                    "mitigation": mit,
                }
            )
    if not policies:
        raise RuntimeError(f"No valid policies parsed from {jsonl_path}")
    return policies


# ---------------------------------------------------------------------------
# CLIP embedding
# ---------------------------------------------------------------------------
class ClipEmbedder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_images(self, paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized image embeddings, one row per readable frame."""
        embs = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for p in batch_paths:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception as e:  # skip unreadable frames
                    print(f"  [warn] could not read {p.name}: {e}")
            if not images:
                continue
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embs.append(feats.cpu().numpy())
        if not embs:
            raise RuntimeError("No frames could be embedded for this clip.")
        return np.concatenate(embs, axis=0).astype(np.float32)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Return L2-normalized text embeddings, one row per input string."""
        embs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,  # CLIP text tower is capped at 77 tokens
            ).to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embs.append(feats.cpu().numpy())
        return np.concatenate(embs, axis=0).astype(np.float32)


def clip_video_embedding(embedder: ClipEmbedder, frame_paths: list[Path]) -> np.ndarray:
    """Mean-pool per-frame CLIP embeddings into one L2-normalized clip vector."""
    frame_embs = embedder.embed_images(frame_paths)
    pooled = frame_embs.mean(axis=0)
    pooled /= np.linalg.norm(pooled) + 1e-12
    return pooled.astype(np.float32)


def policy_text_embeddings(embedder: ClipEmbedder, policies: list[dict]) -> np.ndarray:
    """
    Embed each policy's content. Each of the three fields is encoded separately
    (each fits in CLIP's 77-token window), then averaged and re-normalized so the
    whole policy content is represented without truncation loss.
    """
    fields = ["trigger", "latent_risk", "mitigation"]
    flat = [p[f] for p in policies for f in fields]  # 3 strings per policy, in order
    flat_embs = embedder.embed_texts(flat)  # (3N, D)

    d = flat_embs.shape[1]
    per_policy = flat_embs.reshape(len(policies), len(fields), d).mean(axis=1)  # (N, D)
    per_policy /= np.linalg.norm(per_policy, axis=1, keepdims=True) + 1e-12
    return per_policy.astype(np.float32)


# ---------------------------------------------------------------------------
# Cached policy index  (compute ONCE, reuse every query)
# ---------------------------------------------------------------------------
# The 2017 policies never change between runs, so encoding them with CLIP on
# every call (6051 text forward passes) is pure waste. We embed them once and
# cache the matrix next to the policies file. The cache is keyed on the policy
# file's mtime + the CLIP model, so it auto-rebuilds if either changes.
DEFAULT_INDEX = PROJECT_ROOT / "compute_dist" / "clip_policy_index.npz"


def build_or_load_policy_index(
    embedder: "ClipEmbedder | None",
    policies_path: Path,
    index_path: Path,
    model_name: str,
    rebuild: bool = False,
) -> "PolicyMatcher":
    """Return a PolicyMatcher, loading the cached embeddings if they are valid
    for this (policies file, model) pair, otherwise (re)building them once."""
    pol_mtime = policies_path.stat().st_mtime if policies_path.exists() else 0.0

    if not rebuild and index_path.exists():
        data = np.load(index_path, allow_pickle=True)
        same_model = str(data["model_name"][0]) == model_name
        same_file = float(data["policies_mtime"][0]) == pol_mtime
        if same_model and same_file:
            return PolicyMatcher(
                embeddings=data["embeddings"].astype(np.float32),
                triggers=data["triggers"],
                latent_risks=data["latent_risks"],
                mitigations=data["mitigations"],
                clip_ids=data["clip_ids"],
            )

    # (Re)build — needs a live embedder.
    if embedder is None:
        embedder = ClipEmbedder(model_name, torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))
    policies = load_policies(policies_path)
    print(f"[index] building CLIP policy index for {len(policies)} policies...")
    embs = policy_text_embeddings(embedder, policies)  # (N, D) normalized float32

    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        index_path,
        embeddings=embs,
        triggers=np.array([p["trigger"] for p in policies], dtype=object),
        latent_risks=np.array([p["latent_risk"] for p in policies], dtype=object),
        mitigations=np.array([p["mitigation"] for p in policies], dtype=object),
        clip_ids=np.array([p["clip_id"] for p in policies], dtype=object),
        model_name=np.array([model_name]),
        policies_mtime=np.array([pol_mtime]),
    )
    print(f"[index] saved cache to {index_path}")
    return PolicyMatcher(embs, np.array([p["trigger"] for p in policies], dtype=object),
                         np.array([p["latent_risk"] for p in policies], dtype=object),
                         np.array([p["mitigation"] for p in policies], dtype=object),
                         np.array([p["clip_id"] for p in policies], dtype=object))


class PolicyMatcher:
    """Holds the precomputed policy matrix and answers nearest-policy queries.

    Build once at startup, then call .retrieve(video_emb) per frame/clip in the
    real-time loop. Each query is a single BLAS matrix-vector product plus an
    O(N) top-k partition — exact cosine, sub-millisecond for ~2k policies.
    """

    def __init__(self, embeddings, triggers, latent_risks, mitigations, clip_ids):
        # C-contiguous float32 gives the fastest, exact BLAS gemv.
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.triggers = triggers
        self.latent_risks = latent_risks
        self.mitigations = mitigations
        self.clip_ids = clip_ids

    def __len__(self):
        return len(self.embeddings)

    def retrieve(self, video_emb: np.ndarray, top_k: int = 5) -> list[dict]:
        """Exact top-k by cosine similarity. video_emb must be L2-normalized;
        policy rows already are, so the dot product IS the cosine."""
        q = np.ascontiguousarray(video_emb, dtype=np.float32)
        sims = self.embeddings @ q                       # (N,) one gemv
        k = min(top_k, len(sims))
        # argpartition finds the top-k in O(N) (no full sort), then we sort only k.
        part = np.argpartition(sims, -k)[-k:]
        order = part[np.argsort(sims[part])[::-1]]
        return [
            {
                "index": int(i),
                "score": float(sims[i]),
                "trigger": str(self.triggers[i]),
                "latent_risk": str(self.latent_risks[i]),
                "mitigation": str(self.mitigations[i]),
                "clip_id": str(self.clip_ids[i]),
            }
            for i in order
        ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Find policies closest to a random CCD clip in CLIP space."
    )
    ap.add_argument("--ccd-root", type=Path, default=DEFAULT_CCD_ROOT,
                    help="Folder containing CCD C_*_<frame>.jpg frames.")
    ap.add_argument("--policies", type=Path, default=DEFAULT_POLICIES,
                    help="Path to crash_policies.jsonl.")
    ap.add_argument("--clip-id", type=str, default=None,
                    help="Use this clip instead of a random one (e.g. C_001262).")
    ap.add_argument("--top-k", type=int, default=5, help="How many policies to show.")
    ap.add_argument("--max-frames", type=int, default=50,
                    help="Cap frames embedded per clip (evenly subsampled).")
    ap.add_argument("--model", type=str, default=DEFAULT_CLIP_MODEL,
                    help="CLIP model name.")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                    help="Path to the cached CLIP policy index (.npz).")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Force rebuilding the policy index even if cached.")
    ap.add_argument("--seed", type=int, default=None, help="Seed for random clip pick.")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # 1. Index clips and choose one.
    clips = index_clips(args.ccd_root)
    print(f"[clips] indexed {len(clips)} clips under {args.ccd_root}")

    if args.clip_id is not None:
        if args.clip_id not in clips:
            sys.exit(f"clip-id {args.clip_id!r} not found in dataset.")
        chosen = args.clip_id
    else:
        chosen = random.choice(sorted(clips.keys()))

    frame_paths = clips[chosen]
    if args.max_frames and len(frame_paths) > args.max_frames:
        idx = np.linspace(0, len(frame_paths) - 1, args.max_frames).round().astype(int)
        frame_paths = [frame_paths[i] for i in idx]
    print(f"[clip ] chosen: {chosen}  ({len(frame_paths)} frames embedded)")

    # 2. Load model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] loading {args.model} on {device}")
    embedder = ClipEmbedder(args.model, device)

    # 3. Build/load the cached policy matrix ONCE (no re-encoding on cache hit).
    matcher = build_or_load_policy_index(
        embedder, args.policies, args.index, args.model, rebuild=args.rebuild_index)
    print(f"[policy] matcher ready with {len(matcher)} policies")

    # 4. Embed the video side and run the fast exact query.
    video_emb = clip_video_embedding(embedder, frame_paths)          # (D,)
    hits = matcher.retrieve(video_emb, top_k=args.top_k)

    # 5. Report.
    print("\n" + "=" * 78)
    print(f"Closest {args.top_k} policies to clip {chosen} (CLIP image<->text)")
    print("=" * 78)
    for rank, h in enumerate(hits, 1):
        sim = h["score"]
        print(f"\n#{rank}  cos_sim={sim:.4f}  dist={1 - sim:.4f}  "
              f"(source clip_id={h['clip_id']}, ignored for ranking)")
        print(f"    trigger    : {h['trigger']}")
        print(f"    latent_risk: {h['latent_risk']}")
        print(f"    mitigation : {h['mitigation']}")
    print()


if __name__ == "__main__":
    main()
