"""
dist_covla_vs_policies.py
-------------------------
Same scenario as dist_carCrashDS_vs_policies.py, but the query video is a RANDOM
CoVLA clip instead of a Car Crash Dataset clip.

A CoVLA "video" is one `video_id`; its frames were sampled to 2 Hz and saved by
covla_vla/preprocess.py as:
    covla_preprocessed/frames/<video_id>/<frame_idx>.jpg
    covla_preprocessed/index/{train,val,test}.jsonl   (one sample per frame)

We pick a random video_id, embed all its frames with CLIP's image tower,
mean-pool + L2-normalize into one clip vector, then rank every policy in
crash_policies.jsonl by cosine similarity in CLIP's shared image<->text space.
The policy clip_id / index is IGNORED — ranking is purely by CLIP similarity.

All the CLIP machinery (embedder, video pooling, cached policy matrix, the fast
exact PolicyMatcher) is reused from dist_carCrashDS_vs_policies.py, so the policy
index cache (clip_policy_index.npz) is shared and never recomputed.

Usage:
    python compute_dist/dist_covla_vs_policies.py
    python compute_dist/dist_covla_vs_policies.py --top-k 10 --seed 7
    python compute_dist/dist_covla_vs_policies.py --video-id 2022-07-06--10-43-45_27
    python compute_dist/dist_covla_vs_policies.py --split val
    python compute_dist/dist_covla_vs_policies.py --covla-root "D:/hf/covla_preprocessed"
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
# Make both the sibling CCD script and the covla_vla package importable.
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse everything from the CCD script (no duplicated CLIP logic, shared cache).
from dist_carCrashDS_vs_policies import (  # noqa: E402
    ClipEmbedder,
    clip_video_embedding,
    build_or_load_policy_index,
    DEFAULT_CLIP_MODEL,
    DEFAULT_POLICIES,
    DEFAULT_INDEX,
)

# Where preprocessed CoVLA frames live (see covla_vla/config.py).
try:
    from covla_vla.config import PREPROCESSED_ROOT as _CFG_ROOT  # noqa: E402
    DEFAULT_COVLA_ROOT = Path(_CFG_ROOT)
except Exception:
    DEFAULT_COVLA_ROOT = Path("D:/hf/covla_preprocessed")

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# CoVLA video discovery
# ---------------------------------------------------------------------------
def index_covla_videos(
    covla_root: Path, splits: tuple[str, ...]
) -> tuple[dict[str, list[Path]], dict[str, str]]:
    """Group CoVLA frames into {video_id: [frame paths sorted by frame_idx]}.

    Prefers the preprocessed index/*.jsonl (authoritative, carries captions);
    falls back to scanning frames/<video_id>/<n>.jpg if no index files exist.
    Returns (videos, captions) where captions maps video_id -> a sample caption.
    """
    index_dir = covla_root / "index"
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    captions: dict[str, str] = {}

    index_files = [index_dir / f"{s}.jsonl" for s in splits]
    index_files = [p for p in index_files if p.exists()]

    if index_files:
        for p in index_files:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    s = json.loads(line)
                    vid = s["video_id"]
                    groups[vid].append((int(s["frame_idx"]), covla_root / s["image"]))
                    captions.setdefault(vid, s.get("caption", ""))
    else:
        # Fallback: scan the frames directory directly.
        frames_dir = covla_root / "frames"
        if not frames_dir.exists():
            raise FileNotFoundError(
                f"No CoVLA index/*.jsonl and no frames/ under {covla_root}.\n"
                f"Run `python -m covla_vla.preprocess` first, or pass --covla-root."
            )
        for vid_dir in frames_dir.iterdir():
            if not vid_dir.is_dir():
                continue
            for img in vid_dir.glob("*.jpg"):
                try:
                    groups[vid_dir.name].append((int(img.stem), img))
                except ValueError:
                    continue

    if not groups:
        raise RuntimeError(f"No CoVLA frames discovered under {covla_root}")

    videos = {
        vid: [p for _, p in sorted(frames, key=lambda x: x[0])]
        for vid, frames in groups.items()
    }
    return videos, captions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Find policies closest to a random CoVLA clip in CLIP space."
    )
    ap.add_argument("--covla-root", type=Path, default=DEFAULT_COVLA_ROOT,
                    help="Preprocessed CoVLA root (contains index/ and frames/).")
    ap.add_argument("--split", type=str, default="all",
                    choices=("all", *SPLITS),
                    help="Which preprocessed split(s) to sample a video from.")
    ap.add_argument("--policies", type=Path, default=DEFAULT_POLICIES,
                    help="Path to crash_policies.jsonl.")
    ap.add_argument("--video-id", type=str, default=None,
                    help="Use this CoVLA video instead of a random one.")
    ap.add_argument("--top-k", type=int, default=5, help="How many policies to show.")
    ap.add_argument("--max-frames", type=int, default=50,
                    help="Cap frames embedded per video (evenly subsampled).")
    ap.add_argument("--model", type=str, default=DEFAULT_CLIP_MODEL,
                    help="CLIP model name.")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                    help="Path to the cached CLIP policy index (.npz, shared).")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Force rebuilding the policy index even if cached.")
    ap.add_argument("--seed", type=int, default=None, help="Seed for random pick.")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    splits = SPLITS if args.split == "all" else (args.split,)

    # 1. Discover CoVLA videos and choose one.
    videos, captions = index_covla_videos(args.covla_root, splits)
    print(f"[covla] indexed {len(videos)} videos under {args.covla_root} "
          f"(splits={','.join(splits)})")

    if args.video_id is not None:
        if args.video_id not in videos:
            sys.exit(f"video-id {args.video_id!r} not found in dataset.")
        chosen = args.video_id
    else:
        chosen = random.choice(sorted(videos.keys()))

    frame_paths = videos[chosen]
    if args.max_frames and len(frame_paths) > args.max_frames:
        idx = np.linspace(0, len(frame_paths) - 1, args.max_frames).round().astype(int)
        frame_paths = [frame_paths[i] for i in idx]
    print(f"[video] chosen: {chosen}  ({len(frame_paths)} frames embedded)")
    if captions.get(chosen):
        print(f"[caption] {captions[chosen]}")

    # 2. Load model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] loading {args.model} on {device}")
    embedder = ClipEmbedder(args.model, device)

    # 3. Build/load the cached policy matrix ONCE (shared with the CCD script).
    matcher = build_or_load_policy_index(
        embedder, args.policies, args.index, args.model, rebuild=args.rebuild_index)
    print(f"[policy] matcher ready with {len(matcher)} policies")

    # 4. Embed the video side and run the fast exact query.
    video_emb = clip_video_embedding(embedder, frame_paths)          # (D,)
    hits = matcher.retrieve(video_emb, top_k=args.top_k)

    # 5. Report.
    print("\n" + "=" * 78)
    print(f"Closest {args.top_k} policies to CoVLA video {chosen} (CLIP image<->text)")
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
