"""Downsampling / preprocessing pipeline for CoVLA (run once before training).

Per the CoVLA paper setup: videos are 30 s @ 20 fps with per-frame states and
captions. We downsample to 2 Hz, decode + resize the frames once to JPEG, and
build a flat sample index so the training DataLoader never touches mp4 files.

For each kept frame we store:
  frames/<video_id>/<frame_idx>.jpg          (resized, width=cfg.jpeg_width)
  index/{train,val,test}.jsonl               one sample per line:
      {video_id, frame_idx, image, caption, state: {...}, traj: [[x,y(,z)], ...]}

Samples are skipped when the state has fewer than traj_horizon future
trajectory points (end-of-clip) or the caption is missing.

Usage:
    python -m covla_vla.preprocess [--limit-videos N] [--workers 8]
"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from .config import DATA, MANIFEST_PATH, PREPROCESSED_ROOT, remap_path


def split_of(video_id: str) -> str:
    """Deterministic train/val/test split by video id hash."""
    h = int(hashlib.md5(video_id.encode()).hexdigest(), 16) % 10_000
    if h < DATA.test_frac * 10_000:
        return "test"
    if h < (DATA.test_frac + DATA.val_frac) * 10_000:
        return "val"
    return "train"


def load_per_frame_jsonl(path: Path) -> dict:
    """CoVLA states/captions are JSONL lines like {"<frame_idx>": {...}}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for k, v in rec.items():
                out[int(k)] = v
    return out


def process_video(entry: dict) -> list[dict] | None:
    """Decode one video at 2 Hz, save JPEGs, return sample dicts."""
    video_id = entry["video_id"]
    video_path = remap_path(entry["video_path"])
    states_path = remap_path(entry["states_path"])
    captions_path = remap_path(entry["captions_path"])
    if not (video_path.exists() and states_path.exists() and captions_path.exists()):
        return None

    states = load_per_frame_jsonl(states_path)
    captions = load_per_frame_jsonl(captions_path)

    frame_dir = PREPROCESSED_ROOT / "frames" / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    stride = DATA.frame_stride
    wanted = list(range(0, DATA.frames_per_video, stride))
    wanted_set = set(wanted)

    cap = cv2.VideoCapture(str(video_path))
    samples, idx = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx in wanted_set:
            ok, frame = cap.retrieve()
            if ok:
                sample = build_sample(video_id, idx, frame, states, captions, frame_dir)
                if sample:
                    samples.append(sample)
        idx += 1
        if idx >= DATA.frames_per_video:
            break
    cap.release()
    return samples


def build_sample(video_id, frame_idx, frame, states, captions, frame_dir):
    st = states.get(frame_idx)
    cp = captions.get(frame_idx)
    if st is None or cp is None:
        return None
    caption = cp.get(DATA.caption_field) or cp.get("plain_caption")
    traj = st.get("trajectory")
    if not caption or not traj or st.get("trajectory_count", 0) < DATA.traj_horizon:
        return None

    traj = traj[:DATA.traj_horizon:DATA.traj_subsample]
    if DATA.use_xy_only:
        traj = [[p[0], p[1]] for p in traj]

    state_vec = {k: float(st.get(k, 0.0) or 0.0) for k in DATA.state_keys}

    img_path = frame_dir / f"{frame_idx}.jpg"
    if not img_path.exists():
        h, w = frame.shape[:2]
        new_w = DATA.jpeg_width
        new_h = int(round(h * new_w / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "image": str(img_path.relative_to(PREPROCESSED_ROOT)),
        "caption": caption,
        "state": state_vec,
        "traj": traj,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-videos", type=int, default=0, help="debug: only N videos")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    entries = [json.loads(l) for l in open(MANIFEST_PATH, encoding="utf-8") if l.strip()]
    if args.limit_videos:
        entries = entries[: args.limit_videos]
    print(f"{len(entries)} videos in manifest")

    (PREPROCESSED_ROOT / "index").mkdir(parents=True, exist_ok=True)
    writers = {s: open(PREPROCESSED_ROOT / "index" / f"{s}.jsonl", "w", encoding="utf-8")
               for s in ("train", "val", "test")}
    counts = {s: 0 for s in writers}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_video, e): e["video_id"] for e in entries}
        for i, fut in enumerate(as_completed(futs), 1):
            vid = futs[fut]
            try:
                samples = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {vid}: {e}", file=sys.stderr)
                continue
            if not samples:
                continue
            split = split_of(vid)
            for s in samples:
                writers[split].write(json.dumps(s) + "\n")
            counts[split] += len(samples)
            if i % 100 == 0:
                print(f"  {i}/{len(entries)} videos | samples: {counts}")

    for w in writers.values():
        w.close()
    print(f"done. samples per split: {counts}")
    print(f"index files in {PREPROCESSED_ROOT / 'index'}")


if __name__ == "__main__":
    main()
