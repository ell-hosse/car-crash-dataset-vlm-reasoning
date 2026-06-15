"""Real-time inference pipeline.

Loads the checkpoint saved by train.py and runs a live loop:
  frame -> [async SmolVLM2 captioner thread]  (caption ~1 Hz, never blocks)
        -> trajectory model @ target Hz, using the latest caption embedding
        -> overlay: predicted 3 s trajectory projected into the image + caption

The caption embedding is computed once per new caption and cached, so the
per-frame cost is just vision encoder + fusion (~real-time on Jetson Orin,
hundreds of FPS on H100).

Usage:
    # demo on a CoVLA video (uses GT ego-states if available):
    python -m covla_vla.infer_realtime --ckpt covla_vla/runs/covla_vla_best.pt \
        --video D:/hf/hub/datasets--turing-motors--CoVLA-Dataset/snapshots/<hash>/videos/<id>.mp4

    # webcam / live camera:
    python -m covla_vla.infer_realtime --ckpt ... --camera 0

    # quantitative real-time evaluation on the CoVLA test split
    # (live SmolVLM captions, ADE/FDE vs ground-truth trajectories):
    python -m covla_vla.infer_realtime --ckpt ... --eval --max-videos 50
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import DATA, REALTIME, MANIFEST_PATH, remap_path
from .dataset import preprocess_image, state_to_vec, denormalize_traj
from .model import build_model_and_tokenizer
from .captioner import AsyncCaptioner
from .preprocess import load_per_frame_jsonl, split_of

# CoVLA camera defaults (from states intrinsic/extrinsic): fx=fy=2648,
# cx=964, cy=604 at 1928x1208; camera height ~1.22 m.
CAM = dict(fx=2648.0, fy=2648.0, cx=964.0, cy=604.0, w=1928, h=1208, height=1.22)


def _remap_state_keys(state: dict, model) -> dict:
    """Handle transformers version differences in CLIPTextModel key naming
    (checkpoint may have 'text.encoder...' vs model's 'text.text_model.encoder...'
    or vice versa)."""
    model_keys = set(model.state_dict().keys())
    fixed = {}
    for k, v in state.items():
        if k not in model_keys and k.startswith("text."):
            cand = ("text.text_model." + k[len("text."):]
                    if not k.startswith("text.text_model.")
                    else "text." + k[len("text.text_model."):])
            if cand in model_keys:
                fixed[cand] = v
                continue
        fixed[k] = v
    return fixed


def load_model(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, tokenizer = build_model_and_tokenizer()
    state = _remap_state_keys(ck["model"], model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # non-persistent buffers (e.g. position_ids) may differ across versions
    problems = [k for k in missing + unexpected if "position_ids" not in k]
    if problems:
        raise RuntimeError(f"checkpoint/model key mismatch: {problems[:10]}"
                           f"{' ...' if len(problems) > 10 else ''}")
    model.to(device).eval()
    print(f"loaded {ckpt_path} (epoch {ck.get('epoch')}, "
          f"best val ADE {ck.get('best_ade', float('nan')):.3f} m)")
    return model, tokenizer


class TrajectoryPredictor:
    """Wraps the trained model; caches the caption embedding between updates."""

    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self._text_embed = None
        self._caption_version = -1

    @torch.no_grad()
    def update_caption(self, caption: str, version: int):
        if version == self._caption_version:
            return
        tok = self.tokenizer([caption], padding=True, truncation=True,
                             max_length=77, return_tensors="pt").to(self.device)
        self._text_embed = self.model.encode_text(
            tok["input_ids"], tok["attention_mask"])
        self._caption_version = version

    @torch.no_grad()
    def predict(self, bgr_frame, state: dict) -> np.ndarray:
        img = preprocess_image(bgr_frame).unsqueeze(0).to(self.device)
        st = state_to_vec(state).unsqueeze(0).to(self.device)
        pred = self.model(img, st, text_embed=self._text_embed)
        return denormalize_traj(pred[0].float().cpu().numpy())


def project_traj(traj_xy: np.ndarray, frame_shape) -> np.ndarray:
    """BEV (x fwd, y left) -> image pixels using CoVLA camera geometry."""
    h, w = frame_shape[:2]
    sx, sy = w / CAM["w"], h / CAM["h"]
    pts = []
    for x_fwd, y_left in traj_xy:
        if x_fwd < 1.0:
            continue
        u = CAM["fx"] * (-y_left) / x_fwd + CAM["cx"]
        v = CAM["fy"] * CAM["height"] / x_fwd + CAM["cy"]
        pts.append((int(u * sx), int(v * sy)))
    return np.array(pts, dtype=np.int32)


def draw_overlay(frame, traj_xy, caption, fps, cap_latency):
    pts = project_traj(traj_xy, frame.shape)
    if len(pts) >= 2:
        cv2.polylines(frame, [pts], False, (0, 255, 0), 3)
    for p in pts:
        cv2.circle(frame, tuple(p), 5, (0, 200, 255), -1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(frame, f"{fps:5.1f} FPS | caption latency {cap_latency:.2f}s",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for i in range(0, len(caption), 110):
        cv2.putText(frame, caption[i:i + 110], (10, 48 + 18 * (i // 110)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame


def run_live(args, device):
    model, tokenizer = load_model(args.ckpt, device)
    predictor = TrajectoryPredictor(model, tokenizer, device)
    captioner = AsyncCaptioner(device).start()

    states = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        sp = Path(args.video).with_suffix("")  # try to find matching states
        cand = remap_path(str(Path(str(MANIFEST_PATH)).parent / "states" / (sp.name + ".jsonl")))
        if cand.exists():
            states = load_per_frame_jsonl(cand)
            print(f"using GT ego-states from {cand.name}")
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("could not open video source")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, REALTIME.target_hz, (w, h))

    period = 1.0 / REALTIME.target_hz
    fps_hist = deque(maxlen=30)
    frame_idx = 0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            captioner.submit_frame(frame)
            predictor.update_caption(captioner.caption, captioner.caption_version)
            state = states.get(frame_idx, {}) if states else {}
            traj = predictor.predict(frame, state)

            dt = time.time() - t0
            fps_hist.append(1.0 / max(dt, 1e-6))
            vis = draw_overlay(frame.copy(), traj, captioner.caption,
                               np.mean(fps_hist), captioner.last_latency_s)
            if writer:
                writer.write(vis)
            if REALTIME.display and not args.headless:
                cv2.imshow("CoVLA VLA - realtime", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # pace to target Hz for file sources (camera paces itself)
            if args.video:
                slack = period - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)
            frame_idx += 1
    finally:
        captioner.stop()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    print(f"avg FPS: {np.mean(fps_hist):.1f}")


def run_eval(args, device):
    """Real-time-style evaluation on the CoVLA test split: captions are
    generated live by SmolVLM2 (not GT), trajectories scored vs GT."""
    model, tokenizer = load_model(args.ckpt, device)
    predictor = TrajectoryPredictor(model, tokenizer, device)
    captioner = AsyncCaptioner(device).start()

    entries = [json.loads(l) for l in open(MANIFEST_PATH, encoding="utf-8") if l.strip()]
    test_entries = [e for e in entries if split_of(e["video_id"]) == "test"]
    if args.max_videos:
        test_entries = test_entries[: args.max_videos]
    print(f"evaluating on {len(test_entries)} test videos")

    stride = DATA.frame_stride
    all_ade, all_fde, lat = [], [], []
    try:
        for n, e in enumerate(test_entries, 1):
            vp = remap_path(e["video_path"])
            states = load_per_frame_jsonl(remap_path(e["states_path"]))
            cap = cv2.VideoCapture(str(vp))
            idx = 0
            while True:
                ok = cap.grab()
                if not ok or idx >= DATA.frames_per_video:
                    break
                if idx % stride == 0:
                    ok, frame = cap.retrieve()
                    st = states.get(idx)
                    if ok and st and st.get("trajectory_count", 0) >= DATA.traj_horizon:
                        captioner.submit_frame(frame)
                        predictor.update_caption(captioner.caption,
                                                 captioner.caption_version)
                        t0 = time.time()
                        pred = predictor.predict(frame, st)
                        lat.append(time.time() - t0)
                        gt = np.asarray(st["trajectory"], dtype=np.float32)
                        gt = gt[:DATA.traj_horizon:DATA.traj_subsample, :2]
                        d = np.linalg.norm(pred - gt, axis=-1)
                        all_ade.append(d.mean()); all_fde.append(d[-1])
                idx += 1
            cap.release()
            if n % 10 == 0:
                print(f"  {n}/{len(test_entries)} | ADE={np.mean(all_ade):.3f}m "
                      f"FDE={np.mean(all_fde):.3f}m | "
                      f"traj latency {1e3 * np.mean(lat):.1f}ms "
                      f"(~{1 / np.mean(lat):.0f} Hz)")
    finally:
        captioner.stop()

    print("\n=== real-time evaluation (live SmolVLM captions) ===")
    print(f"samples: {len(all_ade)}")
    print(f"ADE: {np.mean(all_ade):.3f} m   FDE: {np.mean(all_fde):.3f} m")
    print(f"trajectory latency: {1e3 * np.mean(lat):.1f} ms "
          f"(p95 {1e3 * np.percentile(lat, 95):.1f} ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint from train.py")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--video", type=str, help="video file (e.g., a CoVLA mp4)")
    src.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--eval", action="store_true", help="ADE/FDE on test split")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--save", type=str, help="save overlay video to this path")
    ap.add_argument("--headless", action="store_true", help="no display window")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.eval:
        run_eval(args, device)
    else:
        run_live(args, device)


if __name__ == "__main__":
    main()
