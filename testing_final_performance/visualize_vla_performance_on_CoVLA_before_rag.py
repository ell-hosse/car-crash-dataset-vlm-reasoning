"""Visualize VLA trajectory performance (before RAG) on the CoVLA test split.

Mirrors the training pipeline exactly: samples come from the preprocessed
test index (GT captions + GT ego-states, same preprocessing as train.py).

Outputs (in testing_final_performance/viz_before_rag/):
  1. sample_<k>_<video>_<frame>.png  - 10 random test samples: camera frame
     with predicted (green) vs GT (red) trajectory projected into the image,
     plus a BEV plot with ADE/FDE.
  2. overview_grid.png               - all 10 BEV plots in one figure.
  3. video_<id>_rolling.png          - one full test video: every 0.5 s the
     model predicts the next 3 s; each rollout is transformed into a common
     global frame (chained via GT ego-motion) and overlaid on the driven GT
     path. This is how the "rolling 3 s horizon" is shown for a whole video.
  4. video_<id>.mp4                  - animation: frame + BEV side by side,
     stepping through the video at 2 Hz (the training sample rate).

Usage (from repo root, machine with D:/hf data):
    python -m testing_final_performance.visualize_vla_performance_before_rag \
        [--ckpt covla_vla_best.pt] [--num-samples 10] [--seed 0]
        [--video-id <id>] [--no-mp4]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from covla_vla.config import DATA, PREPROCESSED_ROOT                  # noqa: E402
from covla_vla.dataset import preprocess_image, state_to_vec, denormalize_traj  # noqa: E402
from covla_vla.infer_realtime import load_model, project_traj         # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "viz_before_rag"

# time stamps of the 20 kept waypoints: every 3rd of 60 pts @ 20 Hz
WP_T = np.arange(DATA.num_waypoints) * DATA.traj_subsample / DATA.video_fps  # 0..2.85 s
SAMPLE_DT = 1.0 / DATA.sample_hz  # 0.5 s between consecutive training samples


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict_batch(model, tokenizer, samples, device, batch_size=32):
    """Run the model on index samples exactly like the training collate."""
    preds = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        imgs, states = [], []
        for s in chunk:
            bgr = cv2.imread(str(PREPROCESSED_ROOT / s["image"]))
            if bgr is None:
                raise IOError(f"missing frame {s['image']}")
            imgs.append(preprocess_image(bgr))
            states.append(state_to_vec(s["state"]))
        tok = tokenizer([s["caption"] for s in chunk], padding=True,
                        truncation=True, max_length=77, return_tensors="pt").to(device)
        pred = model(torch.stack(imgs).to(device), torch.stack(states).to(device),
                     tok["input_ids"], tok["attention_mask"])
        preds.append(pred.float().cpu().numpy())
    return denormalize_traj(np.concatenate(preds, axis=0))


def gt_traj(sample) -> np.ndarray:
    return np.asarray(sample["traj"], dtype=np.float32)


def ade_fde(pred, gt):
    d = np.linalg.norm(pred - gt, axis=-1)
    return float(d.mean()), float(d[-1])


# ---------------------------------------------------------------- drawing
def draw_image_overlay(sample, pred, gt):
    """Camera frame with GT (red) and predicted (green) trajectories projected."""
    bgr = cv2.imread(str(PREPROCESSED_ROOT / sample["image"]))
    for traj, color in ((gt, (0, 0, 255)), (pred, (0, 255, 0))):
        pts = project_traj(traj, bgr.shape)
        if len(pts) >= 2:
            cv2.polylines(bgr, [pts], False, color, 2)
        for p in pts:
            cv2.circle(bgr, tuple(p), 3, color, -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_bev(ax, pred, gt, title=""):
    """BEV: forward (x) up, ego at origin."""
    ax.plot(-gt[:, 1], gt[:, 0], "o-", color="red", ms=3, lw=1.5, label="GT")
    ax.plot(-pred[:, 1], pred[:, 0], "o-", color="green", ms=3, lw=1.5, label="pred")
    ax.scatter([0], [0], marker="^", s=80, color="black", zorder=5, label="ego")
    ax.set_xlabel("lateral (m, right +)")
    ax.set_ylabel("forward (m)")
    ax.set_title(title, fontsize=9)
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def sample_figure(sample, pred, out_path):
    gt = gt_traj(sample)
    ade, fde = ade_fde(pred, gt)
    fig, (ax_img, ax_bev) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax_img.imshow(draw_image_overlay(sample, pred, gt))
    ax_img.set_title(f"{sample['video_id']}  frame {sample['frame_idx']}  "
                     f"(GT=red, pred=green)", fontsize=9)
    ax_img.axis("off")
    plot_bev(ax_bev, pred, gt, f"ADE {ade:.2f} m | FDE {fde:.2f} m")
    cap = sample["caption"]
    fig.suptitle(cap[:160] + ("..." if len(cap) > 160 else ""),
                 fontsize=7, y=0.02, va="bottom")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return ade, fde


# ------------------------------------------------- whole-video visualization
def interp_pose(traj, t):
    """Pose (x, y, yaw) at time t along one GT 3 s trajectory (ego frame)."""
    x = np.interp(t, WP_T, traj[:, 0])
    y = np.interp(t, WP_T, traj[:, 1])
    eps = 0.075
    dx = np.interp(t + eps, WP_T, traj[:, 0]) - np.interp(t - eps, WP_T, traj[:, 0])
    dy = np.interp(t + eps, WP_T, traj[:, 1]) - np.interp(t - eps, WP_T, traj[:, 1])
    yaw = np.arctan2(dy, dx) if (abs(dx) + abs(dy)) > 1e-4 else 0.0
    return x, y, yaw


def to_global(traj_local, pose):
    """Ego-frame waypoints (x fwd, y left) -> global frame given (X, Y, yaw)."""
    X, Y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    gx = X + traj_local[:, 0] * c - traj_local[:, 1] * s
    gy = Y + traj_local[:, 0] * s + traj_local[:, 1] * c
    return np.stack([gx, gy], axis=1)


def chain_global_poses(samples):
    """Global pose of the ego at each sample, dead-reckoned from GT trajs.

    Consecutive samples are 0.5 s apart; the GT 3 s trajectory at sample k
    gives the ego displacement/heading change over that 0.5 s.
    """
    poses = [(0.0, 0.0, 0.0)]
    for k in range(len(samples) - 1):
        dt_frames = samples[k + 1]["frame_idx"] - samples[k]["frame_idx"]
        t = dt_frames / DATA.video_fps                       # usually 0.5 s
        lx, ly, lyaw = interp_pose(gt_traj(samples[k]), min(t, WP_T[-1]))
        X, Y, yaw = poses[-1]
        c, s = np.cos(yaw), np.sin(yaw)
        poses.append((X + lx * c - ly * s, Y + lx * s + ly * c, yaw + lyaw))
    return poses


def rolling_video_figure(samples, preds, out_path):
    """Spaghetti plot: GT driven path + every 3 s rollout in a global frame."""
    poses = chain_global_poses(samples)
    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(samples)))
    for k, (s, pose) in enumerate(zip(samples, poses)):
        g_gt = to_global(gt_traj(s), pose)
        g_pr = to_global(preds[k], pose)
        ax.plot(-g_gt[:, 1], g_gt[:, 0], color="red", alpha=0.25, lw=1)
        ax.plot(-g_pr[:, 1], g_pr[:, 0], color=cmap[k], alpha=0.55, lw=1.2)
    path = np.array([(p[0], p[1]) for p in poses])
    ax.plot(-path[:, 1], path[:, 0], "k.-", lw=2, ms=4, label="GT driven path")
    ax.plot([], [], color="red", alpha=0.4, label="GT 3 s segments")
    ax.plot([], [], color=cmap[len(cmap) // 2],
            label="pred 3 s rollouts (colored by time)")
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("forward (m)")
    ax.set_title(f"{samples[0]['video_id']} - rolling 3 s predictions every "
                 f"{SAMPLE_DT:.1f} s ({len(samples)} samples)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def rolling_video_mp4(samples, preds, out_path, fps=4):
    """MP4 stepping through the video: overlay frame + BEV per 2 Hz sample."""
    writer = None
    for k, s in enumerate(samples):
        gt = gt_traj(s)
        ade, fde = ade_fde(preds[k], gt)
        fig, (ax_img, ax_bev) = plt.subplots(
            1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
        ax_img.imshow(draw_image_overlay(s, preds[k], gt))
        ax_img.set_title(f"frame {s['frame_idx']}  "
                         f"t={s['frame_idx'] / DATA.video_fps:.1f}s", fontsize=10)
        ax_img.axis("off")
        plot_bev(ax_bev, preds[k], gt, f"ADE {ade:.2f} m | FDE {fde:.2f} m")
        fig.tight_layout()
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        if writer is None:
            h, w = rgb.shape[:2]
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if writer:
        writer.release()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "covla_vla_best.pt"))
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-id", type=str, default=None,
                    help="test video for the whole-video viz (default: random)")
    ap.add_argument("--no-mp4", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.ckpt, device)

    index_path = PREPROCESSED_ROOT / "index" / "test.jsonl"
    test = [json.loads(l) for l in open(index_path, encoding="utf-8")]
    by_video = defaultdict(list)
    for s in test:
        by_video[s["video_id"]].append(s)
    for v in by_video.values():
        v.sort(key=lambda s: s["frame_idx"])
    print(f"test split: {len(test)} samples / {len(by_video)} videos")

    # ---- 1. 10 random samples (spread across distinct videos when possible)
    vids = rng.sample(sorted(by_video), min(args.num_samples, len(by_video)))
    picks = [rng.choice(by_video[v]) for v in vids]
    while len(picks) < args.num_samples:
        picks.append(rng.choice(test))
    preds = predict_batch(model, tokenizer, picks, device)

    fig_g, axes = plt.subplots(2, 5, figsize=(20, 9))
    ades, fdes = [], []
    for k, (s, ax) in enumerate(zip(picks, axes.ravel())):
        out = OUT_DIR / f"sample_{k:02d}_{s['video_id']}_{s['frame_idx']}.png"
        ade, fde = sample_figure(s, preds[k], out)
        ades.append(ade)
        fdes.append(fde)
        plot_bev(ax, preds[k], gt_traj(s),
                 f"{s['video_id'][:14]}.. f{s['frame_idx']}\n"
                 f"ADE {ade:.2f} / FDE {fde:.2f} m")
        print(f"  [{k}] {s['video_id']} f{s['frame_idx']}: "
              f"ADE {ade:.2f} m  FDE {fde:.2f} m  -> {out.name}")
    fig_g.suptitle(f"10 random test samples | mean ADE {np.mean(ades):.2f} m, "
                   f"mean FDE {np.mean(fdes):.2f} m", fontsize=13)
    fig_g.tight_layout(rect=[0, 0, 1, 0.96])
    fig_g.savefig(OUT_DIR / "overview_grid.png", dpi=140)
    plt.close(fig_g)

    # ---- 2. whole-video rolling predictions
    vid = args.video_id or rng.choice(sorted(by_video))
    vsamples = by_video[vid]
    print(f"\nwhole-video viz: {vid} ({len(vsamples)} samples @ {DATA.sample_hz} Hz)")
    vpreds = predict_batch(model, tokenizer, vsamples, device)
    v_ade = np.mean([ade_fde(vpreds[k], gt_traj(s))
                     [0] for k, s in enumerate(vsamples)])
    print(f"  video mean ADE: {v_ade:.2f} m")
    rolling_video_figure(vsamples, vpreds, OUT_DIR / f"video_{vid}_rolling.png")
    if not args.no_mp4:
        rolling_video_mp4(vsamples, vpreds, OUT_DIR / f"video_{vid}.mp4")
        print(f"  wrote video_{vid}.mp4")

    print(f"\nall outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
