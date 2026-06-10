# CoVLA Trajectory-VLA — training + real-time inference

Two separate pipelines sharing one checkpoint:

```
preprocess.py  ──>  train.py (H100)  ──>  runs/covla_vla_best.pt  ──>  infer_realtime.py (Jetson/PC)
                                                                  └──>  export_jetson.py (ONNX/TensorRT)
```

## Architecture

CoVLA-Agent-style VLA sized for real-time edge inference (~50M params):

- **Vision**: DINOv2-small fine-tuned, 224×224 input
- **Language**: frozen CLIP text tower encodes the scene caption (1 token). At
  inference captions are generated **live** by SmolVLM2-256M in a background
  thread (~1 Hz); the embedding is cached so the per-frame path never blocks.
- **State**: MLP over ego signals (vEgo, aEgo, steering, brake, gas, blinkers)
- **Fusion**: 4-layer transformer encoder over [traj-query | vision | text | state]
- **Output**: 20 BEV waypoints (x fwd, y left) covering 3 s — same target as the
  CoVLA paper (60 pts @ 20 Hz, subsampled ×3). Metrics: ADE / FDE (m).

## Downsampling (per CoVLA paper)

Videos are 30 s @ 20 fps (1928×1208) with per-frame states/captions. Preprocess
samples at **2 Hz** (every 10th frame), resizes frames to 480 px JPEGs, drops
end-of-clip frames lacking a full 3 s trajectory, and writes flat
train/val/test indexes (90/5/5 by video id) to `D:/hf/covla_preprocessed/`.

## Run

```bash
pip install -r covla_vla/requirements.txt

# 1. one-time preprocessing (~7.7k videos; use --limit-videos 50 to smoke-test)
python -m covla_vla.preprocess --workers 8

# 2. training (H100)
python -m covla_vla.train --epochs 10 --batch-size 256
#    resume: --resume covla_vla/runs/covla_vla_last.pt

# 3. real-time inference (loads the saved checkpoint)
python -m covla_vla.infer_realtime --ckpt covla_vla/runs/covla_vla_best.pt \
    --video "D:/hf/hub/datasets--turing-motors--CoVLA-Dataset/snapshots/<hash>/videos/<id>.mp4"
#    live camera:        --camera 0
#    save overlay video: --save out.mp4 --headless

# 4. real-time evaluation on the test split (live SmolVLM captions, ADE/FDE vs GT)
python -m covla_vla.infer_realtime --ckpt covla_vla/runs/covla_vla_best.pt --eval --max-videos 50

# 5. Jetson export
python -m covla_vla.export_jetson --ckpt covla_vla/runs/covla_vla_best.pt
```

## Notes

- Training on Linux: set `PATH_REMAP` in `config.py` (manifest stores `D:\hf\...` paths).
- `torch.compile` is on in train.py; comment out if your stack complains.
- Real-time budget: trajectory path is vision+fusion only (caption embedding
  cached) → ~5–10 ms on H100, ~25–40 ms FP16 TensorRT on Orin (≥ 25 FPS).
  SmolVLM2-256M caption refresh ~0.5–1.5 s on Orin, runs fully async.
- For higher caption quality at inference, swap `captioner_model` in
  `config.py` to `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.
