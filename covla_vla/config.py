"""Central configuration for the CoVLA VLA pipelines (training + real-time inference)."""
from dataclasses import dataclass, field
from pathlib import Path
import platform


# ---------------------------------------------------------------------------
# Paths.
# manifest.jsonl stores absolute Windows paths (D:\hf\...). When you train on a
# Linux H100 box, copy/mount the data and set PATH_REMAP accordingly, e.g.
#   PATH_REMAP = {"D:\\hf": "/data/hf"}
# Leave it empty on the Windows machine.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_WINDOWS_ROOT = Path("D:/hf")

# Auto-detect: if the Windows data root is absent, remap to the repo-local copy.
if not _WINDOWS_ROOT.exists():
    PATH_REMAP: dict = {"D:/hf": str(_REPO_ROOT), "D:\\\\hf": str(_REPO_ROOT)}
else:
    PATH_REMAP: dict = {}

DATA_ROOT = Path("D:/hf")
METADATA_ROOT = DATA_ROOT / "CoVLA-metadata"
MANIFEST_PATH = METADATA_ROOT / "manifest.jsonl"

# Where preprocessed (downsampled) frames + index are written.
# On Linux without D:/hf, fall back to the repo-local covla_preprocessed/.
if _WINDOWS_ROOT.exists():
    PREPROCESSED_ROOT = DATA_ROOT / "covla_preprocessed"
else:
    PREPROCESSED_ROOT = _REPO_ROOT / "covla_preprocessed"

# Where checkpoints / exports are saved.
OUTPUT_DIR = Path(__file__).resolve().parent / "runs"


def remap_path(p: str) -> Path:
    """Rewrite a manifest path for the current machine."""
    for src, dst in PATH_REMAP.items():
        if p.startswith(src):
            p = dst + p[len(src):]
    if platform.system() != "Windows":
        p = p.replace("\\", "/")
    return Path(p)


# ---------------------------------------------------------------------------
# Data / downsampling (follows the CoVLA paper setup)
# Videos: 30 s @ 20 fps, 1928x1208. States & captions are per-frame (20 Hz).
# The paper samples scenes at 2 Hz for learning; trajectory target is the
# next 3 s (60 points @ 20 Hz) in the calibrated ego frame (x fwd, y left).
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    sample_hz: float = 2.0            # downsample 20 Hz -> 2 Hz for training samples
    video_fps: float = 20.0
    frames_per_video: int = 600
    image_size: int = 224             # model input resolution
    jpeg_width: int = 480             # stored preprocessed frame width (keeps aspect)
    traj_horizon: int = 60            # 3 s @ 20 Hz future points available in states
    traj_subsample: int = 3           # keep every 3rd point -> 20 waypoints over 3 s
    use_xy_only: bool = True          # predict BEV (x, y); z is ~flat for driving
    caption_field: str = "rich_caption"
    # ego-state features fed to the model (order matters; must match inference)
    state_keys: tuple = ("vEgo", "aEgo", "steeringAngleDeg", "brake", "gas",
                         "leftBlinker", "rightBlinker")
    # split fractions by video id (deterministic)
    val_frac: float = 0.05
    test_frac: float = 0.05

    @property
    def frame_stride(self) -> int:
        return int(round(self.video_fps / self.sample_hz))   # 10

    @property
    def num_waypoints(self) -> int:
        return self.traj_horizon // self.traj_subsample       # 20


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vision_model: str = "facebook/dinov2-small"        # 22M, fast, TRT-friendly
    text_model: str = "openai/clip-vit-base-patch32"   # frozen text tower
    d_model: int = 384
    fusion_layers: int = 4
    fusion_heads: int = 8
    dropout: float = 0.1
    freeze_text: bool = True
    freeze_vision: bool = False        # H100: full fine-tune of the small ViT is fine


# ---------------------------------------------------------------------------
# Training (H100)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256              # H100 80GB handles this at 224px easily
    lr: float = 3e-4
    vision_lr: float = 3e-5            # smaller LR for the pretrained backbone
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    num_workers: int = 12
    bf16: bool = True
    log_every: int = 50
    ckpt_name: str = "covla_vla"
    seed: int = 42


# ---------------------------------------------------------------------------
# Real-time inference
# ---------------------------------------------------------------------------
@dataclass
class RealtimeConfig:
    # Free, fast VLM used to generate captions online (runs in a side thread).
    captioner_model: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    caption_interval_s: float = 1.0    # generate a fresh caption ~1x per second
    caption_max_new_tokens: int = 96
    caption_prompt: str = (
        "You are the perception module of an autonomous vehicle. Describe this "
        "front-camera driving scene in the style: ego vehicle motion, traffic "
        "lights, weather, road type, notable objects/pedestrians, and what the "
        "driver should be careful about. One short paragraph."
    )
    target_hz: float = 10.0            # trajectory model inference rate
    display: bool = True               # draw overlay window (needs a display)


DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
REALTIME = RealtimeConfig()
