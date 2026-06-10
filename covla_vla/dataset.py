"""PyTorch dataset over the preprocessed CoVLA index (see preprocess.py)."""
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DATA, PREPROCESSED_ROOT

# ImageNet stats (DINOv2 preprocessing)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Rough normalization scales for ego-state features (keeps inputs O(1)).
STATE_SCALE = {
    "vEgo": 30.0, "aEgo": 3.0, "steeringAngleDeg": 90.0,
    "brake": 1.0, "gas": 1.0, "leftBlinker": 1.0, "rightBlinker": 1.0,
}
# Trajectory scale (meters): x forward up to ~100 m in 3 s at highway speed.
TRAJ_SCALE = np.array([50.0, 5.0], dtype=np.float32)


def preprocess_image(bgr: np.ndarray, size: int = None) -> torch.Tensor:
    """BGR uint8 -> normalized CHW float tensor. Shared with inference."""
    size = size or DATA.image_size
    img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1))


def state_to_vec(state: dict) -> torch.Tensor:
    """dict -> normalized feature vector in DATA.state_keys order. Shared with inference."""
    vals = [float(state.get(k, 0.0) or 0.0) / STATE_SCALE.get(k, 1.0)
            for k in DATA.state_keys]
    return torch.tensor(vals, dtype=torch.float32)


def normalize_traj(traj: np.ndarray) -> np.ndarray:
    return traj / TRAJ_SCALE


def denormalize_traj(traj: np.ndarray) -> np.ndarray:
    return traj * TRAJ_SCALE


class CoVLADataset(Dataset):
    """Yields dict(image, state, caption, traj). Caption is tokenized in collate."""

    def __init__(self, split: str = "train", augment: bool = False):
        index_path = PREPROCESSED_ROOT / "index" / f"{split}.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found - run `python -m covla_vla.preprocess` first")
        self.samples = [json.loads(l) for l in open(index_path, encoding="utf-8")]
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        bgr = cv2.imread(str(PREPROCESSED_ROOT / s["image"]))
        if bgr is None:
            raise IOError(f"missing frame {s['image']}")
        if self.augment:
            bgr = self._augment(bgr)
        traj = np.asarray(s["traj"], dtype=np.float32)
        return {
            "image": preprocess_image(bgr),
            "state": state_to_vec(s["state"]),
            "caption": s["caption"],
            "traj": torch.from_numpy(normalize_traj(traj)),
        }

    @staticmethod
    def _augment(bgr):
        """Light photometric augmentation only - geometry must stay aligned
        with the trajectory target, so no flips/crops."""
        if np.random.rand() < 0.5:
            bgr = cv2.convertScaleAbs(bgr,
                                      alpha=np.random.uniform(0.85, 1.15),
                                      beta=np.random.uniform(-12, 12))
        return bgr


def make_collate(tokenizer, max_len: int = 77):
    def collate(batch):
        captions = [b["caption"] for b in batch]
        tok = tokenizer(captions, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "state": torch.stack([b["state"] for b in batch]),
            "traj": torch.stack([b["traj"] for b in batch]),
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }
    return collate
