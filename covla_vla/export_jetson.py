"""Export the trained trajectory model to ONNX for Jetson (TensorRT).

The text encoder is exported separately - the caption embedding only changes
~1x/second, so on Jetson you run the vision+fusion engine per frame and the
text engine only when the captioner produces a new caption.

Usage:
    python -m covla_vla.export_jetson --ckpt covla_vla/runs/covla_vla_best.pt

Then on the Jetson:
    trtexec --onnx=covla_traj_core.onnx --fp16 --saveEngine=covla_traj_core.engine
    trtexec --onnx=covla_text_encoder.onnx --fp16 --saveEngine=covla_text.engine
"""
import argparse

import torch

from .config import DATA, OUTPUT_DIR
from .model import build_model_and_tokenizer


class CoreWrapper(torch.nn.Module):
    """image + state + precomputed text_embed -> trajectory (per-frame path)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, state, text_embed):
        return self.model(image, state, text_embed=text_embed)


class TextWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model.encode_text(input_ids, attention_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, tokenizer = build_model_and_tokenizer()
    model.load_state_dict(ck["model"])
    model.eval()
    d = model.traj_query.shape[-1]

    img = torch.randn(1, 3, DATA.image_size, DATA.image_size)
    st = torch.randn(1, len(DATA.state_keys))
    te = torch.randn(1, 1, d)
    core_path = OUTPUT_DIR / "covla_traj_core.onnx"
    torch.onnx.export(
        CoreWrapper(model), (img, st, te), str(core_path),
        input_names=["image", "state", "text_embed"],
        output_names=["trajectory"], opset_version=args.opset)
    print(f"wrote {core_path}")

    tok = tokenizer(["The ego vehicle is driving."], padding="max_length",
                    max_length=77, return_tensors="pt")
    text_path = OUTPUT_DIR / "covla_text_encoder.onnx"
    torch.onnx.export(
        TextWrapper(model), (tok["input_ids"], tok["attention_mask"]),
        str(text_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embed"], opset_version=args.opset)
    print(f"wrote {text_path}")
    print("\nOn the Jetson:")
    print("  trtexec --onnx=covla_traj_core.onnx --fp16 "
          "--saveEngine=covla_traj_core.engine")
    print("  trtexec --onnx=covla_text_encoder.onnx --fp16 "
          "--saveEngine=covla_text.engine")


if __name__ == "__main__":
    main()
