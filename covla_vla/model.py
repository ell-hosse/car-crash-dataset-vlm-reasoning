"""CoVLA trajectory-VLA model.

Architecture (CoVLA-Agent style, sized for real-time edge inference):
  - Vision:  DINOv2-small (22M) -> patch tokens
  - Language: frozen CLIP text tower encodes the scene caption -> 1 token
              (at inference the caption comes from a real-time VLM captioner)
  - State:   MLP over ego signals (speed, accel, steering, ...) -> 1 token
  - Fusion:  TransformerEncoder over [traj_query | vision | text | state]
  - Head:    MLP on the trajectory query -> num_waypoints x 2 (BEV x,y, normalized)

~50M params total; exports cleanly to ONNX/TensorRT for Jetson.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, CLIPTextModel, CLIPTokenizerFast

from .config import DATA, MODEL


class CoVLATrajectoryModel(nn.Module):
    def __init__(self, mcfg=MODEL, dcfg=DATA):
        super().__init__()
        self.num_waypoints = dcfg.num_waypoints
        self.out_dim = 2 if dcfg.use_xy_only else 3
        d = mcfg.d_model

        # --- vision ---
        self.vision = AutoModel.from_pretrained(mcfg.vision_model)
        v_dim = self.vision.config.hidden_size
        if mcfg.freeze_vision:
            self.vision.requires_grad_(False)
        self.v_proj = nn.Linear(v_dim, d)

        # --- language (frozen) ---
        # newer transformers versions don't auto-extract the text config from
        # a full CLIP checkpoint, so pass it explicitly (backward-compatible)
        _cfg = AutoConfig.from_pretrained(mcfg.text_model)
        _text_cfg = getattr(_cfg, "text_config", _cfg)
        self.text = CLIPTextModel.from_pretrained(mcfg.text_model, config=_text_cfg)
        t_dim = self.text.config.hidden_size
        if mcfg.freeze_text:
            self.text.requires_grad_(False)
        self.t_proj = nn.Linear(t_dim, d)

        # --- ego state ---
        self.s_proj = nn.Sequential(
            nn.Linear(len(dcfg.state_keys), d), nn.GELU(), nn.Linear(d, d))

        # --- fusion ---
        self.traj_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(4, d) * 0.02)  # query/vis/txt/state
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=mcfg.fusion_heads, dim_feedforward=4 * d,
            dropout=mcfg.dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=mcfg.fusion_layers)

        # --- head ---
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, self.num_waypoints * self.out_dim))

    def encode_text(self, input_ids, attention_mask):
        """Pooled caption embedding -> (B, 1, d). Cacheable at inference."""
        out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        return self.t_proj(out.pooler_output).unsqueeze(1)

    def forward(self, image, state, input_ids=None, attention_mask=None,
                text_embed=None):
        """text_embed (B,1,d) may be passed directly (cached at inference)."""
        B = image.shape[0]
        vis = self.v_proj(self.vision(pixel_values=image).last_hidden_state)
        if text_embed is None:
            text_embed = self.encode_text(input_ids, attention_mask)
        st = self.s_proj(state).unsqueeze(1)
        q = self.traj_query.expand(B, -1, -1)

        tokens = torch.cat([
            q + self.type_embed[0],
            vis + self.type_embed[1],
            text_embed + self.type_embed[2],
            st + self.type_embed[3],
        ], dim=1)
        fused = self.fusion(tokens)
        pred = self.head(fused[:, 0])
        return pred.view(B, self.num_waypoints, self.out_dim)


def build_model_and_tokenizer(mcfg=MODEL, dcfg=DATA):
    model = CoVLATrajectoryModel(mcfg, dcfg)
    tokenizer = CLIPTokenizerFast.from_pretrained(mcfg.text_model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def ade_fde(pred_m: torch.Tensor, gt_m: torch.Tensor):
    """Average / Final Displacement Error in meters. Inputs (B, T, 2) denormalized."""
    dist = torch.linalg.norm(pred_m - gt_m, dim=-1)   # (B, T)
    return dist.mean().item(), dist[:, -1].mean().item()
