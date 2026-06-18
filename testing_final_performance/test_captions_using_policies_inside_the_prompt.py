"""
test_captions_using_policies_inside_the_prompt.py
-------------------------------------------------
Policy-conditioned captioning ("RAG inside the VLM prompt"), then the VLA
trajectory predicted FROM that caption, drawn against ground truth.

IMPORTANT - data source:
  selected_samples/<group>/<video_id>.mp4 are used ONLY to pick WHICH CoVLA
  videos (and their flagged/safe group). The actual frames, GROUND-TRUTH
  trajectory and REAL ego-state are pulled from the preprocessed CoVLA index
  (PREPROCESSED_ROOT/index/*.jsonl) - NOT from those mp4s, which are themselves
  rendered visualisation montages. This avoids drawing on top of an already
  annotated video.

Per video, at the caption cadence:
  1. CLIP-embed the FRAME IMAGE and retrieve the closest crash policy (the
     reliable route).
  2. Inject that policy into the SmolVLM2 prompt (with a "stay grounded in the
     frame" guard) and generate ONE policy-conditioned caption.
  3. The VLA predicts a 3 s trajectory from that caption + the real ego-state.

Every frame shows the real camera image with THREE trajectories:
      GT          = red
      policy-RAG  = green    (VLA prediction from the policy-conditioned caption)
      no-RAG      = yellow   (VLA prediction from a plain caption, no policy)
plus a BEV of all three, the injected policy, the captions, and ADE/FDE.

Outputs (in testing_final_performance/viz_policy_in_prompt/<group>/):
  <video_id>.mp4
  policy_in_prompt_summary.json

Usage (from repo root, machine with the D:/hf data + checkpoint):
    python -m testing_final_performance.test_captions_using_policies_inside_the_prompt \
        [--caption-interval-s 1.0] [--top-k 3] [--max-frames-per-video N]
"""
import argparse
import json
import sys
import textwrap
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
sys.path.insert(0, str(REPO_ROOT / "compute_dist"))

import covla_vla.config as covla_config                                  # noqa: E402
import covla_vla.dataset as covla_dataset_mod                            # noqa: E402

if not covla_config.PREPROCESSED_ROOT.exists():
    _local = REPO_ROOT / "covla_preprocessed"
    if _local.exists():
        covla_config.PREPROCESSED_ROOT = _local
        covla_dataset_mod.PREPROCESSED_ROOT = _local

from covla_vla.config import REALTIME, DATA                              # noqa: E402
from covla_vla.dataset import (preprocess_image, state_to_vec,           # noqa: E402
                               denormalize_traj)
from covla_vla.infer_realtime import load_model, project_traj           # noqa: E402
from clip_retrieval import (build_or_load_policy_index, ClipEmbedder,    # noqa: E402
                            DEFAULT_CLIP_MODEL, DEFAULT_POLICIES, DEFAULT_INDEX)

PREPROCESSED_ROOT = covla_config.PREPROCESSED_ROOT
SAMPLES_DIR = REPO_ROOT / "selected_samples"
OUT_DIR = Path(__file__).resolve().parent / "viz_policy_in_prompt"

# use the "lh" policy set; keep its own index so the shared crash_policies index
# (clip_policy_index.npz) used by the other scripts is never overwritten.
POLICIES_LH = REPO_ROOT / "crash_policies_lh.jsonl"
INDEX_LH = Path(DEFAULT_INDEX).with_name("clip_policy_index_lh.npz")


def _require(path, what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


def selected_video_ids(root: Path):
    """video_id -> group (flagged/safe), taken from selected_samples filenames."""
    out = {}
    for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for mp4 in sorted(group_dir.glob("*.mp4")):
            out[mp4.stem] = group_dir.name
    return out


def load_all_samples():
    """Load every preprocessed split index and group samples by video_id."""
    by_video = defaultdict(list)
    idx_dir = PREPROCESSED_ROOT / "index"
    found = []
    for split in ("test", "val", "train"):
        f = idx_dir / f"{split}.jsonl"
        if not f.exists():
            continue
        found.append(split)
        for line in open(f, encoding="utf-8"):
            s = json.loads(line)
            s["image"] = s["image"].replace("\\", "/")
            by_video[s["video_id"]].append(s)
    for v in by_video.values():
        v.sort(key=lambda s: s["frame_idx"])
    return by_video, found


# ---------------------------------------------------------------- prompt
BASE_PROMPT = REALTIME.caption_prompt


def build_policy_prompt(hit) -> str:
    """Policy-conditioned captioning prompt, engineered so the safety-relevant
    content lands in the ~77 CLIP text tokens the VLA actually reads: short,
    front-loaded, and phrased as a conservative EGO ACTION (not advice)."""
    return (
        "You are the perception and safety module of an autonomous vehicle operating in a left-hand traffic environment."
        " Look carefully at this front-camera frame.\n\n"
        "A retrieval system flagged a crash pattern that may match this scene "
        "(matched by visual similarity to past incidents -- trust your own eyes "
        "over it):\n"
        f"  - Hazard: {hit['trigger']}\n"
        f"  - Why dangerous: {hit['latent_risk']}\n"
        f"  - Safe response: {hit['mitigation']}\n\n"
        "Write ONE caption, under 45 words, in this exact order:\n"
        "1) The single most safety-critical thing visible right now (lead "
        "vehicle, pedestrian, cyclist, signal, or road/weather condition).\n"
        "2) If that hazard -- or the flagged one -- is visible or plausibly "
        "developing, state that THE EGO VEHICLE slows down, increases following "
        "distance, and is ready to brake or yield as appropriate. Describe the "
        "ego actually doing it.\n"
        "3) Brief remaining context only if space allows.\n\n"
        "Rules: describe only what you can see; never invent objects. But when a "
        "hazard is uncertain, choose the more cautious description and the "
        "slower action. Put the hazard and the cautious maneuver FIRST so they "
        "are never cut off."
    )


# ---------------------------------------------------------------- SmolVLM2
def make_prompt_captioner(device):
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    class PromptVLM:
        def __init__(self):
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(REALTIME.captioner_model)
            self.model = AutoModelForImageTextToText.from_pretrained(
                REALTIME.captioner_model, torch_dtype=dtype).to(device).eval()

        @torch.no_grad()
        def caption(self, bgr, prompt) -> str:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt}]}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(
                    device, dtype=self.model.dtype)
            out = self.model.generate(
                **inputs, max_new_tokens=REALTIME.caption_max_new_tokens,
                do_sample=False)
            text = self.processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0]
            return text.strip()

    return PromptVLM()


# ---------------------------------------------------------------- VLA + metrics
@torch.no_grad()
def predict_traj(model, tokenizer, device, bgr, state_vec, caption):
    img = preprocess_image(bgr).unsqueeze(0).to(device)
    st = state_vec.unsqueeze(0).to(device)
    tok = tokenizer([caption], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    pred = model(img, st, input_ids=tok["input_ids"],
                 attention_mask=tok["attention_mask"])
    return denormalize_traj(pred[0].float().cpu().numpy())


def ade_fde(pred, gt):
    d = np.linalg.norm(pred - gt, axis=-1)
    return float(d.mean()), float(d[-1])


# ---------------------------------------------------------------- drawing
def _wrap(text, width):
    return "\n".join(textwrap.wrap(text or "", width=width)) or "(none)"


def draw_trajs(bgr, gt, pred_rag, pred_base):
    """GT (red) + policy-RAG prediction (green) + no-RAG prediction (yellow)."""
    out = bgr.copy()
    # BGR colors: GT red, policy-RAG green, no-RAG yellow
    for traj, color in ((gt, (0, 0, 255)),
                        (pred_base, (0, 255, 255)),
                        (pred_rag, (0, 200, 0))):
        if traj is None:
            continue
        pts = project_traj(traj, out.shape)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, color, 2)
        for p in pts:
            cv2.circle(out, tuple(p), 3, color, -1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def compose_frame(bgr, gt, pred_rag, pred_base, hit, dist, score, caption,
                  ade, fde, ade_b, fde_b):
    rgb = draw_trajs(bgr, gt, pred_rag, pred_base)
    fig = plt.figure(figsize=(12, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.1, 1.4],
                          width_ratios=[1.0, 2.2], hspace=0.10, wspace=0.04)
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(rgb)
    ax_img.axis("off")
    title = "GT = red    |    policy-RAG = green    |    no-RAG = yellow"
    if hit is not None:
        title += (f"      [policy {hit['clip_id']} dist={dist:.3f}]   "
                  f"ADE green {ade:.2f} / yellow {ade_b:.2f} m")
    ax_img.set_title(title, fontsize=11.5, color="#0a5", fontweight="bold")

    ax_bev = fig.add_subplot(gs[1, 0])
    if gt is not None:
        ax_bev.plot(-gt[:, 1], gt[:, 0], "o-", color="red", ms=3, lw=1.5,
                    label="GT")
    if pred_base is not None:
        ax_bev.plot(-pred_base[:, 1], pred_base[:, 0], "o-", color="gold",
                    ms=3, lw=1.5, label="no-RAG")
    if pred_rag is not None:
        ax_bev.plot(-pred_rag[:, 1], pred_rag[:, 0], "o-", color="green",
                    ms=3, lw=1.5, label="policy-RAG")
    ax_bev.scatter([0], [0], marker="^", s=70, color="black", zorder=5,
                   label="ego")
    ax_bev.set_xlabel("lateral (m, right +)", fontsize=8)
    ax_bev.set_ylabel("forward (m)", fontsize=8)
    ax_bev.set_title("BEV", fontsize=9)
    ax_bev.axis("equal")
    ax_bev.grid(alpha=0.3)
    ax_bev.legend(fontsize=7)

    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")
    pol_block = _wrap(
        f"When {hit['trigger']}  ->  risk: {hit['latent_risk']}  "
        f"mitigation: {hit['mitigation']}", 92) if hit else "(none)"
    ax_txt.text(0.0, 1.0,
                f"INJECTED POLICY (closest by frame image):\n{pol_block}\n\n"
                f"FINAL POLICY-CONDITIONED CAPTION:\n{_wrap(caption, 92)}",
                fontsize=9.3, va="top", ha="left", family="monospace",
                transform=ax_txt.transAxes)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.05)
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return out


# ---------------------------------------------------------------- per-video
def process_video(group, vid, samples, clip, matcher, captioner,
                  model, tokenizer, device, args):
    refresh_every = max(1, int(round(args.caption_interval_s * DATA.sample_hz)))
    out_group = OUT_DIR / group
    out_group.mkdir(parents=True, exist_ok=True)
    out_path = out_group / f"{vid}.mp4"
    writer = None
    print(f"  {group}/{vid}: {len(samples)} samples, refresh every "
          f"{refresh_every} samples (~{args.caption_interval_s:.1f}s)")

    cur_hit = cur_dist = cur_score = None
    cur_caption = None       # policy-conditioned caption  -> green
    cur_base_caption = None  # plain caption (no policy)    -> yellow
    records = []
    ades, fdes, ades_b, fdes_b = [], [], [], []
    n = 0
    for j, s in enumerate(samples):
        if args.max_frames_per_video and n >= args.max_frames_per_video:
            break
        bgr = cv2.imread(str(PREPROCESSED_ROOT / s["image"]))
        if bgr is None:
            print(f"    WARNING missing frame {s['image']}, skipping")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = np.asarray(s["traj"], dtype=np.float32)
        state_vec = state_to_vec(s["state"])

        if (j % refresh_every == 0) or (cur_hit is None):
            img_emb = clip.embed_image_arrays([rgb])[0]
            hits = matcher.retrieve(img_emb, top_k=args.top_k)
            cur_hit = hits[0]
            cur_dist, cur_score = cur_hit["dist"], cur_hit["score"]
            # policy-conditioned caption (green) + plain caption (yellow)
            cur_caption = captioner.caption(bgr, build_policy_prompt(cur_hit))
            cur_base_caption = captioner.caption(bgr, BASE_PROMPT)

        pred = predict_traj(model, tokenizer, device, bgr, state_vec, cur_caption)
        pred_b = predict_traj(model, tokenizer, device, bgr, state_vec,
                              cur_base_caption)
        ade, fde = ade_fde(pred, gt)
        ade_b, fde_b = ade_fde(pred_b, gt)
        ades.append(ade); fdes.append(fde)
        ades_b.append(ade_b); fdes_b.append(fde_b)

        if (j % refresh_every == 0):
            print(f"    f{s['frame_idx']:>4} | policy[{cur_hit['clip_id']}] "
                  f"dist={cur_dist:.3f} | ADE green={ade:.2f} yellow={ade_b:.2f} | "
                  f"cap=\"{cur_caption[:50]}\"")
            records.append({
                "video_id": vid, "group": group, "frame_idx": s["frame_idx"],
                "frame_to_policy_dist": cur_dist, "frame_to_policy_sim": cur_score,
                "injected_policy": {
                    "clip_id": cur_hit["clip_id"], "trigger": cur_hit["trigger"],
                    "latent_risk": cur_hit["latent_risk"],
                    "mitigation": cur_hit["mitigation"]},
                "policy_caption": cur_caption,
                "base_caption": cur_base_caption,
                "ade_policy_rag": ade, "fde_policy_rag": fde,
                "ade_no_rag": ade_b, "fde_no_rag": fde_b,
            })

        frame_rgb = compose_frame(bgr, gt, pred, pred_b, cur_hit, cur_dist,
                                  cur_score, cur_caption, ade, fde, ade_b, fde_b)
        if writer is None:
            h, w = frame_rgb.shape[:2]
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps_out, (w, h))
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        n += 1

    if writer:
        writer.release()
    mean = lambda a: float(np.mean(a)) if a else None
    _r = lambda x: None if x is None else round(x, 3)
    m_ade, m_fde = mean(ades), mean(fdes)
    m_ade_b, m_fde_b = mean(ades_b), mean(fdes_b)
    print(f"    -> wrote {out_path.name} ({n} frames) | mean ADE "
          f"green {_r(m_ade)} / yellow {_r(m_ade_b)} m")
    return {"video_id": vid, "group": group, "n_frames": n,
            "mean_ade_policy_rag": m_ade, "mean_fde_policy_rag": m_fde,
            "mean_ade_no_rag": m_ade_b, "mean_fde_no_rag": m_fde_b,
            "records": records}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples-dir", default=str(SAMPLES_DIR),
                    help="folder whose mp4 names select the CoVLA video ids")
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "covla_vla_best.pt"))
    ap.add_argument("--policies", default=str(POLICIES_LH),
                    help="policy bank (default: crash_policies_lh.jsonl)")
    ap.add_argument("--index", default=str(INDEX_LH),
                    help="cached CLIP index for the policy bank (separate from "
                         "the shared crash_policies index)")
    ap.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--caption-interval-s", type=float,
                    default=REALTIME.caption_interval_s)
    ap.add_argument("--top-k", type=int, default=3,
                    help="policies retrieved; the closest (top-1) is injected")
    ap.add_argument("--max-frames-per-video", type=int, default=None)
    ap.add_argument("--fps-out", type=float, default=4.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"preprocessed root: {PREPROCESSED_ROOT}")
    _require(args.samples_dir, "selected_samples dir")
    _require(args.ckpt, "VLA checkpoint (--ckpt)")
    _require(args.policies, "policies file (--policies)")
    _require(PREPROCESSED_ROOT, "preprocessed CoVLA root (D:/hf or covla_preprocessed)")

    sel = selected_video_ids(Path(args.samples_dir))
    by_video, splits = load_all_samples()
    print(f"loaded splits {splits}: {len(by_video)} videos total")
    print(f"selected {len(sel)} ids from {args.samples_dir}")

    missing = [v for v in sel if v not in by_video]
    if missing:
        print(f"WARNING {len(missing)} selected ids not found in the index: "
              f"{missing}")

    clip = ClipEmbedder(args.clip_model, device=device)
    matcher = build_or_load_policy_index(
        Path(args.policies), Path(args.index), args.clip_model,
        embedder=clip, rebuild=args.rebuild_index)
    print(f"policy matcher ready with {len(matcher)} policies")
    captioner = make_prompt_captioner(device)
    model, tokenizer = load_model(args.ckpt, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for vid, group in sel.items():
        if vid not in by_video:
            continue
        print(f"\nvideo {group}/{vid}")
        summaries.append(process_video(group, vid, by_video[vid], clip, matcher,
                                        captioner, model, tokenizer, device, args))

    out_json = OUT_DIR / "policy_in_prompt_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "caption_interval_s": args.caption_interval_s,
            "top_k": args.top_k,
            "retrieval_route": "frame_image_to_policy",
            "trajectories": {"GT": "red", "policy_RAG_prediction": "green",
                             "no_RAG_prediction": "yellow"},
            "data_source": "preprocessed CoVLA (real GT + ego-state)",
            "videos": summaries,
        }, f, indent=2)
    print(f"\nall outputs in {OUT_DIR}\nsummary -> {out_json}")


if __name__ == "__main__":
    main()
