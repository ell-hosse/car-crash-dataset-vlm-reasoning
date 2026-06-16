"""Correct RAG injection: three strategies vs. the original broken setup.

POLICY SOURCE
-------------
By default uses crash_policies.jsonl (2017 raw per-clip policies).
Pass --policy-source abstract to use abstract_patterns.jsonl instead.

FOUR VARIANTS COMPARED
-----------------------
0. no-RAG        : plain caption → CLIP(caption) → VLA
1. Original      : policy text injected into VLM prompt → VLM writes risk-aware
                   caption → CLIP(caption_with_policy) → VLA   [THE BUG]
2. Fix1-sep      : CLIP(caption) cat CLIP(policy) → (B,2,d) → VLA  [zero-retrain]
3. Fix2-blend    : (1-α)·CLIP(caption) + α·CLIP(policy) → VLA      [zero-retrain]

Usage:
    python -m testing_final_performance.visualize_correct_rag_injection \\
        --ckpt rag/covla_vla_best.pt \\
        [--policy-source crash|abstract] [--num-videos 10] [--seed 0] \\
        [--top-k 5] [--alpha 0.25] [--max-samples N] [--fps 4]
"""
from __future__ import annotations
import argparse, json, random, sys, textwrap, time
from collections import defaultdict
from pathlib import Path

import cv2, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_repo_root(start: Path) -> Path:
    for d in (start, *start.parents):
        if (d / "covla_vla").is_dir() and (d / "crash_policies.jsonl").exists():
            return d
    return start

REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compute_dist"))

from covla_vla.config import DATA, REALTIME, PREPROCESSED_ROOT
from covla_vla.dataset import preprocess_image, state_to_vec, denormalize_traj
from covla_vla.infer_realtime import load_model, project_traj
from testing_final_performance.visualize_vla_performance_on_CoVLA_before_rag import gt_traj, ade_fde
from clip_retrieval import (build_or_load_policy_index, ClipEmbedder,
                             pool_clip_video_embedding, build_vlm_rag_prompt,
                             DEFAULT_CLIP_MODEL, DEFAULT_POLICIES, DEFAULT_INDEX)

OUT_DIR = Path(__file__).resolve().parent / "viz_correct_rag"

# BGR colours for OpenCV overlay / matplotlib colours for BEV
COL_GT    = (0,   0,   255)   # red
COL_NORAG = (0,   255, 0  )   # green
COL_ORIG  = (255, 0,   255)   # magenta  – Original (broken)
COL_SEP   = (255, 255, 0  )   # cyan     – Fix1 separate
COL_BLEND = (0,   165, 255)   # orange   – Fix2 blend

MPL_COLS  = {"GT": "red", "no-RAG": "green",
             "Original": "magenta", "Fix1-sep": "cyan", "Fix2-blend": "orange"}


# ===========================================================================
#  SmolVLM2 captioner  (accepts any prompt string)
# ===========================================================================
class VLMCaptioner:
    def __init__(self, device, cfg=REALTIME):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.cfg, self.device = cfg, device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(cfg.captioner_model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.captioner_model, torch_dtype=dtype).to(device).eval()

    @torch.no_grad()
    def caption(self, bgr: np.ndarray, prompt: str) -> str:
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device, dtype=self.model.dtype)
        out = self.model.generate(
            **inputs, max_new_tokens=self.cfg.caption_max_new_tokens, do_sample=False)
        return self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


# ===========================================================================
#  CLIP scene retriever
# ===========================================================================
class SceneRetriever:
    def __init__(self, clip, matcher, top_k):
        self.clip, self.matcher, self.top_k = clip, matcher, top_k
        self.cad = max(1, int(round(REALTIME.caption_interval_s * DATA.sample_hz)))
        self._cache: dict = {}

    def _frame_emb(self, s):
        key = (s["video_id"], s["frame_idx"])
        if key not in self._cache:
            bgr = cv2.imread(str(PREPROCESSED_ROOT / s["image"]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._cache[key] = self.clip.embed_image_arrays([rgb])[0]
        return self._cache[key]

    def hits_for(self, video_samples, j):
        lo   = max(0, j - self.cad + 1)
        embs = np.stack([self._frame_emb(video_samples[k]) for k in range(lo, j + 1)])
        return self.matcher.retrieve(pool_clip_video_embedding(embs), top_k=self.top_k)


# ===========================================================================
#  Text embedding helpers
# ===========================================================================
def _enc(model, tokenizer, text, device):
    tok = tokenizer([text], padding=True, truncation=True,
                    max_length=77, return_tensors="pt").to(device)
    return model.encode_text(tok["input_ids"], tok["attention_mask"])   # (1,1,d)

def make_sep_embed(model, tok, caps, pols, device):
    caps_e = torch.cat([_enc(model, tok, c, device) for c in caps], dim=0)
    pols_e = torch.cat([_enc(model, tok, p, device) for p in pols], dim=0)
    return torch.cat([caps_e, pols_e], dim=1)   # (B,2,d)

def make_blend_embed(model, tok, caps, pols, device, alpha=0.25):
    blended = [(1-alpha)*_enc(model,tok,c,device) + alpha*_enc(model,tok,p,device)
               for c, p in zip(caps, pols)]
    return torch.cat(blended, dim=0)   # (B,1,d)

def hits_to_policy_text(hits):
    parts = [f"{h.get('latent_risk','')}. {h.get('mitigation','')}".strip(". ")
             for h in hits if h.get("latent_risk") or h.get("mitigation")]
    return " | ".join(parts) if parts else "no policy retrieved"


# ===========================================================================
#  Inference: four variants in one pass
# ===========================================================================
@torch.no_grad()
def predict_all(model, tokenizer, samples, caps_plain, caps_orig,
                policy_texts, device, alpha=0.25, batch_size=32):
    """Returns (p_norag, p_orig, p_sep, p_blend) each (N,T,2) denorm metres."""
    preds = {k: [] for k in ("norag", "orig", "sep", "blend")}

    for i in range(0, len(samples), batch_size):
        sl   = slice(i, i + batch_size)
        samp = samples[sl]
        cp   = caps_plain[sl]
        co   = caps_orig[sl]
        pl   = policy_texts[sl]

        imgs   = torch.stack([preprocess_image(
            cv2.imread(str(PREPROCESSED_ROOT / s["image"]))) for s in samp]).to(device)
        states = torch.stack([state_to_vec(s["state"]) for s in samp]).to(device)

        # no-RAG
        tok = tokenizer(list(cp), padding=True, truncation=True,
                        max_length=77, return_tensors="pt").to(device)
        preds["norag"].append(
            model(imgs, states, tok["input_ids"], tok["attention_mask"])
            .float().cpu().numpy())

        # Original (broken): caption already contains policy language
        tok2 = tokenizer(list(co), padding=True, truncation=True,
                         max_length=77, return_tensors="pt").to(device)
        preds["orig"].append(
            model(imgs, states, tok2["input_ids"], tok2["attention_mask"])
            .float().cpu().numpy())

        # Fix1 – separate embeddings
        preds["sep"].append(
            model(imgs, states,
                  text_embed=make_sep_embed(model, tokenizer, list(cp), list(pl), device))
            .float().cpu().numpy())

        # Fix2 – blended embedding
        preds["blend"].append(
            model(imgs, states,
                  text_embed=make_blend_embed(model, tokenizer, list(cp), list(pl),
                                             device, alpha=alpha))
            .float().cpu().numpy())

    return tuple(denormalize_traj(np.concatenate(preds[k], axis=0))
                 for k in ("norag", "orig", "sep", "blend"))


# ===========================================================================
#  Drawing
# ===========================================================================
def draw_overlay(sample, gt, p_no, p_or, p_se, p_bl):
    bgr = cv2.imread(str(PREPROCESSED_ROOT / sample["image"]))
    for traj, col in ((gt, COL_GT), (p_no, COL_NORAG),
                      (p_or, COL_ORIG), (p_se, COL_SEP), (p_bl, COL_BLEND)):
        pts = project_traj(traj, bgr.shape)
        if len(pts) >= 2:
            cv2.polylines(bgr, [pts], False, col, 2)
        for p in pts:
            cv2.circle(bgr, tuple(p), 3, col, -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_bev(ax, gt, p_no, p_or, p_se, p_bl, title=""):
    for arr, lbl in ((gt,"GT"),(p_no,"no-RAG"),(p_or,"Original"),
                     (p_se,"Fix1-sep"),(p_bl,"Fix2-blend")):
        ax.plot(-arr[:,1], arr[:,0], "o-", color=MPL_COLS[lbl],
                ms=2, lw=1.3, label=lbl)
    ax.scatter([0],[0], marker="^", s=50, color="black", zorder=5, label="ego")
    ax.set_xlabel("lateral (m)"); ax.set_ylabel("forward (m)")
    ax.set_title(title, fontsize=7); ax.axis("equal")
    ax.grid(alpha=0.3); ax.legend(fontsize=5)


def _wrap(label, text, width=70, max_lines=3):
    lines = textwrap.wrap(text, width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]; lines[-1] += " ..."
    return label + ("\n" + " "*len(label)).join(lines)


def render_video(samples, p_no, p_or, p_se, p_bl,
                 caps_plain, caps_orig, policy_texts, hits_list,
                 policy_source_label, out_path, fps=4):
    writer = None
    for k, s in enumerate(samples):
        gt = gt_traj(s)
        an,fn = ade_fde(p_no[k], gt)
        ao,fo = ade_fde(p_or[k], gt)
        as_,fs = ade_fde(p_se[k], gt)
        ab,fb = ade_fde(p_bl[k], gt)

        fig = plt.figure(figsize=(15, 9))
        gs  = fig.add_gridspec(3, 2, height_ratios=[2.8, 0.9, 0.9],
                               width_ratios=[1.7, 1])
        ax_img = fig.add_subplot(gs[0, 0])
        ax_bev = fig.add_subplot(gs[0, 1])
        ax_t1  = fig.add_subplot(gs[1, :])
        ax_t2  = fig.add_subplot(gs[2, :])

        ax_img.imshow(draw_overlay(s, gt, p_no[k], p_or[k], p_se[k], p_bl[k]))
        ax_img.set_title(
            f"{s['video_id']}  frame {s['frame_idx']}  "
            f"t={s['frame_idx']/DATA.video_fps:.1f}s  "
            f"policies: {policy_source_label}\n"
            f"GT=red  no-RAG=green  Original(bug)=magenta  "
            f"Fix1-sep=cyan  Fix2-blend=orange",
            fontsize=7)
        ax_img.axis("off")

        plot_bev(ax_bev, gt, p_no[k], p_or[k], p_se[k], p_bl[k],
                 f"ADE  no-RAG {an:.2f}  Orig {ao:.2f}  Sep {as_:.2f}  Blend {ab:.2f} m\n"
                 f"FDE  no-RAG {fn:.2f}  Orig {fo:.2f}  Sep {fs:.2f}  Blend {fb:.2f} m")

        ids = ", ".join(f"{h['clip_id']}({h['score']:.2f})" for h in hits_list[k])
        ax_t1.axis("off")
        ax_t1.text(0, 1,
            _wrap("Caption (plain):       ", caps_plain[k]) + "\n" +
            _wrap("Caption (orig/prompt): ", caps_orig[k]),
            transform=ax_t1.transAxes, fontsize=7, family="monospace",
            va="top", ha="left")
        ax_t2.axis("off")
        ax_t2.text(0, 1,
            _wrap("Policy text (sep/blend):", policy_texts[k]) + "\n" +
            f"Retrieved:              {ids}",
            transform=ax_t2.transAxes, fontsize=7, family="monospace",
            va="top", ha="left")

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


# ===========================================================================
#  Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",          default=str(REPO_ROOT / "covla_vla_best.pt"))
    ap.add_argument("--policy-source", default="crash",
                    choices=["crash", "abstract"],
                    help="crash = crash_policies.jsonl (2017 raw)  |  "
                         "abstract = abstract_patterns.jsonl (distilled)")
    ap.add_argument("--num-videos",    type=int,   default=10)
    ap.add_argument("--seed",          type=int,   default=0)
    ap.add_argument("--top-k",         type=int,   default=5)
    ap.add_argument("--alpha",         type=float, default=0.25,
                    help="Fix-2 blend weight (0=caption only, 1=policy only)")
    ap.add_argument("--max-samples",   type=int,   default=None)
    ap.add_argument("--fps",           type=int,   default=4)
    ap.add_argument("--clip-model",    default=DEFAULT_CLIP_MODEL)
    args = ap.parse_args()

    # --- resolve policy source ---
    if args.policy_source == "abstract":
        pol_path   = REPO_ROOT / "abstract_patterns.jsonl"
        idx_path   = REPO_ROOT / "compute_dist" / "clip_abstract_index.npz"
        pol_label  = "abstract_patterns"
    else:
        pol_path   = REPO_ROOT / "crash_policies.jsonl"
        idx_path   = Path(DEFAULT_INDEX)
        pol_label  = "crash_policies"

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device         : {device}")
    print(f"repo           : {REPO_ROOT}")
    print(f"policy source  : {pol_label}  ({pol_path.name})")
    print(f"alpha (Fix2)   : {args.alpha}")

    model, tokenizer = load_model(args.ckpt, device)
    clip    = ClipEmbedder(args.clip_model, device=device)
    matcher = build_or_load_policy_index(pol_path, idx_path,
                                         args.clip_model, embedder=clip)
    retr    = SceneRetriever(clip, matcher, args.top_k)
    vlm     = VLMCaptioner(device)
    base_prompt = REALTIME.caption_prompt

    print(f"policy index   : {len(matcher)} entries | "
          f"CLIP window = {retr.cad} frames | "
          f"captioner = {REALTIME.captioner_model}")

    index_path = PREPROCESSED_ROOT / "index" / "test.jsonl"
    test = [json.loads(l) for l in open(index_path, encoding="utf-8")]
    for s in test:
        s["image"] = s["image"].replace("\\", "/")
    by_video: dict[str, list] = defaultdict(list)
    for s in test:
        by_video[s["video_id"]].append(s)
    for v in by_video.values():
        v.sort(key=lambda s: s["frame_idx"])
    print(f"test split     : {len(test)} samples / {len(by_video)} videos\n")

    vids          = rng.sample(sorted(by_video), min(args.num_videos, len(by_video)))
    refresh_every = retr.cad
    all_metrics: dict = {}

    for vi, vid in enumerate(vids):
        samples = by_video[vid]
        if args.max_samples:
            samples = samples[:args.max_samples]
        print(f"[{vi}] {vid}: {len(samples)} samples  "
              f"(caption refresh every {refresh_every} frames)")

        caps_plain, caps_orig, policy_texts, hits_list = [], [], [], []
        cur_plain = cur_orig = cur_pol = None
        cur_hits  = None
        t_cap: list[float] = []

        for j, s in enumerate(samples):
            if (j % refresh_every == 0) or cur_hits is None:
                bgr      = cv2.imread(str(PREPROCESSED_ROOT / s["image"]))
                cur_hits = retr.hits_for(samples, j)

                t0 = time.time()
                # Plain caption – no policy in prompt
                cur_plain = vlm.caption(bgr, base_prompt)
                # Original (broken): policy text injected into VLM prompt
                cur_orig  = vlm.caption(bgr, build_vlm_rag_prompt(base_prompt, cur_hits))
                t_cap.append(time.time() - t0)

                cur_pol = hits_to_policy_text(cur_hits)

            caps_plain.append(cur_plain)
            caps_orig.append(cur_orig)
            policy_texts.append(cur_pol)
            hits_list.append(cur_hits)
            if (j + 1) % 10 == 0:
                print(f"    captioned {j + 1}/{len(samples)} frames")

        caps_plain_arr  = np.array(caps_plain,  dtype=object)
        caps_orig_arr   = np.array(caps_orig,   dtype=object)
        policy_texts_arr= np.array(policy_texts,dtype=object)

        p_no, p_or, p_se, p_bl = predict_all(
            model, tokenizer, samples,
            caps_plain_arr, caps_orig_arr, policy_texts_arr,
            device, alpha=args.alpha)

        def _ade(preds):
            return float(np.mean([ade_fde(preds[k], gt_traj(s))[0]
                                  for k, s in enumerate(samples)]))

        an, ao, as_, ab = _ade(p_no), _ade(p_or), _ade(p_se), _ade(p_bl)
        print(f"    mean ADE:  no-RAG {an:.3f}  |  Original {ao:.3f}  |  "
              f"Fix1-sep {as_:.3f}  |  Fix2-blend {ab:.3f} m  "
              f"(caption {np.mean(t_cap)*1e3:.0f} ms/refresh x2)")

        pick = random.choice(samples)
        out  = OUT_DIR / f"sample_{vi:02d}_{vid}_{pick['frame_idx']}_{pol_label}.mp4"
        render_video(samples, p_no, p_or, p_se, p_bl,
                     caps_plain, caps_orig, policy_texts, hits_list,
                     pol_label, out, fps=args.fps)
        print(f"    wrote {out.name}")

        all_metrics[vid] = []
        for k, s in enumerate(samples):
            g = gt_traj(s)
            r_no,_ = ade_fde(p_no[k], g)
            r_or,_ = ade_fde(p_or[k], g)
            r_se,_ = ade_fde(p_se[k], g)
            r_bl,_ = ade_fde(p_bl[k], g)
            all_metrics[vid].append({
                "frame_idx":       s["frame_idx"],
                "caption_plain":   caps_plain[k],
                "caption_orig":    caps_orig[k],
                "policy_text":     policy_texts[k],
                "retrieved":       [{"clip_id": h["clip_id"], "score": h["score"]}
                                    for h in hits_list[k]],
                "ade_no_rag":      r_no,
                "ade_original":    r_or,
                "ade_fix1_sep":    r_se,
                "ade_fix2_blend":  r_bl,
            })

    metrics_path = OUT_DIR / f"metrics_{pol_label}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(vids)} mp4s + {metrics_path.name}  →  {OUT_DIR}")

    def _agg(key):
        return np.mean([r[key] for v in all_metrics.values() for r in v])

    print(f"\n=== aggregate ADE  [{pol_label}] ===")
    print(f"  no-RAG (baseline)          : {_agg('ade_no_rag'):.3f} m")
    print(f"  Original (bug: cap+policy) : {_agg('ade_original'):.3f} m")
    print(f"  Fix1 – separate embeddings : {_agg('ade_fix1_sep'):.3f} m")
    print(f"  Fix2 – blended embedding   : {_agg('ade_fix2_blend'):.3f} m")


if __name__ == "__main__":
    main()
