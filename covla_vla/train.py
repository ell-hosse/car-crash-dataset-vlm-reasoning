"""Training pipeline (H100). Saves checkpoints consumed by infer_realtime.py.

Usage:
    python -m covla_vla.train [--epochs 10] [--batch-size 256] [--resume PATH]

Checkpoints (in covla_vla/runs/):
    covla_vla_best.pt   - lowest val ADE        <- use this for inference
    covla_vla_last.pt   - latest epoch (for --resume)
Each checkpoint bundles model weights + the data/model configs so the
inference pipeline is self-contained.
"""
import argparse
import dataclasses
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DATA, MODEL, TRAIN, OUTPUT_DIR
from .dataset import CoVLADataset, make_collate, denormalize_traj, TRAJ_SCALE
from .model import build_model_and_tokenizer, ade_fde

_TRAJ_SCALE_T = torch.tensor(TRAJ_SCALE)


def save_ckpt(path, model, optimizer, epoch, best_ade):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_ade": best_ade,
        "data_config": dataclasses.asdict(DATA),
        "model_config": dataclasses.asdict(MODEL),
    }, path)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    scale = _TRAJ_SCALE_T.to(device)
    tot_ade = tot_fde = tot_loss = n = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            pred = model(batch["image"], batch["state"],
                         batch["input_ids"], batch["attention_mask"])
            loss = F.smooth_l1_loss(pred, batch["traj"])
        ade, fde = ade_fde(pred.float() * scale, batch["traj"].float() * scale)
        b = pred.shape[0]
        tot_ade += ade * b; tot_fde += fde * b; tot_loss += loss.item() * b; n += b
    return tot_loss / n, tot_ade / n, tot_fde / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    ap.add_argument("--lr", type=float, default=TRAIN.lr)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(TRAIN.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if TRAIN.bf16 and device.type == "cuda" else torch.float32
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer()
    model.to(device)
    if device.type == "cuda":
        model = torch.compile(model)  # H100: big speedup; comment out if issues

    collate = make_collate(tokenizer)
    dl_kw = dict(batch_size=args.batch_size, num_workers=TRAIN.num_workers,
                 collate_fn=collate, pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(CoVLADataset("train", augment=True), shuffle=True,
                          drop_last=True, **dl_kw)
    val_dl = DataLoader(CoVLADataset("val"), shuffle=False, **dl_kw)
    print(f"train={len(train_dl.dataset)}  val={len(val_dl.dataset)} samples")

    # param groups: lower LR for the pretrained vision backbone
    vis_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (vis_params if ".vision." in name or name.startswith("vision.")
         else other_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": other_params, "lr": args.lr},
         {"params": vis_params, "lr": TRAIN.vision_lr}],
        weight_decay=TRAIN.weight_decay)

    total_steps = len(train_dl) * args.epochs

    def lr_lambda(step):
        if step < TRAIN.warmup_steps:
            return step / max(1, TRAIN.warmup_steps)
        t = (step - TRAIN.warmup_steps) / max(1, total_steps - TRAIN.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch, best_ade = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, best_ade = ck["epoch"] + 1, ck.get("best_ade", best_ade)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    scale = _TRAJ_SCALE_T.to(device)
    step = start_epoch * len(train_dl)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                pred = model(batch["image"], batch["state"],
                             batch["input_ids"], batch["attention_mask"])
                loss = F.smooth_l1_loss(pred, batch["traj"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if i % TRAIN.log_every == 0:
                with torch.no_grad():
                    ade, fde = ade_fde(pred.float() * scale,
                                       batch["traj"].float() * scale)
                ips = (i + 1) * args.batch_size / (time.time() - t0)
                print(f"ep{epoch} it{i}/{len(train_dl)} loss={loss.item():.4f} "
                      f"ADE={ade:.2f}m FDE={fde:.2f}m "
                      f"lr={scheduler.get_last_lr()[0]:.2e} {ips:.0f} img/s")

        val_loss, val_ade, val_fde = evaluate(model, val_dl, device, amp_dtype)
        print(f"== epoch {epoch}: val loss={val_loss:.4f} "
              f"ADE={val_ade:.3f}m FDE={val_fde:.3f}m ==")

        raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        save_ckpt(OUTPUT_DIR / f"{TRAIN.ckpt_name}_last.pt", raw, optimizer,
                  epoch, best_ade)
        if val_ade < best_ade:
            best_ade = val_ade
            save_ckpt(OUTPUT_DIR / f"{TRAIN.ckpt_name}_best.pt", raw, optimizer,
                      epoch, best_ade)
            print(f"   new best (ADE {best_ade:.3f}m) -> "
                  f"{OUTPUT_DIR / (TRAIN.ckpt_name + '_best.pt')}")

    # final test-set numbers
    test_dl = DataLoader(CoVLADataset("test"), shuffle=False, **dl_kw)
    test_loss, test_ade, test_fde = evaluate(model, test_dl, device, amp_dtype)
    print(f"TEST: loss={test_loss:.4f} ADE={test_ade:.3f}m FDE={test_fde:.3f}m")


if __name__ == "__main__":
    main()
