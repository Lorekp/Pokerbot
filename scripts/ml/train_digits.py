#!/usr/bin/env python3
"""Train the digit-strip recognizer (CRNN + CTC) on synthetic data, export ONNX.

The model reads a fixed 32x128 grayscale strip and emits a variable-length numeric
string ("4,320", "720", ...). Conv-only (no LSTM) for fast, portable ONNX export.

Usage:
    python scripts/ml/train_digits.py --steps 4000 --out models/local_vision/digits.onnx
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.ml import synth

IMG_H, IMG_W = 32, 128
T_STEPS = 32  # time steps after conv stack (width preserved)


class DigitCRNN(nn.Module):
    def __init__(self, num_classes: int = synth.NUM_CLASSES):
        super().__init__()
        def block(ci, co, pool):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )
        self.features = nn.Sequential(
            block(1, 32, (2, 2)),    # 32x128 -> 16x64
            block(32, 64, (2, 2)),   # -> 8x32
            block(64, 128, (2, 1)),  # -> 4x32 (preserve width=time)
            block(128, 128, (2, 1)), # -> 2x32
            block(128, 128, (2, 1)), # -> 1x32
        )
        self.classifier = nn.Conv2d(128, num_classes, 1)  # 1x1 over (1 x T)

    def forward(self, x):  # x: (N,1,32,128)
        f = self.features(x)           # (N,128,1,T)
        logits = self.classifier(f)    # (N,C,1,T)
        logits = logits.squeeze(2).permute(0, 2, 1)  # (N,T,C)
        return logits


def load_real_samples(real_dir: Path) -> list[tuple[np.ndarray, str]]:
    """Real labeled digit crops harvested from references (correct client font)."""
    import json

    labels_path = real_dir / "labels.json"
    if not labels_path.is_file():
        return []
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    out = []
    for name, text in labels.items():
        im = cv2.imread(str(real_dir / name))
        if im is not None and all(c in synth.DIGIT_VOCAB for c in text):
            out.append((im, text))
    return out


def _real_strip_to_input(crop: np.ndarray, rng) -> np.ndarray:
    # vary scale/padding a touch, then match synthetic preprocessing
    strip = synth.augment(crop, rng)
    if rng.random() < 0.5:
        padx = rng.randint(0, 6)
        strip = cv2.copyMakeBorder(strip, 0, 0, padx, padx, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32)
    lo, hi = g.min(), g.max()
    return (g - lo) / (hi - lo) if hi - lo > 1e-3 else g * 0


def make_batch(bs: int, bgs, fonts, rng, real=None, p_real: float = 0.0):
    xs, targets, tgt_lens, texts = [], [], [], []
    for _ in range(bs):
        if real and rng.random() < p_real:
            crop, text = rng.choice(real)
            g = _real_strip_to_input(crop, rng)
        else:
            text = synth.random_amount(rng)
            bg = rng.choice(bgs)
            color = rng.choice(["white", "white", "yellow", "grey"])
            strip = synth.render_number_strip(text, bg, rng.choice(fonts), rng, color=color)
            strip = synth.augment(strip, rng)
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            g = gray.astype(np.float32)
            lo, hi = g.min(), g.max()
            g = (g - lo) / (hi - lo) if hi - lo > 1e-3 else g * 0
        xs.append(g)
        lab = synth.text_to_labels(text)
        targets.extend(lab)
        tgt_lens.append(len(lab))
        texts.append(text)
    x = torch.from_numpy(np.stack(xs)[:, None, :, :]).float()
    return x, torch.tensor(targets), torch.tensor(tgt_lens), texts


def greedy_decode(logits: np.ndarray) -> str:
    idx = logits.argmax(-1)  # (T,)
    out, prev = [], 0
    for i in idx:
        if i != prev and i != 0:
            out.append(int(i))
        prev = int(i)
    return synth.labels_to_text(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "local_vision" / "digits.onnx"))
    ap.add_argument("--real-dir", default=str(REPO_ROOT / "data" / "ml" / "digits"))
    ap.add_argument("--p-real", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    print("harvesting backgrounds...")
    bgs = synth.harvest_dark_backgrounds(n=500, hw=(IMG_H, IMG_W))
    fonts = synth.discover_fonts()
    print(f"{len(bgs)} backgrounds, {len(fonts)} fonts")

    real_all = load_real_samples(Path(args.real_dir))
    rng.shuffle(real_all)
    n_val = max(8, len(real_all) // 6)
    real_val, real_train = real_all[:n_val], real_all[n_val:]
    p_real = args.p_real if real_train else 0.0
    print(f"real crops: {len(real_train)} train / {len(real_val)} val (p_real={p_real})")

    model = DigitCRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def real_val_acc() -> float:
        if not real_val:
            return float("nan")
        model.eval()
        ok = 0
        with torch.no_grad():
            for crop, text in real_val:
                g = _real_strip_to_input(crop, random.Random(7))  # deterministic
                xt = torch.from_numpy(g[None, None]).float().to(device)
                pred = greedy_decode(model(xt).cpu().numpy()[0])
                ok += int(pred == text)
        model.train()
        return ok / len(real_val)

    model.train()
    for step in range(1, args.steps + 1):
        x, targets, tgt_lens, _ = make_batch(args.batch, bgs, fonts, rng, real_train, p_real)
        x, targets, tgt_lens = x.to(device), targets.to(device), tgt_lens.to(device)
        logits = model(x)                       # (N,T,C)
        logp = F.log_softmax(logits, -1).permute(1, 0, 2)  # (T,N,C)
        in_lens = torch.full((x.size(0),), logits.size(1), dtype=torch.long, device=device)
        loss = ctc(logp, targets, in_lens, tgt_lens)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, _, _, vtexts = make_batch(64, bgs, fonts, rng)
                vlogits = model(vx.to(device)).cpu().numpy()
                preds = [greedy_decode(vlogits[i]) for i in range(len(vtexts))]
                sacc = sum(p == t for p, t in zip(preds, vtexts)) / len(vtexts)
            model.train()
            print(f"step {step:5d} loss {loss.item():.3f} synth_acc {sacc:.2%} real_val_acc {real_val_acc():.2%}")

    # final eval
    model.eval()
    with torch.no_grad():
        vx, _, _, vtexts = make_batch(512, bgs, fonts, rng)
        vlogits = model(vx.to(device)).cpu().numpy()
        preds = [greedy_decode(vlogits[i]) for i in range(len(vtexts))]
        acc = sum(p == t for p, t in zip(preds, vtexts)) / len(vtexts)
    print(f"FINAL synthetic exact-string acc: {acc:.2%} | real_val acc: {real_val_acc():.2%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out_path.with_suffix(".pt")
    torch.save(model.state_dict(), ckpt)
    print(f"saved checkpoint {ckpt}")

    class ExportWrap(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            return F.log_softmax(self.m(x), -1)

    wrap = ExportWrap(model).to("cpu").eval()
    dummy = torch.zeros(1, 1, IMG_H, IMG_W)
    # Stable TorchScript exporter (dynamo=False) avoids the onnxscript dependency.
    torch.onnx.export(
        wrap, dummy, str(out_path),
        input_names=["input"], output_names=["logprobs"],
        dynamic_axes=None, opset_version=17, dynamo=False,
    )
    print(f"exported {out_path} ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
