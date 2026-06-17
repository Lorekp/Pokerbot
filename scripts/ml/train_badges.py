#!/usr/bin/env python3
"""Train the action-badge classifier on synthetic + real crops, export ONNX.

Classes: fold, call, raise, check, bet, sb, bb, all_in, none. Synthetic rendered
badges (balanced) close the class-imbalance gap in the small real harvest; real
crops anchor the true client appearance. Color input (suit/hue matters).

Usage:
    python scripts/ml/train_badges.py --steps 3000 --out models/local_vision/badges.onnx
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from scripts.ml import synth

IMG_H, IMG_W = 28, 64
CLASSES = ["fold", "call", "raise", "check", "bet", "sb", "bb", "all_in", "none"]
NONE_IDX = CLASSES.index("none")
RENDER_LABELS = [c for c in CLASSES if c != "none"]


def load_real(real_dir: Path) -> dict[str, list[np.ndarray]]:
    by: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    lp = real_dir / "labels.json"
    if not lp.is_file():
        return by
    labels = json.loads(lp.read_text(encoding="utf-8"))
    for name, lab in labels.items():
        if lab in by:
            im = cv2.imread(str(real_dir / name))
            if im is not None:
                by[lab].append(im)
    return by


def _to_input(img: np.ndarray, rng) -> np.ndarray:
    img = synth.augment(img, rng)
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def make_batch(bs, bgs, fonts, real, rng, p_real=0.4):
    xs, ys = [], []
    for _ in range(bs):
        if rng.random() < 0.1:  # none
            img = _to_input(rng.choice(bgs), rng)
            label = NONE_IDX
        else:
            label_name = rng.choice(RENDER_LABELS)
            use_real = real.get(label_name) and rng.random() < p_real
            if use_real:
                img = _to_input(rng.choice(real[label_name]), rng)
            else:
                badge = synth.render_badge(label_name, rng.choice(bgs), rng.choice(fonts), rng)
                img = _to_input(badge, rng)
            label = CLASSES.index(label_name)
        xs.append(img)
        ys.append(label)
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float()
    return x, torch.tensor(ys)


class BadgeCNN(nn.Module):
    def __init__(self, n=len(CLASSES)):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3, 24), block(24, 48), block(48, 96))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(96, 96), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(96, n),
        )

    def forward(self, x):
        return self.head(self.features(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "local_vision" / "badges.onnx"))
    ap.add_argument("--real-dir", default=str(REPO_ROOT / "data" / "ml" / "badges"))
    ap.add_argument("--p-real", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    bgs = synth.harvest_backgrounds(n=300, hw=(IMG_H, IMG_W), min_mean=20, max_mean=150)
    fonts = synth.discover_fonts()
    real_all = load_real(Path(args.real_dir))
    # hold out ~20% of each class with >=4 samples for real-val
    real_train, real_val = {c: [] for c in CLASSES}, []
    for c, imgs in real_all.items():
        rng.shuffle(imgs)
        k = len(imgs) // 5 if len(imgs) >= 5 else 0
        for im in imgs[:k]:
            real_val.append((im, c))
        real_train[c] = imgs[k:]
    print(f"real per-class: {{ {', '.join(f'{c}:{len(v)}' for c,v in real_train.items() if v)} }}; "
          f"{len(real_val)} val")

    model = BadgeCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    lossf = nn.CrossEntropyLoss()

    def real_val_acc():
        if not real_val:
            return float("nan")
        model.eval(); ok = 0
        with torch.no_grad():
            for im, c in real_val:
                xi = torch.from_numpy(_to_input(im, random.Random(3))).permute(2, 0, 1)[None].to(device)
                ok += int(CLASSES[model(xi).argmax(1).item()] == c)
        model.train()
        return ok / len(real_val)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(args.batch, bgs, fonts, real_train, rng, args.p_real)
        loss = lossf(model(x.to(device)), y.to(device))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = make_batch(256, bgs, fonts, real_train, rng, args.p_real)
                acc = (model(vx.to(device)).argmax(1).cpu() == vy).float().mean().item()
            model.train()
            print(f"step {step:5d} loss {loss.item():.3f} mix_acc {acc:.2%} real_val_acc {real_val_acc():.2%}")

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path.with_suffix(".pt"))
    model.to("cpu").eval()
    torch.onnx.export(
        model, torch.zeros(1, 3, IMG_H, IMG_W), str(out_path),
        input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False,
    )
    (out_path.parent / "badges_classes.txt").write_text("\n".join(CLASSES), encoding="utf-8")
    print(f"exported {out_path} ({out_path.stat().st_size/1024:.0f} KB) + classes file")


if __name__ == "__main__":
    main()
