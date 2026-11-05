"""P2 · 生成合成数据集，用于在没照片时验证整条流程.

用法：
    .venv\\Scripts\\python.exe p2_finetune\\make_demo_data.py
    .venv\\Scripts\\python.exe p2_finetune\\make_demo_data.py --per-class 60 --out data/_demo

产出 data/_demo/<类别>/*.jpg（含少量 .heic，用来验证 HEIC 链路）。

【讲解】每个类画一种形状（圆 / 方 / 三角 / 十字 / 斜纹），背景、位置、大小、旋转
随机；每类前 3 张存成 .heic，覆盖 HEIC 读取路径。形状可分，正常应接近 100%。

注意：这些图只是验证工具，真实照片放 data/my_photos/，两者别混。

讲解段落用【讲解】标记。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

pillow_heif.register_heif_opener()

SHAPES = ("circle", "square", "triangle", "cross", "stripe")


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, box, color) -> None:
    x0, y0, x1, y1 = box
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    elif shape == "triangle":
        cx = (x0 + x1) / 2
        draw.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=color)
    elif shape == "cross":
        w = (x1 - x0) // 3
        h = (y1 - y0) // 3
        draw.rectangle((x0 + w, y0, x1 - w, y1), fill=color)
        draw.rectangle((x0, y0 + h, x1, y1 - h), fill=color)
    elif shape == "stripe":
        # 【讲解】斜纹 = n 个平行四边形依次排开。
        n = 4
        step = (x1 - x0) // n
        for i in range(n):
            sx = x0 + i * step
            draw.polygon([(sx, y1), (sx + step // 2, y0),
                          (sx + step, y1), (sx + step // 2, y1)], fill=color)


def make_image(shape: str, rng: np.random.Generator, size: int = 256) -> Image.Image:
    """一张随机背景 + 随机位置/大小/旋转的图形。"""
    bg = tuple(int(v) for v in rng.integers(60, 220, 3))
    fg = tuple(int(v) for v in rng.integers(0, 256, 3))
    # 保证前景和背景有足够对比度，否则图会"隐形"
    while abs(sum(fg) - sum(bg)) < 180:
        fg = tuple(int(v) for v in rng.integers(0, 256, 3))

    img = Image.new("RGB", (size, size), bg)

    # 在透明图层上画图形，再旋转贴回去 —— 旋转整个图会连背景一起转，不自然
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    side = int(rng.integers(size // 3, size * 3 // 4))
    cx, cy = rng.integers(side // 2, size - side // 2, 2)
    box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
    draw_shape(d, shape, box, fg + (255,))
    layer = layer.rotate(float(rng.integers(0, 360)), resample=Image.BICUBIC)

    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

    # 加噪点：模拟真实照片的传感器噪声
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-12, 13, arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 P2 合成数据集")
    ap.add_argument("--out", default="data/_demo")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = (HERE.parent / out).resolve()

    rng = np.random.default_rng(args.seed)
    print(f"生成到 {out}")

    for ci, shape in enumerate(SHAPES):
        d = out / shape
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            img = make_image(shape, rng)
            # 【讲解】每类前 3 张存成 .heic，覆盖 HEIC 读取路径。
            if i < 3:
                img.save(d / f"{i:03d}.heic", format="HEIF")
            else:
                img.save(d / f"{i:03d}.jpg", quality=92)
        print(f"  {shape:<9} {args.per_class} 张（其中 3 张 heic）")

    total = args.per_class * len(SHAPES)
    print(f"\n完成：{len(SHAPES)} 类，共 {total} 张 → {out}")
    print("\n下一步：")
    print("  .venv\\Scripts\\python.exe p2_finetune\\check_data.py --data-root data/_demo")
    print("  .venv\\Scripts\\python.exe p2_finetune\\train.py --data-root data/_demo --out runs_p2/demo")


if __name__ == "__main__":
    main()
