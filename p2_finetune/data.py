"""P2 · 数据加载：ImageFolder + 手机照片兼容 + 自动切分.

这个文件负责五件事：
  1. 读 root/类名/*.jpg 目录结构（ImageFolder 约定）
  2. 让 ImageFolder 接受 .heic / .heif
  3. 读图时按 EXIF 把方向转正
  4. 按类别分层切出训练集 / 验证集（固定种子，可复现）
  5. 训练 / 评估两套变换（224x224 + ImageNet 归一化）

讲解段落用【讲解】标记，PyCharm 里 Ctrl+Shift+F 搜 "【讲解】" 可跳读。

单独运行这个文件可以体检你的数据：
    .venv\\Scripts\\python.exe p2_finetune\\data.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pillow_heif
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import folder as tv_folder

# ─────────────────────────────────────────────────────────────
# 1. 手机照片兼容
# ─────────────────────────────────────────────────────────────

# 【讲解】注册 HEIC 解码器，再把 .heic/.heif 加进 torchvision 的扩展名白名单。
#   IMG_EXTENSIONS 是 tuple，不能 append，只能整体重绑定；
#   再把新 tuple 同步到 ImageFolder 的类属性上。
pillow_heif.register_heif_opener()
tv_folder.IMG_EXTENSIONS = tuple(tv_folder.IMG_EXTENSIONS) + (".heic", ".heif")
ImageFolder.IMG_EXTENSIONS = tv_folder.IMG_EXTENSIONS   # 类属性同步一份

# 【讲解】ImageNet 的均值 / 标准差，供 ToTensor 之后做 Normalize。
#   用预训练权重时必须是这套数值。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "my_photos"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}


def pil_loader_exif(path: str) -> Image.Image:
    """读图 → 按 EXIF 转正 → 转 RGB。

    【讲解】Image.open() 给出的是传感器原始像素，方向记在 EXIF 的 Orientation
    标签里。exif_transpose() 按标签把像素真的转过去，并清掉该标记；
    load() 在文件句柄关闭前先把像素读进内存。
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()                       # 必须 load，否则文件句柄关掉后读不到像素
        return ImageOps.exif_transpose(img).convert("RGB")


def is_valid_file(path: str) -> bool:
    """过滤 0 字节 / 隐藏文件。手机传过来的文件偶尔会有半截的。"""
    name = os.path.basename(path)
    if name.startswith("."):
        return False
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────
# 2. 变换
# ─────────────────────────────────────────────────────────────

def build_transforms(mode: str = "eval", strong: bool = False):
    """mode: 'train' | 'eval'。

    【讲解】eval：Resize(256) → CenterCrop(224) → ToTensor → Normalize。
    train：RandomResizedCrop(224) → 水平翻转 → 颜色抖动 →（strong 时追加
    RandAugment）→ ToTensor → Normalize → RandomErasing。
    """
    if mode == "eval":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    ops = [
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
    ]
    if strong:
        # 【讲解】strong=True 时在颜色抖动之后追加 RandAugment，得到更强的增强。
        ops.append(transforms.RandAugment(num_ops=2, magnitude=7))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2),   # 必须在 Normalize 之后
    ]
    return transforms.Compose(ops)


# ─────────────────────────────────────────────────────────────
# 3. 数据集与切分
# ─────────────────────────────────────────────────────────────

def list_classes(root: str | Path) -> list[str]:
    """列出类别目录（按名字排序，和 ImageFolder 的 class_to_idx 一致）。"""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def build_datasets(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    val_ratio: float = 0.2,
    seed: int = 0,
    strong_aug: bool = False,
):
    """返回 (train_ds, val_ds, classes)。

    【讲解】建两份 ImageFolder，两者文件顺序完全一致：full_aug 带训练增强，
    full_plain 不带。下面用同一组下标，分别从两份里取 train / val。
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise SystemExit(
            f"\n[找不到照片目录] {data_root} 不存在\n\n"
            "请把照片按下面的结构放好：\n"
            f"  {data_root}\\类名\\xxx.jpg\n"
            f"  {data_root}\\类名\\yyy.png\n"
            "每个类别一个文件夹，文件夹名就是类别名。"
        )

    classes = list_classes(data_root)
    if not classes:
        raise SystemExit(
            f"\n[目录里没有类别文件夹] {data_root}\n\n"
            "现在这个目录下只有文件、没有子文件夹。需要这样建：\n"
            f"  {data_root}\\cup\\1.jpg\n"
            f"  {data_root}\\cup\\2.jpg\n"
            f"  {data_root}\\book\\1.jpg\n"
            "文件夹名 = 类别名，图片直接放在里面。"
        )

    common = dict(loader=pil_loader_exif, is_valid_file=is_valid_file)
    try:
        full_aug = ImageFolder(str(data_root), transform=build_transforms("train", strong_aug), **common)
        full_plain = ImageFolder(str(data_root), transform=build_transforms("eval"), **common)
    except FileNotFoundError as e:
        # 常见于"某个类别文件夹是空的"
        raise SystemExit(
            f"\n[数据不完整] {e}\n\n"
            "最常见的原因：某个类别文件夹里没有图片。\n"
            "ImageFolder 要求**每个类别文件夹都至少有一张图**，否则直接报错。\n"
            "解决：要么把空文件夹删掉，要么往里放图。"
        ) from e

    n = len(full_aug)
    if n < 10:
        raise SystemExit(f"\n[数据太少] 一共只有 {n} 张图。每类至少 30 张、总计至少 100 张再开始。")

    # 【讲解】按类别分层切分：先按 target 把样本下标分组，每组内部用固定种子的
    #   randperm 打乱，取前 n_take 个进验证集。每类各自切，所以 val 里每个类别
    #   都有样本，且每次运行切出同一批图。
    by_class: dict[int, list[int]] = {}
    for idx, target in enumerate(full_aug.targets):
        by_class.setdefault(target, []).append(idx)

    # 固定种子的 randperm：每次运行切出**同一批**验证集。
    # 否则两次实验 val 集不同，指标不可比，调参全靠撞运气。
    g = torch.Generator().manual_seed(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    split_info: list[tuple[str, int, int]] = []
    for target, idxs in sorted(by_class.items()):
        order = torch.randperm(len(idxs), generator=g).tolist()
        shuffled = [idxs[k] for k in order]
        if len(shuffled) <= 1:
            n_take = 0                       # 只有 1 张就别切了，全给训练集
        else:
            n_take = max(1, round(len(shuffled) * val_ratio))
            n_take = min(n_take, len(shuffled) - 1)   # 至少留 1 张给训练
        val_idx += shuffled[:n_take]
        train_idx += shuffled[n_take:]
        split_info.append((full_aug.classes[target], len(shuffled) - n_take, n_take))

    if not val_idx:
        raise SystemExit("\n[切不出验证集] 每个类别都只有 1 张图。请每类至少准备 10 张。")

    train_ds = Subset(full_aug, train_idx)
    val_ds = Subset(full_plain, val_idx)

    # 检查验证集是否漏掉了某个类别（小数据集常见）
    val_labels = {full_plain.samples[i][1] for i in val_idx}
    missing = [c for i, c in enumerate(full_plain.classes) if i not in val_labels]
    if missing:
        print(f"[警告] 验证集里没有这些类别: {missing} —— 它们的指标无法评估。"
              f"建议每个类至少 10 张，或调大 --val-ratio")

    print("[切分] " + "   ".join(f"{c}: {tr}/{va}" for c, tr, va in split_info)
          + "   (训练/验证)")
    return train_ds, val_ds, full_aug.classes


def build_loaders(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    batch_size: int = 32,
    workers: int = 4,
    val_ratio: float = 0.2,
    seed: int = 0,
    strong_aug: bool = False,
):
    train_ds, val_ds, classes = build_datasets(data_root, val_ratio, seed, strong_aug)

    # 【讲解】train：shuffle + drop_last（不足一个 batch 的尾巴丢掉，避免 BN 遇到
    #   batch=1）；val：不 shuffle、不丢尾，保证每张都算进去。
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch_size), shuffle=False,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader, classes


if __name__ == "__main__":
    # 体检：类别、数量、单张形状。数据有问题会在这里就暴露，不用等到训练中途
    tr, va, classes = build_loaders(workers=0)
    print(f"类别 ({len(classes)}): {classes}")
    print(f"训练集 {len(tr.dataset)} 张 / 验证集 {len(va.dataset)} 张")

    xb, yb = next(iter(tr))
    print(f"一个 batch: x={tuple(xb.shape)} {xb.dtype}  y={tuple(yb.shape)}")
    print(f"归一化后 mean={xb.mean():.3f} std={xb.std():.3f}  (应接近 0 和 1)")
    print(f"标签范围: {int(yb.min())} ~ {int(yb.max())}  (应为 0 ~ {len(classes)-1})")
