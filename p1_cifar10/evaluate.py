"""P1 · 评估：把"准确率 91%"拆开看清楚模型到底错在哪.

用法（项目根目录下）：
    .venv\\Scripts\\python.exe p1_cifar10\\evaluate.py --ckpt runs/baseline/best.pt

产出三样东西（都在 --out 目录，默认跟权重同目录）：
    confusion_matrix.png   10x10 混淆矩阵 —— 看"哪两类被搞混"
    misclassified.png      48 张判错的图 —— 看"错在什么程度上"
    eval_report.json       每类准确率 / 最常见的混淆对

【讲解】在准确率之外，额外输出混淆矩阵（每类被判成了什么）、误判样本墙和每类准确率。

所有讲解段落用【讲解】标记，PyCharm 里 Ctrl+Shift+F 搜 "【讲解】" 可跳读。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

# 【讲解】必须在 import pyplot 之前设置后端。Agg 是只画图不显示的无头后端，用于存 PNG。
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
import torch                                       # noqa: E402
from sklearn.metrics import confusion_matrix       # noqa: E402
from torch.utils.data import DataLoader            # noqa: E402
from torchvision import datasets                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import CIFAR_MEAN, CIFAR_STD, CLASSES, DEFAULT_DATA_ROOT, build_transforms  # noqa: E402
from model import SmallVGG, count_params  # noqa: E402

# 【讲解】中文字体按列表顺序尝试；unicode_minus=False 让负号用 ASCII 的 "-"，
#   否则中文字体里没有 U+2212，负号会显示成方块。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def denormalize(t: torch.Tensor) -> np.ndarray:
    """把归一化后的张量还原成能看的 RGB 图 (H, W, 3)，范围 [0, 1]。

    【讲解】做 Normalize 的逆运算：t * std + mean，clamp 到 [0, 1]，
    再把 (C, H, W) 换成 (H, W, C) 供 imshow 显示。
    """
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)   # 从 (3,) 变 (3,1,1) 才能广播到 (3,H,W)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    # permute(1,2,0)：(C,H,W) → (H,W,C)，因为 matplotlib 要的是"高、宽、通道"顺序
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def collect(model, loader, device, want_wrong: int = 48):
    """跑完整个测试集，收集真实标签、预测标签，以及前 want_wrong 张判错的图。

    【讲解】一次遍历同时收集预测结果和错判样本：用布尔 mask 索引，
    只留下判错的图，不把 1 万张全存进内存。
    """
    model.eval()
    preds, labels = [], []
    wrong_imgs, wrong_trues, wrong_preds = [], [], []

    for x, y in loader:
        logits = model(x.to(device))
        p = logits.argmax(1).cpu()
        preds.append(p)
        labels.append(y)

        # 只在还没收集够的时候才做筛选，避免后面几千个 batch 白干活
        if len(wrong_imgs) < want_wrong:
            mask = p != y
            for img, t, pr in zip(x[mask], y[mask], p[mask]):
                if len(wrong_imgs) >= want_wrong:
                    break
                wrong_imgs.append(denormalize(img))
                wrong_trues.append(int(t))
                wrong_preds.append(int(pr))

    return torch.cat(labels).numpy(), torch.cat(preds).numpy(), wrong_imgs, wrong_trues, wrong_preds


def plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    """画混淆矩阵热力图。

    【讲解】按行归一化成百分比（每行加起来 = 100%）；格子里的字随底色深浅换成白或深色。
    """
    fig, ax = plt.subplots(figsize=(8.2, 7))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    # rotation=45 是因为 10 个类名横着放会重叠（automobile 很长）
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（按行归一化：每行加起来 = 100%）")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] == 0:
                continue      # 0 的格子不写字，否则 100 个数字糊成一片
            color = "white" if cm_norm[i, j] > 0.5 else "#2C2C2A"
            ax.text(j, i, f"{cm_norm[i, j]*100:.0f}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="占该真实类别的比例")
    fig.tight_layout()          # 自动调整边距，否则 45 度的类名会被裁掉
    fig.savefig(out_path, dpi=150)
    plt.close(fig)              # 【讲解】一定要 close，否则循环存大量图会内存泄漏


def plot_wrong(imgs, trues, preds, out_path: Path) -> None:
    """把判错的图拼成 8 列的网格墙，标题写"真实/判成"。"""
    n = len(imgs)
    if n == 0:
        return
    cols = 8
    rows = (n + cols - 1) // cols      # 向上取整，保证够放
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.75))

    # 【讲解】reshape(-1) 把 axes 统一成一维：rows 或 cols 为 1 时 axes 本身就是一维数组，
    #   不统一会让后面 zip 时图配错标题。
    flat = np.array(axes).reshape(-1)
    for ax in flat:
        ax.axis("off")                 # 先把所有格子关掉边框，没图的格子就是空白

    for ax, img, t, p in zip(flat, imgs, trues, preds):
        ax.imshow(img)
        ax.set_title(f"真:{CLASSES[t]}\n判:{CLASSES[p]}", fontsize=7, color="#A32D2D")

    fig.suptitle("判错的样本（前 48 张）—— 看错得离谱还是情有可原", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="P1 评估")
    ap.add_argument("--ckpt", default="runs/baseline/best.pt")
    ap.add_argument("--out", default="", help="输出目录，默认与权重同目录")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--data-root", default="", help="数据集根目录，默认 <项目>/data")
    args = ap.parse_args()

    root = HERE.parent
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"找不到权重 {ckpt_path}\n先跑 train.py 再评估。")

    out_dir = Path(args.out) if args.out else ckpt_path.parent
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 【讲解】checkpoint 里混着 state_dict 和 args（普通 dict），
    #   所以加载自己的文件时用 weights_only=False。
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    # 【讲解】按训练时的结构（类别数、dropout）重建模型，再 load_state_dict。
    model = SmallVGG(num_classes=len(CLASSES), dropout=float(saved.get("dropout", 0.0)))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"载入 {ckpt_path}  (epoch {ckpt.get('epoch')}, "
          f"val acc {ckpt.get('val_acc', float('nan'))*100:.2f}%)")
    print(f"参数量 {count_params(model)/1e6:.3f} M")

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    if not data_root.is_absolute():
        data_root = (root / data_root).resolve()
    if not (data_root / "cifar-10-batches-py").exists():
        raise SystemExit(f"找不到 CIFAR-10：{data_root / 'cifar-10-batches-py'} 不存在，检查 --data-root")
    test_set = datasets.CIFAR10(str(data_root), train=False, download=False,
                                transform=build_transforms("eval"))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_true, y_pred, imgs, trues, preds = collect(model, test_loader, device)

    # 【讲解】这里用 fp32 全精度评估（train.py 里在 bf16 autocast 下算），
    #   两个数字会差零点几个点，属正常。
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    per_class = (cm.diagonal() / cm.sum(axis=1)).tolist()

    plot_confusion(cm, out_dir / "confusion_matrix.png")
    plot_wrong(imgs, trues, preds, out_dir / "misclassified.png")

    # 找出最容易混淆的类别对（排除对角线上的"判对"）
    pairs = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), CLASSES[i], CLASSES[j]))
    pairs.sort(reverse=True)

    report = dict(
        ckpt=str(ckpt_path), test_acc=acc,
        per_class_acc={c: round(v, 4) for c, v in zip(CLASSES, per_class)},
        top_confusions=[{"count": c, "true": t, "pred": p} for c, t, p in pairs[:5]],
    )
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\ntest acc = {acc*100:.2f}%\n")
    print(f"{'类别':<12}{'准确率':>8}")
    print("-" * 22)
    # 按准确率升序排 —— 最差的排最前面，这才是你该关注的部分
    for c, v in sorted(zip(CLASSES, per_class), key=lambda kv: kv[1]):
        print(f"{c:<12}{v*100:>7.2f}%")
    print("\n最容易混淆的 5 对（真实 -> 判成）:")
    for c, t, p in pairs[:5]:
        print(f"  {t} -> {p}: {c} 张")
    print(f"\n图已保存:\n  {out_dir / 'confusion_matrix.png'}\n  {out_dir / 'misclassified.png'}")


if __name__ == "__main__":
    main()
