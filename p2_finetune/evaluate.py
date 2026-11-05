"""P2 · 评估：混淆矩阵 + 误判样本墙 + 每类精确率/召回率.

用法：
    .venv\\Scripts\\python.exe p2_finetune\\evaluate.py
    .venv\\Scripts\\python.exe p2_finetune\\evaluate.py --ckpt runs_p2/scratch/best.pt --out runs_p2/scratch

产出（默认写到 ckpt 同目录）：
    confusion_matrix.png   混淆矩阵（按行归一化）
    misclassified.png      判错的样本 + 模型当时给的置信度
    per_class.png          每类准确率柱状图
    eval_report.json       完整指标（含 sklearn 的 precision/recall/f1）

【讲解】类别数和类名从 checkpoint 与数据集读取，图幅、标签都按 len(classes) 动态生成。

讲解段落用【讲解】标记。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
import torch                                        # noqa: E402
import torch.nn as nn                               # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader             # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import DEFAULT_DATA_ROOT, IMAGENET_MEAN, IMAGENET_STD, build_datasets  # noqa: E402
from model import build_model                        # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def denormalize(t: torch.Tensor) -> np.ndarray:
    """把 ImageNet 归一化后的张量还原成可显示的 RGB。

    【讲解】做 Normalize 的逆运算：t * std + mean，clamp 到 [0, 1]，
    再把通道维从 (C, H, W) 换到 (H, W, C) 供 imshow 显示。
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def collect(model, loader, device, want_wrong: int = 24):
    model.eval()
    preds, labels, probs = [], [], []
    wrong = []      # (img, true_idx, pred_idx, confidence)

    for x, y in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits.float(), dim=1).cpu()      # 概率，用于展示置信度
        pred = p.argmax(1)
        preds.append(pred)
        labels.append(y)
        probs.append(p)

        # 【讲解】只收集前 want_wrong 张错判样本，
        #   每条记录 (原图, 真实下标, 预测下标, 该预测的置信度)。
        if len(wrong) < want_wrong:
            mask = pred != y
            for img, ti, pi, conf in zip(x[mask], y[mask], pred[mask], p[mask].max(1).values):
                if len(wrong) >= want_wrong:
                    break
                wrong.append((denormalize(img), int(ti), int(pi), float(conf)))

    return torch.cat(labels).numpy(), torch.cat(preds).numpy(), torch.cat(probs).numpy(), wrong


def plot_confusion(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    n = len(classes)
    # 【讲解】图幅按类别数取 6~14 英寸，避免类别多时标签重叠。
    size = max(6.0, min(14.0, n * 1.1 + 2.5))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(n), classes, rotation=45, ha="right")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（按行归一化：每行加起来 = 100%）")

    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            color = "white" if cm_norm[i, j] > 0.5 else "#2C2C2A"
            # 同时标百分比和原始张数：百分比看规律，张数看样本是否够
            ax.text(j, i, f"{cm_norm[i, j]*100:.0f}%\n({cm[i, j]})", ha="center", va="center",
                    fontsize=max(6, 9 - n // 6), color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="占该真实类别的比例")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wrong(wrong, classes: list[str], out_path: Path) -> None:
    if not wrong:
        return
    n = len(wrong)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.3))
    flat = np.array(axes).reshape(-1)
    for ax in flat:
        ax.axis("off")
    for ax, (img, ti, pi, conf) in zip(flat, wrong):
        ax.imshow(img)
        # 【讲解】标题写"真实类 / 判成什么 + 置信度"，置信度是 softmax 概率。
        ax.set_title(f"真:{classes[ti]}\n判:{classes[pi]} {conf:.2f}", fontsize=8, color="#A32D2D")
    fig.suptitle(f"判错的样本（前 {n} 张）—— 看错得离谱还是情有可原", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class(per_class_acc: dict, out_path: Path) -> None:
    items = sorted(per_class_acc.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v * 100 for _, v in items]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9 + 2), 4))
    bars = ax.bar(names, vals, color="#1D9E75")
    # 【讲解】最差类比最好类低 5 个点以上时，把最低的那根柱标成橙色。
    if vals and min(vals) < max(vals) - 5:
        bars[0].set_color("#EF9F27")
        title = "每类准确率（按从低到高排序，橙色 = 最需要补数据的那类）"
    else:
        title = "每类准确率（各类表现接近，没有明显掉队的类别）"
    ax.set_ylim(0, 105)
    ax.set_ylabel("准确率 (%)")
    ax.set_title(title)
    ax.axhline(100, color="#B4B2A9", linewidth=0.5, linestyle="--")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 评估")
    ap.add_argument("--ckpt", default="runs_p2/baseline/best.pt")
    ap.add_argument("--out", default="", help="输出目录，默认与权重同目录")
    ap.add_argument("--data-root", default="", help="照片目录，默认取训练时记录的路径")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    root = HERE.parent
    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else root / args.ckpt
    if not ckpt_path.exists():
        raise SystemExit(f"找不到权重 {ckpt_path}\n先跑 train.py 再评估。")

    out_dir = Path(args.out) if args.out else ckpt_path.parent
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 【讲解】从 checkpoint 里取四项：classes（输出下标 → 类名）、arch（重建网络结构）、
    #   val_ratio / seed（复现训练时切出的同一批验证集）。
    classes = ckpt["classes"]
    arch = ckpt.get("arch", "resnet18")
    saved = ckpt.get("args", {})
    val_ratio = float(saved.get("val_ratio", 0.2))
    seed = int(saved.get("seed", 0))

    print(f"载入 {ckpt_path}")
    print(f"  结构 {arch} | 类别 {classes}")
    print(f"  最佳 epoch {ckpt.get('epoch')} | val acc {ckpt.get('val_acc', float('nan'))*100:.2f}%")

    model, _ = build_model(arch, num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    data_root = Path(args.data_root) if args.data_root else (
        Path(saved["data_root"]) if saved.get("data_root") else DEFAULT_DATA_ROOT
    )
    if not data_root.is_absolute():
        data_root = (root / data_root).resolve()

    full_aug, val_ds, classes_now = build_datasets(data_root, val_ratio, seed, False)
    if classes_now != classes:
        raise SystemExit(f"类别对不上：权重里是 {classes}，目录里是 {classes_now}。\n"
                         f"是不是照片文件夹改过了？")
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_true, y_pred, y_prob, wrong = collect(model, loader, device)
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    per_class = (cm.diagonal() / np.maximum(cm.sum(axis=1), 1)).tolist()

    plot_confusion(cm, classes, out_dir / "confusion_matrix.png")
    plot_wrong(wrong, classes, out_dir / "misclassified.png")
    plot_per_class(dict(zip(classes, per_class)), out_dir / "per_class.png")

    # 【讲解】用 sklearn 的 classification_report 输出每类的 precision / recall / f1。
    report_txt = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                       target_names=classes, digits=3, zero_division=0)

    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), classes[i], classes[j]))
    pairs.sort(reverse=True)

    report = dict(
        ckpt=str(ckpt_path), arch=arch, classes=classes,
        n_val=int(len(y_true)), val_acc=acc,
        per_class_acc={c: round(v, 4) for c, v in zip(classes, per_class)},
        top_confusions=[{"count": c, "true": t, "pred": p} for c, t, p in pairs[:8]],
        classification_report=report_txt,
    )
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n验证集 {len(y_true)} 张，准确率 = {acc*100:.2f}%\n")
    print(report_txt)
    if pairs:
        print("最容易混淆的类别对（真实 -> 判成）:")
        for c, t, p in pairs[:5]:
            print(f"  {t} -> {p}: {c} 张")
    else:
        print("没有任何错判 —— 验证集全对。")
        print("  注意：这通常说明任务太简单（合成数据）或验证集太小，")
        print("  不代表真实场景也这么准。")

    saved = [out_dir / "confusion_matrix.png", out_dir / "per_class.png"]
    if wrong:
        saved.insert(1, out_dir / "misclassified.png")
    else:
        print("  （全对，没有错题，所以没有生成 misclassified.png）")
    print("图已保存:")
    for s in saved:
        print(f"  {s}")


if __name__ == "__main__":
    main()
