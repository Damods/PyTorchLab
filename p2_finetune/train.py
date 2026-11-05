"""P2 · 训练：冻结 warmup → 解冻微调（两段式迁移学习）.

用法（项目根目录下）：

    :: 先用 3 个 batch 验链路，20 秒出结果
    .venv\\Scripts\\python.exe p2_finetune\\train.py --epochs 1 --limit-batches 3

    :: 正式训练（默认 15 epoch：前 5 轮冻结 backbone，后 10 轮解冻微调）
    .venv\\Scripts\\python.exe p2_finetune\\train.py

    :: 对照实验：从零训练（不加载 ImageNet 权重），看预训练值多少
    .venv\\Scripts\\python.exe p2_finetune\\train.py --scratch --out runs_p2/scratch

    :: 换更大的 backbone
    .venv\\Scripts\\python.exe p2_finetune\\train.py --arch resnet50 --batch-size 16

看曲线：
    .venv\\Scripts\\tensorboard.exe --logdir runs_p2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import DEFAULT_DATA_ROOT, build_loaders  # noqa: E402
from model import build_model, set_backbone_frozen, set_bn_eval, split_params  # noqa: E402

PROJECT_ROOT = HERE.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2: 迁移学习·自己的照片分类")

    # ---- 数据 ----
    p.add_argument("--data-root", default="", help="照片根目录（类名文件夹所在层），默认 data/my_photos")
    p.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    p.add_argument("--strong-aug", action="store_true", help="开 RandAugment（更强的增强）")

    # ---- 模型 ----
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--scratch", action="store_true",
                   help="不加载预训练权重，从零训练（对照实验用）")

    # ---- 训练预算 ----
    # 【讲解】默认 15 轮：数据集小时一个 epoch 只有几十个 batch，
    #   足够跑到收敛或过拟合。数据多了可以调大。
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32,
                   help="显存很宽裕（实测 resnet18 bs32 只占 0.57 GB），别设太小")
    p.add_argument("--freeze-epochs", type=int, default=5,
                   help="前 N 轮冻结 backbone 只训分类头；0 = 全程解冻直接微调")

    # ---- 优化器：两组学习率 ----
    # 【讲解】两组学习率：--lr 是 head 的峰值，
    #   backbone 的实际 lr = --lr × --backbone-lr-scale。
    p.add_argument("--lr", type=float, default=1e-3, help="分类头（head）的峰值学习率")
    p.add_argument("--backbone-lr-scale", type=float, default=0.1,
                   help="backbone 的 lr = head_lr × 这个系数，默认 1/10")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--label-smoothing", type=float, default=0.1)

    # ---- 工程 ----
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--limit-batches", type=int, default=0, help="只跑前 N 个 batch（调试）")
    p.add_argument("--print-every", type=int, default=1, help="每 N 轮打印一行")
    p.add_argument("--out", default="runs_p2/baseline")
    return p.parse_args()


def lr_scale(epoch: int, args: argparse.Namespace) -> float:
    """返回相对峰值学习率的倍数（0~1）：warmup 线性升温 + 余弦退火。

    【讲解】只返回倍数，head 和 backbone 分别乘自己的峰值 lr，
    一条曲线同时控制两组参数，比例保持不变。
    """
    if epoch < args.warmup:
        return (epoch + 1) / max(1, args.warmup)
    progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_one_epoch(model, loader, optimizer, criterion, device, amp,
                    limit_batches=0, frozen_backbone=False):
    model.train()
    # 【讲解】冻结阶段在 model.train() 之后、前向之前把 BN 切回 eval，
    #   让 backbone 的 BN 用 running stats，而不是当前 batch 的统计量。
    if frozen_backbone:
        set_bn_eval(model)

    loss_sum, correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp, limit_batches=0):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()

    # ---- 数据 ----
    train_loader, val_loader, classes = build_loaders(
        data_root=data_root, batch_size=args.batch_size, workers=args.workers,
        val_ratio=args.val_ratio, seed=args.seed, strong_aug=args.strong_aug,
    )
    n_cls = len(classes)

    # 【讲解】输出目录放在数据加载成功之后创建：数据有问题时提前退出，
    #   不会在 runs_p2/ 下留下空目录。
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 【讲解】打印本次运行的数据路径、类别、样本数、模型和超参。
    print(f"数据       : {data_root}")
    print(f"类别 ({n_cls})  : {classes}")
    print(f"样本       : train {len(train_loader.dataset)} / val {len(val_loader.dataset)}")
    print(f"模型       : {args.arch}  {'从零训练' if args.scratch else 'ImageNet 预训练'}")
    print(f"超参       : epochs={args.epochs} bs={args.batch_size} "
          f"head_lr={args.lr} backbone_lr={args.lr*args.backbone_lr_scale:.1e} "
          f"freeze={args.freeze_epochs} amp={args.amp}")

    # ---- 模型 ----
    model, head = build_model(args.arch, num_classes=n_cls, pretrained=not args.scratch)
    model = model.to(device)

    # 【讲解】用 param_groups 给两组参数分别设 lr 和 weight_decay：
    #   第 0 组 backbone（lr × scale），第 1 组 head（lr，wd = 0）。
    backbone_params, head_params = split_params(model, head)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale, "weight_decay": args.wd},
        {"params": head_params, "lr": args.lr, "weight_decay": 0.0},
    ])

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_path = out_dir / "best.pt"
    best_acc, best_epoch = 0.0, 0
    history = []

    print(f"\n{'epoch':>6} {'阶段':>6} {'lr(head)':>9} {'train_loss':>11} {'train_acc':>10} "
          f"{'val_loss':>9} {'val_acc':>8} {'time':>7}")
    print("-" * 78)

    for epoch in range(args.epochs):
        # ---- 1. 决定本轮的冻结状态 ----
        # 【讲解】epoch 从 0 开始，epoch < freeze_epochs 的轮次冻结 backbone；
        #   之后 requires_grad 恢复 True，进入全量微调。
        frozen = epoch < args.freeze_epochs
        set_backbone_frozen(model, head, frozen)
        phase = "冻结" if frozen else "微调"

        # ---- 2. 设置本轮学习率（两组一起缩放）----
        scale = lr_scale(epoch, args)
        optimizer.param_groups[0]["lr"] = args.lr * args.backbone_lr_scale * scale  # backbone
        optimizer.param_groups[1]["lr"] = args.lr * scale                          # head

        # ---- 3. 训练一轮 ----
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.amp,
            args.limit_batches, frozen_backbone=frozen,
        )

        # ---- 4. 验证 ----
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, args.amp, args.limit_batches)
        dt = time.time() - t0

        if (epoch + 1) % max(1, args.print_every) == 0 or epoch == args.epochs - 1:
            print(f"{epoch+1:>6} {phase:>6} {optimizer.param_groups[1]['lr']:>9.1e} "
                  f"{tr_loss:>11.4f} {tr_acc*100:>9.2f}% {va_loss:>9.4f} {va_acc*100:>7.2f}% {dt:>6.1f}s")

        writer.add_scalar("loss/train", tr_loss, epoch + 1)
        writer.add_scalar("loss/val", va_loss, epoch + 1)
        writer.add_scalar("acc/train", tr_acc, epoch + 1)
        writer.add_scalar("acc/val", va_acc, epoch + 1)
        writer.add_scalar("lr/head", optimizer.param_groups[1]["lr"], epoch + 1)
        writer.add_scalar("lr/backbone", optimizer.param_groups[0]["lr"], epoch + 1)
        history.append(dict(epoch=epoch + 1, phase=phase, frozen=frozen,
                            lr_head=optimizer.param_groups[1]["lr"],
                            lr_backbone=optimizer.param_groups[0]["lr"],
                            train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, sec=dt))

        # ---- 5. 存最佳权重 ----
        # 【讲解】val acc 刷新时保存 best.pt，内容是 state_dict + arch + classes
        #   + epoch + val_acc + 本次运行的参数。
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch + 1
            torch.save({
                "model": model.state_dict(),
                "arch": args.arch,
                "classes": classes,
                "epoch": epoch + 1,
                "val_acc": va_acc,
                "args": vars(args),
            }, best_path)

    # ---- 收尾 ----
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    final_val = history[-1]["val_acc"]

    report = dict(
        args=vars(args), classes=classes, best_val_acc=best_acc, best_epoch=best_epoch,
        final_val_acc=final_val, peak_vram_gb=peak, history=history,
    )
    (out_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-" * 78)
    print(f"最佳 val acc : {best_acc*100:.2f}%  (epoch {best_epoch})")
    print(f"末轮 val acc : {final_val*100:.2f}%")
    print(f"显存峰值     : {peak:.2f} GB")
    print(f"权重         : {best_path}")
    print()
    # 【讲解】没有独立 test 集：val 既用来挑最佳轮次，也作为最终成绩输出。
    print("注意：P2 数据量小，没有独立 test 集 —— 上面的 val acc 就是最终成绩。")
    print("      所以不要反复拿 val 调参（调十次再报最好那次 = 数字虚高）。")
    print(f"\n看混淆矩阵：.venv\\Scripts\\python.exe p2_finetune\\evaluate.py --ckpt {args.out}/best.pt")

    writer.add_hparams(
        {k: str(v) for k, v in vars(args).items()},
        {"hparam/best_val_acc": best_acc, "hparam/final_val_acc": final_val},
    )
    writer.close()


if __name__ == "__main__":
    main()
