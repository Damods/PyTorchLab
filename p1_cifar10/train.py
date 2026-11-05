"""P1 · 训练循环（全部手写，不用任何训练框架）.

训练骨架就是 train_one_epoch() 里的五行：

    梯度清零 → 前向 → 算损失 → 反向 → 更新参数

其余部分是数据、日志、存盘和调度。所有讲解段落用【讲解】标记，
PyCharm 里 Ctrl+Shift+F 搜 "【讲解】" 可跳读。

用法（在项目根目录 D:\\project\\workplace pycharm\\PyTorchLab 下执行）：

    :: 先验证链路通不通，20 秒内出结果
    .venv\\Scripts\\python.exe p1_cifar10\\train.py --epochs 1 --limit-batches 20

    :: 完整训练（30 epoch，实测 3 分钟，目标 test acc >= 90%）
    .venv\\Scripts\\python.exe p1_cifar10\\train.py

    :: 三个对照实验（README 里有实测结论）
    .venv\\Scripts\\python.exe p1_cifar10\\train.py --aug none   --out runs/no_aug
    .venv\\Scripts\\python.exe p1_cifar10\\train.py --lr 0.5     --out runs/lr_high
    .venv\\Scripts\\python.exe p1_cifar10\\train.py --aug none --dropout 0.5 --out runs/dropout

    :: 每 5 个 epoch 才打印一行（控制台清爽些；曲线仍然逐轮完整记录）
    .venv\\Scripts\\python.exe p1_cifar10\\train.py --print-every 5

看曲线：
    .venv\\Scripts\\tensorboard.exe --logdir runs
  然后浏览器打开 http://localhost:6006
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
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

# 让 `python p1_cifar10/train.py` 这种直接脚本运行也能 import 同目录的模块
# （不这么做的话，只有把 p1_cifar10 当包、用 -m 运行才找得到 data/model）
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import CLASSES, DEFAULT_DATA_ROOT, build_loaders  # noqa: E402
from model import SmallVGG, count_params  # noqa: E402

PROJECT_ROOT = HERE.parent


def parse_args() -> argparse.Namespace:
    """所有可调参数都放这里，方便逐个改做对照实验。

    【讲解】每个超参一个独立开关；结果写进 runs/<名字>/train_report.json，可直接横向对比。
    """
    p = argparse.ArgumentParser(description="P1: 从零训练 CIFAR-10 分类器")

    # ---- 训练预算 ----
    p.add_argument("--epochs", type=int, default=30,
                   help="30 epoch 实测 3 分钟、test 91.5%%，但曲线末轮还在涨，没收敛")
    p.add_argument("--batch-size", type=int, default=128,
                   help="越大梯度越稳、越省时间；显存不够就减半（这模型只占 0.33 GB）")

    # ---- 优化器 ----
    # 【讲解】这是 SGD 的峰值学习率；每一轮实际用多少由 lr_at() 算。
    p.add_argument("--lr", type=float, default=0.1, help="SGD 峰值学习率（warmup 后）")
    p.add_argument("--wd", type=float, default=5e-4, help="weight decay（L2 正则强度）")
    p.add_argument("--warmup", type=int, default=3, help="前几个 epoch 线性升温")

    # ---- 正则化 / 数据 ----
    p.add_argument("--aug", choices=["none", "basic", "strong"], default="basic")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="分类头前的 dropout 概率，默认 0 表示不丢弃")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="把 one-hot 标签软化，抑制过度自信，通常小幅稳定提升")

    # ---- 工程 ----
    p.add_argument("--workers", type=int, default=6, help="DataLoader 子进程数，0 会慢十几倍")
    p.add_argument("--data-root", default="", help="数据集根目录，默认 <项目>/data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    # 【讲解】--amp 默认开，另提供 --no-amp 关闭：store_true 默认 False，
    #   这里把 default 设为 True，再用 store_false 提供关闭开关。
    p.add_argument("--amp", dest="amp", action="store_true", default=True,
                   help="bf16 混合精度（默认开）：权重/激活用 16 位算，显存和速度都收益")
    p.add_argument("--no-amp", dest="amp", action="store_false")

    # ---- 调试 / 输出 ----
    p.add_argument("--limit-batches", type=int, default=0,
                   help="只跑前 N 个 batch（调试用）。改完代码先跑 5 个 batch 验链路")
    p.add_argument("--print-every", type=int, default=1,
                   help="每完成 N 个 epoch 打印一行（默认 1 = 每轮都打印）。"
                        "只影响控制台输出，TensorBoard 和 train_report.json 始终逐轮记录")
    p.add_argument("--out", default="runs/baseline", help="输出目录（权重 / 日志）")
    return p.parse_args()


def lr_at(epoch: int, args: argparse.Namespace) -> float:
    """warmup 线性升温 + 余弦退火到 0。

    【讲解】epoch < warmup 时返回 args.lr * (epoch + 1) / warmup；
    之后按余弦曲线从 args.lr 衰减到 0。
    """
    if epoch < args.warmup:
        return args.lr * (epoch + 1) / args.warmup
    progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_one_epoch(model, loader, optimizer, criterion, device, amp, limit_batches=0):
    """跑一个 epoch 的训练，返回 (平均损失, 准确率)。

    【讲解】每个 batch 依次执行：model.train() 已在循环外调用；
    zero_grad(set_to_none=True) → autocast 下前向并算损失 → backward → step，
    然后累加 loss 和正确预测数。
    """
    model.train()

    # 【讲解】loss 按 batch 内样本数加权累加，最后除以总样本数，得到按样本加权的平均损失。
    loss_sum, correct, total = 0.0, 0, 0

    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break

        # non_blocking=True：配合 pin_memory，让 H2D 拷贝和 CPU 计算重叠（异步传输）
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 【讲解】autocast 只包住前向和损失计算，dtype 用 bfloat16
        #   （bf16 的指数位与 fp32 相同，不需要 GradScaler）。
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        # 【讲解】backward() 从标量 loss 反向求导，把每个参数的梯度写进 param.grad。
        loss.backward()

        # 【讲解】step() 用梯度更新参数。SGD 的更新式：p = p - lr * (grad + wd * p)。
        optimizer.step()

        # ---- 统计（不参与计算图，只是记账）----
        loss_sum += loss.item() * y.size(0)
        # 【讲解】argmax(1) 取打分最大的位置作为预测类别（不需要 softmax），累加正确数。
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()  # 【讲解】整个函数不建计算图：省显存、也更快
def evaluate(model, loader, criterion, device, amp, limit_batches=0):
    """评估：返回 (平均损失, 准确率)。

    【讲解】@torch.no_grad() 关掉计算图；model.eval() 让 BN 用累积的滑动平均、
    Dropout 变成恒等映射。两者缺一，指标就不可靠。
    """
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

    # 【讲解】没有可用 GPU 时退回 CPU，代码仍能跑（只是慢）。
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 【讲解】设置随机种子，让同样的配置尽量得到同样的结果
    #   （cuDNN 少数卷积算法非确定性，只能接近、不能完全一致）。
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        # 【讲解】benchmark=True 让 cuDNN 为固定输入尺寸挑最快的卷积算法，
        #   第一次迭代会多花几秒做 autotune。
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()   # 清零显存峰值计数，否则统计会被前序操作污染

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device = {device}  amp = {args.amp}  aug = {args.aug}")

    # 【讲解】--data-root 的相对路径按项目根解析，从任何目录执行都指向同一位置。
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()
    print(f"数据目录 = {data_root}")

    train_loader, val_loader, test_loader = build_loaders(
        data_root=data_root, batch_size=args.batch_size, workers=args.workers,
        aug=args.aug, seed=args.seed,
    )
    print(f"train {len(train_loader.dataset)} 张 / val {len(val_loader.dataset)} 张 / "
          f"test {len(test_loader.dataset)} 张")

    model = SmallVGG(num_classes=len(CLASSES), dropout=args.dropout).to(device)
    print(f"参数量 {count_params(model)/1e6:.3f} M")

    # 【讲解】label_smoothing=0.1 把 one-hot 标签软化，避免模型把概率压到 0/1。
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 【讲解】SGD + momentum=0.9 + nesterov，weight_decay=args.wd 加到参数更新上。
    optimizer = SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True
    )

    # TensorBoard 记录器。log_dir 下每个 run 一个子目录，多个实验可以叠在一张图上对比
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_path = out_dir / "best.pt"
    # Save the first valid checkpoint even when validation accuracy is zero.
    best_acc = -1.0
    history = []      # 逐 epoch 的记录，最后写进 train_report.json，方便脚本化对比

    print(f"\n{'epoch':>6} {'lr':>7} {'train_loss':>11} {'train_acc':>10} "
          f"{'val_loss':>9} {'val_acc':>8} {'time':>7}")
    print("-" * 66)

    for epoch in range(args.epochs):
        # ---- 1. 设置本轮学习率 ----
        lr = lr_at(epoch, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ---- 2. 训练一轮 ----
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.amp, args.limit_batches
        )

        # ---- 3. 在 val 上验证 ----
        # 【讲解】每个 epoch 都在 val 上评估一次，便于和 train 曲线对比。
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, args.amp, args.limit_batches)
        dt = time.time() - t0

        # 【讲解】--print-every 只控制打印频率，TensorBoard 和 train_report.json 仍然逐轮记录；
        #   最后一轮一定打印。
        if (epoch + 1) % max(1, args.print_every) == 0 or epoch == args.epochs - 1:
            print(f"{epoch+1:>6} {lr:>7.4f} {tr_loss:>11.4f} {tr_acc*100:>9.2f}% "
                  f"{va_loss:>9.4f} {va_acc*100:>7.2f}% {dt:>6.1f}s")

        # ---- 4. 记录到 TensorBoard ----
        # 【讲解】train / val 的 loss 和 acc 都写入 TensorBoard，另记 lr 和本轮耗时。
        writer.add_scalar("loss/train", tr_loss, epoch + 1)
        writer.add_scalar("loss/val", va_loss, epoch + 1)
        writer.add_scalar("acc/train", tr_acc, epoch + 1)
        writer.add_scalar("acc/val", va_acc, epoch + 1)
        writer.add_scalar("lr", lr, epoch + 1)
        history.append(dict(epoch=epoch + 1, lr=lr, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, sec=dt))

        # ---- 5. 只在 val 变好时存盘 ----
        # 【讲解】val acc 刷新时保存 best.pt，内容是 state_dict、epoch、val_acc 和本次参数。
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_acc": va_acc, "args": vars(args)}, best_path)

    # ---- 训练结束：用 val 上挑出的最佳权重，在 test 上只评估一次 ----
    # 【讲解】加载 best.pt，在 test 集上评估一次，得到最终 test acc。
    te_acc = float("nan")
    ckpt = {"epoch": "?"}
    if best_path.exists():
        # weights_only=False：checkpoint 里存了 args（普通 dict，不是张量），
        # 严格模式下会被拒。只有在加载自己生成的文件时才这么放开。
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        _, te_acc = evaluate(model, test_loader, criterion, device, args.amp, args.limit_batches)

    # 【讲解】读 CUDA 显存峰值（GB），用于判断还能不能加大 batch。
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0

    report = dict(args=vars(args), best_val_acc=best_acc, test_acc=te_acc,
                  peak_vram_gb=peak, history=history)
    (out_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-" * 66)
    print(f"最佳 val acc : {best_acc*100:.2f}%  (epoch {ckpt.get('epoch', '?')})")
    print(f"test acc     : {te_acc*100:.2f}%   <- 交付物看这个数，全文只算这一次")
    print(f"显存峰值     : {peak:.2f} GB")
    print(f"权重         : {best_path}")
    print(f"曲线         : .venv\\Scripts\\tensorboard.exe --logdir {PROJECT_ROOT / 'runs'}")
    print(f"\n下一步：.venv\\Scripts\\python.exe p1_cifar10\\evaluate.py --ckpt {args.out}/best.pt")

    # 【讲解】把超参和最终指标写进 TensorBoard 的 hparams 面板；值统一转成 str，
    #   因为 add_hparams 要求同名列类型一致。
    writer.add_hparams(
        {k: str(v) for k, v in vars(args).items()},
        {"hparam/test_acc": te_acc, "hparam/best_val_acc": best_acc},
    )
    writer.close()   # 不关的话最后的标量可能还没落盘就被进程退出丢掉


# 【讲解】Windows 上 DataLoader 用 spawn 启动子进程，子进程会重新 import 本文件；
#   没有这个 guard 会反复创建进程。Linux 用 fork，不写也能跑。
if __name__ == "__main__":
    main()
