"""P1 · CIFAR-10 数据加载与增强.

这个文件把磁盘上的 CIFAR-10 变成 DataLoader：定义均值/方差和类名、
训练与评估两套变换，并从 5 万张训练集里切出验证集。

所有讲解段落用【讲解】标记，PyCharm 里 Ctrl+Shift+F 搜 "【讲解】" 可跳读。

单独运行这个文件可以检验数据管道：
    .venv\\Scripts\\python.exe p1_cifar10\\data.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 【讲解】CIFAR-10 训练集逐通道统计出的均值 / 标准差，供 Normalize 使用。
#   换成自己的数据时要重新算这三个数，直接套 CIFAR 的值是错的。
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# 【讲解】类别顺序由数据集固定：标签 0~9 依次对应下面十个名字，
#   评估画混淆矩阵时用它把数字翻译成类名。
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# 数据集放在项目根目录的 data/ 下（约 178 MB），不在 p1_cifar10/ 里
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def build_transforms(mode: str = "eval", aug: str = "basic"):
    """返回一套图像变换。mode: 'train' | 'eval'；aug: 'none' | 'basic' | 'strong'。

    【讲解】eval 只有 ToTensor + Normalize（确定性）。
    train 按 aug 选增强：none 不做增强，basic 是随机裁剪 + 水平翻转，
    strong 再追加 RandAugment 和 RandomErasing。
    """
    if mode == "eval":
        # 评估只做确定性变换：ToTensor（HWC uint8 → CHW float32，并除以 255）+ Normalize
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    if aug == "none":
        # 对照组：和 eval 相同的变换，不做任何增强
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    if aug == "strong":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # 【讲解】RandAugment 从增强池里随机挑 num_ops 个操作，magnitude 控制强度。
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            # 【讲解】RandomErasing 随机抠一块矩形填噪声，必须放在 Normalize 之后。
            transforms.RandomErasing(p=0.25),
        ])

    # basic：随机裁剪 + 水平翻转
    return transforms.Compose([
        # 【讲解】RandomCrop(32, padding=4)：四边各补 4 像素到 40x40，再随机裁回 32x32，
        #   裁到补的边时输入均值会偏低，属正常现象。
        transforms.RandomCrop(32, padding=4),
        # 【讲解】RandomHorizontalFlip：随机水平翻转，分类任务的标签不跟着变。
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def build_loaders(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    batch_size: int = 128,
    workers: int = 6,
    val_ratio: float = 0.1,
    aug: str = "basic",
    seed: int = 0,
):
    """返回 (train_loader, val_loader, test_loader)。

    50000 张训练集里切出 val_ratio 做验证，test 10000 张全程不动。
    """
    data_root = Path(data_root)

    # 先确认数据已解压到位，否则 torchvision 会开始下载 170 MB
    batches = data_root / "cifar-10-batches-py"
    if not batches.exists():
        raise SystemExit(
            f"\n[找不到 CIFAR-10] 期望看到: {batches}\n\n"
            "数据集已经放在 <项目>/data/ 下，正常情况下不需要动它。\n"
            "如果它在别的位置，用 --data-root 指向那个目录"
            "（该目录下要能直接看到 cifar-10-batches-py 文件夹）。\n\n"
            "确实要重新下载（本机单连接被限速到 55 KB/s，约 50 分钟）：\n"
            '  curl -L -o "%s/cifar-10-python.tar.gz" https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz\n'
            '  tar -xzf "%s/cifar-10-python.tar.gz" -C "%s"\n' % (data_root, data_root, data_root)
        )

    try:
        train_aug = datasets.CIFAR10(
            str(data_root), train=True, download=False, transform=build_transforms("train", aug)
        )
        test_set = datasets.CIFAR10(
            str(data_root), train=False, download=False, transform=build_transforms("eval")
        )
    except Exception as e:  # 文件损坏 / 被删了一部分
        raise SystemExit(
            f"\n[数据加载失败] {type(e).__name__}: {e}\n"
            f"检查 {batches} 是否完整（应有 data_batch_1..5、test_batch、batches.meta）。\n"
        ) from e

    # 【讲解】再建一份不带增强的 train_plain（同一个目录）。用固定种子的 randperm
    #   生成一份下标，前 n_val 个给 val（从 plain 取），其余给 train（从带增强的取），
    #   这样两份数据集切出的是同一批文件。
    train_plain = datasets.CIFAR10(
        str(data_root), train=True, download=False, transform=build_transforms("eval")
    )
    n = len(train_aug)
    n_val = int(n * val_ratio)
    g = torch.Generator().manual_seed(seed)          # 固定种子 → 每次切出同一批 val
    perm = torch.randperm(n, generator=g).tolist()
    train_set = Subset(train_aug, perm[n_val:])      # 剩下的 90% 当训练集
    val_set = Subset(train_plain, perm[:n_val])      # 前 10% 当验证集，注意用 plain

    # 【讲解】num_workers 是并行读图的子进程数；pin_memory 把 batch 放进锁页内存，
    #   拷到 GPU 更快；persistent_workers 让 epoch 之间保留 worker 进程。
    common = dict(num_workers=workers, pin_memory=True)
    if workers > 0:
        common["persistent_workers"] = True

    # 【讲解】train 用 drop_last=True，丢掉不足一个 batch 的尾巴（避免 BN 遇到 batch=1）；
    #   val / test 不丢尾，保证覆盖全部样本。
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **common
    )
    # 验证/测试用更大的 batch（不参与反向，显存压力小），且不需要打乱
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **common)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 单独跑这个文件就是给数据管道做体检：形状、数值范围、类别分布
    import torchvision

    tr, va, te = build_loaders(workers=0)
    print(f"train batches = {len(tr)}  val batches = {len(va)}  test batches = {len(te)}")
    x, y = next(iter(tr))
    print(f"一个 batch: x={tuple(x.shape)} {x.dtype}  y={tuple(y.shape)}")
    counts = torch.bincount(y, minlength=10).tolist()
    print("这个 batch 的类别分布:", dict(zip(CLASSES, counts)))

    # 【讲解】归一化自检用不带增强的 test 集，避免 RandomCrop 补的边影响均值。
    raw = torchvision.datasets.CIFAR10(
        str(DEFAULT_DATA_ROOT), train=False, download=False, transform=build_transforms("eval")
    )
    xs = torch.stack([raw[i][0] for i in range(1000)])
    print(f"\n[自检] 无增强 1000 张: mean={xs.mean():.4f} std={xs.std():.4f}  (应接近 0 和 1)")
    print(f"[说明] 上面训练 batch 的 mean={x.mean():.3f} 偏低，是 RandomCrop(padding=4)")
    print("       补的边造成的，属于正常；关掉增强（--aug none）后会回到 0 附近。")
