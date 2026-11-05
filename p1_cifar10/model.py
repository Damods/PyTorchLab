"""P1 · 手写的 CNN 结构（约 1.15 M 参数）.

不用 torchvision 的现成模型，每一层都在这个文件里搭出来。
所有讲解段落都用【讲解】标记，在 PyCharm 里 Ctrl+Shift+F 搜 "【讲解】" 可以跳着看。

结构（尺寸变化是最容易搞混的地方，先记这张图）：

    输入            (N, 3, 32, 32)
    ConvBlock(3→64)  (N, 64, 16, 16)     ← 两次 3x3 卷积保持 32x32，池化减半
    ConvBlock(64→128)(N, 128, 8, 8)
    ConvBlock(128→256)(N, 256, 4, 4)
    AdaptiveAvgPool1 (N, 256, 1, 1)      ← 全局平均池化，把 4x4 压成 1 个值
    Linear(256→10)   (N, 10)             ← 10 个类别的打分（logits）

单独运行这个文件可以打印逐层形状和参数量：
    .venv\\Scripts\\python.exe p1_cifar10\\model.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv-BN-ReLU x2 + MaxPool。输入输出都是 4D 张量 (N, C, H, W)。

    【讲解】两次 3x3 卷积（感受野等价于 5x5，参数 18 个）接一次 2x2 池化：
    通道数从 c_in 变 c_out，空间尺寸减半。
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            # 【讲解】bias=False：紧跟的 BatchNorm 会减均值再加 beta，卷积的偏置会被消掉。
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            # 【讲解】BatchNorm2d(c_out)：按通道归一化输出。running stats 只在 train()
            #   模式下更新，eval() 模式用累积的滑动平均。
            nn.BatchNorm2d(c_out),
            # 【讲解】inplace=True 原地写入，省一份中间张量。
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            # 【讲解】MaxPool2d(2)：2x2 窗口、步长 2，空间尺寸精确减半（32→16→8→4）。
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallVGG(nn.Module):
    """3 个卷积块 + 全局平均池化 + 单层线性分类头。

    【讲解】AdaptiveAvgPool2d(1) 把每个通道在 4x4 上取平均，得到 256 维，
    再接 Linear(256, 10)；比 Flatten 后接 Linear(4096, 10) 少得多参数。
    """

    def __init__(self, num_classes: int = 10, widths=(64, 128, 256), dropout: float = 0.0):
        super().__init__()
        w0, w1, w2 = widths        # 通道数逐块翻倍：64 → 128 → 256
        self.features = nn.Sequential(
            ConvBlock(3, w0),      # 3 通道 RGB 进，64 出，32x32 → 16x16
            ConvBlock(w0, w1),     # 16x16 → 8x8
            ConvBlock(w1, w2),     # 8x8 → 4x4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (N, 256, 4, 4) → (N, 256, 1, 1)
            nn.Flatten(),              # (N, 256, 1, 1) → (N, 256)
            # 【讲解】dropout 作用在这 256 维特征上，概率由 --dropout 控制（默认 0 = 关闭）。
            nn.Dropout(dropout),
            nn.Linear(w2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 【讲解】forward 里不接 softmax：CrossEntropyLoss 内部已含 log_softmax + NLLLoss，
        #   模型直接输出 logits。
        return self.head(self.features(x))


def count_params(model: nn.Module) -> int:
    """只统计需要梯度的参数（requires_grad=True）。冻结层不参与训练，不该算进来。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 自检：形状对不对、参数量多少、每层把尺寸变成了什么
    m = SmallVGG()
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    print(f"参数量: {count_params(m)/1e6:.3f} M")
    print(f"输入 {tuple(x.shape)} -> 输出 {tuple(y.shape)}  (应为 (4, 10))")

    # 【讲解】依次过三个 ConvBlock，打印每块的输出形状（下采样就发生在这里）。
    h = x
    for i, blk in enumerate(m.features):
        h = blk(h)
        print(f"  block{i}: {tuple(h.shape)}")
