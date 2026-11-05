"""P2 · 模型：加载预训练 backbone，换掉分类头.

本文件四个函数：
  build_model          取 resnet18/resnet50，把 fc 换成 num_classes 类的新线性层
  split_params         按"是否属于 head"把参数分成两组
  set_backbone_frozen  冻结 / 解冻 backbone（requires_grad）
  set_bn_eval          把 BatchNorm 层切到 eval（只读 running stats）

讲解段落用【讲解】标记。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# 支持的骨架 —— 权重都已经下好在本地 cache，不会联网
SUPPORTED = ("resnet18", "resnet50")


def build_model(arch: str = "resnet18", num_classes: int = 5, pretrained: bool = True):
    """返回 (model, head)。

    【讲解】head 是新建的分类头模块，train.py 会给它单独设更大的学习率。
    pretrained=False 时不加载 ImageNet 权重，供 --scratch 对照实验使用。
    """
    if arch not in SUPPORTED:
        raise SystemExit(f"不支持的模型 {arch}，可选：{SUPPORTED}")

    weights = "DEFAULT" if pretrained else None
    try:
        model = getattr(models, arch)(weights=weights)
    except Exception as e:
        raise SystemExit(
            f"\n[权重加载失败] {type(e).__name__}: {e}\n"
            "预训练权重已下在 C:\\Users\\11859\\.cache\\torch\\hub\\checkpoints\\\n"
            "如果确实缺失，用 --scratch 从零训练。"
        ) from e

    # 【讲解】取原 fc 的输入维度，新建 Linear(in_features, num_classes)，
    #   赋回 model.fc —— 原层输出 1000 类，这里换成你的类别数。
    in_features = model.fc.in_features
    head = nn.Linear(in_features, num_classes)
    model.fc = head

    return model, head


def split_params(model: nn.Module, head: nn.Module):
    """把参数分成 (backbone_params, head_params)。

    【讲解】先收集 head 参数的 id，再按 id 是否在集合里，把 model.parameters()
    分成两组 —— 不依赖 head 在具体架构里叫什么名字。
    """
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head_ps = [p for p in model.parameters() if id(p) in head_ids]
    return backbone, head_ps


def set_backbone_frozen(model: nn.Module, head: nn.Module, frozen: bool) -> None:
    """冻结 / 解冻 backbone。

    【讲解】遍历 backbone 参数，把 requires_grad 设为 not frozen：
    冻结时这些参数不求梯度，优化器也就不会更新它们。
    """
    for p in split_params(model, head)[0]:
        p.requires_grad = not frozen


def set_bn_eval(model: nn.Module) -> None:
    """把模型里所有 BatchNorm2d 切到 eval 模式（只读 running stats，不更新）。

    【讲解】遍历 model.modules()，对每个 BatchNorm2d 调 eval()。
    调用点在 model.train() 之后、第一次前向之前（train_one_epoch 里就是这么做的）。
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
