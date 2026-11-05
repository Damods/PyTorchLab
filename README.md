# PyTorchLab

PyTorch 训练练习：CIFAR-10 分类、照片迁移学习、StrawDI 草莓实例分割。

仓库仅保存源码、依赖和使用说明。数据集、预训练模型、训练断点、日志和结果图片均在本机下载或生成，不上传到 GitHub。

## 在另一台电脑重新开始

本项目的版本锁定环境使用 **Python 3.14**。NVIDIA 显卡需安装兼容 CUDA 13.0 的驱动；没有适配显卡时可安装 CPU 环境。不同硬件的速度和训练结果可能不同。

```powershell
git clone https://github.com/Damods/PyTorchLab.git
cd PyTorchLab

# NVIDIA CUDA 13.0 环境（与原训练环境相同）
python tools/setup_env.py --backend cu130

# 只有 CPU 时，改用下面这条，不需要执行上面的安装命令
# python tools/setup_env.py --backend cpu

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate

python check_env.py
python check_vision.py
```

如果 PowerShell 不允许激活脚本，可直接用 `.\.venv\Scripts\python.exe` 替代后面各命令中的 `python`。`check_env.py`、`check_vision.py` 包含 GPU 体检；CPU 环境使用下面的最小检查：

```powershell
python -c "import torch, torchvision, cv2, ultralytics; print(torch.__version__, torch.cuda.is_available())"
```

`tools/setup_env.py` 会先安装匹配的 PyTorch/torchvision，再安装 `requirements.txt` 并运行 `pip check`。`requirements-lock.txt` 是原 Windows 环境的完整包版本记录，不是跨平台安装入口。

## 草莓实例分割：从数据到训练

```powershell
# 1. 从原作者提供的地址下载 StrawDI_Db1，支持分块重试和 SHA-256 校验
python tools/download_data.py strawdi

# 2. 转换成 YOLO 实例分割格式，生成可跨目录使用的 data.yaml
python p4_strawdi/convert.py

# 3. 检查数据与训练参数
python p4_strawdi/train.py --dry-run

# 4. 先做小规模训练确认流程
python p4_strawdi/train.py --fraction 0.01 --epochs 1 --imgsz 640 --batch 2 --workers 0 --cache none --name smoke

# 5. 正式训练：YOLO11s-seg，1024，batch 4，100 轮，AdamW，lr0=0.002
python p4_strawdi/train.py
```

首次训练会自动下载官方 `yolo11s-seg.pt` 作为初始化模型；它不是此前训练的断点。完全随机初始化、也不下载预训练权重时：

```powershell
python p4_strawdi/train.py --weights yolo11s-seg.yaml
```

默认自动选择 CUDA 或 CPU；也可显式传 `--device 0` 或 `--device cpu`。显存不足可降低 `--batch` / `--imgsz`；内存不足传 `--cache none`。新电脑需重新训练，不会恢复本机第 40 轮的状态。

训练结束后，权重和记录生成在本机 `runs_p4/<实验名>/`。以后在同一台机器上续训、评估：

```powershell
python p4_strawdi/train.py --weights runs_p4/strawdi_1024/weights/last.pt --resume
python p4_strawdi/evaluate.py --weights runs_p4/strawdi_1024/weights/best.pt --device cpu
```

数据来自 [StrawDI 官方网站](https://strawdi.github.io/)，约 4.39 GB 原始压缩包。使用时遵守原作者的数据许可并在研究中引用；数据不在本仓库分发。下载服务不可用时，可从官网手动下载为 `data/StrawDI_Db1.zip`，再执行转换命令。转换后另需数 GB 磁盘空间。

## CIFAR-10 分类

```powershell
python tools/download_data.py cifar10
python p1_cifar10/train.py --epochs 1 --limit-batches 20 --workers 0
python p1_cifar10/train.py
```

详见 [p1_cifar10/README.md](p1_cifar10/README.md)。

## 照片迁移学习

真实照片需自己按 `data/my_photos/<类别名>/<图片>` 准备；这部分个人数据不会随代码上传。没有照片时，可以生成合成数据验证训练流程：

```powershell
python p2_finetune/make_demo_data.py
python p2_finetune/train.py --data-root data/_demo --scratch --epochs 1 --freeze-epochs 0 --workers 0 --limit-batches 2 --out runs_p2/demo_smoke
```

正式使用自己的照片时运行 `python p2_finetune/train.py`，会自动下载官方 ImageNet 预训练模型。详见 [p2_finetune/README.md](p2_finetune/README.md)。

## 目录

- `p1_cifar10/`：分类网络、训练、评估。
- `p2_finetune/`：照片数据处理、迁移学习、数据体检、合成数据。
- `p4_strawdi/`：实例掩码转换、训练、评估、GPU 基准。
- `tools/download_data.py`：从数据原始来源下载。
- `tools/setup_env.py`：重建本机 Python 环境。
- `smoke_train.py`：使用合成数据的训练链路检查。

数据、权重及训练输出已加入 `.gitignore`。训练模型与第三方数据沿用各自原有许可。
