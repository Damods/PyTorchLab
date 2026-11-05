# StrawDI 草莓实例分割

从项目根目录运行以下命令。完整安装步骤见 [项目 README](../README.md)。

```powershell
python tools/download_data.py strawdi
python p4_strawdi/convert.py
python p4_strawdi/train.py --dry-run
python p4_strawdi/train.py
```

默认使用官方 `yolo11s-seg.pt` 初始化，首次自动下载；训练参数是 `imgsz=1024`、`batch=4`、`epochs=100`、`AdamW`、`lr0=0.002`、`patience=30`、混合精度开启、`seed=0`。默认缓存到内存，设备自动选择 CUDA/CPU。路径根据当前项目目录解析，不依赖旧电脑。

完全从随机初始化开始：

```powershell
python p4_strawdi/train.py --weights yolo11s-seg.yaml
```

小规模检查、CPU、显存或内存较小的配置：

```powershell
python p4_strawdi/train.py --fraction 0.01 --epochs 1 --imgsz 640 --batch 2 --workers 0 --cache none --name smoke
python p4_strawdi/train.py --device cpu --imgsz 640 --batch 2 --workers 0 --cache none --no-amp
```

数据转换使用实例 ID 掩码：0 是背景，一个非零 ID 对应一个草莓。同一实例有多个不连通轮廓时保留最大轮廓；坐标归一化后保存为 YOLO 多边形。`prepare_strawdi.py` 是同一转换器的兼容入口。

```powershell
python p4_strawdi/convert.py --dry-run
python p4_strawdi/convert.py --verify
python p4_strawdi/convert.py --limit 20 --out data/_yolo_check
```

`--overwrite` 会允许更新转换输出，应仅在确认目标目录后使用。

本机训练生成的结果在 `runs_p4/<name>/`，不会被 Git 跟踪。已有本地断点才使用 `--resume`：

```powershell
python p4_strawdi/train.py --weights runs_p4/strawdi_1024/weights/last.pt --resume
python p4_strawdi/evaluate.py --weights runs_p4/strawdi_1024/weights/best.pt --device cpu
```

GPU 参数基准（需要 NVIDIA CUDA 环境）：`python p4_strawdi/bench.py --all`。

原始数据与许可见 [StrawDI 官网](https://strawdi.github.io/)。仓库不分发数据或训练权重，不保证不同硬件、软件版本或随机初始化产生完全相同的指标。
