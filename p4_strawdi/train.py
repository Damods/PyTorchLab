"""StrawDI / 草莓实例分割训练脚本（封装 ultralytics，参数全部显式暴露）。

默认值是按本机实测选的（RTX 5060 Laptop 8 GB，见 README「五、实测基准」）：
  imgsz=1024, batch=4, amp=True → 显存 3.11 GiB，12.7 img/s
原图 1008x756，用 640 会把果实压小一圈，故默认按原分辨率训练。

用法：
    python train.py --dry-run                 # 只检查数据 + 打印最终参数，不训练
    python train.py                           # 默认配置正式训练，输出到 runs_p4/
    python train.py --paper-fast              # 论文输入尺寸与批量，启用混合精度与 RAM 缓存
    python train.py --name v1_1024 --epochs 300 --cache ram
    python train.py --weights runs_p4/v1_1024/weights/last.pt --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "StrawDI_yolo" / "data.yaml"
DEFAULT_WEIGHTS = "yolo11s-seg.pt"
DEFAULT_PROJECT = PROJECT_ROOT / "runs_p4"


def count_split(data_yaml: Path, split: str) -> tuple[int, int]:
    """数一个 split 下的图片与标签数量，标签缺失/为空也能发现。"""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    img_dir = root / cfg[split]
    lab_dir = root / "labels" / Path(cfg[split]).name
    imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    labs = list(lab_dir.glob("*.txt"))
    empty = sum(1 for p in labs if not p.read_text().strip())
    return len(imgs), len(labs) - empty


def check(data_yaml: Path) -> None:
    print(f"数据集配置：{data_yaml}")
    for split in ("train", "val", "test"):
        try:
            n_img, n_lab = count_split(data_yaml, split)
        except KeyError:
            print(f"  {split:5s} 未在 data.yaml 中定义，跳过")
            continue
        flag = "OK" if n_img and n_lab else "!! 缺文件"
        print(f"  {split:5s} 图片 {n_img:5d}  有效标签 {n_lab:5d}  {flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO11-seg 训练（StrawDI 草莓实例分割）")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="官方模型名（首次自动下载）、本地权重，或 yolo11s-seg.yaml 从零初始化")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30, help="验证指标多少轮不涨就早停")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto 自动选择 CUDA/CPU，也可填 0 或 cpu")
    ap.add_argument("--no-amp", action="store_true", help="关闭混合精度（默认开）")
    ap.add_argument("--cache", choices=["none", "ram", "disk"], default="ram",
                    help="ram: 解码后的图放内存（约 7 GB），消除每轮重复读盘")
    ap.add_argument("--fraction", type=float, default=1.0, help="只用一部分训练集（调试）")
    ap.add_argument("--lr0", type=float, default=0.002, help="初始学习率")
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--cos-lr", action="store_true")
    ap.add_argument("--close-mosaic", type=int, default=10, help="最后 N 轮关闭 mosaic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--name", default="strawdi_1024")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只校验数据并打印参数")
    ap.add_argument("--paper-fast", action="store_true",
                    help="使用 640/batch 8/100 轮/AdamW，启用混合精度与 RAM 缓存")
    args = ap.parse_args()
    if args.paper_fast:
        ap.set_defaults(imgsz=640, batch=8, epochs=100, patience=0, workers=0,
                        cache="ram", optimizer="AdamW", lr0=0.002, name="paper_640_amp")
        args = ap.parse_args()

    if not args.data.is_file():
        sys.exit(f"找不到 {args.data}\n先跑 p4_strawdi/convert.py 生成数据集")
    if args.resume and not Path(args.weights).is_file():
        sys.exit(f"断点续训需要本地文件：{args.weights}")
    if ("/" in args.weights or "\\" in args.weights) and not Path(args.weights).is_file():
        sys.exit(f"找不到模型文件：{args.weights}")

    import torch
    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"

    check(args.data)

    kwargs = dict(
        data=str(args.data), imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
        patience=args.patience, workers=args.workers, device=args.device,
        amp=not args.no_amp, cache=False if args.cache == "none" else args.cache,
        fraction=args.fraction, optimizer=args.optimizer, cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic, seed=args.seed, deterministic=True,
        project=str(args.project), name=args.name, plots=True, save=True,
        val=True, verbose=True,
    )
    if args.lr0 is not None:
        kwargs["lr0"] = args.lr0

    print("\n—— 最终训练参数 ——")
    for k, v in kwargs.items():
        print(f"  {k:14s} {v}")
    print(f"  {'weights':14s} {args.weights}")

    if args.dry_run:
        print("\n（--dry-run：未启动训练）")
        return

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    if args.resume:
        model.train(resume=True)
    else:
        model.train(**kwargs)

    save_dir = Path(getattr(model.trainer, "save_dir", args.project / args.name))
    print(f"\n训练结束，输出目录：{save_dir}")
    print(f"最佳权重：{save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
