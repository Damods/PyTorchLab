"""把 Windows 上训出来的 last.pt 改造成能在 Linux 上续训的版本。

为什么需要这步：ultralytics 的 checkpoint 里存着 `train_args`（本次训练的全部参数），
其中 `data` / `project` / `model` 都是**绝对路径**。在本地是
`D:\\project\\workplace pycharm\\PyTorchLab\\data\\StrawDI_yolo\\data.yaml`，
搬到 Linux 上这个路径不存在，`--resume` 会直接报 Dataset not found。

本脚本只改这几个路径字段 + 打开 exist_ok（避免续训时新建 run 目录导致 results.csv 断掉），
优化器状态、EMA、epoch 计数、LR 调度进度一律不动 —— 所以续上去是真正的"接着第 40 轮"。

用法：
    python prepare_resume.py                       # 就地改（先自动备份 .orig）
    python prepare_resume.py --out last_cloud.pt   # 输出到新文件，不动原件
    python prepare_resume.py --name strawdi_1024 --data data/StrawDI_yolo/data.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = PROJECT_ROOT / "runs_p4" / "strawdi_1024" / "weights" / "last.pt"
DEFAULT_DATA = PROJECT_ROOT / "data" / "StrawDI_yolo" / "data.yaml"

PATH_KEYS = ("data", "project", "model", "resume")


def main() -> None:
    ap = argparse.ArgumentParser(description="改造 last.pt 以便跨机器/跨系统续训")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="续训要用的 data.yaml（本机绝对或相对路径都行）")
    ap.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs_p4")
    ap.add_argument("--name", default=None, help="run 名，默认沿用 checkpoint 里的")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出文件名；默认就地覆盖（会先备份成 xxx.orig）")
    ap.add_argument("--batch", type=int, default=None,
                    help="续训时改 batch（ultralytics 明确允许，A10 24G 可以给 8）")
    ap.add_argument("--workers", type=int, default=None, help="续训时改 DataLoader 进程数")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="续训时改分辨率（不推荐：会改变训练分布，慎用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要改动的内容")
    args = ap.parse_args()

    ckpt_path = args.ckpt.expanduser().resolve()
    if not ckpt_path.is_file():
        sys.exit(f"找不到 checkpoint：{ckpt_path}")

    data_path = args.data.expanduser()
    data_path = (data_path if data_path.is_absolute() else (Path.cwd() / data_path)).resolve()
    if not data_path.is_file():
        sys.exit(f"找不到 data.yaml：{data_path}")

    project_path = args.project.expanduser()
    project_path = (project_path if project_path.is_absolute() else (Path.cwd() / project_path)).resolve()

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "train_args" not in ck:
        sys.exit("这个 checkpoint 里没有 train_args，不是 ultralytics 训练产物，无法续训")

    ta = dict(ck["train_args"])
    done = int(ck.get("epoch", 0)) + 1        # checkpoint 里的 epoch 是 0 基索引
    total = ta.get("epochs", "?")
    print(f"checkpoint : {ckpt_path}")
    print(f"已完成轮数 : {done} / {total}（epoch 索引 {ck.get('epoch', '?')}）")
    print(f"原始 project/name : {ta.get('project')} / {ta.get('name')}")

    updates = {
        "data": str(data_path),
        "project": str(project_path),
        "model": str(ckpt_path),
        "resume": str(ckpt_path),
        # save_dir 也在 train_args 里，且是绝对路径。不改的话云端会建出一个
        # 名字叫 "D:\project\..." 的目录（Linux 上反斜杠是合法文件名字符）。
        "save_dir": str(project_path / str(args.name or ta.get("name"))),
        "exist_ok": True,   # 关键：否则续训可能新建 strawdi_10242，results.csv 断掉
    }
    if args.name:
        updates["name"] = args.name
    # 这三项 ultralytics 在 resume 时是允许改的（见 engine/trainer.py 的 allowed 集合）
    if args.batch:
        updates["batch"] = args.batch
    if args.workers:
        updates["workers"] = args.workers
    if args.imgsz:
        updates["imgsz"] = args.imgsz

    print("\n将修改的字段：")
    for k, v in updates.items():
        old = ta.get(k, "<缺失>")
        flag = "" if str(old) == str(v) else "  ← 改"
        print(f"  {k:10s} {old}  ->  {v}{flag}")

    if args.dry_run:
        print("\n（--dry-run：未写盘）")
        return

    ta.update(updates)
    ck["train_args"] = ta

    out_path = args.out.expanduser() if args.out else ckpt_path
    out_path = out_path if out_path.is_absolute() else (Path.cwd() / out_path)
    if out_path == ckpt_path:
        backup = ckpt_path.with_suffix(".pt.orig")
        if not backup.exists():
            shutil.copy2(ckpt_path, backup)
            print(f"\n原件已备份：{backup}")
    torch.save(ck, out_path)
    print(f"\n已写出：{out_path}")

    name = ta.get("name")
    print("\n下一步这样续训（在同一台机器、同一个路径下）：")
    print(f"  python p4_strawdi/train.py --weights {out_path} --resume")
    print(f"  期望日志：Resuming training from ... ，第一行显示 {(done) + 1}/{total}")
    print(f"  输出目录：{project_path / str(name)}")


if __name__ == "__main__":
    main()
