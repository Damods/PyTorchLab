"""训练配置基准：对若干 (imgsz, batch, amp) 组合各跑一小段真训练，量显存峰值与迭代速度。

每个组合单独起一个子进程 —— torch.cuda.max_memory_allocated() 是进程级高水位，
跑在同一个进程里会互相污染。

用法：
    python bench.py --all
    python bench.py --one --imgsz 1024 --batch 4 --amp 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = "yolo11s-seg.pt"
DATA = PROJECT_ROOT / "data" / "StrawDI_yolo" / "data.yaml"
OUT = PROJECT_ROOT / ".tmp" / "bench"

# fraction 0.12 ≈ 336 张 / batch 8 ≈ 42 个迭代，足够让 cuDNN 定型又能快速出结果
CONFIGS = [
    dict(imgsz=640, batch=8, amp=1, fraction=0.12),
    dict(imgsz=1024, batch=8, amp=1, fraction=0.12),
    dict(imgsz=1024, batch=4, amp=1, fraction=0.24),
    dict(imgsz=1024, batch=4, amp=0, fraction=0.24),
]


def run_one(imgsz: int, batch: int, amp: int, fraction: float) -> None:
    import torch
    from ultralytics import YOLO

    tag = f"i{imgsz}_b{batch}_amp{amp}"
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(WEIGHTS))
    try:
        model.train(
            data=str(DATA), epochs=1, imgsz=imgsz, batch=batch, amp=bool(amp),
            fraction=fraction, workers=0, device=0, val=False, plots=False,
            save=False, cache=False, project=str(OUT), name=tag, exist_ok=True, verbose=False,
        )
    except RuntimeError as exc:
        print(json.dumps(dict(tag=tag, imgsz=imgsz, batch=batch, amp=amp, error=str(exc)[:120])))
        return

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    rows = (OUT / tag / "results.csv").read_text().strip().splitlines()
    header = rows[0].split(",")
    last = rows[-1].split(",")
    rec = dict(zip(header, last))
    secs = float(rec["time"])
    iters = int(fraction * 2800) // batch
    print(json.dumps(dict(
        tag=tag, imgsz=imgsz, batch=batch, amp=amp,
        peak_gib=round(peak, 2),
        epoch_s=round(secs, 1),
        iters=iters,
        s_per_iter=round(secs / max(iters, 1), 3),
        img_per_s=round(iters * batch / secs, 1),
    )))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--fraction", type=float, default=0.12)
    args = ap.parse_args()

    if args.one:
        run_one(args.imgsz, args.batch, args.amp, args.fraction)
        return

    results = []
    for cfg in CONFIGS:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--one",
               "--imgsz", str(cfg["imgsz"]), "--batch", str(cfg["batch"]),
               "--amp", str(cfg["amp"]), "--fraction", str(cfg["fraction"])]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        line = (p.stdout.strip().splitlines() or [""])[-1]
        print(line, flush=True)
        if line.startswith("{"):
            results.append(json.loads(line))
        else:
            print(p.stderr[-800:], file=sys.stderr)

    print("\n—— 汇总 ——")
    print(f"{'config':22s} {'peak GiB':>9s} {'s/iter':>7s} {'img/s':>7s}")
    for r in results:
        if "error" in r:
            print(f"{r['tag']:22s} {'OOM/ERR':>9s}  {r['error'][:60]}")
        else:
            print(f"{r['tag']:22s} {r['peak_gib']:9.2f} {r['s_per_iter']:7.3f} {r['img_per_s']:7.1f}")


if __name__ == "__main__":
    main()
