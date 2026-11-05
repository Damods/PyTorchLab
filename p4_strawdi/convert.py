"""StrawDI_Db1 -> YOLO-seg 数据集转换脚本。

StrawDI_Db1 的标注是「实例 ID 图」：单通道 uint8，0 是背景，每个草莓一个独立整数值。
YOLO-seg 需要的是「一张图一个 txt，每行 类别 + 归一化多边形点序列」，且每个实例只能是一个环。

转换规则：
  1. 按 ID 升序逐个实例处理，类别统一为 0 = strawberry；
  2. 同一 ID 若被叶片/果实遮挡切成多块，只保留面积最大的一块，其余碎片丢弃并计数
     （YOLO 的多边形格式放不下不相连的多个环；把碎片点硬拼成一个环会连出跨块的
      无效多边形，data/StrawDI_yolo 的旧标签就是这样，约 6% 的实例受影响）；
  3. 取外轮廓原始点（CHAIN_APPROX_SIMPLE），不做 approxPolyDP 简化；
  4. 坐标除以 (宽, 高) 归一化，保留 6 位小数。

用法：
    python convert.py --dry-run                  # 只统计，不写盘
    python convert.py                            # 全量转换（目标目录非空需 --overwrite）
    python convert.py --labels-only --overwrite  # 只重算 txt，图片目录保持不动
    python convert.py --verify                   # 与现有标签逐值比对，不写盘
    python convert.py --limit 20 --out data/_yolo_check   # 小样本试转
"""

from __future__ import annotations

import argparse
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = PROJECT_ROOT / "data" / "StrawDI_Db1.zip"
DEFAULT_OUT = PROJECT_ROOT / "data" / "StrawDI_yolo"
SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "strawberry"}


def parse_label(inst: np.ndarray, min_area: float) -> tuple[list[np.ndarray], Counter]:
    """实例 ID 图 -> 归一化多边形列表 + 统计信息。"""
    h, w = inst.shape[:2]
    polys: list[np.ndarray] = []
    stats: Counter = Counter()

    for iid in np.unique(inst):
        if iid == 0:
            continue
        mask = (inst == iid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            stats["id_no_contour"] += 1
            continue

        areas = [cv2.contourArea(c) for c in contours]
        best = int(np.argmax(areas))
        if len(contours) > 1:
            stats["fragmented_ids"] += 1
            stats["fragments_dropped"] += len(contours) - 1
        if areas[best] < min_area or len(contours[best]) < 3:
            stats["too_small_or_degenerate"] += 1
            continue

        pts = contours[best].reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        polys.append(pts)

    return polys, stats


def label_text(polys: list[np.ndarray]) -> str:
    lines = ["0 " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)) for pts in polys]
    return "\n".join(lines) + ("\n" if lines else "")


def collect(zf: zipfile.ZipFile, split: str, limit: int | None) -> list[str]:
    names = [
        n for n in zf.namelist()
        if f"/{split}/img/" in n and n.lower().endswith(".png") and not n.endswith("/")
    ]
    names.sort()
    return names[:limit] if limit else names


def write_yaml(out: Path, splits: tuple[str, ...]) -> None:
    lines = ["# 不写 path：ultralytics 缺省用本 yaml 所在目录，本地/云端都不用改",
             "names:"] + [f"  {k}: {v}" for k, v in CLASS_NAMES.items()]
    lines += [f"{s}: images/{s}" for s in splits]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")


def convert(zip_path: Path, out: Path, splits: tuple[str, ...], limit: int | None,
            min_area: float, overwrite: bool, labels_only: bool, dry_run: bool) -> None:
    if not labels_only and not dry_run and out.exists() and any(out.iterdir()) and not overwrite:
        sys.exit(f"目标目录已存在且非空：{out}\n确认要覆盖请加 --overwrite")

    totals: Counter = Counter()
    per_image: list[int] = []
    changed = 0

    with zipfile.ZipFile(zip_path) as zf:
        for split in splits:
            imgs = collect(zf, split, limit)
            if not imgs:
                print(f"[{split}] 压缩包里没有图片，跳过")
                continue
            if not dry_run:
                if not labels_only:
                    (out / "images" / split).mkdir(parents=True, exist_ok=True)
                (out / "labels" / split).mkdir(parents=True, exist_ok=True)

            for i, name in enumerate(imgs, 1):
                stem = Path(name).stem
                inst = cv2.imdecode(np.frombuffer(zf.read(name.replace("/img/", "/label/")), np.uint8),
                                    cv2.IMREAD_UNCHANGED)
                if inst is None:
                    sys.exit(f"标签无法解码：{name}")
                if inst.ndim == 3:
                    inst = inst[..., 0]

                polys, stats = parse_label(inst, min_area)
                totals.update(stats)
                totals[f"{split}_images"] += 1
                totals[f"{split}_instances"] += len(polys)
                per_image.append(len(polys))
                text = label_text(polys)

                if not dry_run:
                    if not labels_only:
                        (out / "images" / split / f"{stem}.png").write_bytes(zf.read(name))
                    target = out / "labels" / split / f"{stem}.txt"
                    if target.is_file() and target.read_text() == text:
                        continue
                    target.write_text(text)
                    changed += 1

                if i % 200 == 0:
                    print(f"[{split}] {i}/{len(imgs)}", flush=True)

    if not dry_run:
        write_yaml(out, splits if len(splits) == 3 else SPLITS)

    report(totals, per_image)
    if dry_run:
        print("\n（--dry-run：未写入任何文件）")
    else:
        print(f"\n输出目录：{out}")
        print(f"本次写入/更新标签 {changed} 个")


def report(totals: Counter, per_image: list[int]) -> None:
    print("\n—— 转换统计 ——")
    for split in SPLITS:
        if totals[f"{split}_images"]:
            print(f"  {split:5s} 图片 {totals[f'{split}_images']:5d}  实例 {totals[f'{split}_instances']:6d}")
    inst = sum(totals[f"{s}_instances"] for s in SPLITS)
    if per_image:
        print(f"  每图实例数 min/median/max = {min(per_image)}/"
              f"{statistics.median(per_image):.0f}/{max(per_image)}")
    if inst:
        print(f"  被遮挡切成多块的实例 {totals['fragmented_ids']} 个"
              f"（占 {totals['fragmented_ids'] / inst:.1%}），丢弃碎片 {totals['fragments_dropped']} 块")
    print(f"  因过小/退化被丢弃的实例 {totals['too_small_or_degenerate']} 个")


def verify(zip_path: Path, out: Path, splits: tuple[str, ...], limit: int | None,
           min_area: float, tol: float = 1e-6) -> None:
    """重新计算标签并与现有 txt 逐值比对（行顺序无关），不写盘。"""
    checked = 0
    missing: list[str] = []
    bad: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        for split in splits:
            for name in collect(zf, split, limit):
                stem = Path(name).stem
                ref_path = out / "labels" / split / f"{stem}.txt"
                if not ref_path.is_file():
                    missing.append(f"{split}/{stem}")
                    continue
                inst = cv2.imdecode(np.frombuffer(zf.read(name.replace("/img/", "/label/")), np.uint8),
                                    cv2.IMREAD_UNCHANGED)
                if inst.ndim == 3:
                    inst = inst[..., 0]
                polys, _ = parse_label(inst, min_area)
                got_rows = label_text(polys).splitlines()
                ref_rows = [ln for ln in ref_path.read_text().splitlines() if ln.strip()]

                ok = len(got_rows) == len(ref_rows)
                if ok:
                    used: set[int] = set()
                    for r in ref_rows:
                        rv = np.array(r.split()[1:], float)
                        for j, gl in enumerate(got_rows):
                            if j in used:
                                continue
                            gv = np.array(gl.split()[1:], float)
                            if gv.shape == rv.shape and np.abs(gv - rv).max() <= tol:
                                used.add(j)
                                break
                        else:
                            ok = False
                            break
                checked += 1
                if not ok:
                    bad.append(f"{split}/{stem}({len(got_rows)} vs {len(ref_rows)})")

    print(f"已比对 {checked} 张，不一致 {len(bad)} 张")
    if missing:
        print(f"缺少参考标签 {len(missing)} 张，例如 {missing[:5]}")
    if bad:
        print("不一致样例：", bad[:10])
        sys.exit(1)
    print("全部一致。")


def main() -> None:
    ap = argparse.ArgumentParser(description="StrawDI_Db1 -> YOLO-seg 转换")
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--limit", type=int, default=None, help="每个 split 只处理前 N 张（调试用）")
    ap.add_argument("--min-area", type=float, default=1.0, help="实例最小面积（像素），默认 1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--labels-only", action="store_true", help="只写 labels/，不动 images/")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    ap.add_argument("--verify", action="store_true", help="与现有标签比对，不写盘")
    args = ap.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if not args.zip.is_file():
        sys.exit(f"找不到压缩包：{args.zip}")

    if args.verify:
        verify(args.zip, args.out, splits, args.limit, args.min_area)
    else:
        convert(args.zip, args.out, splits, args.limit, args.min_area,
                args.overwrite, args.labels_only, args.dry_run)


if __name__ == "__main__":
    main()
