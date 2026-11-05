"""P2 · 数据体检：训练之前先扫一遍，把问题挡在训练之外.

用法：
    .venv\\Scripts\\python.exe p2_finetune\\check_data.py
    .venv\\Scripts\\python.exe p2_finetune\\check_data.py --data-root D:\\照片

检查六件事：
    1. 每类多少张、是否严重失衡
    2. 有没有空类别文件夹（会让 ImageFolder 直接报错）
    3. 有没有读不出来的文件（损坏 / 半截 / 扩展名骗人）
    4. 有没有完全重复的图（同一张复制多份会让 val 虚高）
    5. 格式分布（有多少 .heic 需要特殊处理）
    6. 分辨率分布（太小的图放大到 224 只会更糊）

【讲解】逐类统计张数、格式、分辨率，逐张实际解码一次并算 md5 查重；
结果分成"必须解决"和"建议处理"两类打印，有阻塞性问题时以非零码退出。

讲解段落用【讲解】标记。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# 复用 data.py 里的 heic 注册与合法扩展名判断
import pillow_heif  # noqa: E402
from data import DEFAULT_DATA_ROOT, IMG_EXTS, pil_loader_exif  # noqa: E402

pillow_heif.register_heif_opener()


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 数据体检")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--min-per-class", type=int, default=20, help="每类低于这个数就警告")
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    if not root.is_absolute():
        root = (HERE.parent / root).resolve()

    print(f"扫描目录: {root}")
    if not root.exists():
        raise SystemExit("目录不存在。先按 类名/图片 的结构把照片放好。")

    # 【讲解】类别只取根目录下的直接子文件夹（和 ImageFolder 一致）；
    #   根目录里的散图记到 loose，单独提示。
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    loose = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    if not class_dirs:
        raise SystemExit(f"没有找到类别文件夹。需要 {root}\\类名\\图片.jpg 这样的结构。")

    problems: list[str] = []
    warnings: list[str] = []

    print(f"\n{'类别':<16}{'张数':>6}  {'格式分布':<28}{'分辨率范围'}")
    print("-" * 78)

    per_class: dict[str, int] = {}
    fmt_counter: Counter = Counter()
    sizes: list[tuple[int, int]] = []
    hash_map: dict[str, list[str]] = defaultdict(list)
    corrrupt: list[str] = []
    empty: list[str] = []

    for d in class_dirs:
        files = sorted([p for p in d.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stat().st_size > 0])
        per_class[d.name] = len(files)
        if not files:
            empty.append(d.name)
            continue

        fmts = Counter()
        wh = []
        for p in files:
            fmt_counter[p.suffix.lower()] += 1
            fmts[p.suffix.lower()] += 1
            try:
                # 【讲解】实际解码一次（load()），确认文件能完整读出，并记录尺寸。
                img = pil_loader_exif(str(p))
                img.load()
                wh.append(img.size)
                sizes.append(img.size)
            except Exception as e:
                corrrupt.append(f"{p.relative_to(root)}  ({type(e).__name__})")
                continue
            hash_map[md5_of(p)].append(str(p.relative_to(root)))

        fmt_str = " ".join(f"{k}×{v}" for k, v in sorted(fmts.items()))
        if wh:
            ws = [s[0] for s in wh]
            hs = [s[1] for s in wh]
            rng = f"{min(ws)}x{min(hs)} ~ {max(ws)}x{max(hs)}"
        else:
            rng = "-"
        print(f"{d.name:<16}{len(files):>6}  {fmt_str:<28}{rng}")

    # ---- 汇总判断 ----
    counts = [v for v in per_class.values() if v > 0]
    total = sum(counts)

    print("\n" + "=" * 60)
    if empty:
        problems.append(f"这些类别文件夹是空的：{', '.join(empty)} —— ImageFolder 会直接报错。"
                        "要么放图，要么删掉文件夹。")
    if loose:
        warnings.append(f"根目录下有 {len(loose)} 张散图不会被读到（ImageFolder 只认子文件夹）。"
                        f"挪进某个类别文件夹里。")
    if corrrupt:
        problems.append(f"{len(corrrupt)} 个文件读不出来：\n    " + "\n    ".join(corrrupt[:10]))

    dup_groups = {h: v for h, v in hash_map.items() if len(v) > 1}
    if dup_groups:
        n_dup = sum(len(v) - 1 for v in dup_groups.values())
        detail = list(dup_groups.values())[:3]
        warnings.append(f"{n_dup} 张图完全重复（md5 相同）。重复图会让验证集虚高。示例：\n    "
                        + "\n    ".join(" = ".join(v) for v in detail))

    if counts:
        mn, mx = min(counts), max(counts)
        if mn < args.min_per_class:
            problems.append(f"最少的一类只有 {mn} 张（建议 ≥ {args.min_per_class}）。")
        if mx >= mn * 3 and mx > 30:
            warnings.append(f"类别失衡：最多 {mx} 张，最少 {mn} 张（差 {mx/mn:.1f} 倍）。"
                            "模型会偏向样本多的类，少数类 recall 会明显偏低。")

    if len(counts) < 2:
        problems.append(f"只有 {len(counts)} 个有图的类别，至少要 2 类才能做分类。")
    if total < 100:
        warnings.append(f"总样本 {total} 张偏少，建议总量 ≥ 200 张。")

    if sizes:
        small = [s for s in sizes if min(s) < 100]
        if small:
            warnings.append(f"{len(small)} 张图最短边 < 100 像素，放大到 224 会很模糊。")

    heic_n = fmt_counter.get(".heic", 0) + fmt_counter.get(".heif", 0)
    if heic_n:
        print(f"检测到 {heic_n} 张 HEIC —— 代码已装 pillow-heif 并打了 ImageFolder 扩展名补丁，可直接训练。")

    print(f"总计 {total} 张，{len(counts)} 个有效类别")
    print(f"格式分布: {dict(fmt_counter)}")

    if problems:
        print("\n[必须解决]")
        for p in problems:
            print(f"  ✗ {p}")
    if warnings:
        print("\n[建议处理]")
        for w in warnings:
            print(f"  ! {w}")
    if not problems and not warnings:
        print("\n[OK] 数据看起来没问题，可以开始训练。")
    elif not problems:
        print("\n[OK] 没有阻塞性问题，可以训练（上面的建议能不做也能跑）。")

    print("\n下一步：.venv\\Scripts\\python.exe p2_finetune\\train.py")
    # 【讲解】有阻塞性问题时返回非零退出码。
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
