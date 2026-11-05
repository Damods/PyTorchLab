"""test 集评估 + 证据图。只在最后跑一次，避免拿 test 调参。

产出（默认 runs_p4/<name>/test_eval/）：
  eval_report.json   指标与推理耗时
  test_evidence.png  「原图 / GT / 预测」三联图
  val_batch*.jpg     由 ultralytics 生成的预测可视化

用法：
    python evaluate.py --weights runs_p4/strawdi_1024/weights/best.pt
    python evaluate.py --weights ... --samples 6 --conf 0.4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "StrawDI_yolo" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs_p4"


def load_gt_polys(label_path: Path, w: int, h: int) -> list[np.ndarray]:
    polys = []
    for line in label_path.read_text().splitlines():
        v = line.split()
        if len(v) >= 7:
            pts = np.array(v[1:], float).reshape(-1, 2) * [w, h]
            polys.append(np.round(pts).astype(np.int32))
    return polys


def overlay(img: np.ndarray, polys: list[np.ndarray], color: tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    layer = np.zeros_like(img)
    for p in polys:
        cv2.fillPoly(layer, [p], color)
    cv2.addWeighted(layer, 0.45, out, 0.55, 0, out)
    for p in polys:
        cv2.polylines(out, [p], True, color, 2, cv2.LINE_AA)
    return out


def make_evidence(weights: Path, data: Path, out_dir: Path, imgsz: int, conf: float,
                  samples: int, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    from ultralytics import YOLO

    cfg = yaml.safe_load(data.read_text())
    root = Path(cfg.get("path", data.parent))
    img_dir = root / cfg["test"]
    lab_dir = root / "labels" / Path(cfg["test"]).name
    images = sorted(img_dir.glob("*"))
    picks = random.Random(seed).sample(images, min(samples, len(images)))

    model = YOLO(str(weights))
    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 3.4 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, path in enumerate(picks):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gt = load_gt_polys(lab_dir / f"{path.stem}.txt", w, h)
        res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        pred = [np.round(p).astype(np.int32) for p in (res.masks.xy if res.masks is not None else [])]

        for col, (title, canvas) in enumerate([
            ("image", img),
            (f"GT ({len(gt)})", overlay(img, gt, (0, 255, 0))),
            (f"pred ({len(pred)})", overlay(img, pred, (255, 120, 0))),
        ]):
            axes[row, col].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")

    fig.tight_layout()
    target = out_dir / "test_evidence.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    print(f"证据图：{target}")


def main() -> None:
    ap = argparse.ArgumentParser(description="test 集评估")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25, help="画证据图时的置信度阈值")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--name", default=None, help="输出子目录名，默认 <权重所属run>_test_eval")
    args = ap.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"找不到权重 {args.weights}")

    run_name = args.name or f"{args.weights.parents[1].name}_{args.split}_eval"
    out_dir = args.project / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    m = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
                  device=args.device, plots=True, save_json=True, project=str(out_dir.parent),
                  name=out_dir.name, exist_ok=True)

    speed = {k: round(v, 2) for k, v in m.speed.items()}
    total_ms = speed.get("preprocess", 0) + speed.get("inference", 0) + speed.get("postprocess", 0)

    import yaml
    cfg = yaml.safe_load(args.data.read_text())
    img_dir = Path(cfg.get("path", args.data.parent)) / cfg[args.split]
    n_images = len([p for p in img_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    report = {
        "weights": str(args.weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "n_images": n_images,
        "box": {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
                "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)},
        "mask": {"mAP50": round(float(m.seg.map50), 4), "mAP50-95": round(float(m.seg.map), 4),
                 "precision": round(float(m.seg.mp), 4), "recall": round(float(m.seg.mr), 4)},
        "speed_ms_per_image": speed,
        "fps": round(1000 / total_ms, 1) if total_ms else None,
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n—— test 集结果 ——")
    print(f"  box  mAP50 {report['box']['mAP50']:.4f}   mAP50-95 {report['box']['mAP50-95']:.4f}   "
          f"P {report['box']['precision']:.4f}  R {report['box']['recall']:.4f}")
    print(f"  mask mAP50 {report['mask']['mAP50']:.4f}   mAP50-95 {report['mask']['mAP50-95']:.4f}   "
          f"P {report['mask']['precision']:.4f}  R {report['mask']['recall']:.4f}")
    print(f"  推理 {total_ms:.1f} ms/张（{report['fps']} FPS），"
          f"其中 pre {speed.get('preprocess')} / inf {speed.get('inference')} / post {speed.get('postprocess')}")
    print(f"  报告：{out_dir / 'eval_report.json'}")

    make_evidence(args.weights, args.data, out_dir, args.imgsz, args.conf, args.samples, args.seed)


if __name__ == "__main__":
    main()
