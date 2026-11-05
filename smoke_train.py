"""YOLO 端到端训练冒烟测试：合成小数据集，1 epoch，全程离线。

验证 dataloader -> 前向 -> 损失 -> 反向 -> 优化器 -> 验证 这条链在 CUDA 上是否通。
产物写到系统临时目录，不污染项目。

    python smoke_train.py
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(tempfile.gettempdir()) / "pylab_smoke"
(RROOT := ROOT / "ds").mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
for split, n in (("train", 8), ("val", 4)):
    (RROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (RROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((320, 320, 3), 40, np.uint8)
        # 画一个类 0 的方块
        w, h = rng.integers(60, 120, 2)
        x1, y1 = rng.integers(20, 320 - w - 20), rng.integers(20, 320 - h - 20)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (60, 200, 240), -1)
        cv2.imwrite(str(RROOT / "images" / split / f"{i}.jpg"), img)
        cx, cy = (x1 + w / 2) / 320, (y1 + h / 2) / 320
        (RROOT / "labels" / split / f"{i}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {w / 320:.6f} {h / 320:.6f}\n")

yaml_path = RROOT / "data.yaml"
yaml_path.write_text(
    f"path: {RROOT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: box\n"
)

os.environ["YOLO_VERBOSE"] = "false"

from ultralytics import YOLO  # noqa: E402

model = YOLO("yolo11n.yaml")
res = model.train(
    data=str(yaml_path),
    epochs=1,
    imgsz=320,
    batch=4,
    device=0,
    workers=0,
    amp=False,       # 关掉 AMP 检查，避免它去下载 yolo11n.pt 做自检
    plots=False,
    val=True,
    project=str(ROOT / "runs"),
    name="smoke",
    exist_ok=True,
    verbose=False,
)

print("\n" + "=" * 46)
print("结论")
print("=" * 46)
metrics = res.results_dict if hasattr(res, "results_dict") else {}
print(f"训练完成, 最终指标: { {k: round(float(v), 4) for k, v in metrics.items()} }")
print(f"权重: {res.save_dir}")
print("[OK] 训练链路（数据加载 -> 前向 -> 损失 -> 反向 -> 验证）在 CUDA 上跑通。")
