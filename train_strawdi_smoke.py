"""Run a one-epoch YOLO11s-seg training and validation check from PyCharm."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
BATCH_SIZE = 8


def main():
    dataset = ROOT / "data" / "StrawDI_yolo" / "data.yaml"
    if not dataset.is_file():
        raise FileNotFoundError("Run p4_strawdi/convert.py to create data/StrawDI_yolo/data.yaml first.")

    import torch
    from ultralytics import YOLO

    model = YOLO("yolo11s-seg.pt")
    model.train(
        data=str(dataset), device=0 if torch.cuda.is_available() else "cpu", imgsz=640, batch=BATCH_SIZE, epochs=1,
        optimizer="AdamW", lr0=0.002, workers=0, amp=False, plots=False,
        project=str(ROOT / "runs"), name="strawdi_smoke",
    )
    print(f"Saved weights: {model.trainer.best}")


if __name__ == "__main__":
    main()
