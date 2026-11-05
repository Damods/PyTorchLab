"""PyTorchLab 视觉栈验证：torchvision / OpenCV / ultralytics 是否真能在 CUDA 上跑。

全部离线，不下载任何预训练权重（仓库只从 yaml 现场构建）。

    .venv\\Scripts\\python.exe check_vision.py
"""

import sys
import traceback

FAIL = []


def section(title):
    print("\n" + "=" * 46)
    print(title)
    print("=" * 46)


def ok(label, msg):
    print(f"{label:<24}: OK   {msg}")


def fail(label, msg):
    FAIL.append(f"{label}: {msg}")
    print(f"{label:<24}: FAIL {msg}")


# ---------------------------------------------------------------- torch
section("torch（应当没被改动）")
import torch

print(f"torch   : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"可用    : {torch.cuda.is_available()}")
print(f"设备    : {torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
if torch.version.cuda is None:
    fail("torch", "这是 CPU 版，被覆盖了！")
else:
    ok("torch.cuda 构建", torch.version.cuda)

# ------------------------------------------------------------ torchvision
section("torchvision（重点：必须带 CUDA 内核）")
import torchvision

print(f"torchvision: {torchvision.__version__}")
try:
    from torchvision.ops import nms
    from torchvision.transforms import v2

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]],
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    keep = nms(boxes, scores, 0.5)
    ok("torchvision.ops.nms", f"cuda 张量跑通，keep={keep.tolist()}")
except Exception as e:  # noqa: BLE001
    fail("torchvision.ops.nms", f"{type(e).__name__}: {e}")
    traceback.print_exc()

try:
    t = v2.Compose([v2.Resize((224, 224)), v2.ToDtype(torch.float32, scale=True)])(
        torch.randint(0, 255, (3, 320, 320), dtype=torch.uint8)
    )
    ok("transforms.v2", f"输出 shape={tuple(t.shape)} dtype={t.dtype}")
except Exception as e:  # noqa: BLE001
    fail("transforms.v2", f"{type(e).__name__}: {e}")

try:
    from torchvision.models import resnet18

    net = resnet18(weights=None).cuda().eval()
    with torch.inference_mode():
        y = net(torch.randn(1, 3, 224, 224, device="cuda"))
    ok("resnet18 前向", f"输出 shape={tuple(y.shape)}  device={y.device}")
except Exception as e:  # noqa: BLE001
    fail("resnet18 前向", f"{type(e).__name__}: {e}")

# ------------------------------------------------------------------- cv2
section("OpenCV")
import cv2
import numpy as np

print(f"cv2     : {cv2.__version__}")
try:
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    small = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    ok("resize/cvtColor/Blur", f"{img.shape} -> {small.shape}, 边缘像素 {int((edges > 0).sum())}")
except Exception as e:  # noqa: BLE001
    fail("基础图像算子", f"{type(e).__name__}: {e}")

try:
    enc_ok, buf = cv2.imencode(".png", small)
    back = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    ok("PNG 编码/解码", f"{len(buf)} 字节, 回读 shape={back.shape}")
except Exception as e:  # noqa: BLE001
    fail("PNG 编码/解码", f"{type(e).__name__}: {e}")

try:
    print(f"CUDA 模块: {'有' if hasattr(cv2, 'cuda') else '无（属正常，pip 版 OpenCV 不含 cudawarping）'}")
    if hasattr(cv2, "cuda"):
        print(f"cv2.cuda.getCudaEnabledDeviceCount() = {cv2.cuda.getCudaEnabledDeviceCount()}")
except Exception as e:  # noqa: BLE001
    print(f"CUDA 模块探测跳过: {e}")

# -------------------------------------------------------------- ultralytics
section("ultralytics（离线从 yaml 建模型，在 GPU 上真跑一次推理）")
try:
    import ultralytics

    print(f"ultralytics: {ultralytics.__version__}")
    from ultralytics import YOLO

    model = YOLO("yolo11n.yaml")  # 不下载权重，现场构建
    n_par = sum(p.numel() for p in model.model.parameters())
    ok("YOLO(yaml) 构建", f"{n_par / 1e6:.2f} M 参数")

    fake = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    res = model.predict(fake, device=0, verbose=False, conf=0.25)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    r = res[0]
    ok("predict(device=0)", f"返回 {len(res)} 个结果, boxes={tuple(r.boxes.data.shape)}")
    print(f"{'检测框为空':<24}: 正常（yaml 构建的是未训练模型）")
    # predictor 内部 AutoBackend 会 deepcopy 模型到目标设备，原 model.model 仍留在 cpu，这是设计如此
    dev = model.predictor.device
    print(f"{'predictor 实际设备':<24}: {dev}")
    print(f"{'推理峰值显存':<24}: {peak:.1f} MB")
    if str(dev).startswith("cuda") and peak > 0:
        ok("在 GPU 上推理", f"{dev}，峰值占用 {peak:.1f} MB")
    else:
        fail("在 GPU 上推理", f"device={dev}, peak={peak:.1f} MB")
except Exception as e:  # noqa: BLE001
    fail("ultralytics 推理", f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------- 结论
section("结论")
if FAIL:
    print("[FAIL] 有问题：")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("[OK] torchvision / OpenCV / ultralytics 全部就绪，可直接用于 YOLO 训练与图像处理。")
