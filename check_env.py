"""环境自查：确认 PyTorch 能用上 RTX 5060 的 GPU 算力。

在 PyCharm 里右键 Run 这个文件即可。全部 OK 才说明装对了。
"""

import sys
import time


def line(title):
    print("\n" + "=" * 46)
    print(title)
    print("=" * 46)


line("Python")
print("解释器 :", sys.executable)
print("版本   :", sys.version.split()[0])

line("PyTorch")
try:
    import torch
except ImportError:
    print("[X] 没有装 torch。请在 PyCharm 的 Terminal 里执行：")
    print('    python -m pip install torch --index-url https://download.pytorch.org/whl/cu130')
    sys.exit(1)

print("torch  :", torch.__version__)
print("编译CUDA:", torch.version.cuda)
print("线程数 :", torch.get_num_threads())

if "+cu" not in torch.__version__ and "+cpu" in torch.__version__:
    print("[X] 这是纯 CPU 版！说明装错了包，GPU 用不上。")

line("CUDA 可用性")
ok = torch.cuda.is_available()
print("is_available :", ok)

if not ok:
    print("[X] CUDA 不可用。常见原因：")
    print("    1) 装的是 CPU 版 torch（PyPI 默认那个 118MB 的包）")
    print("    2) 显卡驱动过旧")
    sys.exit(1)

idx = torch.cuda.current_device()
name = torch.cuda.get_device_name(idx)
prop = torch.cuda.get_device_properties(idx)
print("设备     :", name)
print("算力     : sm_%d%d" % (prop.major, prop.minor))
print("显存总量 : %.1f GB" % (prop.total_memory / 1024 ** 3))
free, total = torch.cuda.mem_get_info()
print("当前可用 : %.1f GB / %.1f GB" % (free / 1024 ** 3, total / 1024 ** 3))

line("真实算子测试（这一步才算数）")
# is_available() 只检查驱动握手，不检查内核里有没有 sm_120 的代码。
# 真的跑一次浮点矩阵乘法，才能证明内核镜像里有你的架构。
try:
    a = torch.randn(3, 3, device="cuda")
    print("randn().cuda()        : OK   sum=%.4f" % a.sum().item())

    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    y = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    z = x @ y
    print("fp16 1024x1024 矩阵乘  : OK   mean=%.5f" % z.mean().item())

    torch.cuda.synchronize()
except RuntimeError as e:
    msg = str(e)
    print("[X] 算子执行失败：", msg)
    if "no kernel image" in msg:
        print("    -> 这就是 sm_120 内核缺失。必须用 cu128 或 cu130 的 wheel，")
        print("       cu126/cu124 不支持 RTX 50 系。")
    sys.exit(1)

line("NumPy 互操作")
try:
    import numpy as np
    print("numpy  :", np.__version__)
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = torch.from_numpy(arr).cuda()
    print("numpy -> tensor -> cuda : OK   sum=%.1f" % t.sum().item())
except ImportError:
    print("[!] 没装 numpy。torch.from_numpy() 和 tensor.numpy() 会不可用。")
    print('    补装：python -m pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple')

line("速度参考")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(200):
    x @ y
torch.cuda.synchronize()
dt = time.time() - t0
flops = 200 * 2 * 1024 ** 3
print("200 次 fp16 1024³ matmul: %.2f 秒" % dt)
print("实测吞吐               : %.2f TFLOPS" % (flops / dt / 1e12))

line("结论")
print("[OK] 环境正常，PyTorch 可以调用 RTX 5060 的 CUDA 核心。")
print("     记得训练前关掉占显存的程序（浏览器、Obsidian 等）。")
