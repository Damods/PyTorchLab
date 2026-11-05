# P2 · 迁移学习：训练你自己的照片分类器

**目标**：拿 ImageNet 上训好的 ResNet，只教它"你这几类怎么分"。
200 张照片就能跑出可用结果 —— 这件事 P1 那种从零训练是做不到的。

**交付标准**：每类准确率看得懂、混淆矩阵能说出"该补哪类的数据"、
能讲清楚预训练到底值多少个点。

---

## 一、现在的状态（重要）

`data/my_photos/` 里已经建好 5 个类别文件夹（cup / bottle / book / shoe / keys），
**但还没有图片**。所以你第一次跑会看到：

```
[数据不完整] Found no valid file for the classes book, bottle, cup, keys, shoe.
最常见的原因：某个类别文件夹里没有图片。
```

这是预期行为。**在放照片之前，先用合成数据验证整条流程**：

```bash
# 生成 5 类合成图（每类 40 张，含 15 张 heic 用来验证 HEIC 链路）
.venv\Scripts\python.exe p2_finetune\make_demo_data.py

# 体检
.venv\Scripts\python.exe p2_finetune\check_data.py --data-root data/_demo

# 训练
.venv\Scripts\python.exe p2_finetune\train.py --data-root data/_demo --out runs_p2/demo

# 出图
.venv\Scripts\python.exe p2_finetune\evaluate.py --ckpt runs_p2/demo/best.pt
```

`data/_demo` 是**验证工具，不是你的项目数据**。照片拍好后换成：

```bash
.venv\Scripts\python.exe p2_finetune\check_data.py          # 默认扫 data/my_photos
.venv\Scripts\python.exe p2_finetune\train.py              # 默认也是 data/my_photos
```

---

## 二、照片怎么放

```
data\my_photos\           ← 默认目录
├── cup\
│   ├── 001.jpg
│   └── 002.jpg
├── bottle\
├── book\
├── shoe\
└── keys\
```

- **文件夹名 = 类别名**，图直接放进去
- 每类 **30~50 张**，少于 20 张学不出东西
- 5~10 类最合适
- jpg / png / webp / bmp / **heic** 都能读（已装 pillow-heif 并打了扩展名补丁）
- **不用手动分 train/val**，代码按类别分层切 8:2

放好先跑 `check_data.py`，它会告诉你：每类多少张、有没有损坏文件、
有没有重复图、类别是否失衡、有多少 heic。

---

## 三、代码导读

代码注释用 **`【讲解】`** 标记，共 38 段，只说明"这段代码在做什么"。PyCharm 里 `Ctrl+Shift+F` 搜 `【讲解】`。
（"为什么这么做"的来龙去脉在本文档里，不在代码里。）

| 顺序 | 文件 | 行数 | 主要内容 |
|---|---|---|---|
| 1 | `data.py` | 248 | 手机照片兼容、变换、分层切分、DataLoader |
| 2 | `model.py` | 80 | 换掉 fc 层、参数分组、冻结 backbone、切 BN |
| 3 | `train.py` | 285 | 两段式训练、两组学习率、AMP、存 checkpoint |
| 4 | `evaluate.py` | 255 | 混淆矩阵、误判样本墙、每类指标 |
| 5 | `check_data.py` | 186 | 训练前的数据体检 |
| — | `make_demo_data.py` | 120 | 生成合成数据验证管道 |

**三个最容易写错的地方**（注释里各有说明）：

1. 冻结阶段必须在**前向之前**把 BN 切 eval（写在 forward 之后等于没做）
2. `IMG_EXTENSIONS` 是 tuple，只能重绑定不能 append
3. 验证集的切分必须和训练时**完全一致**（evaluate.py 从 checkpoint 读 `val_ratio` 和 `seed`）

---

## 四、四个实验（都已实跑，数字在下面）

```bash
# A. 从零训练（不用预训练权重）—— 看预训练值多少
.venv\Scripts\python.exe p2_finetune\train.py --data-root data/_demo --scratch --out runs_p2/scratch

# B. 全程冻结，只训分类头（等价 linear probe）
.venv\Scripts\python.exe p2_finetune\train.py --data-root data/_demo --freeze-epochs 10 --out runs_p2/freeze_all

# C. 全程解冻，一上来就全量微调
.venv\Scripts\python.exe p2_finetune\train.py --data-root data/_demo --freeze-epochs 0 --out runs_p2/full_all

# D. 开强增强
.venv\Scripts\python.exe p2_finetune\train.py --data-root data/_demo --strong-aug --out runs_p2/strong_aug
```

### 实测结果（合成数据，5 类 200 张，10 epoch）

| 配置 | 最佳 val | 末轮 val | 说明 |
|---|---|---|---|
| **预训练 + 两段式**（`--freeze-epochs 3`） | **100.0%** | 100.0% | 先冻 3 轮再解冻 |
| 预训练 + 全程冻结（`--freeze-epochs 10`） | 95.0% | 95.0% | 只训那一个 Linear 层 |
| 预训练 + 全程解冻（`--freeze-epochs 0`） | 100.0% | 100.0% | 直接全量微调 |
| **从零训练**（`--scratch`） | **27.5%** | 25.0% | 5 类随机猜 = 20% |

**第一眼就该看第 1 行和第 4 行的差距：100% vs 27.5%。**

从零训练在 200 张图上基本停在"瞎猜"水平，它连"什么是边缘"都还没学会；
预训练模型第 3 轮（还完全冻结着，只训了一个 2565 参数的线性层）就到 95%。
**这就是迁移学习的全部意义。**

> 两段式（默认）和"直接全解冻"在这份合成数据上都是 100%，看不出差别——
> 因为形状任务太简单了。两段式的优势在**真实照片 + 更少样本**时才明显：
> 先让随机初始化的 head 收敛，再解冻 backbone，避免大梯度把预训练特征带偏。
> 你用自己的照片跑 A/B/C 三组，大概率能看到差别。

### 从零训练失败的样子（值得看一眼）

`runs_p2/scratch/confusion_matrix.png` 里，从零训练的模型**把几乎所有东西都判成 square**：
circle→square 5 张、triangle→square 5 张、stripe→square 5 张。

这叫 **mode collapse**（模式崩塌）：小数据 + 从零训练时，模型发现
"全猜成最常见的那类"就能让 loss 降一点，于是放弃学习特征。
准确率 27.5% 看起来比 20% 好，其实是假的。

---

## 五、性能与显存（实测）

| 模型 | batch | 每步 | 吞吐 | 显存峰值 |
|---|---|---|---|---|
| ResNet18 | 32 | 29 ms | 1118 img/s | 0.60 GB |
| ResNet50 | 32 | 87 ms | 367 img/s | 1.64 GB |

8 GB 显存非常宽裕，**ResNet50 + batch 64 也放得下**。别怕设大 batch。

200 张照片跑 10 epoch 约 1 分钟（第一个 epoch 有 8~10 秒启动开销）。

---

## 六、你会踩的坑

| 现象 | 原因 |
|---|---|
| `Found no valid file for the classes ...` | 某个类别文件夹是空的 |
| 准确率 20% 左右不动 | 用了 `--scratch`，或标签/类别目录搞错了 |
| 准确率 99% 但换张图就错 | 每类照片背景太单一，模型学的是背景 |
| `DataLoader worker ... exited` | 自己写的脚本漏了 `if __name__ == "__main__":` |
| 样本数比实际少几张 | .heic 被 ImageFolder 静默跳过（本项目的 data.py 已打补丁，自己另写代码要注意） |
| TensorBoard 上曲线叠成两条 | 重跑时没换 `--out` |

---

## 七、完成清单

- [ ] `check_data.py` 跑一遍，能说出每类有多少张
- [ ] 合成数据上跑通训练 + 评估，`val acc ≥ 95%`
- [ ] 跑 `--scratch` 对照，能说出差了多少个点
- [ ] 拍好照片，`check_data.py` 报 OK
- [ ] 用自己照片训练，`val acc ≥ 85%`
- [ ] 打开 `confusion_matrix.png`，指出最混淆的两类，并解释为什么
- [ ] 打开 `misclassified.png`，找一张"高置信度却判错"的图，猜原因
- [ ] 一句话说清：**这次训练里，预训练贡献了多少，数据质量贡献了多少**

---

## 八、真实数据实测结果（2026-10-27，210 张 / 5 类）

数据来自 Wikimedia Commons（161 张，自由许可）+ Open Images V5（49 张，bbox 裁出的单物体图）。
来源、作者、许可记录在 `data/my_photos/_sources.csv`。

### 四组对照

| 配置 | 最佳 val | 末轮 val | 说明 |
|---|---|---|---|
| **预训练 + 两段式**（默认） | 87.80% | 87.80% | 训练准确率 99.4% —— 明显过拟合 |
| 预训练 + 全程冻结（`--freeze-epochs 15`） | 87.80% | 85.37% | 冻住 backbone 也没更好 |
| **预训练 + 强增强**（`--strong-aug`） | **90.24%** | 87.80% | book 召回 62.5% → 87.5% |
| 从零训练（`--scratch`） | **36.59%** | 26.83% | 5 类随机猜 = 20% |

**预训练 vs 从零：90.24% vs 36.59%，差 53.6 个点。**

### ⚠️ 但 90.24% 和 87.80% 不能算"提升"

验证集只有 41 张，**1 张图就是 2.44 个百分点**。这两个配置的差别就是 1 张图，
完全在噪声范围内。真实结论只能是：

> 强增强**方向正确**（book 类召回从 5/8 提到 7/8），但这么小的验证集证明不了它更好。
> 想证明，得把每类扩到 100 张以上。

这是小数据集的固有问题：**指标分辨率不够**。±2~3 个点以内的差异，别看，看也没用。

### 诊断：问题集中在 book 类

混淆矩阵里 book 只有 62% 召回（基线版本），判错的 3 张是：

| 判错样本 | 被判成 | 置信度 | 原因 |
|---|---|---|---|
| 书页边缘特写 | shoe | 0.32 | 只剩纸张纹理，看不出"书" |
| 一摞书（红/棕书脊） | keys | 0.36 | 侧面视角，和"深色金属小物"混了 |
| WORLD ATLAS 深色封面 | keys | 0.43 | 深色封面 + 图案，被当成了杂物 |

**共同点：置信度全都低于 0.5。** 模型不是"自信地错"，而是"拿不准"。
这说明这些图本身对模型就是模糊的 —— book 类内部形态太散：
特写书页、书脊侧面、封面、整摞书，四种视角混在一个类里。

反过来，keys 和 shoe 都是 100%，因为它们的形态高度一致。

### 另一个信号：疑似标签噪声

误判墙里有一张标成 `bottle` 的图，内容是**一堆金属瓶盖/小瓶**（判成 keys，置信度 0.46）。
如果那不是瓶子，就是标错了。**建议人工复查 bottle 类里那几张**——
数据集的标签错误会稳定地拉低上限，而且模型学不到任何东西。

### 想再往上走的三个方向（按性价比）

1. **补 book 类，但要补"形态统一"的** —— 如果统一用封面照，就别混书脊特写。
   类别内部的形态一致性，比数量更重要。
2. **复查 bottle 类的边界** —— 瓶盖、小药瓶算不算 bottle？先定义清楚。
3. **把验证集做大** —— 每类 100 张以上，指标才有 ±1% 的分辨率，才谈得上比较配置。

> 数据从 210 张加到 500 张，比换 ResNet50 或调 lr 有用得多。
> 这不是鸡汤：`--scratch` 那组已经证明了，瓶颈从来不在模型。
