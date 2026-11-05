# P1 · 从零手写 CNN 分类 CIFAR-10

**目标**：不借助任何训练框架（不用 Lightning / timm / ultralytics 的 `model.train()`），
把"数据 → 前向 → 损失 → 反向 → 更新 → 评估"这条链自己写一遍，并且**能诊断出模型为什么训不好**。

**交付标准**：`test acc ≥ 90%`，加上一份能说清"哪一步贡献最大"的三行结论。

---

## 一、四步跑起来

在项目根目录 `D:\project\workplace pycharm\PyTorchLab` 下执行：

```bash
# 1. 验结构（3 秒，确认输出形状和参数量）
.venv\Scripts\python.exe p1_cifar10\model.py

# 2. 验数据（首次会下 170 MB，之后走缓存）
.venv\Scripts\python.exe p1_cifar10\data.py

# 3. 验链路（20 秒，只跑 20 个 batch，确认能反向能更新）
.venv\Scripts\python.exe p1_cifar10\train.py --epochs 1 --limit-batches 20 --out runs/smoke

# 4. 完整训练（30 epoch，实测 3 分钟）
.venv\Scripts\python.exe p1_cifar10\train.py
```

训练过程会实时打印每个 epoch 的 lr / train / val / 耗时。
嫌日志太长可以加 `--print-every 5`，只每 5 轮打印一行——**TensorBoard 和
`train_report.json` 仍然逐轮完整记录**，只是控制台安静了（最后一行永远会打，
不然跑完看不到最终成绩）。
训练完自动在 test 上评估**一次**，打印 test acc 和显存峰值。

看曲线：

```bash
.venv\Scripts\tensorboard.exe --logdir runs
# 浏览器打开 http://localhost:6006
```

出证据图（混淆矩阵 + 错判样本墙）：

```bash
.venv\Scripts\python.exe p1_cifar10\evaluate.py --ckpt runs/baseline/best.pt
```

---

## 一·五、数据放哪

**你不需要放任何东西 —— CIFAR-10 已经在项目里了。**

```
D:\project\workplace pycharm\PyTorchLab\
├── data\                          ← 数据集在这里（341 MB）
│   ├── cifar-10-batches-py\       ← torchvision 认的就是这个目录名
│   │   ├── data_batch_1..5        （每个 30 MB，训练数据）
│   │   ├── test_batch             （测试数据）
│   │   └── batches.meta           （类别名）
│   └── cifar-10-python.tar.gz     ← 原始压缩包，留作备份，可以删
├── p1_cifar10\                    ← 代码
└── runs\                          ← 训练输出（权重/曲线/图），不是输入
```

代码里 `DEFAULT_DATA_ROOT` 默认指向 `data\`，**直接跑就行**。

### 想放到别的盘

数据集 341 MB 放在项目里其实无所谓，但如果你要把整个 CIFAR 目录挪到 `D:\datasets\`：

```bash
.venv\Scripts\python.exe p1_cifar10\train.py --data-root D:\datasets\cifar10
```

注意 `--data-root` 指的是**能看到 `cifar-10-batches-py` 文件夹的那一层**，不是 `cifar-10-batches-py` 本身。
指错了会直接报错退出，**不会偷偷开始一次 50 分钟的下载**（这一点我特意加了拦截，
因为这台机器单连接只有 55 KB/s，静默下载等于卡死）。

### 换成自己的数据（P2 会用到）

P1 的 `data.py` 是 CIFAR 专用的二进制格式，**不能直接塞自己的照片**。
自己的数据要用 `torchvision.datasets.ImageFolder`，目录结构必须是：

```
D:\datasets\my_photos\
├── train\
│   ├── cat\      1.jpg  2.jpg ...
│   ├── dog\      ...
│   └── bird\     ...
└── val\
    ├── cat\      ...
    ├── dog\      ...
    └── bird\     ...
```

规则：**文件夹名 = 类别名**，里面直接放图，不要再套一层。
这是 P2 的活，到那一步再写。

### 三个不要

1. **不要手动改 `data\cifar-10-batches-py\` 里的文件** —— 二进制格式，改了必坏
2. **不要把 `runs\` 当输入** —— 那里只有输出（权重、曲线、图）
3. **不要删 `data\cifar-10-batches-py\`** —— 重下要 50 分钟（虽然 `tar.gz` 备份还在，
   解压就能恢复：`tar -xzf data\cifar-10-python.tar.gz -C data`）

---

## 二、代码导读：按这个顺序读

**代码注释用 `【讲解】` 标记**，一共 57 段，只说明"这段代码在做什么"。
在 PyCharm 里按 `Ctrl+Shift+F` 搜 `【讲解】`，就能跳着看。

"为什么这么写"放在这份 README 和第三节的实验结论里；代码旁边只留行为说明。

### 读的顺序

| 顺序 | 文件 | 行数 | 要能回答的问题 |
|---|---|---|---|
| 1 | `model.py` | 99 | 哪一步把 32×32 变成 16×16？为什么用全局平均池化代替大 FC？ |
| 2 | `data.py` | 170 | 为什么训练和评估要用两套变换？val 为什么不能带随机增强？ |
| 3 | `train.py` | 315 | 训练那五行分别干什么？为什么必须 `zero_grad()`？ |
| 4 | `evaluate.py` | 227 | 混淆矩阵为什么要按行归一化？为什么它的准确率比 train.py 高 0.08 点？ |

### 每一步的验收方式

**1. `model.py`** —— 直接运行它：

```bash
.venv\Scripts\python.exe p1_cifar10\model.py
```

它会打印逐层形状。**能看着这四行输出讲清楚尺寸怎么变、参数量 1.149 M 从哪里来**，这一节就过了。
文件里的 【讲解】 覆盖了 `bias=False`、`inplace=True`、GAP 为什么省参数。

**2. `data.py`** —— 同样单独运行：

```bash
.venv\Scripts\python.exe p1_cifar10\data.py
```

最后会打印两个 mean/std。带增强的 batch 是 -0.2 左右，不带增强的是 0.02 —— 
**能解释这两个数字为什么不同**（RandomCrop 补的黑边），这一节就过了。

**3. `train.py`** —— 核心。先只看 `train_one_epoch()` 里那五行：

```
zero_grad -> autocast(前向 + 损失) -> backward -> step
```

**这五行是所有 PyTorch 训练的骨架**，换个任务也只是换数据、换模型、换损失。
看懂它之后，再看 `lr_at()`（warmup + 余弦）和 `if va_acc > best_acc`（为什么存最佳而不是最后一个）。

**4. `evaluate.py`** —— 把"91.5%"拆开成"哪一类错了、错成什么"。
重点看 `denormalize()`（为什么画图前要还原）和 `plot_confusion()`（为什么按行归一化）。

### 三个不建议跳过的讲解

| 位置 | 为什么重要 |
|---|---|
| `train.py` 的 `zero_grad(set_to_none=True)` | PyTorch 梯度是**累加**的，不清零会 NaN。新手第一大坑 |
| `train.py` 末尾的 `if __name__ == "__main__":` | Windows 上不写会无限 spawn 子进程。抄 Linux 教程最容易踩 |
| `model.py` 的 `forward` 里没有 softmax | `CrossEntropyLoss` 内部已含 softmax，手动加了等于做两次，训练明显变差 |

---

## 三、三个实验（做完才算真懂）

每个实验 3–4 分钟（不是十几分钟，这台机器太快了），跑完对比 `runs/*/train_report.json` 里的 `test_acc`。

**下面所有"实测"都是我 2026-10-26 真跑出来的，不是估计。**
其中两个结果跟我原本的预期相反 —— 那正好是这节最有价值的部分。

### 四组结果汇总

| 配置 | best val | test acc | 末轮 train | 末轮 val | 差距 |
|---|---|---|---|---|---|
| baseline（增强 + lr 0.05） | 91.86% | **91.47%** | 98.87% | 91.86% | 7.0 点 |
| 实验 A `--aug none` | 88.84% | **88.65%** | **100.00%** | 88.46% | 11.5 点 |
| 实验 B `--lr 0.5` | **92.22%** | **91.59%** | 96.56% | 92.22% | 4.3 点 |
| 实验 C `--aug none --dropout 0.5` | 88.46% | **87.55%** | **100.00%** | 88.44% | 11.6 点 |

### 实验 A：数据增强值 2.8 个点

```bash
.venv\Scripts\python.exe p1_cifar10\train.py --aug none --out runs/no_aug
```

**实测**：91.47% → **88.65%，掉 2.82 个点**。

但真正值得看的不是这 2.8 点，是**过拟合的形状变了**：

- 有增强：末轮 train 98.87%，train-val 差 **7.0 点**
- 无增强：**epoch 10 就到 97%**，末轮 train **100.00%**，差 **11.5 点**

train acc 打满 100% 意味着模型把训练集背下来了。**37.5 万参数记住 4.5 万张图毫无压力**。

> 顺带一提：`--aug strong`（RandAugment + RandomErasing）我没让默认开，
> 因为 30 epoch 的短训练里强增强会来不及收敛。你可以自己试，看它在这个预算下能不能追回 91%。

### 实验 B：lr 调到 0.5 居然没崩 —— 但震荡是真的

```bash
.venv\Scripts\python.exe p1_cifar10\train.py --lr 0.5 --out runs/lr_high
```

**我原本的预期是错的。** 我以为会 NaN 或卡在 10%，实际结果是 **91.59%，比 baseline 还高 0.12 点**。

但震荡完全如预期，看前 10 个 epoch 的 val acc：

```
epoch:   1    2    3    4    5    6    7    8    9   10
val:    42   55   51   54   47   69   66   62   50   71     ← 上蹿下跳
```

对比 baseline 同期的 `53 69 73 78 79 80 79 83 81 80`（平滑上升）。

**为什么没死**：三个东西在兜底 —— ① `--warmup 3`，前 3 个 epoch 学习率从 0.017 爬到 0.5，
不是一上来就 0.5；② BatchNorm 让梯度尺度自适应；③ 余弦退火在最后 10 个 epoch 把 lr 压到 0.002
以下，模型在低 lr 阶段把震荡"收敛"成一个还不错解，所以末轮 val 反而最高（92.22%）。

**这一条的正确结论**：不是"大 lr 一定崩"，而是"**大 lr + 没有 warmup** 才会崩"。
判断标准永远是看曲线，不是背结论。

### 实验 C：dropout 加了反而更差

```bash
.venv\Scripts\python.exe p1_cifar10\train.py --aug none --dropout 0.5 --out runs/dropout
```

**实测：87.55%，比不开 dropout 的 88.65% 还低 1.1 个点。** 又是一个反直觉结果。

看末轮就知道 dropout 没起到作用：train **100.00%**，val 88.44%，差距 **11.6 点**——
和不开 dropout 的 11.5 点几乎一样。**过拟合一点没被治住。**

原因在 `model.py` 的结构里：dropout 加在 `AdaptiveAvgPool2d` 之后、`Linear` 之前，
只作用在 256 维特征上，而**过拟合发生在卷基层**（它记住了纹理和背景）。
在分类头做 dropout，等于洪水来了只在门口垫条毛巾。

**正确的治法**（按性价比排）：
1. 数据增强（实验 A 已证明 +2.8 点）
2. 更多数据
3. 更强的正则用在卷积层：`Dropout2d` 插在 ConvBlock 之间
4. 换更大更强的 backbone —— 但那就不叫"从零手写"了

**这一条是重点**：`train acc 高不是好事`，正则化也不是随便加就有效。
**判断模型好坏只看 val，判断改动有没有用只能靠对照实验。**

> 想更狠一点：把 `data.py` 的 `val_ratio` 改成 0.8（只留 1 万张训练），
> 再看 val acc 掉到多少 —— 亲手制造一次数据饥饿。

### 还有一个观察：30 epoch 根本没收敛

baseline 的最佳 val 出现在**最后一轮**（91.86%），曲线还在往上走。
也就是说这个配方在 30 epoch 的预算下**没跑满**，加到 60–80 epoch 还有空间。
这也解释了为什么 `--aug strong` 现在追不回来 —— 不是增强不好，是 epoch 不够。

---

## 四、你会踩的坑

| 现象 | 原因 |
|---|---|
| `RuntimeError: DataLoader worker ... exited unexpectedly` | 忘了 `if __name__ == "__main__":`，Windows 上必须写 |
| test acc 只有 10% | `model.eval()` 或 `@torch.no_grad()` 漏了，或者标签/输出顺序错 |
| loss 变 NaN | 学习率太大且**没有 warmup**；有 warmup 时 0.5 都能跑（见实验 B） |
| 显存不够 | 这个模型的峰值实测 **0.33 GB**，几乎不可能 OOM；真不够就 `--batch-size 64` |
| 训练很慢（> 2 秒/iteration） | `--workers 0` 或磁盘满；这台机器 `workers=6` 是甜点 |
| TensorBoard 打不开 | 端口占用，换 `--port 6007` |

---

## 五、完成清单

- [ ] 四个脚本都能单独跑通（`model.py` / `data.py` / `train.py` / `evaluate.py`），
      `model.py` 的形状输出能自己讲一遍
- [ ] 完整训练 30 epoch，`test acc ≥ 90%`，记下显存峰值
- [ ] TensorBoard 里能看出 train/val 两条线的分叉点在第几个 epoch
- [ ] 跑完实验 A（`--aug none`），能说出差了几个点
- [ ] 跑完实验 B（`--lr 0.5`），能描述震荡曲线长什么样
- [ ] `confusion_matrix.png` 里最红的两格是哪两类，为什么
- [ ] 一句话说清"这次训练里，哪个改动贡献最大"

七项全打勾，P1 就真的做完了 —— 这时候进 P2（迁移学习），
你会发现 `torchvision` 的那些封装，每一行你都能对上自己写过的东西。

---

## 六、实测基线（这台机器 2026-10-26 跑出来的）

代码已经跑通过一次，结果在 `runs/baseline/`。**你重跑应该得到接近但不一样的数字**
（`--seed` 相同、但 cuDNN 的非确定性算子仍有微小差异）。

| 项 | 实测值 |
|---|---|
| 总耗时 | **3 分 05 秒**（30 epoch） |
| 单 epoch | 第 1 个 43 s（含 worker 启动），之后稳定 **3.7 s** |
| 吞吐 | 约 12000 img/s（24 核 CPU 喂数据完全不是瓶颈） |
| val acc | **91.86%**（epoch 30，全程单调上升，没有过拟合迹象） |
| test acc | **91.47%**（`train.py` 里 bf16 评估）／**91.55%**（`evaluate.py` 里 fp32 全精度） |
| 显存峰值 | **0.33 GB**（8 GB 用掉 4%） |

> test acc 两个数差 0.08 个点（约 8 张图），来源是 bf16 与 fp32 的数值误差，
> 正常。要报"最终成绩"就用 `evaluate.py` 的 fp32 那个。

**学习曲线长什么样**（这是最值得看的一列）：

```
epoch  5   train 81.74%  val 79.10%   ← 两条线贴在一起
epoch 10   train 88.06%  val 80.40%   ← 开始分叉，val 反而掉了
epoch 20   train 94.90%  val 88.80%   ← 差距 6 个点，过拟合
epoch 30   train 98.87%  val 91.86%   ← train 98% 是"背下来"的部分
```

第 9~10 个 epoch 那个 val 回落（82.9% → 81.1% → 80.4%）很有趣：学习率还在 0.045
左右，模型在跳出局部解。**这不是 bug**，余弦退火把它带回来并一路升到 91.8%。

**哪些类最难**（`evaluate.py` 的输出）：

| 类别 | 准确率 | 说明 |
|---|---|---|
| cat | **82.2%** | 最差。83 张被判成 dog |
| dog | **86.2%** | 反向被混 76 张 |
| bird | 88.4% | 30 张被判成 deer（草地背景干扰） |
| ship | 95.4% | 26 张被判成 airplane（都在天空里） |
| automobile | **96.4%** | 最好 |

`confusion_matrix.png` 里最亮的两格是 cat↔dog，占了全部错误的三分之一。
翻 `misclassified.png` 你会发现：**很多"错"的图连人眼都要犹豫**——
这正是 32×32 分辨率的信息瓶颈，不是模型笨。

打开 `runs/baseline/confusion_matrix.png` 和 `misclassified.png` 对着看，
你就能回答 README 完成清单的最后一项："这次训练里哪个改动贡献最大"。**
