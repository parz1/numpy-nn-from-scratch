# numpy-nn-from-scratch

只用 **NumPy** 从零学习显式反向传播。没有 PyTorch、TensorFlow、自动求导或现成训练器。
这是一个练习仓库：所有 `forward` / `backward`、SGD 更新与梯度清零、训练循环和评估计算均只保留函数签名、docstring 和 TODO 注释。初始化、接口、数据下载、CLI 和测试已经写好。

**当前状态：尚未实现，训练不会产生学习结果，核心测试应当失败。**
MNIST 测试准确率严格 `>95%` 是实现完成后的验收目标，并非此模板已达到的成绩。

## 安装与运行（PowerShell）

```powershell
cd numpy-nn-from-scratch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest tests/test_data.py -q
```

Python 3.10+；运行依赖只有 NumPy，开发测试依赖 pytest。若 PowerShell 禁止激活脚本，直接用 `.\.venv\Scripts\python.exe` 替代上述 `python` 即可。

## 文件结构

```text
nn/
  __init__.py       导出公共类
  layers.py         Linear、ReLU（forward/backward 待实现）
  loss.py           MSE、SoftmaxCrossEntropy（forward/backward 待实现）
  optim.py          SGD（step/zero_grad 待实现）
  data.py           完整下载、缓存校验、IDX 读取、归一化与 minibatch
train_toy.py         单隐藏神经元练习，默认 50 步
train_mnist.py       两层全连接网络、训练/评估练习和完整命令行
tests/
  gradcheck.py       完整中心差分工具
  test_layers.py     层的输入和参数梯度检查
  test_loss.py       损失梯度、归一化和稳定性
  test_optim.py      SGD 更新、清零、参数引用
  test_data.py       无网络依赖的数据测试
  test_training.py   组合梯度、玩具网络、小数据训练、可选 MNIST 验收
```

## 接口约定（先读）

1. 批次在第一个维度：`x` 为 `(N,D)`。`Linear.W` 为 `(D,H)`，`b` 为 `(H,)`；输出为 `(N,H)`。`bias=False` 时 `b`、`db` 为 `None`。
2. 层的 `backward(grad_output)` 返回输入梯度；`Linear` 同时把参数梯度写入 `dW`、`db`。调用 backward 前必须 forward，否则抛 `RuntimeError`。
3. `parameters()` 返回活的 `(参数数组, 梯度数组)` 引用。**在原数组上写入梯度**（如使用切片赋值），不要替换 `dW` / `db` 对象，否则提前构造的 SGD 会继续使用旧数组。每次 backward 覆盖梯度，不累积。
4. 损失 `forward` 返回标量，`backward()` 返回对预测值或 logits 的梯度。均值归一化只在损失处做一次；Linear 不再除以 batch size。
5. MSE 对所有元素取均值，无 `1/2` 系数。交叉熵对样本取均值，标签为整数 `(N,)`，输入为原始 logits `(N,C)`，不在最后一层额外调用 softmax。
6. ReLU 在零点的导数约定为 0；输入和上游梯度不得被修改。loss backward 重复调用应返回相同数值。
7. 所有层的 backward 完成后才调用 SGD；不能边传播边更新参数。

## 建议实现顺序

### 第 1 步：Linear

在 `nn/layers.py` 缓存输入，完成批量仿射变换。反向传播时分别考虑输入、权重、偏置的导数，以及偏置广播在反向时怎样还原。使用矩阵乘法和按轴求和，核对每一个结果的形状；参数梯度原地覆盖。

```powershell
python -m pytest tests/test_layers.py -k linear -q
```

### 第 2 步：ReLU

完成元素级前向与缓存。反向用缓存判断哪些位置允许梯度通过。不要用数值差分在零点判断“对错”：这里不可微，测试单独检查约定，其余数值检查刻意避开零点。

```powershell
python -m pytest tests/test_layers.py -q
```

### 第 3 步：MSE

检查预测和目标形状完全一致、非空，避免广播隐藏错误。先对每个元素写出平方误差的导数，再处理所有元素的均值。缓存前向所需信息。

```powershell
python -m pytest tests/test_loss.py -k mse -q
```

### 第 4 步：SGD

在 `nn/optim.py` 完成原地参数更新与原地梯度清零。学习率已在构造函数中检查；优化器持有数组引用，不要重新绑定局部变量代替更新。

```powershell
python -m pytest tests/test_optim.py -q
```

### 第 5 步：50 步玩具训练

在 `train_toy.py` 实现 `train_toy`：

- 网络为 `y_hat = v × ReLU(w × x + b)`；`x=2`、`y=10`、初值 `w=1`、`b=1`、`v=2`、`lr=0.01`。三个参数都训练。
- 用两个 `Linear(1,1)` 包住 ReLU；输出 Linear 设 `bias=False`，因此只代表 v，整个网络只有一个隐藏神经元。
- 每步依次清零、前向、MSE、反向、更新；打印更新前的 loss。恰好更新 50 次，最后额外前向一次取得 `final_loss`。
- 按 docstring 返回 losses 和最终 w/b/v；支持测试用的 `steps`、`lr`、`verbose` 参数。

手算检查：初始隐藏值 3，预测 6，MSE 为 16。第一步参数梯度应为 `dw=-32`、`db=-16`、`dv=-24`，更新后 `w=1.32`、`b=1.16`、`v=2.24`。这些值用于定位链式求导问题。

```powershell
python -m pytest tests/test_training.py -k toy -q
python train_toy.py
```

预期恰好打印 50 行 `step=001 loss=...` 到 `step=050 loss=...`，loss 快速下降。

### 第 6 步：稳定 softmax 交叉熵

在 `nn/loss.py` 实现 `SoftmaxCrossEntropy`：先验证二维非空 logits，以及形状正确、取值合法的整数标签。每行减去最大 logit，再用 log-sum-exp 计算负对数似然。不要先求概率再直接 `log`，目标概率极小时会下溢；不要裁剪概率来掩盖问题。

缓存概率和标签，推导 softmax 与交叉熵合并后的梯度，最后按样本数归一化。backward 操作副本，避免破坏缓存。

```powershell
python -m pytest tests/test_loss.py -q
```

### 第 7 步：两层网络和通用训练循环

在 `train_mnist.py` 按顺序完成：

1. `forward`：依次通过模型各层，返回 logits。
2. `backward`：按相反顺序传递梯度，返回输入梯度。
3. `evaluate`：不打乱数据，分批计算 CE 与 argmax 正确数。用实际样本数加权 batch loss，不能直接平均各批次均值；不执行反向或更新。
4. `train`：只接收训练集。创建一个固定 seed 的 Generator，在各 epoch 持续使用它打乱数据；按 `zero_grad → forward → loss → backward → step` 训练。包含最后不足 batch size 的批次，每 epoch 按样本数加权 loss 并记录/打印。

```powershell
python -m pytest tests/test_training.py -k "composed or evaluate or small" -q
python -m pytest -q
```

默认测试离线运行，不下载 MNIST。所有普通测试通过后，再进行完整数据训练。

### 第 8 步：MNIST 验收

完整数据管道已经实现，可提前单独运行下载：

```powershell
python train_mnist.py --download-only
python train_mnist.py --epochs 20 --batch-size 128 --hidden-size 128 --lr 0.1 --seed 42
```

默认网络为 `784 → Linear(128) → ReLU → Linear(10)`，两个全连接层；权重使用 He 初始化，偏置为零，计算用 float32。图像除以 255 后范围为 `[0,1]`；标签用 int64。训练集与测试集独立，测试集只在训练结束用于报告，不能参与参数更新或超参数选择。

这组配置是达到 `>95%` 的起点；模板未包含已实现网络或准确率实测。完成后脚本会输出 test loss/accuracy；若准确率 `<=95%`，以非零退出码提示未达标。

完整 pytest 验收（会下载并训练，耗时取决于 CPU / NumPy BLAS）：

```powershell
$env:RUN_MNIST = "1"
python -m pytest -m mnist -q -s
Remove-Item Env:RUN_MNIST
```

可用 `--data-dir PATH` 更改缓存位置，用 `--no-download` 强制只读取缓存；pytest 的缓存目录可通过 `MNIST_DATA_DIR` 环境变量设置。数据由 [CVDF 的 MNIST 镜像](https://github.com/cvdfoundation/mnist) 下载；四个 gzip 文件逐一校验 MD5、解析 IDX。支持缓存复用和损坏文件重新下载。离线时可把对应四个原始 gzip 文件放入缓存目录。

如果准确率不足，先确认普通测试通过，检查 batch 均值、参数引用、标签对应、训练是否覆盖全部样本和 loss 是否下降。确需调参时，从训练集中留出验证集，用验证结果选择学习率、epoch 数或隐藏宽度，再最终评估测试集。

## 数值梯度检查怎样工作

对某个元素作 `+ε` 和 `-ε` 扰动，用中心差分 `(f(θ+ε)-f(θ-ε))/(2ε)` 与 backward 的结果比较。测试使用 float64、`ε=1e-6`，每次扰动后恢复原值。
层的输出不是标量，所以测试把输出与随机上游梯度作内积，将其变成标量目标；这样检查的是实际反向传播需要的向量-Jacobian 乘积。Linear 检查输入、W、b；ReLU 检查输入；两种损失检查预测/logits；组合网络额外检查链式传播的输入及全部参数梯度。

容差为 `rtol=2e-5, atol=2e-7`。初始失败是练习尚未完成的正常现象，只有完整 MNIST 验收默认跳过；不要把未实现测试标成 xfail 或放宽容差来绕过失败。
