---
title: Torch
categories:
  - Python
tags:
  - Python
  - Torch
  - Machine Learning
date: 2025-03-11 15:26:53
updated: 2025-03-25 11:18:12
toc: true
mathjax: true
description: 
---

##  PyTorch

-   *PyTorch*：基于 *GPU*、*CPU* 用于
    -   *PyTorch* 核心功能（模块）
        -   `torch.Tensor` 张量
        -   `torch.autograd` 自动微分
        -   `torch.nn` 网络模块
        -   `torch.nn.functional` 函数
        -   `torch.optim` 优化器
    -   *PyTorch* 特点
        -   动态图计算
        -   支持在不同设备之间移动张量
        -   `torchvision`、`torchaudio`、`torchtext` 等包生态

### `torch.Tensor` 张量

-   `torch.Tensor` 张量：类似 `np.ndarray` 记录模型输入、输出、参数的数据结构，支持包括 GPU 在内的硬件加速
    -   `Tensor` 支持类似于 `np.ndarray` 的各种 API
        -   `torch` 命名空间下类似 `np` 的各种初始化函数
        -   类似 `np.ndarray` 的索引、切片、连接操作
        -   支持算术运算、矩阵运算、代数运算
        -   支持在位运算（在位运算方法名常为普通运算方法名后跟 `_` 后缀）
    -   `Tensor` 运算支持 `CUDA`、`MPS`、`MTIA`、`XPU` 等加速器
        -   `Tensor` 默认创建在 *CPU* 上
        -   （需）可通过 `.to` 方法显式移至加速器
    -   *CPU* `Tensor` 可与 `np.ndarray` 共享底层存储（同步修改）
        -   通过 `Tensor.numpy`、`torch.from_numpy` 在 `Tensor`、`np.ndarray` 间切换

> - *Tensors*：<https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html>

### `torch.autograd` 自动微分

| 微分属性、方法                                    | 描述                                            |
|---------------------------------------------------|-------------------------------------------------|
| `autograd.backward(tensors, grad_tensors, ...)`   | 反向传播                                        |
| `autograd.grad(outputs, inputs, ...)`             | 计算梯度                                        |
| `Tensor.grad`                                     | 梯度                                            |
| `Tensor.grad_fn`                                  | 反向传播函数                                    |
| `Tensor.requires_grad`                            | 梯度跟踪（计算）标志                            |
| `Tensor.is_leaf`                                  | 叶子节点标志，直接创建、`requires_grad` 置位    |
| `Tensor.requires_grad_()`                         | 切换梯度跟踪标志                                |
| `Tensor.detach()`                                 | 创建张量副本（脱离原图）                        |
| `Tensor.detach_()`                                | 从图中分离张量（即置为叶子节点，不再跟踪梯度）  |
| `Tensor.register_hook(hook)`                      | 注册反向传播钩子（钩子函数入参为 `.grad` 梯度） |
| `Tensor.register_post_accumulate_grad_hook(hook)` | 注册（反向传播中）梯度累加完后后钩子            |
| `Tensor.retain_grad()`                            | 保留（非叶子）节点梯度                          |
| `Tensor.backward([gradient,...])`                 | 反向传播                                        |
| `torch.no_grad()`                                 | 禁止梯度跟踪（上下文环境）                      |

-   `torch.autograd` 自动微分模块：*PyTorch* 内建的微分引擎，支持任意（计算） *DAG* 的自动微分
    -   从根节点回溯（反向传播）至叶子节点，即可通过链式法则自动计算梯度
        -   图中叶子节点即输入张量，根节点即输出张量
        -   自动微分机制在前向传播中（向模型输入数据、张量运算）
            -   执行张量计算
            -   维护梯度函数
        -   自动微分机制在反向传播中（调用根节点张量 `.backward()` 方法）
            -   调用张量 `.grad_fn` 反向传播函数计算梯度
            -   维护（累加）张量 `.grad` 梯度属性
            -   依据链式法则反向传播至叶子节点
    -   *DAG* 中非叶子节点（边）即张量运算（函数），负责在前向传播中执行张量运算、在反向传播中计算梯度
        -   `autograd.Function` （计算图）函数类可用于自定义张量运算函数
            -   子类需实现 `forward(ctx, i)`、`backward(ctx, grad_output)` 两个静态方法
            -   执行运算时应调用 `apply` 静态方法（而不是直接调用 `forward`）
        -   *PyTorch* 预定义张量运算函数有多种类型（并非 `autograd.Function` 子类）
            -   `builtin_function_method`：`torch.add`、`torch.matmul`
            -   `function`：`nn.functional.binary_cross_entropy_with_logits`
            -   预定定义运算回自动存储所需张量用于反向传播中计算梯度，可通过相应张量 `.grad_fn._saved_<XXX>` 访问
    -   对不可微函数，按如下顺序取梯度
        -   可微，直接取梯度
        -   凸函数（局部），取最小范数次梯度，即最速下降方向
        -   凹函数（局部），取最大范数次梯度
        -   若函数有定义，按连续性取值，存在多值则任取其一
        -   若函数未定义，取任意值，通常为 `NaN`
        -   非函数映射在反向传播时将报错

> - *AutoMatic Differentiation with `torch.autograd`*：<https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html>
> - *Autograd Mechanics*：<https://pytorch.org/docs/stable/notes/autograd.html>
> - *Automatic Differentiation Packaged - `torch.autograd`*：<https://pytorch.org/docs/stable/autograd.html>
> - 自动微分机制：<https://jackiexiao.github.io/eat_pytorch_in_20_days/2.%E6%A0%B8%E5%BF%83%E6%A6%82%E5%BF%B5/2-2%2C%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E6%9C%BA%E5%88%B6/>
> - *Extending torch.autograd*：<https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd>

####    *Grad Modes*

-   梯度模式：*PyTorch* 包含可通过上下文管理器切换、影响自动梯度计算逻辑的梯度模式
    -   `grad` 梯度（默认）模式
        -   `requires_grad` 设置仅在此模式下生效，在其他模式下均被覆盖为 `False`
    -   `with torch.no_grad()` 无梯度模式：不记录运算
        -   适合不应记录运算本身，但需要运算结果场合
            -   如在优化器中更新参数时，不应记录更新操作，但需更新参数用于后续轮次前向
    -   `with torch.inference_mode()` 推断模式：不记录运算，且推断模式中创建的张量不能无法在退出后被使用
        -   极端无梯度模式，开销较无梯度模式更小
    -   注意，`nn.Module.eval()`、`nn.Module.train(False)` 为将模型切换至评估模式，与梯度模式无关
        -   评估模式适用于模式依赖 `nn.Dropout`、`nn.BatchNorm2d` 场合，避免模型更新数据统计值
        -   梯度模式决定是否在前向传播中记录运算，供反向传播时更新梯度
        -   而评估模式决定特定层（模型）在 **前向传播是否更新参数**

> - *Autograd Grad Modes*：<https://pytorch.org/docs/stable/notes/autograd.html#grad-modes>

####    *Autograd Function*

```python
class MyCube(torch.autograd.Function):
    @staticmethod
    def forward(x):
        # We wish to save dx for backward. In order to do so, it must
        # be returned as an output.
        dx = 3 * x ** 2
        result = x ** 3
        return result, dx

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        result, dx = output
        ctx.save_for_backward(x, dx)

    @staticmethod
    def backward(ctx, grad_output, grad_dx):
        x, dx = ctx.saved_tensors
        # In order for the autograd.Function to work with higher-order
        # gradients, we must add the gradient contribution of `dx`,
        # which is grad_dx * 6 * x.
        result = grad_output * dx + grad_dx * 6 * x
        return result

# Wrap MyCube in a function so that it is clearer what the output is
def my_cube(x):
    result, dx = MyCube.apply(x)
    return result
```

-   `autograd.function.Function` （计算图）函数类：用于自定义张量运算函数，常用于实现不可微张量运算、或运算依赖非 *PyTorch* 库（提升效率、减少开销）
    -   `forward([ctx,]...)` 前向传播：执行运算，**原函数**
        -   `forward` 支持任意数量、类型参数，未实现 `setup_context` 方法时，首个参数须为 `ctx`
        -   `forward` 可返回任意值，无需仅为运算结果
        -   `forward` 中运算不会被记录至图中
            -   也即，其中运算不会被自动微分机制记录，不会在反向传播时被触发对应梯度计算
            -   仅输入张量参数（不包含列表、字典等容器中张量元素）、函数类整体将被注册至计算图中
        -   `forward` 方法不应被直接调用，应使用 `apply` （静态）方法触发
    -   `setup_context(ctx, input, output)` 设置上下文：仅负责维护 `ctx` 上下文变动，不执行计算
        -   `setup_context` 可省略，并整合至 `forward`，此时 `forward` 首个参数应为 `ctx` 上下文对象，并且负责上下文
        -   （*PyTorch 2.0* 后）`setup_context` 与 `forward` 拆分更接近 *PyTorch* 原生运算工作方式，子系统兼容性更好
    -   `backward(ctx, ...)` 反向传播：计算梯度，**导数函数**
        -   `backward` 首个参数须为 `ctx`，后续参数为 `forward` 输出张量的梯度（即链式法则反向传播梯度）
            -   `backward` 不应在位修改入参
        -   `backward` 应计算、并返回 `forward` 输入对应梯度
            -   即，返回值数量应同 `forward` 入参
            -   即，`backward` 实现函数自身梯度逻辑，并（应）结合链式法则计算 `forward` 输入对应梯度
            -   无需计算梯度入参、非张量入参、不可微类型（整形）张量可（须）对应输出 `None`
        -   `backward` 方法也不应被直接调用，常在张量 `Tensor.backward()` 引起的反向传播中被链式触发
            -   注意，`Function.backward` 不是 `Tensor.backward`
            -   事实上，`Function.backward` 方法将被包装为 `forward` 方法输出的张量的 `grad_fn` 属性
-   `autograd.function.Function` 实现、说明
    -   `ctx` 为 `autograd.function.FunctionCtx` 实例，暂存信息用于反向传播的上下文对象
    -   *PyTorch* 预定义张量运算函数有多种类型（并非 `autograd.Function` 子类）
        -   `builtin_function_method`：`torch.add`、`torch.matmul`
        -   `function`：`nn.functional` 中定义函数，如 `F.binary_cross_entropy_with_logits`
    -   二次反向传播
        -   `create_graph` 置位时，`backward` 中运算会被记录
            -   若 `backward` 中运算可被自动微分机制记录，则 `Function` 支持二次反向传播
            -   即，`Function` 导函数支持求导，即支持二阶梯度
        -   可使用 `autograd.function.once_differentiable()` 装饰 `backward` 申明反向传播仅允许一次（不支持二阶梯度）
    -   逻辑检查
        -   可使用 `autograd.gradcheck()` 方法检查 `backward` 中的梯度计算逻辑
        -   可使用 `autograd.gradgradcheck()` 方法检查 `backward` 二阶梯度计算支持、逻辑

| `FunctionCtx` 方法、属性             | 描述                                                             |
|--------------------------------------|------------------------------------------------------------------|
| `ctx.save_for_backward(*tesnor)`     | 暂存张量，仅应在 `forward` 内调用 1 次                           |
| `ctx.mark_dirty(*args)`              | 标记张量已经被修改（执行在位运算），仅应在 `forward` 内调用 1 次 |
| `ctx.mark_non_differentiable(*args)` | 标记 **输出** 为不可微，仅应在 `forward` 内调用 1 次             |
| `ctx.needs_input_grad`               | 元组，指示 `forward` 各入参是否需计算梯度                        |
| `ctx.saved_tensors`                  | 元组，`save_for_backward` 方法暂存的张量                         |

> - *Autograd Function*：<https://pytorch.org/docs/stable/autograd.html#function>
> - *Extending torch.autograd*：<https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd>
> - 自定义操作 `torch.autograd.Function`：<https://zhuanlan.zhihu.com/p/344802526>
> - *Double Backward with Custom Functions*：<https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>

####    *Computational Graphs*

-   计算图：张量、`Function` 代表节点，张量和 `Function` 之间（输入、输出）关系代表边
    -   *PyTorch* 为动态图机制，训练中 **每次迭代** 均会构建新计算图
        -   语句动态在计算图中动态添加节点、边，**并立即执行（前向传播）**
            -   *PyTorch* 通计算图捕获尝试从全局视角优化计算，在迭代的靠前（首个）轮次需要即时编译，可能较慢
        -   计算图默认在反向传播后立即被销毁
            -   即，无法二次调用 `Tensor.backward()`
    -   叶子节点 `is_leaf`：直接创建（非 `Function` 运算结果）、`require_grad` 置位的张量
        -   反向传播过程中，仅叶子节点梯度 `.grad` 被保留（用户一般只关心直接创建的张量的梯度）
        -   而，非叶子节点梯度 `.grad` 将默认被清空（仅在计算过程中被用到）
            -   置位 `retain_grad` 将保留非叶子节点梯度
            -   可利用 `Tensor.register_hook()` 注册钩子查看反向传播过程中梯度
    -   图中张量更应视为 **对应子图代表的（复合）函数**（及函数在具体输入下的取值）
        -   张量反向传播即 **复合函数链式求导**
            -   事实上，图根节点绝对值无意义，仅因其作为极小化目标（损失）而在正向传播中计算
            -   即，为极小化目标函数（根节点值）将参数沿负梯度方向优化
        -   对反向传播中某个（中间）函数（运算），**其中参数梯度仅依赖输入，即数学上的梯度的数值化带入计算**
            -   即，需要上下文暂存输入、中间结果（用于简化计算）
            -   而，函数输出对当前函数中参数梯度计算往往无意义，而作为下层函数输入
        -   反向传播简化 **参数梯度数值解** 计算
            -   仅需分别给出（简单）中间函数（内）参数偏导，应用链式法则、从根节点开始累积梯度即可
            -   否则，需得到根节点（复合函数）对各参数的偏导解析式、带入计算，即正向求解
    -   *Double Backward* 二次反向（传播）：在首次反向传播得到的计算图上再次反向传播，即二阶梯度
        -   要求计算图中 `Function` 均支持二次反向传播
        -   即，自定义函数 `Backward` 中运算均可被自动微分机制记录

> - 动态计算图：<https://jackiexiao.github.io/eat_pytorch_in_20_days/2.%E6%A0%B8%E5%BF%83%E6%A6%82%E5%BF%B5/2-3%2C%E5%8A%A8%E6%80%81%E8%AE%A1%E7%AE%97%E5%9B%BE/>
> - *How Computational Graphs are Constructed in PyTorch*：<https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/>
> - *PyTorch* 中计算图是如何构建的：<https://pytorch.ac.cn/blog/computational-graphs-constructed-in-pytorch/>
> - 计算图捕获：<https://zhuanlan.zhihu.com/p/644590863>
> - *Double Backward with Custom Functions*：<https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>

### `torch.nn` 神经网络

-   `torch.nn` 中包含构建神经网络所需的基本模块
    -   `nn` 命名空间已引入
        -   `nn.modules` 包、子包中定义的常用模块类
        -   `nn.parameters` 中参数类
    -   `nn.functioncal` 中函数未被引入 `nn` 命名空间

> - *What is `torch.nn` really*：<https://pytorch.org/tutorials/beginner/nn_tutorial.html>
> - `torch.nn` *API*：<https://pytorch.org/docs/stable/nn.html>
> - `torch.nn.functional` *API*：<https://pytorch.org/docs/stable/nn.functional.html>

####    `nn.modules.module.Module`

```python
def DigitsRecogNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

-   `nn.Module` 模块基类：*PyTorch* 中所有模块（层）均为此子类
    -   模块支持嵌套，即将其他模块实例作为自身属性
    -   `nn.Module` 类似 `autograd.Function` 可（用于）自定义数据运算
        -   但，**拥有并维护状态**，即更新、维护内部参数
        -   事实上，`nn.modules` 包、子包中预定义模块类实现即调用对应 `F.<Function>`
            -   模块类包含、维护函数所需参数，可与容器模块嵌套使用、注册钩子函数监控前后向传播
            -   函数自由度更高，可实现（数据）参数共享等功能
        -   一般的，任选模块、函数均可
            -   需维护参数（模型）、超参（函数配置参数）、与容器模块嵌套时，**模块实例化** 可能更方便
            -   但，注意 `nn.Dropout`、`nn.BatchNorm2D` 受 `Module.eval()` 切换评估模式影响，而 `F.dropout`、`F.BatchNorm2D` 不会
    -   `nn.Parameter` 参数：`torch.Tensor` 子类
        -   作为模块属性的 `Parameter` 实例将被注册至模块参数列表，可通过 `Module.parameters()` 访问
        -   依赖 `nn.Parameter` 参数类，被嵌套模块参数可被嵌套模块统一管理（运算由自动微分机制记录）
    -   除预定义模块外，可继承实现自定义模块
        -   `Module.forward` 前向传播（方法）
            -   不应直接调用，向模型实例传入数据将自动调用此方法

| 模块相关类         | 描述                                       |
|--------------------|--------------------------------------------|
| `nn.Sequential`    | 顺序封装，可视为单独模块                   |
| `nn.ModuleList`    | 模块列表，不被视为单独模块，仅用于注册参数 |
| `nn.ModuleDict`    | 模块字典，同上                             |
| `nn.ParameterList` | 参数列表，同上                             |
| `nn.ParameterDict` | 参数字典，同上                             |

> - *Build the Neural Network*：<https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html>
> - `torch.nn` *API*：<https://pytorch.org/docs/stable/nn.html>
> - `nn` 与 `nn.functional` 区别：<https://www.zhihu.com/question/66782101>
> - `nn.Parameter`：<https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html>
> - `torch.nn.functional` *API*：<https://pytorch.org/docs/stable/nn.functional.html>

####    常用模块、函数

| 函数                             | 模块                     | 描述             |
|----------------------------------|--------------------------|------------------|
| `F.conv1d`                       | `nn.Conv1d`              | 卷积             |
| `F.conv_transpose1d`             | `nn.ConvTranspose1d`     | 转置卷积         |
|                                  | `nn.LazyConv1d`          | 懒卷积           |
| `F.unfold`                       | `nn.Unfold`              | 卷积拆分         |
|                                  | `nn.ReflectionPad1d`     | 填充             |
| `F.avg_pool1d`                   | `nn.AvgPool1d`           | 池化             |
| `F.lp_pool1d`                    | `nn.LPPool1d`            | 指数平均池化     |
| `F.adaptive_avg_pool1d`          | `nn.AdaptiveAvgPool1d`   | 自适应池化       |
| `F.fractional_max_pool2d`        | `nn.FractionalMaxPool2d` | 分数阶最大值池化 |
| `F.threshold`                    | `nn.Threshold`           | 线性激活函数     |
| `F.threshold_`                   |                          | 在位激活         |
| `F.softmax`                      | `nn.Softmax`             | 非线性激活函数   |
| `F.nomarlize`                    |                          | 正则化           |
|                                  | `nn.BatchNorm1d`         | 批正则化         |
|                                  | `nn.LazyBatchNorm1d`     | 懒正则化         |
| `F.linear`                       | `nn.Linear`              | 线性变换         |
| `F.dropout`                      | `nn.Dropout`             | 随机丢弃         |
| `F.embedding`                    | `nn.Embedding`           | 嵌入             |
| `F.pairwise_distance`            | `nn.PairwiseDistance`    | 配对距离         |
| `F.binary_cross_entropy`         | `nn.CrossEntropyLoss`    | 损失函数         |
| `F.pixel_shuffle`                | `nn.Pixelshuffle`        | 图像像素混淆     |
| `nn.parallel.data_parallel`      | `nn.DataParallel`        | 数据并行         |
|                                  | `nn.RNN`                 | *RNN*            |
| `F.scaled_dot_product_attention` |                          | 注意力点乘       |
|                                  | `nn.Transformer`         | *Transformer*    |

> - `torch.nn` *API*：<https://pytorch.org/docs/stable/nn.html>
> - `torch.nn.functional` *API*：<https://pytorch.org/docs/stable/nn.functional.html>

####    `torch.optim.Optimizer`

| `Optimizer` 常用方法 | 描述               |
|----------------------|--------------------|
| `step()`             | 执行优化，更新参数 |
| `zero_grad()`        | 清空待优化参数梯度 |
| `add_param_group()`  | 添加参数组         |
| `load_state_dict()`  | 载入优化器状态     |
| `state_dict()`       | 返回优化器状态     |

-   `torch.optim`：包、子包实现有最优化算法
    -   `optim.optimizer.Optimizer` 是所有优化器基类
        -   （内置）优化器首个形参 `params` 需为 `nn.Parameter` 待优化参数迭代器，后续参数为优化器配置参数
            -   `Iterable[nn.Paramter]`：最常即 `Module.parameters()` 优化模型中所有参数
            -   `[(<NAME>, nn.Parameter),...]`：最常即 `Module.named_parameters()` 优化具名参数
            -   `[{"param": Iterable[nn.Parameter],...},...]`：为参数指定不同优化器参数（依旧可以指定优化器级配置参数作为默认值）
        -   `Optimizer.step()`：更新参数
            -   不带参调用：适合大部分优化器，常用于 `Tensor.backward()` 反向传播更新梯度之后更新参数
            -   带闭包函数调用：适合 *LBFGS* 等每轮迭代中需要多次计算损失的优化器
                -   闭包函数（仅）需清空梯度、计算并返回损失（张量）
        -   `Optimizer.zero_grad()`：清空待优化参数梯度
    -   `torch.optim` 内置优化器处于性能、可靠性（泛用性）有多种实现（默认会使用在当前设备上最快实现）
        -   *For-loop*：遍历各参数更新
            -   直观、慢
        -   *Foreach*：将参数组组合为 *multi-tensor* 整体更新参数组
            -   较快、内存占用高、内核调用少
            -   大部分优化器的默认实现
        -   *Fused*：在一次内核调用中更新所有参数
            -   最快、融合多种操作以减少开销
            -   依赖特定硬件加速，已实现优化器较少
    -   `optim.lr_scheduler` 中提供多种学习率（动态）调整器
        -   `lr_scheduler.LRScheduler`：（及其子类）根据 *epoch* 轮次动态调整
        -   `lr_scheduler.ReduceLROnPlateau`：（及其子类）根据验证结果动态调整
        -   `<XXX>Scheduler.step()`：调整学习率
            -   学习率调整应在优化器更新后应用
            -   多个学习率调整可以背靠背应用

> - `torch.optim`：<https://pytorch.org/docs/stable/optim.html>
> - *Optimizer in PyTorch*：<https://zhuanlan.zhihu.com/p/684067397>

### 交互

#### 数据加载

```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

training_data = CustomImageDataset(labels, imgdir)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
```

-   *PyTorch* 包含数据集相关处理功能 `torch.utils.data` 以解耦数据预处理、模型训练
    -   `data.Dataset` 数据集：封装样本、标签
        -   映射风格数据集：实现 `__getitem__`、`__len__` 协议，代表键（位序下标）、样本映射
            -   可通过键、下表直接访问样本
            -   可直接继承 `data.Dataset` 实现自定义数据集，代表样本迭代器
        -   迭代风格数据集：实现 `__iter__` 协议
            -   适合随机读取开销大、块大小依赖获取数据的场合
            -   可继承 `data.IterableDataset` （继承自 `data.Dataset`）实现自定义数据集
    -   `data.DataLoader`：封装 `data.Dataset` 为可迭代对象，方便使用
        -   支持映射风格、迭代风格数据集
        -   支持自定义（映射风格）数据加载顺序
            -   迭代风格数据集加载顺序由数据集自身决定
            -   映射风格由 `data.Sampler` 类负责排列样本顺序（`shuffle` 参数置位时将在内部自动构造）
        -   自动分块、整理数据
            -   `batch_size`、`batch_sampler` 均 `None` 时禁用自动分块
            -   `collate_fn` 在自动分块时处理块数据，否则处理单个样本（缺省将 `np.ndarray` 转换为张量）
        -   支持多进程数据加载
        -   支持内存固定

> - *Datasets & DataLoader*：<https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>
> - `torch.utils.data`：<https://pytorch.org/docs/stable/data.html>

####    可视化

| `SummaryWriter` 方法                                                  | 描述                     |
|-----------------------------------------------------------------------|--------------------------|
| `SummaryWriter.add_scalar(tag,scalar_value,global_step,...)`          | 添加标量（折线图）       |
| `SummaryWriter.add_scalars(main_tag,tag_scalar_dict,global_step,...)` | 添加多组标量（折线图）   |
| `SummaryWriter.add_histogram(tag,values,global_step,...)`             | 添加直方图               |
| `SummaryWriter.add_image(tag,img_tensor,global_step,...)`             | 添加图片                 |
| `SummaryWriter.add_images(tag,img_tensor,global_step,...)`            | 添加多张图片             |
| `SummaryWriter.add_figure(tag,figure,global_step,...)`                | 添加 `matplotlib` 图表   |
| `SummaryWriter.add_video(tag,vid_tensor,global_step,...)`             | 添加视频                 |
| `SummaryWriter.add_audio(tag,snd_tensor,global_step,...)`             | 添加音频                 |
| `SummaryWriter.add_text(tag,snd_tensor,global_step,...)`              | 添加文本                 |
| `SummaryWriter.add_graph(model,input_to_model,...)`                   | 添加 `nn.Module` 计算图  |
| `SummaryWriter.add_embedding(mat,metadata,...)`                       | 添加 *embedding* 投影    |
| `SummaryWriter.add_pr_curve(tag,labels,predictions,...)`              | 添加 *PR* 曲线           |
| `SummaryWriter.add_custom_scalars(layout)`                            | 创建 `scalar` 表格布局   |
| `SummaryWriter.add_mesh(tag,vertices,colors,...)`                     | 添加网格、点云           |
| `SummaryWriter.add_hparams(hparam_dict,metric_dict,,...)`             | 添加超参、评估指标对比表 |
| `SummaryWriter.flush()`                                               | 刷新缓冲区至磁盘         |
| `SummaryWriter.close()`                                               | 关闭                     |

-   *PyTorch* 支持 *TensorBoard* 的可视化展示
    -   `torch.utils.tensorboard.SummaryWriter(log_dir,...)`：*TensorBoard* 可视化数据记录器
        -   `SummaryWriter` 支持多种方法添加多种类型数据，且会 **自动将原始数据加工成对应图表所需结果**
            -   `tag`：数据标签，用于区分不同数据
            -   `global_step`：全局迭代轮次，用于区分多次同名标签记录

> - *Visualizing Models, Data and Training with TensorBoard*：<https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html>
> - *How two use TensorBoard with PyTorch*：<https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html>
> - `torch.utils.tensorboard`：<https://pytorch.org/docs/stable/tensorboard.html>
