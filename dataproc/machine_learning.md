---
title: Data Science
categories:
  - ML Theory
tags:
  - Machine Learning
  - Supervised Learning
  - Unsupervised Learning
  - Semi-Supervised Learning
  - Reinforcement Learning
date: 2019-07-14 20:02:42
updated: 2024-07-18 19:09:54
toc: true
mathjax: true
comments: true
description: 机器学习范畴
---

##  *Data Science*

###  数据科学范畴

-   *Statistic - Frequentist and Bayesian* 统计：数学分支，概率论和优化的交集，是数据科学其他分支的理论基础
    -   分析方法：验证式分析
        -   统计建模：基于数据构建统计模型，并验证假设
        -   模型预测：运用模型对数据进行预测、分析
    -   理论依据：模型驱动，严格的数理支撑
        -   理论体系
            -   概率论、信息论、计算理论、最优化理论、计算机学科等多个领域的交叉学科
            -   并在发展中形成独自的理论体系、方法论
        -   基本假设：同类数据具有一定的统计规律性，可以用概率统计方法加以处理，推断总体特征，如
            -   随机变量描述数据特征
            -   概率分布描述数据统计规律
    -   分析对象：以样本为分析对象
        -   从数据出发，提取数据特征、抽象数据模型、发现数据知识，再回到对数据的分析与预测
        -   数据多种多样，包括数字、文字、图像、音视频及其组合
        -   假设数据独立同分布产生
        -   训练数据集往往是人工给出的

-   *Machine Learning* 机器学习：从有限观测数据中学习一般性规律，并将规律应用到未观测样本中进行预测（最基本的就是在不确定中得出结论）
    -   分析方法：归纳式、探索式分析
    -   理论依据：数据驱动，从数据中中学习知识
    -   分析对象：对样本要求低，样本往往不具有随机样本的特征

-   *Data Mining* 数据挖掘：从现有的信息中提取数据的 *pattern*、*model*
    -   即精选最重要、可理解的、有价值的信息
        -   核心目的在于找到数据变量之间的关系
        -   不是证明假说的方法，而是构建假说的方法
        -   大数据的发展，传统的数据分析方式无法处理大量“不相关”数据
    -   说明
        -   统计、机器学习可视为工具、方法
        -   数据挖掘可视为使用工具
        -   人工只能视为目标、成果

-   *Artificial Intelligence* 人工智能：研究如何创造智能 *Agent*，并不一定涉及学习、归纳
    -   但是大部分情况下，**智能** 需要从过去的经验中进行归纳，所以 *AI* 中很大一部分是 *ML*

### *ML* 特征

####    *Representing Learning*

-   *Representing Learning* 表示学习：自动学习有效特征、提高最终机器学习模型性能的学习
    -   好的学习标准
        -   较强的表示能力：同样大小向量可以表示更多信息
        -   简化后续学习任务：需要包含更高层次语义信息
        -   具有一般性，是任务、领域独立的：期望学到的表示可以容易迁移到其他任务
    -   要学习好的高层语义（分布式表示），需要从底层特征开始，经过多步非线程转换得到
        -   深层结构的优点式可以增加特征重用性，指数级增加表示能力
        -   所以表示学习的关键是构建具有一定深度、多层次特征表示
    -   传统机器学习中也有关于特征学习的算法，如：主成分分析、线性判别分析、独立成分分析
        -   通过认为设计准则，用于选取有效特征
        -   特征学习、最终预测模型的学习是分开进行的，学习到的特征不一定可以用于提升最终模型分类性能
    -   *Local Representation* 局部表示：每个特征作为高维局部表示空间中轴上点
        -   离散表示、符号表示，通常可以表示为 *one-hot* 向量形式
            -   *one-hot* 维数很高、不方便扩展
            -   不同特征取值相似度无法衡量
    -   *Distributed Representation*：分布式表示：特征分散在整个低维嵌入空间中中
        -   通常可以表示为 **低维、稠密** 向量
            -   表示能力强于局部表示
            -   维数低
            -   容易计算相似度

> - *Semantic Gap*：语义鸿沟，输入数据底层特征和高层语义信息之间不一致性、差异性

####    *Deep Learning*

-   深度：原始数据进行 **非线性特征转换的次数**
    -   将深度学习系统看作有向图结构，深度可以看作是从输入节点到输出节点经过最长路径长度
    -   *Deep Learning* 深度学习：将原始数据特征通过多步特征转换得到更高层次、抽象的特征表示，进一步输入到预测函数得到最终结果
        ![deep_learning_procedures](imgs/deep_learning_procedures.png)
        -   主要目的是从数据中自动学习到 **有效的特征表示**
            -   替代人工设计的特征，避免 “特征” 工程
            -   模型深度不断增加，特征表示能力越强，后续预测更容易
        -   相较于浅层学习：需要解决的关键问题是 **贡献度分配问题**（参数、权重）
            -   从某种意义上说，深度学习也可以视为强化学习
            -   内部组件不能直接得到监督信息，需要通过整个模型的最终监督信息得到，有延时
            -   目前深度学习主流的神经网络模型可以使用反向传播算法，较好的解决贡献度分配问题
    -   *Shallow Learning* 浅层学习：不涉及特征学习，特征抽取依靠人工经验、特征转换方法
        ![shallowing_learning_procedures.png](imgs/shallowing_learning_procedures.png)
        -   通过对数据分布的假设，建立精巧的数学模型描述数据规律
            -   模型结构简单，参数数量较少，训练简单
            -   有良好可解释性，模型参数本身往往可以体现数据关系
        -   步骤
            -   数据预处理
            -   特征提取
            -   特征转换
            -   预测

> - *Credit Assignment Problem*：贡献度分配问题，系统中不同组件、参数对最终系统输出结果的贡献、影响

#### *End-to-End Learning*

-   *End-to-End Learning* 端到端学习/训练：学习过程中不进行分模块、分阶段训练，直接优化任务的总体目标
    -   不需要给出不同模块、阶段功能，中间过程不需要认为干预
    -   训练数据为“输入-输出”对形式，无需提供其他额外信息
    -   和深度学习一样，都是要解决“贡献度分配”问题
        -   大部分神经网络模型的深度学习可以看作是端到端学习

###  *Supervised Learning*

-   *Supervised Learning* 有监督学习：从有标记的数据中学习、训练、建立模型，用于对可能的输入预测输出
    -   *Generative Approach* 生成方法：学习联合概率分布 $P(X, Y)$，然后求出条件概率分布 $P(Y|X)$ 作为 *Generative Model*
        -   即，学习输入 $X$、输出 $Y$ 之间的生成关系（联合概率分布）
        -   *Generative Model*：生成模型，由生成方法学习到的模型 $P(Y|X) = \frac {P(X, Y)} {P(X}$
            -   朴素贝叶斯法
            -   隐马尔可夫模型
        -   特点
            -   可以还原联合概率分布 $P(X, Y)$
            -   生成方法学习收敛速度快，样本容量增加时，学习到的模型可以快速收敛到真实模型
            -   存在隐变量时，仍可以使用生成方法学习
    -   *Discriminative Approach* 判别方法：直接学习决策函数 $f(x)$、条件概率分布 $P(Y|X)$ 作为 *Discriminative Model*
        -  即，对给定输入 $X$ 预测输出$Y$
        -   *Discriminative Model*：判别模型
            -   *KNN*
            -   感知机
            -   决策树
            -   逻辑回归
            -   最大熵模型
            -   支持向量机
            -   提升方法
            -   条件随机场
        -   特点
            -   直接学习条件概率、决策函数
            -   直面预测，学习准确率更高
            -   可以对数据进行各种程度抽象、定义特征、使用特征，简化学习问题
    -   数据空间
        -   *input space*：输入空间 $\chi$，所有输入 $X$ 可能取值的集合
        -   *output space*：输出空间 $\gamma$，所有输出 $Y$ 可能取值集合
        -   *feature space*：特征空间，表示输入实例 *Feature Vector* 存在的空间
            -   特征空间每维对应一个特征
            -   模型实际上是定义在特征空间上的
            -   特征空间是输入空间的象集，有时等于输入空间

-   有监督学习常用于解决问题
    -   *Classification* 分类问题：输出变量 $Y$ 为有限个离散变量（即类别）
        -   目标：根据样本特征判断样本所属类别
            -   训练：根据已有数据集训练分类器 $P(Y|X))$、$Y=F(X)$
                -   不存在分类能力弱于随机预测的分类器（若有，结论取反）
            -   推理：利用学习的分类器对新输入实例进行分类
        -   分类问题模型
            -   *KNN*
            -   感知机
            -   朴素贝叶斯
            -   决策树
            -   决策列表
            -   逻辑回归
            -   支持向量机
            -   提升方法
            -   贝叶斯网络
            -   神经网络
    -   *Tagging* 标注问题：输入、输出均为变量序列（即标签）
        -   目标：可认为是分类问题的一个推广、更复杂 *structure prediction* 简单形式
            -   训练：利用已知训练数据集构建条件概率分布模型 $P(Y^{(1)}, Y^{(2)}, \cdots, Y^{(n)}|X^{(1)}, X^{(2)}, \cdots, X^{(n)})$
            -   推理：按照学习到的条件概率分布，标记新的输入观测序列
        -   标注问题模型
            -   隐马尔可夫模型
            -   条件随机场
    -   *Regression* 回归问题：输入（自变量）、输出（因变量）均为连续变量
        -   回归模型的拟合等价于函数拟合：选择函数曲线很好的拟合已知数据，且很好的预测未知数据
            -   学习过程：基于训练数据构架模型（函数）$Y=f(X)$
            -   预测过程：根据学习到函数模型确定相应输出
        -   回归问题模型
            -   线性回归
            -   广义线性回归

> - 监督学习：<https://zh.wikipedia.org/wiki/%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0>

####  二分类到多分类

-   多分类策略
    -   *1 vs n-1*：对类 $k=1,...,n$ 分别训练当前类对剩余类分类器
        -   分类器数据量有偏，可以在负类样本中进行抽样
        -   训练 $n$ 个分类器
    -   *1 vs 1*：对 $k=1,...,n$ 类别两两训练分类器，预测时取各分类器投票多数
        -   需要训练 $\frac {n(n-1)} 2$ 给分类器
    -   *DAG*：对 $k=1,...,n$ 类别两两训练分类器，根据预先设计的、可以使用 *DAG* 表示的分类器预测顺序依次预测
        -   即排除法排除较为不可能类别
        -   一旦某次预测失误，之后分类器无法弥补
            -   但是错误率可控
            -   设计 *DAG* 时可以每次选择相差最大的类别优先判别

###  *Unsupervised Learning*

-   *Unsupervised Learning* 无监督学习：自动对无标记的数据进行分类、分群
    -   无监督模型一般比有监督模型表现差
    -   无监督学习主要用于数据预划分、识别、编码，供其他任务使用
        -   *Clustering Analysis* 聚类分析
            -   *Hierarchy Clustering*
            -   *K-means*
            -   *Mixture Models*
            -   *DBSCAN*
            -   *OPTICS Algorithm*
        -   *Anomaly Detection* 异常检测
            -   *Local Outlier Factor*
        -   *Encoder* 编码器
            -   *Auto-encoders*
            -   *Deep Belief Nets*
            -   *Hebbian Learning*
            -   *Generative Adversarial Networks*
            -   *Self-organizing Map*
        -   隐变量学习
            -   *Expectation-maximization Algorithm*
            -   *Methods of Moments* 矩估计
            -   *Bind Signal Separation Techniques* 带信号分离
                -   *Principal Component Analysis*
                -   *Independent Component Analysis*
                -   *Non-negative Matrix Factorization*
                -   *Singular Value Decomposition*

-   *Semi-Supervised Learning* 半监督学习：利用少量标注数据和大量无标注数据进行学习的方式
    -   可以利用大量无标注数据提高监督学习的效果

> - 无监督学习：<https://zh.wikipedia.org/wiki/%E7%84%A1%E7%9B%A3%E7%9D%A3%E5%AD%B8%E7%BF%92>

###  *Reinforcement Learning*

-   *Reinforcement Learning* 强化学习：与环境交互并获得延迟奖励，并据此调整策略以获得最大回报
    -   从与环境交互中不断学习的问题、以及解决这类问题的方法
        -   每个动作不能直接得到监督信息，需要通过整个模型的最终监督信息得到，且具有时延性
        -   给出的监督信息也非 “正确” 策略，而是策略的延迟回报，并通过调整策略以取得最大化期望回报

-   强化学习要素
    -   *State* 环境状态
    -   *Action* 个体动作
    -   *Reward* 环境奖励：$t$ 时刻个体在状态 $S_t$ 采取动作 $A_t$ 将延时在 $t+1$ 时刻得到奖励 $R_{t+1}$
    -   *Policy* 个体策略：个体采取动作的依据，常通过条件概率分布 $\pi(A|S)$ 表示
    -   *Value* 行动价值：个体在状态 $S_t$ 下采取行动后的价值，综合考虑当前、后续的延时奖励整体期望
    -   *Decay Rate* 奖励衰减因子：后续延时奖励对行动价值的权重
    -   环境状态转化模型：状态 $S_t$、动作 $A_t$ 构成概率状态机
    -   探索率：动作选择概率

> - 强化学习基础：<https://www.cnblogs.com/pinard/p/9385570.html>

###  *ML Components*

-   *ML* 包括 3 个组成部分
    -   *Model* 模型：问题的抽象描述、刻画
        -   对某类模型整体，可从以下方面进行描述、对比
            -   问题刻画能力
            -   模型复杂程度
            -   参数可解释性
        -   对具体完成训练的模型实体（可视为策略、算法、训练、超参的统称），需要注意
            -   模型推理能力
            -   模型泛化能力
    -   *Policy* 策略：模型的可求解变换
        -   大部分均转化为为最优化问题，即损失函数
            -   损失函数的设置需考虑问题特点、求解难易程度
            -   一般来说选择自由度不大，给定模型的损失函数基本确定
        -   策略结果本身也可以视为对模型的评价指标
    -   *Alogrithm* 算法：策略的求解方法
        -   一般即对应损失函数的最优化算法
        -   可从以下方面评价
            -   全局最优性
            -   求解效率：收敛速度
            -   稀疏性

####    *Model*

-   *Model* 模型：待学习的条件概率分布 $P(Y|X)$、决策函数 $Y=f(X)$
    -   模型可用决策函数、条件概率分布函数表示
        -   概率模型：用条件概率分布 $P(Y|X)$ 表示的模型
        -   非概率模型：用决策函数 $Y=f(x)$ 表示的模型
    -   *Hypothesis Space* 假设空间：特征空间（输入空间）到输出空间的映射集合
        -   假设空间可以定义为决策函数、条件概率的集合，通常是由参数向量 $\theta$ 决定的函数/条件分布族
            -   假设空间包含所有可能的条件概率分布或决策函数
            -   假设空间的确定意味着学习范围的确定
        -   概率模型假设空间可表示为：$F=\{P|P_{\theta}(Y|X), \theta \in R^n\}$
        -   非概率模型假设空间可表示为：$F=\{f|Y=f(x),\Theta \in R^n \}$

-   模型信息来源
    -   训练数据包含信息
    -   模型形成过程中提供的先验信息
        -   模型：采用特定内在结构（如深度学习不同网络结构）、条件假设、其他约束条件（正则项）
        -   数据：调整、变换、扩展训练数据，让其展现更多、更有用的信息

> - 模型也被称为 *Hypothesis* 假设、*Opimizee* 优化对象、*Learner* 学习器

#####    *Over-Fitting*

-   过拟合：学习时选择的所包含的模型复杂度大（参数过多），导致模型对已知数据预测很好，对未知数据预测效果很差
    -   若在假设空间中存在 “真模型”，则选择的模型应该逼近真模型（参数个数相近）
        -   一味追求对训练集的预测能力，复杂度往往会比“真模型”更高
    -   过拟合解决方法
        -   减少预测变量数量
            -   最优子集回归：选择合适评价函数（带罚）选择最优模型
            -   验证集挑选模型：将训练集使用 *抽样技术* 分出部分作为 *validation set*，使用额外验证集挑选使得损失最小的模型
            -   正则化（罚、结构化风险最小策略）
                -   岭回归：平方损失，$L_2$ 范数
                -   *LASSO*：绝对值损失，$L_1$ 范数
                -   *Elastic Net*
        -   减弱变量特化程度：仅适合迭代求参数的方法
            -   *EarlyStop*：提前终止模型训练
            -   *Dropout*：每次训练部分神经元

####    *Policy*

-   *Policy*、*Goal* 策略、目标：根据 *Evaluation Criterion* 从假设空间钟选择最优模型，使得其在给定评价准则下有最优预测
    -   可视为对模型中参数进行估计的方案
    -   对有监督学习，选择合适策略即将问题变为损失（风险）函数最优化问题
        -   *Empirical Risk Minimiation* 经验风险最小化
            -   按经验风险最小化求最优模型，等价于求最优化问题
                $$ \min_{f \in F} \frac 1 N \sum_{i=1}^N L(y_i, f(x_i)) $$
            -   样本容量足够大时，经验风险最小化能保证有较好的学习效果，现实中也被广泛采用
        -   *Structural Risk Minimization* 结构风险最小化
            -   结构风险最小的模型是最优模型，则求解最优模型等价于求解最优化问题
                $$ arg \min_{f \in F} \frac 1 N \sum_{i=1}^N L(y_i, f(x_i)) + \lambda J(f) $$
            -   结构风险小需要经验风险与模型复杂度同时小
                -   此时模型往往对训练数据、未知的测试数据都有较好的预测
                    -   在一定程度可以避免过拟合
                -   结构化风险最小策略符合 *Occam's Razor* 原理
    -   最优化问题说明
        -   最优化问题目标函数也有可能不是损失函数，如：*SVM* 中损失函数是和模型紧密相关的损失函数，但逻辑类似
        -   *Well-Posed Problem*：好解问题，指问题解应该满足以下条件
            -   解存在
            -   解唯一
            -   解行为随着初值**连续变化**
        -   *Ill-Posed Problem*：病态问题，解不满足以上三个条件

-   除损失函数最优化问题之外
    -   矩估计：**建立参数和总体矩的关系**，求解参数
        -   除非参数本身即为样本矩，否则基本无应用价值
        -   应用场合
            -   均值：对应二次损失 $\arg\min_{\mu} \sum_{i=1}^N (x_i - \mu)^2$
            -   方差：对应二次损失?
    -   极大似然估计：极大化似然函数，求解概率上最合理参数
        -   需知道（假设）总体 **概率分布形式**
        -   似然函数形式复杂，求解困难
            -   往往无法直接给出参数的解析解，只能求数值解
        -   应用场合
            -   估计回归参数：对数损失
                $\mathop{\arg\min}_{\beta} \sum_{i=1}^N lnP(y_i|x_i, \beta)$
    -   随机模拟估计参数：设计随机模拟实验估计参数
        -   应用场合
            -   蒙特卡洛类似算法：随机化损失
    -   最大熵原理（估计）：选择熵最大的概率（分布）模型
        -	最大熵原理思路：熵最大概率模型即最好模型
            -	概率模型要满足已有事实（约束条件）
            -	没有更多信息的情况下，不确定部分是等可能的
            -   信息熵可代表概率分布的混乱程度
            -   则信息熵最大即为给定条件下，非必要限制最小、最自然的情况
        -   应用场合
            -   最大熵模型

> - 参数估计都可以找到合适损失函数，通过迭代求解损失最小化
> - *Occam's Razor*：奥卡姆剃刀原理，在所有可能选择的模型中，能够很好的解释已知数据并且十分简单才是最好的模型

####    *Algorithm*

-   *Algorithm*、*Optimizer* 算法、优化器：学习模型（选择、求解最优模型）的具体计算方法（求解最优化问题）
    -   若最优化问题有显式解析解，则可直接求出解析解
        -   最小二乘估计：线性回归模型中 *MSE* 作为损失函数时
    -   但通常解析解不存在，需要用数值计算方法迭代求解
        -   全局最优性
        -   求解效率：收敛速度
        -   稀疏性

## 模型策略

###    *Expected Risk* / *Expected Loss* / *Generalization Loss*

$$ R_{exp}(f) = E_p[L(Y, f(X))] = \int_{x*y} L(y,f(x))P(x,y) dxdy $$
> - $P(X, Y)$：随机变量 $(X, Y)$ 遵循的联合分布，未知

-   *Expected Risk* 期望风险（函数）、泛化误差：损失函数 $L(Y, f(X))$（随机变量）期望
    -   风险函数值度量模型预测错误程度
        -   代表学习方法的泛化能力
        -   评价标准（**监督学习目标**）就应该是选择期望风险最小
    -   *Generalization Ability* 泛化能力：方法学习到的模型对未知数据的预测能力，是学习方法本质、重要的性质
    -   *Generalization Error Bound* 泛化误差上界：泛化误差的概率上界，常用于评估学习方法的泛化能力
        -   损失函数 $L(Y, f(x))$ 联合分布未知，无法直接计算期望风险
        -   是样本容量函数，样本容量增加时，泛化误差上界趋于 0
        -   是假设空间容量（模型复杂程度）函数，假设空间容量越大，模型越难学习，泛化误差上界越大

####    二分类泛化误差上界

-   对二分类问题，根据 *Hoeffding* 不等式有（随即变量只能取 $0, 1$）
    $$\begin{align*}
    P(|R(h) - \hat R(h)| \geq \epsilon) & \leq 2 e^{-2 N \epsilon^2} \\
    P(\forall h \in H: |R(h) - \hat R(h)| \leq \epsilon|)
        & = 1 - P(\exists h \in H: |R(h) - \hat R(h)| \geq \epsilon) \\
    & = 1 - P(\bigcup_{h \in H} \{R(h) - \hat R(h) \}) \\
    & \geq 1 - \sum_{h \in H} P(|R(h) - \hat R(h)| \geq \epsilon) \\
    & \geq 1 - 2|H|e^{-2N \epsilon^2}
    \end{align*}$$
    > - $H$：假设空间
    > - $N$：样本数量
    > - $R(h) := R_{exp}(h)$
    > - $\hat R(h) := R_{emp}(h)$
    -   则令 $\sigma = |H| exp(-2N\epsilon^2)$，则至少以概率 $1-\sigma$ 满足如下，即得到泛化误差上界
        $$\begin{align*}
        E(h)  & \leq \hat E(h) + \epsilon(|H|, N, \sigma) \\
        \epsilon(|H|, N, \sigma) & = \sqrt {\frac 1 {2N} (log |H| + log \frac 1 {\sigma})}
        \end{align*}$$
    -   对任意 $\epsilon$，随样本数量 $m$ 增大，泛化误差 $\Delta(h) \leq \epsilon$ 概率增大
        -   即，经验风险依概率收敛至期望风险，可以使用经验风险近似期望风险

> - 二分类泛化误差上界：<https://www.cnblogs.com/aichemistar/p/13720564.html>

####   *Probably Approximate Correct* 可学习

-   *PAC* 可学习：在短时间内利用少量（多项式级别）样本能够找到模型 $h^{'}$，满足
    $$ P(E(h^{'}) \leq \epsilon) \geq 1 - \sigma, 0 < \epsilon, \sigma < 1 $$
    -   即，模型需要满足两个 *PAC* 辨识条件
        -   近似条件：泛化误差 $E(h^{'})$ 足够小
        -   可能正确：满足近似条件概率足够大
    -   *PAC* 学习理论关心能否从假设空间 $H$ 中学习到好的假设 $h$
    -   对二分类问题，取 $\sigma = 2|H|e^{-2N\epsilon^2}$，则样本量满足 $N = \frac {ln \frac {2|H|} \sigma} {2 \epsilon^2}$ 时，模型是 *PAC* 可学习的

###    *Empirical Risk* / *Empirical Loss*

$$\begin{align*}
R_{emp}(f) & = \sum_{i=1}^N D_i L(y_i, f(x_i;\theta)) \\
E(R_{emp}(f)) & = R_{exp}(f)
\end{align*}$$
> - $\theta$：模型参数
> - $D_i$：样本损失权重，常为 $\frac 1 N$，在 *Boosting* 框架中不同

-   *Empirical Risk* 经验风险：模型关于给定训练数据集的平均损失
    -   经验风险损失是模型 $f(x)$ 的函数
        -   训练时，模型是模型参数的函数
        -   即其为模型参数函数
    -   根据大数定律，样本量容量 $N$ 趋于无穷时，$R_{emp}(f)$ 趋于 $R_{exp}(f)$
        -   但是现实中训练样本数目有限、很小，经验风险不能完全代表模型的的泛化能力
        -   利用经验风险估计期望常常并不理想，需要对经验风险进行矫正
    -   例子
        -   *maximum probability estimation*：极大似然估计
            -   模型：条件概率分布（贝叶斯生成模型、逻辑回归）
            -   损失函数：对数损失函数

####    训练、测试误差

-   *Training Error* 训练误差：模型在训练集上的误差，损失函数 $L(Y, F(X))$ 均值
    $$ e_{train} = R_{emp}(\hat f) = \frac 1 N \sum_{i=1}^N L(y_i, \hat {f(x_i)}) $$
    > - $\hat f$：学习到的模型
    > - $N$：训练样本容量
    -   训练时采用的损失函数和评估时一致时，训练误差等于经验风险
    -   训练误差对盘对给定问题是否容易学习是有意义的，但是本质上不重要
        -   模型训练本身就以最小化训练误差为标准，如：最小化 *MSE*、最大化预测准确率，一般偏低，不能作为模型预测误差的估计
        -   训练误差随模型复杂度增加单调下降（不考虑模型中随机因素）

-   *Test Error* 测试误差：模型在测试集上的误差，损失函数 $L(Y, f(X))$ 均值
    $$ e_{test} = \frac 1 {N^{'}} \sum_{i=1}^{N^{'}} L(y_i,\hat {f(x_i)}) $$
    > - $\hat f$：学习到的模型
    > - $N$：测试样本容量
    -   测试误差反映了学习方法对未知测试数据集的预测能力，可以作为模型泛化误差估计，度量模型 *Generalization Ability*
    -   测试误差随模型复杂度增加呈U型
        -   偏差降低程度大于方差增加程度，测试误差降低
        -   偏差降低程度小于方差增加程度，测试误差增大
    -   训练误差小但测试误差大表明模型过拟合，使测试误差最小的模型为理想模型

###    *Structual Risk* / *Structual Loss*

$$ R_{srm} = \frac 1 N \sum_{i=1}^N L(y_i, f(x_i)) + \lambda J(f) $$
> - $J(f)$：模型复杂度，定义在假设空间$F$上的泛函
> - $\lambda$：权衡经验风险、模型复杂度的系数

-   *Structual Risk* 结构风险：在经验风险上加上表示模型复杂度的 *Regularizer*（*Penalty Term*）
    -   模型复杂度 $J(f)$ 表示对复杂模型的惩罚
        -   模型 $f$ 越复杂，复杂项 $J(f)$ 越大
        -   模型复杂度作为成为惩罚项被优化，可
            -   提高学习模型的泛化能力、避免过拟合
            -   学习简单模型：稀疏模型、引入组结构
    -   案例
        -   *maximum posterior probability estimation*：最大后验概率估计
            -   损失函数：对数损失函数
            -   模型复杂度：模型先验概率对数后取负
            -   先验概率对应模型复杂度，先验概率越小，复杂度越大
        -   岭回归：平方损失 + $L_2$ 正则化
            $\mathop{\arg\min}_{\beta} \sum_{i=1}^N (y_i - f(x_i, \beta))^2 + \|\beta\|$
        -   *LASSO*：平方损失 + $L_1$ 正则化
            $\mathop{\arg\min}_{\beta} \sum_{i=1}^N (y_i - f(x_i, \beta))^2 + \|\beta\|_1$

####  模型复杂度

-   简单模型低方差高偏差，复杂模型低偏差高方差
    -   模型复杂度越高对问题的刻画能力越强
        -   低偏差：对训练集的拟合充分
        -   高方差：模型紧跟特定数据点，受其影响较大，预测结果不稳定
        -   远离真实关系，模型在来自同系统中其他尚未观测的数据集上预测误差大
    -   而训练集、测试集往往不完全相同
        -   复杂度较高的模型（过拟合）在测试集上往往由于其高方差效果不好，而建立模型最终目的是用于预测未知数据
        -   所以要兼顾偏差和方差，通过不同建模策略，找到恰当模型，其复杂度不太大且误差在可接受的水平
        -   使得模型更贴近真实关系，泛化能力较好

> - *approximation error*：近似误差，模型偏差，代表模型对训练集的拟合程度
> - *estimation error*：估计误差，模型方差，代表模型对训练集波动的稳健性

-   衡量模型复杂度方式有很多种
    -   函数光滑限制
        -   多项式最高次数
    -   向量空间范数
        -   $\mathcal{L_0} - norm$：参数个数
            -   稀疏化约束
            -   解 $\mathcal{L_0}$ 范数正则化是 *NP-hard* 问题
        -   $\mathcal{L_1} - norm$：参数绝对值和
            -   $\mathcal{L_1}$ 范数可以通过凸松弛得到 $\mathcal{L_0}$ 的近似解
            -   有时候出现解不唯一的情况
            -   $\mathcal{L_1}$ 范数凸但不严格可导，可以使用依赖次梯度的方法求解极小化问题
        -   $\mathcal{L_2}- norm$：参数平方和
            -   凸且严格可导，极小化问题有解析解
        -   $\mathcal{L_1 + L_2}$
            -   有组效应，相关变量权重倾向于相同
    -   树模型中叶子节点数量

####    $L_1$ 范数类正则项

-   $L_1$ 范数稀疏性推广应用
    -   正负差异化：在正负设置权重不同的 $L_1$，赋予在正负不同的压缩能力，甚至某侧完全不压缩
    -   分段函数压缩：即只要保证在 0 点附近包含 $L_1$ 用于产生稀疏解，远离 0 处可以设计为常数等不影响精确解的值
        -   *Smoothly Clipped Absolute Deviation* 平滑检查绝对偏差
            $$ R(x|\lambda, \gamma) = \left \{ \begin{array} {l}
                \lambda|x| \qquad & if |x| \leq \lambda \\
                \frac {2\gamma\lambda|x| - x^2 - {\lambda}^2 } {2(\gamma - 1)} & if \gamma< |x| <\gamma\lambda \\
                \frac { {\lambda}^2(\gamma+1)} 2 & if |x| \geq \gamma\lambda
            \end{array} \right.$$
        -   *Derivate of SCAD*
            $$ R(x; \lambda, \gamma) = \left \{ \begin{array} {l}
                \lambda \qquad & if |x| \leq \gamma \\
                \frac {\gamma\lambda - |x|} {\gamma - 1} &
                    if \lambda < |x| < \gamma\lambda \\
                0 & if |x| \geq \gamma\lambda
            \end{array} \right.  $$
        -   *Minimax Concave Penalty*
            $$ R_{\gamma}(x;\lambda) = \left \{ \begin{array} {l}
                \lambda|x| - \frac {x^2} {2\gamma} \qquad &
                    if |x| \leq \gamma\lambda \\
                \frac 1 2 \gamma{\lambda}^2 &
                    if |x| > \gamma\lambda
            \end{array} \right. $$
    -   分指标：对不同指标动态设置 $\mathcal{L_0}$ 系数
        -   *Adaptive Lasso*：$\lambda \sum_J w_jx_j$

### 损失函数

-   损失函数可以视为**模型与真实的距离**的度量
    -   因此损失函数设计关键即，寻找可以代表模型与真实的距离的统计量
        -   对有监督学习：**真实** 已知，可以直接设计损失函数
        -   对无监督学习：**真实** 未知，需要给定 **真实标准**
            -   *NLP*：需要给出语言模型
            -   *EM* 算法：熵最大原理
    -   同时为求解方便，应该损失函数最好应满足导数存在
        -   *Surrogate Loss* 代理损失函数：用优化方便的损失函数代替难以优化的损失函数，间接达到优化原损失函数的目标
            -   如 0-1 损失难以优化，考虑使用二次损失、交叉熵损失替代

| 损失函数           | 逻辑                             | 适用场合             |
|--------------------|----------------------------------|----------------------|
|*0-1 Loss*          | $$
L(y, f(x)) = \left \{ \begin{array}{l}
    1, & y \neq f(x) \\
    0, & y = f(x)
\end{array} \right. $$                                  | 分类                 |
|*Squared Error Loss*| $$
L(y, f(x)) = \frac 1 2 (y - f(x))^2 $$                  | 回归、二分类         |
|*Logistic SE*       | $$
L(y, f(x)) = \frac 1 2 (y - \sigma(f(x)))^2 $$          | 二分类               |
|*Cross Entropy*     | $$\begin{align*}
L(y, f(x)) & = -ylog(f(x)) \\
& = - \sum_{k=1}^K y_k log f(x)_k
\end{align*}$$                                          | 分类、标签           |
|*Hinge Loss*        | $$\begin{align*}
L(y, f(x)) & = [1 - yf(x)]_{+} \\
[z]_{+} & = \left \{ \begin{array}{l}
    z, & z > 0 \\
    0, & z \leq 0
\end{array} \right.
\end{align*}$$                                          | *SVM*                |
|*Pseudo Loss*       | $$
L(y, f(x)) = \frac 1 2 \sum_{y^{(j)} \neq f(x)} w_j (1 - f(x, y) + f(x, y^{(j)})) $$ | *Adaboost.M2*|
| *Absolute Loss*    | $$ L(y, f(x)) = |y-f(x)| $$      | 回归                 |
| *Logarithmic Loss* | $$ L(y, P(y|x)) = -logP(y|x) $$  | 贝叶斯生成、逻辑回归 |
| *Exponential Loss* | $$ L(y, f(x)) = exp\{-yf(x)\} $$ | 前向分步算法         |
| *Absolute Loss*    | $$ L(y, f(x)) = |y - f(x)| $$    |                      |
| *Huber Loss*       | $$
L(y, f(x)) = \left \{ \begin{array}{l}
    \frac 1 2 (y - f(x))^2, & |y - f(x)| \leq \sigma \\
    \sigma (|y - f(x)| - \frac 2 {\sigma}), & |y - f(x)| > \sigma \\
\end{array} \right.$$                                   |                      |
| 分位数损失         | $$ L(y, f(x)) = \sum_{y \geq f(x)} \theta|y - f(x)|
+ \sum_{y < f(x)} (1-\theta)|y - f(x)| $$               | 分位数回归           |


![01_se_ce_hinge_loss](imgs/01_se_ce_hinge_loss.png)

####    *0-1 Loss*

$$ L(y, f(x)) = \left \{ \begin{array}{l}
    1, & y \neq f(x) \\
    0, & y = f(x)
\end{array} \right. $$

-   *0-1 Loss*
    -   0-1 损失函数梯度要么为 0、要么不存在，无法通过梯度下降方法优化 0-1 损失
    -   适用场合
        -   二分类：*Adaboost*
        -   多分类：*Adaboost.M1*

####    *Quadratic* / *Squared Error Loss*

$$\begin{align*}
L(y, f(x)) &= \frac 1 2 (y - f(x))^2 \\
L(y, f(x)) &= \frac 1 2 (y - \sigma(f(x)))^2
\end{align*}$$

-   *Quadratic*、*Squared Error Loss* 平方损失
    -   平方损失函数可导，可以基于梯度下降算法优化损失函数
    -   适用场合
        -   回归预测：线性回归
        -   分类预测：0-1 二分类（根据预测得分、阈值划分）

-   *Logistic SE* 对数平方损失：进行 *sigmoid* 变换后再应用平方损失
    -   若二分类模型输出大于 1，平方损失会错估模型，进行不必要、开销较大的优化
    -   *Logistic SE* 损失函数曲线对二分类拟合优于平方损失
        -   但负区间存在饱和问题，损失最大只有 0.5

####    *Cross Entropy*

$$\begin{align*}
L(y, f(x)) & = -ylog(f(x)) \\
& = - \sum_{k=1}^K y_k log f(x)_k
\end{align*}$$
> - $y$：样本实际值 0-1 向量
> - $f(x)$：各类别预测概率
> - $K$：分类数目

-   *Cross Entropy* 交叉熵损失
    -   交叉熵损失综合平方损失、*logistic SE* 优势，以正样本为例
        -   预测值较大时：损失接近 0，避免无效优化
        -   预测值较小时：损失偏导趋近于 -1，不会出现饱和现象
    -   适合场合
        -   分类问题：此时交叉熵损失同对数损失（负对数极大似然函数）
        -   标签问题

####    *Hinge Loss*

$$\begin{align*}
L(y, f(x)) & = [1 - yf(x)]_{+} \\
[z]_{+} & = \left \{ \begin{array}{l}
    z, & z > 0 \\
    0, & z \leq 0
\end{array} \right.
\end{align*}$$
> - $y \in \{-1, +1\}$

-   *Hinge Loss* 合页损失函数
    -   0-1 损失函数的上界，效果类似交叉熵损失函数
        -   要求分类不仅正确，还要求确信度足够高损失才为 0
        -   即对学习有更高的要求
    -   适用场合
        -   二分类：线性支持向量机

####    *Absolute Loss*

$$ L(y, f(x)) = |y-f(x)| $$

-   *Absolute Loss* 绝对损失函数
    -   适用场合
        -   回归预测

####    *Logarithmic Loss*

$$ L(y, P(y|x)) = -logP(y|x) $$

-   *Logarithmic Loss* 对数损失函数（负对数极大似然损失函数）
    -   适用场合
        -   多分类：贝叶斯生成模型、逻辑回归

####    *Exponential Loss*

$$ L(y, f(x)) = exp\{-yf(x)\} $$

-   *Exponential Loss* 指数函数函数
    -   适用场合
        -   二分类：前向分步算法

####    *Pseudo Loss*

$$ L(y, f(x)) = \frac 1 2 \sum_{y^{(j)} \neq f(x)} w_j (1 - f(x, y) + f(x, y^{(j)})) $$
> - $w_j$：样本个体错误标签权重，对不同个体分布可不同
> - $f(x, y^{(j)})$：分类器将输入 $x$ 预测为第 $j$ 类 $y^{(j)}$ 的置信度

-   *Pseudo Loss* 伪损失
    -   考虑个体损失 $(x_i, y_i)$ 如下，据此构造伪损失
        -   $h(x_i, y_i)=1, \sum h(x_i, y)=0$：完全正确预测
        -   $h(x_i, y_i)=0, \sum h(x_i, y)=1$：完全错误预测
        -   $h(x_i, y_i)=1/M$：随机预测（M为分类数目）
    -   伪损失函数考虑了预测 **标签** 的权重分布
        -   通过改变此分布，能够更明确的关注难以预测的个体标签，而不仅仅个体
    -   伪损失随着分类器预测准确率增加而减小
        -   分类器 $f$ 对所有可能类别输出置信度相同时，伪损失最大达到 0.5，此时就是随机预测
        -   伪损失大于 0.5 时，应该将使用 $1-f$
    -   适用场景
        -   多分类：*Adaboost.M2*

### 模型评价

-   给定损失函数时，基于损失函数的误差显然可以用于评估模型实例
    -   但损失函数的设计更多需要考虑求解，含义不够清晰
    -   回归预测模型
        -   *Squared Error*
            -   *MSE*
            -   $R^2$、$R^2_{Adj}$
            -   *AIC*
            -   *BIC*
        -   *Absolute Error*
            -   *MAE*
            -   *MAPE*
            -   *SMAPE*
    -   分类预测模型：模型误差主要是分类错误率 *ERR=1-ACC*
        -   混淆矩阵
            -   *F-Measure*
            -   *TPR*、*FPR*
        -   *PR Curve*
        -   *AUC*
            -   *ROC*
    -   *Tagging* 标注问题：类似分类问题
