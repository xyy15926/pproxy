---
title: 算法分析
categories:
  - Algorithm
tags:
  - Algorithm
date: 2019-03-21 17:27:15
updated: 2024-07-09 11:16:17
toc: true
mathjax: true
comments: true
description: 算法分析
---

##  基础

-   算法：一系列解决问题的 **明确** 指令
    -   即，对于符合一定规范的输入，能够在有限时间内获得要求的输出
        -   无歧义：算法每步都必须没有歧义
        -   给定输入：算法所处理的输入需确定
    -   算法说明
        -   同一问题可存在不同的算法，对应不同思路、不同速度
        -   同一算法可用不同的形式描述

### 算法正确性

-   算法正确性证明
    -   常用数学归纳法证明：算法的迭代过程本身就符合其所需的一系列步骤
        -   证明过程简单、或复杂取决于算法
        -   不应根据特定输入下算法行为判断算法正确性
        -   算法错误则仅需反例
    -   对近似算法，常常证明算法误差不超出预定义的误差

### 算法确定性

-   *Deterministic Alogrithm* 确定算法：利用问题解析性质，产生确定的有限、无限序列使其收敛于全局最优解
    -   依某确定性策略搜索局部极小，试图跳跃已获得的局部极小而达到某个全局最优点
    -   能充分利用问题解析性质，从计算效率高

-   *Nondeterministic Algorithm* 不确定算法：算法行为不确定，具体行为取决于问题的实例 *L*
    -   不确定算法每轮包括两个阶段：对问题实例 *L*
        -   猜测（非确定）阶段：*L* 作为输入，生成任意串 *S* 作为 *L* 候选解
        -   验证（确定）阶段：把 *L*、*S* 作为输入
            -   若 *S* 是 *L* 解，则问题实例 *L* 有解 *S*
            -   否则，回到猜测阶段、或无解终止算法
    -   称算法能求解此问题，当且仅当对问题每个真实例，不确定算法会在某次执行中返回是
    -   *Nondeterministic Polynominal Algorithm*：验证阶段时间效率是多项式级的不确定算法

### 算法分析框架

-   算法分析方向
    -   *Time Efficiency* 时间效率：运行速度
        -   算法分析框架中最重点方向
    -   *Space Efficiency* 空间效率：所需额外的存储空间
        -   算法分析框架中次重点方向
        -   但随硬件进步，重要性逐渐下降
    -   *Simplicity* 简单性
        -   容易理解实现
        -   *Bug* 更少
    -   *Generality* 一般性
        -   所解决问题的一般性
        -   所接受输入的一般性
    -   *Optimality* 最优性：仅与待解决问题的复杂度有关，与具体算法效率无关
    -   可解决性：问题是否可用算法解决

##  算法时间效率

-   算法运行时间效率影响因素
    -   输入规模 $N$：大部分算法运行时间随输入规模同步增长
        -   输入规模度量方式取决于问题
        -   但，常用二进制位数 $b=\lfloor {log_{2}^{N}} \rfloor + 1$ 表示
    -   输入细节：算法的运行时间有时取决于特定输入的细节，从此角度算法时间效率可分为
        -   *Worst-case Efficiency*：最坏情况下的效率
        -   *Best-case Efficiency*：最优情况下的效率
        -   *Average-case Efficiency*：平均效率
            -   可将输入按执行效率按分类，根据输入细节的概率分布计算加权平均效率
        -   *Amortized Efficiency*：均摊效率，可视为特殊的平均效率
            -   均摊效率仅强调某些算法效率变化有周期规律，而据此计算平均效率
            -   即，仅在某些算法（场合）中存在均摊效率（即平均效率）

> - 平均、均摊复杂度说明：<http://laomst.site/article/20>

-   时间效率衡量：*Basic Operation* 运行次数
    -   基本操作运行次数可用 $T(n)=c_{op}C(n)$ 计算
        -   $c_{op}$：特定基本操作的执行时间
        -   $C(n)$：算法执行基本操作的次数
            -   排序基本操作为键比较
            -   数学问题则是四则运算（但除法、乘法、加减法耗时不同）
    -   优、缺点
        -   不依赖于其他无关因素
        -   对总运行时间贡献最大，不需要统计算法每步操作执行次数
        -   $C(n), c_{op}$ 中均为估计结果，不可靠
    -   不采用算法实际运行时间，因为
        -   算法实际运行时间依赖于：计算机速度、程序实现质量、编译器
        -   计时困难

### 渐进效率

-   算法的渐进效率：基本操作的 *Order of Growth* 、乘法常量
    -   对于实际类型输入，大部分算法中乘法常量相差不大
        -   小规模输入运行时间差别难以区分高效算法、低效算法
        -   大规模输入忽略乘法常量
        -   即使是中等规模的输入，较优渐进效率类型的算法时间效率也较高
    -   渐进符号常用于描述算法的渐进效率
        -   忽略函数中增长较慢部分、各项系数
        -   保留可用于表明函数增长趋势的重要部分

> - *OI-Wiki* 算法复杂度：<https://oi-wiki.org/basic/complexity/>

####    渐进符号

-   渐进符号：函数阶的严格规范描述
    -   $T(n) \in \Theta(g(n))$ 当且仅当 $\exists c_1, c_2, n_0 > 0$，使得 $\forall n \geq n_0, 0 \leq c_1 g(n) \leq T(n) \leq c_1 g(n)$
        -   同时给出函数的上下界，即能找到正数 $c_1, c_2$ 使得 $T(n)$ 夹在 $c_1 g(n)$、$c_2 g(n)$ 中间
        -   即，$\Theta(g(n))$ 为增长次数等于 $g(n), n \rightarrow \infty$ （及常数倍）的函数集合
        -   另外，$T(n) \in \Theta(g(n))$ 等价于 $T(n) \in O(g(n)) \wedge T(n) \in \Omega(g(n))$
        -   若，函数之间严格不等，则可改写为 $T(n) \in \theta(g(n))$
    -   $T(n) \in O(g(n))$ 当且仅当 $\exists c, n_0 > 0$，使得 $\forall n \geq n_0, 0 \leq T(n) \leq c g(n)$
        -   仅给出函数上界，研究算法复杂度时常用（复杂度通常更关心上界）
        -   即，$O(g(n))$ 为增长次数小于 $g(n), n \rightarrow \infty$ （及常数倍）的函数集合
        -   若，函数之间严格不等，则可改写为 $T(n) \in o(g(n))$
    -   $T(n) \in \Omega(g(n))$ 当且仅当 $\exists c, n_0 > 0$，使得 $\forall n \geq n_0, 0 \leq c g(n) \leq T(n)$
        -   仅给出函数下界
        -   即，$\Omega(g(n))$ 为增长次数大于 $g(n), n \rightarrow \infty$ （及常数倍）的函数集合
        -   若，函数之间严格不等，则可改写为 $T(n) \in \omega(g(n))$

####    渐进效率计算

-   可利用比值极限比较函数增长次数
    -   比直接利用定义判断算法的增长次数方便，可以使用微积分技术计算极限

    $$
    \lim_{n \rightarrow \infty} \frac {t(n)} {g_n}
    \left\{
        \begin{array}\\
        0       & t(n)的增长次数比g(n)小，t(n) \in O(g(n))\\
        c>0     & t(n)的增长次数同g(n)，t(n) \in \Theta(g(n)) \\
        \infty  & t(n)的增长次数比g(n)大，t(n) \in \Omega(g(n))\\
        不存在 \\
        \end{array}
    \right.
    $$

####    基本渐进效率类型

|类型|名称|注释|
|------|------|------|
|$1$|常量|很少，效率最高|
|$log_{n}$|对数|算法的每次循环都会消去问题规模的常数因子，对数算法不可能关注输入的每个部分|
|$n$|线性|能关注输入每个部分的算法至少是线性运行时间|
|$nlog_{n}$|线性对数|许多分治算法都属于此类型|
|$n^{2}$|平方|包含两重嵌套循环的典型效率|
|$n^{3}$|立方|包含三重嵌套循环的典型效率|
|$2^{n}$|指数|求n个元素的所有子集|
|$n!$|阶乘|n个元素集合的全排列|

### 算法的数学分析

-   非递归算法分析通用方案
    -   找出算法的基本操作：一般位于算法最内层循环
    -   检查算法基本操作执行次数是否只依赖于输入规模
        -   决定表示输入规模的参数
        -   若和其他特性有关，需要分别研究最差、最优、平均效率
    -   建立算法基本操作执行次数的求和表达式（或者是递推关系）
        -   利用求和运算的标准公式、法则建立操作次数的闭合公式
        -   或，至少确定其增长次数

-   递归算法常通过建立、求解递推式确定时间复杂度
    -   分析算法特征，建立算法复杂度递推式、确定初始值
    -   求解递推式方法
        -   猜测通项公式、验证、数学归纳法证明
        -   直接根据递推式求解

####    *Master Theorem*

-   *Master Theorem* 主定理：若 $T(n) = aT(\frac n b) + f(n), f(n) \in \Theta(n^d)$，则有如下递推式

    $$ T(n) \in \left\{ \begin{array}\\
        \Theta(n^d)         & a<b^d \\
        \Theta(n^d log^n)   & a=b^d \\
        \Theta(n^{log_b^a}) & a>b^d \\
    \end{array} \right. $$

    -   此版本主定理清晰、简单，但注意
        -   $f(n)$ 被预设为特定阶数，可分类讨论 $f(n)$ 与 $n^{log_b^a}$ 阶数
        -   应满足条件：$a>1, b>1, T(1)>0$
    -   主定理可方便对分治法、减常因子法效率进行分析

##  问题（的算法）下界

-   问题（的算法）下界：求解问题的时间效率极限
    -   可用于评价问题的某具体算法的效率
    -   寻找问题的更优算法时，可根据算法下界确定期望获得的改进
        -   若算法下界是紧密的，则改进至多不过是常数因子
        -   若算法下界和算法仍有差距，则可能存在更快算法，或者是证明更好的下界

-   问题下界的讨论方向
    -   *Trivial Lower Bound* 平凡下界：任何算法只要要“读取”所有要处理的项、“写”全部输出，对其计数即可得到平凡下界
        -   大部分情况下，此下界过小，用处不大
        -   例
            -   输出全排列的平凡下界 $\Omega(n!)$ 为紧密下界
            -   方阵乘法的平凡下界 $\Omega(n^2)$
    -   *Information-Theoretic Lower Bound* 信息论下界：通过算法必须处理的信息量（比特数）建立的效率下界
        -   例
            -   猜整数的信息论下界：整数的不确定信息量 $\lceil \log_2 n \rceil$（数字二进制位数）
    -   *Adversary Lower Bound* 敌手下界：基于恶意、一致性迫使算法尽可能多执行的下界
        -   也即算法最差效率下界
            -   恶意将算法推向最消耗时间的路径
            -   一致要求问题、解答已确定
        -   例
            -   猜整数中每次排除均只能排除数字较少集合
            -   归并排序中的归并步骤中数据比较次数拉满

### 转换问题确定下界

|问题|下界|紧密性|
|-----|-----|-----|
|排序|$\Omega(nlogn)$|是|
|有序数组查找|$\Omega(logn)$|是|
|元素惟一性|$\Omega(nlogn)$|是|
|n 位整数乘法|$\Omega(n)$|未知|
|n 阶方程点乘|$\Omega(n^2)$|未知|

-   问题转换：若问题 *Q* 下界已知，可将问题 *Q* 转换为 *P*，得到 *P* 下界
    -   任意 *Q* 问题实例可以转换为 *P* 问题，即 $Q \subseteq P$
    -   但问题复杂度的直观判断和问题表现形式相关，问题复杂性不清楚时不可靠
    -   例
        -   平面欧几里得最小生成树：使用元素唯一性问题作为下界已知问题
        -   矩阵乘法：使用方阵乘法作为下界已知问题

            $$
            X =  \begin{bmatrix} 0 & A \\ A^T & 0 \\ \end{bmatrix},
            Y = \begin{bmatrix} 0 & B^T \\ B & 0 \\ \end{bmatrix} \\
            XY = \begin{bmatrix} AB & 0 \\ 0 & A^TB^T \\ \end{bmatrix}
            $$

### 基于比较算法的问题下界

-   可以使用二叉树研究基于比较的算法性能
    -   算法操作可视为沿着决策树根到叶子节点路径
        -   非叶子节点：代表一次键值比较
        -   叶子节点：代表一次结果输出
            -   个数大于等于输出
            -   不同叶子节点可以产生相同输出
    -   则，最坏情况下比较次数等于算法决策树高度（规模为 $n$）
        -   即得到信息论下界 $h \leqslant \lceil log_2 l \rceil$

####    案例：线性表排序

-   线性表排序（基于比较）问题
    -   $n$ 个元素列表排序输出数量等于 $n!$，则最坏情况下比较次数

        $$ \begin{align*}
        C_{worst} & \geqslant \lceil log_2 n! \rceil \\
            & \approx log_2 \sqrt {2\pi n} (n/e)^n  \\
            & = nlog_2n - nlog_2e + \frac {log_2n + log_22\pi} 2 \\
            & \approx nlog_2n
        \end{align*} $$

    -   归并排序、堆排序在最坏情况下大约必须要做 $nlog_2^n$ 次比较，所以其渐进效率最优
        -   也即渐进下界 $\lceil log_2n! \rceil$ 是紧密的，不能继续改进
        -   此只是基于二叉决策树的渐进下界，对于具体值估计可能不准

-   也可使用二叉树分析基于比较的排序算法的平均性能，即决策树叶子节点平均深度
    -   基于排序的所有输出都不特殊的标准假设，可以证明平均比较次数下界 $C_{avg}(n) \geqslant log_2n!$
    -   这个平均是建立在所有输出都不特殊假设上，所以这个其实应该是**不同算法平均比较次数下界的上界**
    -   对于单独排序算法，平均效率会明显好于最差效率

####    案例：有序线性表查找

-   有序线性表查找（基于比较）：最主要算法是折半查找，其在最坏情况下下效率

    $$C_{worst}^{bs} = \lfloor log_2n \rfloor + 1 = \lceil log{(n+1)} \rceil$$

    -   折半查找使用的三路比较（小于、等于、大于），可使用三叉查找树表示
        -   三叉查找树会有 $2n+1$ 个节点：$n$ 个查找成功节点、$n+1$ 个查找失败节点
        -   所以在最坏情况下，比较次数下界 $C_{worst}(n) \geqslant \lceil log_3{(2n+1)} \rceil$
            小于折半查找最坏情况下比较次数（渐进）
    -   事实上，可以删除三叉查找树各节点等于分支，得到二叉树
        -   非叶子节点同样表示三路比较，只是同时作为查找成功终点
        -   可以得到一个新的下界 $C_{worst}(n) \geqslant \lceil log_2{n+1} \rceil$

-   更复杂的分析表明：标准查找假设下，折半查找平均情况比较次数是最少的
    -   查找成功时 $log_2n - 1$
    -   查找失败时 $log_2(n+1)$

##  *P*、*NP*、完全 *NP* 问题

### 复杂性理论

-   如果算法的最差时间效率 $\in O(p(n))$，其中 $p(n)$ 为问题输入规模 $n$ 的多项式函数，则称算法能在多项式时间内对问题求解
    -   *Tractable*：易解的，可以在多项式时间内求解的问题
    -   *Intractable*：难解的，不能在多项式内求解的问题

-   使用多项式函数理由
    -   多项式函数具有方便的特性：多项式加和、组合也为多项式
    -   多项式类型可以发展出 *Computational Complexity* 利用
        -   该理论试图对问题内在复杂性进行分类
        -   只要使用一种主要计算模型描述问题，并用合理编码方案描述输入，问题难解性都是相同的
    -   对实用类型的算法而言，其多项式次数很少大于 3
        -   虽然多项式次数相差很大时运行时间也会有巨大差别
        -   否则，无法保证在合理时间内对难解问题所有实例求解，除非问题实例非常小

### *Decision Problem*

-   判定问题：寻求一种可行的、机械的算法，能够对某类问题在有穷步骤内确定是否具有某性质
    -   *Undecidable*问题：不可判定问题，不能使用任何算法求解的判定问题
        -   停机问题：给定程序、输入，判断程序对于输入停止还是无限运行下去
    -   *Decidable*问题：可判定问题，能用算法求解的问题
        -   可判定、难解问题存在，但非常少
        -   很多判定问题（或者可以转化为等价判定问题），既没有找到多项式类型算法，也没有证明这样算法不存在，即无法判断是否难解

### *P*、*NP*、*NPC*、*NP-Hard*

-   *Polynomial* 类型问题：能够用确定性算法在多项式时间内求解的
    -   也即，易解的问题，即能够在多项式时间内求解的问题（计算机范畴）
        -   多项式时间内求解：排除难解问题
        -   判定问题：很多重要问题可以化简为更容易研究的判断问题，虽然原始表达形式不是判定问题

-   *Nondeterministic Polynomial* 类型问题：可以用不确定多项式算法求解（即多项式时间内验证）的判定问题
    -   *NP* 问题计算上求解困难，但计算上判定待定结果简单：可以在多项式时间内完成
    -   大多数判断问题都是属于 *NP* 类型的
        -   $P \subseteq NP$
        -   以下没有找到、且未证明不存在多项式算法的组合优化问题的判定版本
            -   哈密顿回路问题
            -   旅行商问题
            -   背包问题
            -   划分问题
            -   装箱问题
            -   图着色问题
            -   整数线性规划问题
    -   $P \overset ? = NP$：*P* 问题、*NP* 问题是否一致有待证明

-   *Polynomially Reducible* 多项式规约：判定问题 $D_1$ 可以多项式规约为判定问题 $D_2$，仅当存在函数 $f$ 将 $D_1$ 实例转换为 $D_2$ 实例，满足
    -   $f$ 将 $D_1$ 所有真实例映射为 $D_2$ 真实例，把 $D_1$ 所有假实例映射为 $D_2$ 假实例
    -   $f$ 可用多项式算法计算

####    *NPC*、*NP-Hard*

![p_np_npc_nphard](imgs/p_np_npc_nphard.png)

-   *NP Complete* 问题：*NP* 中其他任何问题（已知或未知）可以在多项式时间内规约为 *NPC* 问题
    -   属于 *NP* 问题，所有 *NPC* 问题难度一致
    -   若，任意 *NPC* 问题存在多项式确定算法
        -   则所有 *NP* 问题可以在多项式时间内求解，即 $P = NP$
        -   即，对于所有类型判定问题，验证待定解、在多项式时间内求解在复杂性上没有本质区别
        -   而，*NPC* 问题可以被其他 *NP* 问题转换，*NPC* 问题目前不存在对所有实例通用的多项式时间算法
    -   直接证明任何 *NP* 问题都可以在多项式时间内化简为当前问题比较困难
        -   常常利用多项式规约特性，证明某个确定 *NPC* 问题可以多项式规约为当前问题
        -   例：哈密顿回路问题转换为旅商问题
            -   哈密顿回路中图 $G$ 映射为加权图 $G^{'}$，其中存在边在权重为 1，不存在边权重为 2
            -   则，哈密顿回路问题转换为 $G^{'}$ 是否存在长度不超过 $|V|$ 的哈密顿回路，即旅商问题的等价
    -   *NPC* 问题案例
        -   *CNF-Satisfied Problem* 合取范式满足性问题：能否设置合取范式类型的布尔表达式中布尔变量值，使得整个表达式值为真
            -   首个被发现 *NPC* 问题
            -   每个布尔表达式都可以表达为合取范式
        -   前述组合优化 *NP* 问题的判定版本

-   *NP-Hard* 问题：可以在多项式时间内规约为 *NPC* 问题的非 *NP* 问题
    -   即，与 *NPC* 问题同样困难的非判定问题
    -   *NP-Hard* 问题包括 *NPC* 问题
        -   前述组合优化问题最优版本即 *NP-Hard* 问题

##  算法设计思路
#TODO

-   通用设计技术
    -   复用结果
        -   回溯：栈嵌套控制调用跳转、结果复用
        -   动态规划：键查询控制调用跳转、结果复用
            -   动态规划表存储历史结果
    -   剪枝：缩小搜索空间
        -   固有约束：问题中包含约束条件
        -   分支界限：已搜索分支中最优解动态限制，仅适用于最优解问题
    -   变治：问题变换
        -   规模划分：划分问题，求解子问题
            -   分治：划分问题，缩减问题规模，独立解决子问题、合并子问题解
                -   合并子问题时，根据问题性质仅考虑子问题间部分可行解
                -   即，分治仅可缩减 **子问题间可行解** 搜索空间
                -   即，分治适合优化超线性复杂度、最优化问题
            -   减治：划分问题为需求解、无需求解两部分，缩减问题规模
        -   输入增强：利用数据结构秩序性简化问题
            -   预排序表
            -   有序树：二叉树、堆
            -   Hash 数据结构：查询是常数时间，适合缓冲记录结果，记录存在性、累计数量
        -   位运算：将问题转换为二进制位运算，可能同时结合了规模划分、输入增强特点
            -   输入增强：原始问题转换为位运算问题
            -   规模划分：问题划分为单位长度个子问题
                -   但位运算的规模划分往往不缩减搜索空间，仅利用计算机硬件特性降低常数复杂度

##  注意事项

### 溢出

-   数值运算
    -   运算溢出
        -   正整数：32 位边界值 `1`、`0x7FFF FFFF`、`0xFFFF FFFF`
        -   负整数：32 位边界值 `0x8000 0000`、`0xFFFF FFFF`
        -   忽略语言特性：如 `long` 类型常量不加 `LL`
    -   浮点型相等比较
        -   避免使用精确相等 `==`，应该做差与常数比较大小
            ```python
            a, b = 1.11111, 1.11111111
            if abs(a - b) < 0.00001:
                print("equal")
            ```

### *Corner Cases*

-   字符串
    -   空字符串
    -   长度1字符串、长度2字符串
    -   字符相同字符串

-   普通序列
    -   空列表
        -   若在循环外即已取值，应该提前判断列表是否空
    -   长度1列表、长度2列表
    -   元素相同列表

-   树、二叉树
    -   空树、只有根元素
    -   只有左子树、右子树

-   文件边界条件
    -   首个字符
    -   最末字符、倒数第二个字符

### 终止条件

-   循环终止条件
    -   `>=`、`<=` 逻辑判断
        -   同时使用 `<=`、`>=`，可能造成死循环、遗漏边界等
        -   若可行，避免同时使用 `>=`、`<=`
    -   常用终止条件
        -   是否为初值
        -   是否越界
    -   常用初值
        -   数值型：`0`、`-1`、`sys.maxsize`、`float('inf')`
        -   字符串型：`""`
        -   迭代过程中可能取值：输出列表首个元素

-   递归终止条件主要有两种设计方案
    -   最后元素：判断元素是否是最后元素，终止递归调用
    -   空（无效）元素：判断元素是空（无效）元素，终止递归调用
        -   需要确保最终能够进入该分支，而不是停止在最后一步

### 风格

-   风格、习惯
    -   保持程序设计风格：把经常使用的工具性代码编辑成已验证
    -   用规范的格式处理、保存数据
    -   区分代码与数据：与代码无关的数据应该尽可能区分开来，尽量把数据保存在常量数组中
        -   避免将大局部变量保存到栈中，少量递归即会发生栈溢出

####    输入、输出优化

-   输入、输出优化
    -   将输入、输出流重定向到文件中，避免频繁测试时频繁输入
        -   输入放在 `in.txt` 文件中
        -   输出到 `out.txt` 中
    -   输出执行时间
    -   大量输入、输出时，避免使用 `cin`、`cout` 等高级输入、输出方式

    ```cpp
    #ifdef SUBMIT
    freopen("in.txt", "r", stdin);
        // input data
    freopen("out.txt", "w", stdout);
        // output
    long _begin_time = clock();
    #endif

    // put code here

    #ifdef SUBMIT
    long _end_time = clock();
    printf("time = %ld ms\n", _end_time - begin_time);
    #endif
    ```

    ```python
    import sys
    import time
    sys.stdin = open("in.txt", "r")
    sys.stdout = open("out.txt", "w")
    __tstart = time.time()

     # code here

    __trange = time.time() - __tstart
    print("time used: %f" % __trange)
    ```
