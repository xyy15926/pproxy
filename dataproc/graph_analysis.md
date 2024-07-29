---
title: 
categories:
  - 
tags:
  - 
date: 2024-07-24 11:26:34
updated: 2024-07-29 22:34:20
toc: true
mathjax: true
description: 
---

##  现实网络

-   现实世界的任何问题都可以用复杂关系网络近似模拟
    -   *Social Network*：社交网络，表示个体之间的相互关系
        -   *Node*、*Vertex* 节点：研究问题中主体、人
        -   *Link*、*Edge*：人与人之间的 *Relation*，可以有标签、权重、方向
        -   个体社会影响力：社交网络中节点中心性
    -   无标度性：现实网络大多为 *Scale-Free Network* 无标度网络，即网络中节点度服从幂律分布
        -   幂次参数大多 $\gamma \in [2, 3]$ 
            -   *Triangle Power Law*：网络中三角形数量服从幂律分布
            -   *Tigenvalue Power Law*：网络邻接矩阵的特征值服从幂律分布
        -   网络中大部分节点度很小，小部分 hub 节点有很大的度
        -   对随机攻击稳健，但对目的攻击脆弱
    -   小世界性：短路径长度、大聚类系数
        -   小直径：与六度分离实验相符
        -   高聚类特性：具有较大点聚类系数
        -   存在巨大连通分支
    -   社区结构：明显的模块结构
    -   自相似性：绝大多数现实网络、网络结构模型虽然不能直观看出自相似性，但是在某种 *Length-Scale* 下确实具有自相似性
        -   万维网
        -   社会网络
        -   蛋白质交互作用网络
        -   细胞网络

> - *Giant Connected Component*：巨大连通分支，即规模达到 $O(N)$ 的连通分支
> - （网络结构）自相似性：局部在某种意义上与整体相似

-   社交网络：人、人与人之间的关系确定的网络结构
    -   有人类行为存在的任何领域都可以转化为社交网络形式
        -   *Offline Social Networks*：线下社交网络，现实面对面接触中的人类行为产生，人类起源时即出现
        -   *Online Social Networks*、*Social Webs*：在线社交网络
        -   *Social Media Websites*：多媒体网社交网
    -   定性特征：由于社交网络中人类主观因素的存在，定性特征可以用于社交网络分析
        -   关系强弱
        -   信任值
    -   量化指标：对网络结构的分析的数量化指标可以分析社交网络的基本特征
        -   度、度分布
        -   聚类系数
        -   路径长度
        -   网络直径
    -   数据分析类型
        -   *Content Data*：内容数据分析，文本、图像、其他多媒体数据
        -   *Linkage Data*：链接数据分析，网络的动力学行为：网络结构、个体之间沟通交流

-   社交网络中社区发现
    -   现实世界网络普遍具有模块、社区结构特性
        -   内部联系紧密、外部连接稀疏
        -   提取社区结构，研究其特性有助于在网络动态演化过程中理解、预测其自然出现的、关键的、具有因果关系的本质特性
    -   社区发现难点
        -   现实问题对应的关系网络
            -   拓扑结构类型未知
            -   大部分为随时间变化网络
            -   规模庞大
        -   现有技术方法应用受到限制
            -   多数方法适用静态无向图，研究有向网络、随时间动态
                演化网络形式技术方法较少
            -   传统算法可能不适用超大规模网络
    -   社区发现/探测重要性
        -   社区结构刻画了网络中连边关系的局部聚集特性，体现了连边的分布不均匀性
        -   社区通常由功能相近、性质相似的网络节点组成
            -   有助于揭示网络结构和功能之间的关系
            -   有助于更加有效的理解、开发网络

### 网络统计特性

-   基本统计指标、特性
    -   *Subnetwork*、*Subgraph*
        -   *Singleton*：单点集，没有边的子图
        -   *Clique*：派系，任意两个节点间均有连边的子图
    -   *Degree* 度：与节点相连的边数目
        -   *Average Degree*：网络平均度，所有节点度的算术平均
        -   *Degree Distribution*：网络度分布，概率分布函数 $P(k)$
        -   说明
            -   对有向图可以分为 *Out-Degree*、*In-Degree*
    -   *Path*
        -   *Path Length*：路径长度
        -   *Shortest Path*：节点间最短路径
        -   *Distance*：节点距离，节点间最短路径长度
        -   *Diameter*：网络直径，任意两个节点间距离最大值
    -   *Density* 网络密度：实际存在边数与可容纳边数上限壁比值
        -   说明
            -   刻画节点间相互连边的密集程度，衡量社交网络中的社交关系的密集程度及演化趋势
    -   *Clustering Coefficient* 聚类系数：与同一节点相连的节点也互为相邻节点的程度
        -   说明
            -   刻画社交网络的聚集性
    -   *Betweeness* 介数：节点承载图中所有最短路径的数量
        -   说明
            -   评价节点在网络信息传递中的重要程度

####    *Clustering Coefficient*

-   *Clustering Coefficient*
    -   *Node Clustering Coefficient*：点聚类系数
        $$\begin{align*}
        NC_i & = \frac {triangle_i} {triple_i} \\
        NC & = \frac {\sum_i NC_i} N
        \end{align*}$$
        > - $triangle_i$：包含节点 $i$ 三角形数量
        > - $triple_i$：与节点 $i$ 相连三元组，包含节点 $i$ 的三个节点，且至少存在节点 $i$ 到其他两个节点的两条边
        > - $NC_i$：节点 $i$ 聚类系数
        > - $NC$：整个网络聚类系数
    -   *Edge Clustering Coefficient*：边聚类系数
        $$ EC_{ij} = \frac {|包含边<i,j>三角形|} {min\{(d_i-1), (d_j-1)\}} $$
        > - $d_i$：节点 $i$ 度，即分母为边 $<i,j>$ 最大可能存在于的三角形数量

-   *Clique* 派系、社团：节点数大于等于 3 的全连通子图
    -   *N-Clique* 派系：任意两个顶点最多可以通过 $N-1$ 个中介点连接
        -   对派系定义的弱化
        -   允许两派系的重叠

####    *Edge Centrality*

-   *Edge Centrality* 边中心性
    -   *Edge Betweenness* 边介数：图中所有最短路径中包含该边的路径数量

-   边介数的计算
    -   从源节点 $i$ 出发，为每个节点 $j$ 维护距离源节点最短路径 $d_j$、从源节点出发经过其到达其他节点最短路径数目 $w_j$
        -   定义源节点 $i$ 距离 $d_i=0$、权值 $w_i=1$
        -   对源节点 $i$ 的邻接节点 $j$，定义其距离 $d_j=d_i+1$、权值 $w_j=w_i=1$
        -   对节点 $j$ 的任意邻接节点 $k$
            -   若 $k$ 未被指定距离，则指定其距离 $d_k=d_j+1$、权值 $w_k=w_j$
            -   若 $k$ 被指定距离且 $d_k=d_j+1$，则原权值增加 1，即 $w_k=w_k+1$
            -   若 $k$ 被指定距离且 $d_k<d_j+1$，则跳过
        -   重复以上直至网络中包含节点的连通子图中节点均被指定距离、权重
            -   此时连通子图中得到一棵广度优先搜索最小生成树
            -   树中存在叶子节点，不被任何从源节点出发到其他节点的最短路径经过
    -   从节点 $k$ 经过节点 $j$ 到达源节点 $i$ 的最短路径数目、与节点 $k$ 到达源节点 $i$ 的最短路径数目之比为 $\frac {w_i} {w_j}$
        -   从叶子节点 $l$ 开始，若叶子节点 $l$ 节点 $i$ 相邻，则将权值 $\frac {w_i} {w_l}$ 赋给边 $(i,l)$
        -   从底至上，边 $(i,j)$ 赋值为该边之下的邻边权值之和加 1 乘以 $\frac {w_i} {w_j}$
        -   重复直至遍历图中所有节点

####    *Node Centrality*

-   *Node Cetrality* 节点中心性：刻画节点处于网络中心地位的程度
    -   描述整个网络是否存在核心、核心的状态
    -   基于度的节点中心性
        -   *Degree Centrality*：度中心性
            $$ DC_i = \frac {d_i} {N-1} $$
            > - $d_i$：节点 $i$ 的度
            -   衡量节点对促进网络传播过程发挥的作用
        -   *Eigenvector Centrality*：特征向量中心性
        -   *Subgraph Centrality*：子图中心性
    -   基于路径数的节点中心性
        -   *Betweenness Centrality*：介数中心性
            $$ BC_i = \frac 2 {(N-1)(N-2)} \sum_{j<k, j,k \neq i} \frac {p_{j,k}(i)} {p_{j,k}} $$
            > - $p_{j,k}$：节点 $j,k$ 间路径数量
            > - $p_{j,k}(i)$：节点 $j,k$ 间路径经过节点 $i$ 路径数量
            -   衡量节点对其他节点间信息传输的潜在控制能力
        -   *Closeness Centrality*

### *Community Structure*

-   社区结构：内部联系紧密、外部联系稀疏（通过边数量体现）的子图，即满足如下条件子图 $G$
    $$\begin{align*}
    \sigma_{in}(S) & = \frac {E(S)}  {V(S)(V(S)-1)/2} \\
    \sigma_{out}(S) & = \frac {E(G-S)} {(V(G) - V(S))(V(G) - V(S) - 1) / 2} \\
    \sigma(G) & = \frac {E(G)} {V(G)(V(G)-1) / 2} \\
    \sigma_{in}(S) &> \sigma(G) > \sigma_{out}(S)
    \end{align*}$$
    > - $G, S$：全图、子图
    > - $E, V, E(S), V(S)$：全图、子图边数、节点数
    > - $\sigma_{in}(S), \sigma_{out}(S)$：子图 $S$ 的内部、与外部连接密度
    -   强社区结构
        $$ E_{in}(S, i) > E_{out}(S, i), \forall i \in S $$
        > - $E_{in}(S, i)$：节点 $i$ 和子图 $S$ 内节点连边数
        > - $E_{out}(S, i)$：节点 $i$ 和子图 $S$ 外节点连边数
    -   弱社区结构
        $$ \sum_{i \in S} E_{in}(S, i) > \sum_{i \in S} E_{out}(S, i), \forall i \in S $$
    -   最弱社区结构
        $$ \forall i \in S_j, E_{in}(S_j, i) > E(S_j, i, S_k), j \neq k, k=1,2,\cdots,M $$
        > - 社区 $S_1,S_2,\cdots,S_M$ 是网络 $G$ 中社区
        > - $E(S_j, i, S_k)$：子图 $S_j$ 中节点 $i$ 与子图 $S_k$ 之间连边数
    -   改进的弱社区结构：同时满足弱社区结构、最弱社区结构
    -   *LS* 集：任何真子集与集合内部连边数都多于与集合外部连边数的节点集合

####    *Modularity*

-   *Modulariyt* 模块度 *Q1*：社区内部实际边数量与期望边数量的差值
    $$\begin{align*}
    Q1 & = \sum_{i} (e_{i,i} - \hat e_{i,i}) \\
    & = \sum_{i} (e_{i,i} - a_i^2) \\
    & = \sum_{i} (e_{i,i} - \sum_j e_{i,j}^2) \\
    & = Tr(e) - \sum_{i,j} e_{i,j}^2
    \end{align*}$$
    > - $e_{i,j}$：社区 $i,j$ 中节点间连边数在所有边中所占比例
    > - $\hat e_{i,j} = a_i a_j$：随机网络中社区 $i,j$ 连边数占比期望，特别的 $\hat e_{i,i} = a_i^2$
    > - $a_i$：社区 $i$ 中节点全部连边数占比，显然有 $a_i^2 = \sum_j e_{i,j}^2$
    > - $Tr(e) = \sum_i e_{i,i}$：社区边占比矩阵 $[e_{i,j}]$ 的迹，即社区内边占比
    -   理论上 $Q1 \in [-1, 1]$，划分对应 *Q* 值越大，划分效果越好
        -   但，端点 $\pm$ 不易取得
        -   $Q1 > 0$ 时即表明社区结构存在
        -   特别的，$Q1 = 0$ 时即为将所有点放在同一社区的平凡划分

-   考虑节点度的模块度 *Q2*：在 *Q1* 基础上考虑单个节点度，而不是社区整体边
    $$\begin{align*}
    Q2 &= \frac 1 m (\frac 1 2 \sum_{i,j} A_{i,j} \sigma_{i,j} - \frac 1 2 \sum_{i,j} P_{i,j} \sigma_{i,j}) \\
    & = \frac 1 {2m} \sum_{i,j}(A_{i,j} - P_{i,j})\sigma_{i,j} \\
    & = \frac 1 2 \sum_{i,j} (A_{i,j} - \frac {d_i d_j} {2m}) \sigma_{i,j}
    \end{align*}$$
    > - $A_{i,j}$：节点 $i,j$ 是否存在边，即 $A$ 为邻接矩阵
    > - $\sigma_{i,j}$：节点 $i,j$ 是否属于同一社区
    > - $P_{i,j}$：节点 $i,j$ 相连接的概率
    > - $d_i$：节点 $i$ 的度
    -   理论上 $Q2 \in [-0.5, 1]$
        -   实际网络中，好的划分时 $Q2 \in [0.3, 0.7]$
        -   特别的，$Q2 = 0$ 时即为将所有点放在同一社区的平凡划分
    -   问题
        -   *Q* 值分辨能力有限，网络中规模较大社区会掩盖小社区，即使其内部连接紧密
        -   采用完全随机形式，无法避免重边、自环的存在，而现实网络研究常采用简单图，所以 *Q* 值存在局限

-   广义模块度 *Generalized Modularity*
    $$ Q_{\gamma} = \frac 1 2 \sum_{i,j} (A_{i,j} - \gamma \frac {d_i d_j} {2m}) \sigma_{i,j} $$
    > - $\gamma$：分辨率参数，取较小值时倾向将小社团合并为大社团

> - 模块度：<https://juejin.cn/post/6844904120202035213>
> - 模块度 *Modularity* 的发展历程：<https://qinyuenlp.com/article/fba09bc9bda7/>（其中 *Q2* 中 $\sigma$ 函数定义有问题）

####    其他评价

-   社区密度 *D*：社区内部连边、社区间连边之差与社区节点总数之比
    $$\begin{align*}
    D & = \sum_{i=1}^M d(S_i) \\
    & = \sum_{i=1}^M \frac {E_{in}(S_i) - E_{out}(S_i)} {V_{in}(S_i)}
    \end{align*}$$
    > - $M$：社区数量

-   社区度 *C*
    $$ C = \frac 1 M \sum_{i=1}^M [\frac {E_{in}(S_i)}
        {V(S_i)(V(S_i) - 1) / 2} - \frac {E_{out}(S_i)}
        {V(S_i) (V - V(S_i))})] $$
    > - $\frac {E_{in}(S_i)} {V(S_i)(V(S_i)-1)/2}$：社区 $S_i$ 的簇内密度
    > - $\frac {E_{out}(S_i)} {V(S_i)(V-V(S_i))}$：社区 $S_i$ 与其他社区的间密度

-   *Fitness* 函数
    $$\begin{align*}
    f_i & = \frac {d_{in}(S_i)} {d_{in}(S_i) + d_{out}(S_i)} \\
    & = \frac {2 * E_{in}(S_i)} {2 * E_{in}(S_i) + E_{out}(S_i)} \\
    \bar f &= \frac 1 M \sum_{i=1}^M f_i
    \end{align*}$$
    > - $d_{in}(S_i) = 2 * E_{in}(S_i)$：社区 $S_i$ 内部度
    > - $d_{out}(S_i) = E_{out}(S_i)$：社区 $S_i$ 外部度
    > - $\bar f$：整个网络社区划分的 *Fitness* 函数

### 社区发现算法

-   社区发现算法
    -   *Agglomerative Method*：凝聚算法
        -   凝聚算法流程可用世系图表示
            -   最初每个节点各自成为独立社区
            -   按某种方法计算各社区之间相似性，选择相似性最高的社区合并
                -   相关系数
                -   路径长度
                -   矩阵方法
            -   不断重复，直至满足某个度量标准，此时节点聚合情况即为网络中社区结构
        -   典型算法
            -   *Newman Fast* 算法
            -   *Walk Trap* 随机游走算法
    -   *Division Method*：分裂算法
        -   分裂算法流程同凝聚算法相反
        -   典型算法
            -   *Girvan-Newman* 算法
            -   *Edge-Clustering Detection Algorithm* 边聚类探测算法
        -   *Label Propagation* 标签扩散算法

####    *Girvan-Newman*算法

-   *Girvan-Newman* 算法
    -   流程
        -   计算网络中各边相对于可能源节点的边介数
        -   删除网络中边介数较大的边，每当分裂出新社区（即产生新连通分支）
            -   计算网络的社区结构评价指标
            -   记录对应网络结构
        -   重复直到网络中边都被删除，每个节点为单独社区，选择最优评价指标的网络结构作为网络最终分裂状态
    -   缺点
        -   计算速度满，边介数计算开销大，只适合处理中小规模网络

####    *Newman Fast Algorithm*

-   *NF* 快速算法：
    -   流程
        -   初始化网络中各个节点为独立社区、矩阵 $E=\{e_{i,j}\}$
            $$\begin{align*}
            e_{i,j} & = \left \{ \begin{array}{l}
                \frac 1 {2M}, & 边(i,j)存在 \\
                0, & 节点间不存在边
            \end{array} \right. \\
            a_i & = \frac {d_i} {2M}
            \end{align*}$$
            > - $M$：网络中边总数
            > - $e_{i,j}$：网络中社区 $i,j$ 节点边在所有边中占比
            > - $a_i$：与社区 $i$ 中节点相连边在所有边中占比
        -   依次合并有边相连的社区对，计算合并后模块度增量
            $$ \Delta Q = e_{i,j} + e_{j,i} = 2(e_{i,j} - a_i a_j) $$
            -   根据贪婪思想，每次沿使得 $Q$ 增加最多、减少最小方向进行
            -   每次合并后更新元素 $e_{i,j}$，将合并社区相关行、列相加
            -   计算网络社区结构评价指标、网络结构
        -   重复直至整个网络合并成为一个社区，选择最优评价指标对应网络社区结构

-   说明
    -   *GN* 算法、*NF* 算法大多使用无权网络，一个可行的方案是计算无权情况下各边介数，加权网络中各边介数为无权情况下个边介数除以边权重
        -   此时，边权重越大介数越小，被移除概率越小，符合社区结构划分定义

####    *Edge-Clustering Detection Algorithm*

-   *Edge-Clustering Detection Algorithm* 边聚类探测算法
    -   流程
        -   计算网络中尚存的边聚类系数值
        -   移除边聚类系数值最小者 $(i,j)$，每当分裂出新社区（即产生新连通分支）
            -   计算网络社区评价指标 *fitness*、*modularity*
            -   记录对应网络结构
        -   重复直到网络中边都被删除，每个节点为单独社区，选择最优评价指标的网络结构作为网络最终分裂状态

##  *Random Walk*

-   （网络）随机游走：
    -   游走形式
        -   *Unbiased Random Walks*：无偏随机游走，等概率游走
        -   *Biased Random Walks*：有偏随机游走，正比于节点度
        -   *Self-Avoid Walks*：自规避随机游走
        -   *Quantum Walks*：量子游走
    -   研究内容
        -   *First-Passage Time*：平均首达时间
        -   *Mean Commute Time*：平均转移时间
        -   *Mean Return Time*：平均返回时间
    -   用途
        -   *Community Detection*：社区探测
        -   *Recommendation Systems*：推荐系统
        -   *Electrical Networks*：电力系统
        -   *Spanning Trees*：生成树
        -   *Infomation Retrieval*：信息检索
        -   *Natural Language Proessing*：自然语言处理
        -   *Graph Partitioning*：图像分割
        -   *Random Walk Hypothesis*：随机游走假设（经济学）
        -   *Pagerank Algorithm*：PageRank算法

