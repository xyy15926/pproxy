---
title: 计算机视觉
categories:
  - 
tags:
  - 
date: 2024-07-22 20:38:17
updated: 2024-07-22 21:41:12
toc: true
mathjax: true
description: 
---

#  图像特征提取

> - 传统的图像特征提取方法由人工预设算子权重，现在常用 *CNN* 自动学习特征提取算子权重

##  *Local Binary Pattern*

-   *Local Binary Pattern* 局部二值模式：描述图像局部纹理的特征算子
    -	具有旋转不变性、灰度不变性
        -	通过对窗口中心的、领域点关系进行比较，重新编码形成新特征以消除外界场景对图像影响
        -   一定程度上解决了复杂场景下（光照变换）特征描述问题
    -   *Classic LBP* 算子：3 * 3 正方向窗口
        -   *Sobel Operator*
        -   *Laplace Operator*
        -   *Canny Edge Detector*
    -	*Circular LBP*：任意圆形领域

###	缩略图 *Hash*

-   缩略图 *Hash*：对图像进行特征提取得到 0、1 特征向量
    -   通过比较图片向量特征间汉明距离即可计算图片之间相似度

-   *Average Hashing* 平均哈希算法：利用平均值作为阈值二值化
    -   步骤
        -	缩放图片：将图像缩放到 8 * 8=64 像素
            -	保留结构、去掉细节
            -	去除大小、纵横比差异
        -	灰度化：把缩放后图转换为 256 阶灰度图
        -	计算平均值：计算灰度图像素点平均值
        -	二值化：遍历 64 个像素点，大于平均值者记为 1、否则为 0

-   *Perceptual Hashing*感知哈希算法：利用离散余弦变换降低频率，去除成分较少的高频特征
    -   步骤
        -	缩放图片：将图片缩放至 32 * 32
        -	灰度化：将缩放后图片转换为 256 阶灰度图
        -	计算 *DCT*：把图片分离成频率集合
        -	缩小 *DCT*：保留 32 * 32 左上角 8 * 8 代表图片最低频率
        -	计算平均值：计算缩小 *DCT* 均值
        -	二值化：遍历 64 个点，大于平均值者记为 1、否则为 0
    -	特点
        -	相较于 *Average Hash* 更稳定

-   *Differential Hashing* 差异哈希算法：基于差分（渐变）二值化
    -   步骤
        -	缩放图片：将图片缩放至 9 * 8
        -	灰度化：将缩放后图片转换为 256 阶灰度图
        -	计算差异值：对每行像素计算和左侧像素点差异，得到 8 * 8
        -	二值化：遍历 64 个点，大于 0 记为 1、否则为 0
    -	特点
        -	相较于 *Average Hash* 效果好
        -	相较于 *Perceptual Hash* 快



## 角点特征提取

-   角点特征检测
    -   *Corner Point* 角点：邻域各方向上灰度变化值足够高的点，是图像边缘曲线上曲率极大值的点
    -   焦点算法类型
        -	基于灰度图像的角点检测
            -	基于梯度：计算边缘曲率判断角点
            -	基于模板：考虑像素邻域点的灰度变化，将领域点亮度对比足够大的点定义为角点
            -	基于模板、梯度组合
        -	基于二值图像的角点检测：将二值图像作为单独的检测目标，可使用各种基于灰度图像的角点检测方法
        -	基于轮廓曲线的角点检测：通过角点强度、曲线曲率提取角点
    -   角点检测算子
        -   *Moravec*
        -   *Harris*
        -   *Good Feature to Track*
        -   *Feature from Accelerated Segment Test*：加速分割测试获得特征

-   思想、步骤
    -	使用角点检测算子，对图像每个像素计算 *Cornner Response Function* 值
        $$ E(u, v) = \sum_{(x,y)} w(x,y)[I(x+u, y+v) - I(x,y)]^2 $$
        > - $w(x,y)$：*window function*，窗口函数
        > - $I(x,y)$：图像梯度
        > - $E(x,y)$：角点响应函数，体现灰度变化剧烈程度，变化程度剧烈则窗口中心就是角点
    -	阈值化角点响应函数值	
        -	根据实际情况选择阈值 $T$
        -	小于阈值 $T$ 者设置为 0
    -	在窗口范围内对角点响应函数值进行非极大值抑制
        -	窗口内非响应函数值极大像素点置0
    -	获取非零点作为角点

###	*Moravec*

-   *Moravec*
    -   步骤
        -	取偏移量 $(\Delta x, \Delta y)$ 为 $(1,0), (1,1), (0,1), (-1,1)$，分别计算每个像素点灰度变化
        -	对每个像素点 $(x_i, y_i)$ 计算角点响应函数 $R(x) = min \{E\}$
        -	设定阈值 $T$，小于阈值者置0
        -	进行非极大值抑制，选择非 0 点作为角点检测结果
    -   特点
        -	二值窗口函数：角点响应函数不够光滑
        -	只在 4 个方向（偏移量）上计算灰度值变化：角点响应函数会在多处都有较大响应值
        -	对每个点只考虑响应函数值最小值：算法对边缘敏感

##  基于尺度空间特征

-   基于尺度空间特征提取算法
    -   *Scale-Invariant Feature Transform*
    -   *Speeded Up Robust Feature*
        -   对 *SIFT* 算法的改进，降低了时间复杂度，提高了稳健性
            -	高斯二阶维分模型简化，卷积平滑操作仅需要转换为加减运算
            -	最终生成特征向量维度从 128 维减少为64维
    -   *Brief*
    -   *Oriented Brief*

### *HOG* 特征

-   方向梯度直方图特征：通过计算、统计图像局部区域梯度方向直方图实现特征描述
    -   步骤
        -   归一化处理：图像转换为灰度图像，再利用伽马校正实现
            -   提高图像特征描述对光照、环境变量稳健性
            -   降低图像局部阴影、局部曝光、纹理失真
            -   尽可能抵制噪声干扰
        -   计算图像梯度
        -   统计梯度方向
        -   特征向量归一化（块内）
            -   克服光照不均匀变化及前景、背景对比差异
        -   生成特征向量

###	*Scale-Invariant Feature Transform*

![sift_procedure](imgs/sift_procedure.png)

-   *SIFT*：检测局部特征的算法
    -	*SIFT* 在不同尺度空间查找关键点，计算关键点大小、方向、尺度信息，组成对关键点的描述，并对进行图像特征点匹配
        -	*SIFT* 查找的关键点为突出、稳定的特征点，不会因光照、仿射变换、噪声等因素而改变
            -	角点
            -	边缘点
            -	暗区亮点
            -	亮区暗点
        -	匹配过程就是对比特征点过程
    -	优点
        -	稳定性：具有旋转、尺度、平移、视角、亮度不变性，
            利于对目标特征信息进行有效表达
        -	独特性：信息量丰富，适合海量特征数据中进行匹配
        -	多量性：少数物体也可以产生大量SIFT特征向量
        -	可扩展性：可以方便同其它形式特征向量联合
        -	对参数调整稳健性好：可以根据场景调整特征点数量进行
            特征描述、方便特征分析
    -	缺点
        -	不借助硬件加速、专门图像处理器难以实现

####	构建尺度空间

![image_scale_space](imgs/image_scale_space.png)

-   构建尺度空间
    -   图像的尺度空间：解决如何对图像在所有尺度下描述的问题
    -	思想：对原始图像进行尺度变换，获得多尺度空间下图像表示序列，**模拟图像数据的多尺度特征**
        -	对序列进行尺度空间主轮的提取
        -	以主轮廓作为特征向量，实现边缘、角点检测、不同分辨率上**稳定关键点**提取
    -	对高斯金字塔生成的 $O$ 组、$L$ 层不同尺度图像，$(O, L)$ 就构成高斯金字塔的尺度空间
        -	即以高斯金字塔组 $O$、层 $L$ 作为坐标
        -	给定一对 $(o,l)$ 即可唯一确定一幅图像

#####    图像金字塔

-   图像金字塔：以**多分辨率解释**图像的结构
    ![image_pyramid](imgs/image_pyramid.png)
    -	通过对原始图像进行**多尺度像素采样**方式生成 $N$ 个不同分辨率的图像
        -	图像分辨率从下至上逐渐减小
        -	直至金字塔顶部只包含一个像素
    -	获取图像金字塔步骤
        -	利用低通滤波器平滑图像
        -	对平滑图像进行采样
            -	上采样：分辨率逐渐升高
            -	下采样：分辨率逐渐降低

-   高斯金字塔：由很多组图像金字塔构成，每组金字塔包含若干层
    ![image_gaussian_pyramid](imgs/image_gaussian_pyramid.png)
    -	同一组金字塔中
        -	每层图像尺寸相同
        -	仅高斯平滑系数 $\sigma$ 不同，后一层图像是前一层 $k$ 倍
    -	不同组金字塔中
        -	后一组图像第一个图像是前一组倒数第三个图像二分之一采样
        -	图像大小是前一组一半
    -   构建过程
        -	构建第 1 组图像金字塔
            -	第 1 层：将原图扩大一倍得到
            -	第 2 层：第 1 层图像经过高斯卷积得到（*SIFT* 算子中，高斯平滑参数 $\sigma=1.6$）
            -	第 k 层：$k\sigma$ 作为第 $k$ 层图像高斯卷积平滑参数
            -	不断重复得到 $L$ 层图像
        -	构建第 k 组图像金字塔
            -	第 1 层：将第 k-1 组金字塔倒数第 3 层做比例因子为 2 的降采样得到
            -	之后同第 1 组图像金字塔
        -	不断重复得到 $O$ 组图像金字塔，共计 $O * L$ 个图像

-   *Difference of Gaussian* 金字塔：差分金字塔
    ![image_dog_pyramid](imgs/image_dog_pyramid.png)
    -	*DOG* 金字塔第 0 组第 k 层由高斯金字塔第 0 组第 k+1层减去第k层得到
        -	*DOG* 金字塔每组比高斯金字塔少一层
        -	按高斯金字塔逐组生成 $O * (L-1)$ 个差分图像
    -	*DOG* 包含信息需要归一化才可见）
        -	在不同 *DOG* 层（即不同模糊程度、不同尺度）都存在的特征即 *SIFT* 要提取的稳定特征
        -	后续 *SIFT* 特征点都是在 *DOG* 金字塔中进行
    ![image_dog_pyramid_instance](imgs/image_dog_pyramid_instance.png)

#### 关键点定位、检测

-   空间极值点检测：关键点初步查探
    -	寻找 *DOG* 图像极值点：每个像素点和其所有相邻点比较
        -	需要同时比较 **图像域、尺度空间域** 相邻点
        -	保证关键点在尺度空间、二维图像空间上都是局部极值点
    -	对二维图像空间，对中心点
        -	图像域：与 3 * 3 领域内 8 个点比较
        -	**同组尺度空间**：和上下两层图像中 2 * 9 个点比较
    -	极值点是在不同尺度空间下提取的，保证了关键点尺度不变性

-   精确定位：稳定关键点精确定位
    -	*DOG* 值对噪声、边缘敏感，需要对局部极值进一步筛选，去除不稳定、错误检测极值点
    -	构建高斯金字塔时采用下采样图像，需要求出下采样图像中极值点对应在原始图像中确切位置

-   方向信息分配：为关键点分配方向（梯度）信息赋予关键点旋转不变性
    -	计算关键点各方向梯度幅度值，绘制梯度方向直方图给出关键点梯度方向
        ![gradient_orientation_histgram](imgs/gradient_orientation_histgram.png)
        -	梯度幅度值
            $$ m(x, y) = \sqrt {(L(x+1,y) - L(x-1,y))^2 + (L(x,y+1) - L(x,y-1))^2} $$
        -	梯度方向
            $$ \theta(x,y) = tan^{-1} (\frac {L(x,y+1) - L(x,y-1)} {L(x+1,y) - L(x-1,y)}) $$
    -   具体步骤
        -	计算关键点为中心领域内所有点梯度方向
        -	把所有梯度方向划分到 36 个区域，每个方向代表 10 度
        -	累计每个方向关键点数目，生成梯度方向直方图
        -	将直方图中峰值代表方向作为关键点主方向
        -	若存在相当于峰值 80% 大小的方向，则作为辅方向
            -	辅方向可以增强匹配的鲁棒性
            -   *Lowe* 指出：大概 15% 关键点具有辅方向，且这些关键点对稳定匹配起关键作用

-   关键点描述：以数学方式定义关键点的过程，包括关键点周围对其有贡献的领域点
    -	对关键点周围像素区域分块
        -	计算块内梯度直方图
        -	生成具有独特性的向量，作为对该区域图像信息的抽象表述
    -	如下图
        ![descriptor_of_critical_point](imgs/descriptor_of_critical_point.png)
        -	将关键点周围分为 2 * 2 块
        -	对每块所有像素点梯度做高斯加权（拉开差距）
        -	每块最终取 8 个方向，得到 2 * 2 * 8 维向量，作为中心关键点数学描述
    > - *Lowe* 实验表明：采用 4 * 4 * 8 共 128 维描述子表征关键点，综合效果最好
        ![descriptor_of_critical_point_by_lowe](imgs/descriptor_of_critical_point_by_lowe.png)

-   特征点匹配：计算两组特征点描述向量（128 维）的欧式距离
    -	欧式距离越小、相似度越高，小于指定阈值时既可认为匹配成功

