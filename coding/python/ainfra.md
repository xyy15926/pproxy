---
title: AI Infra
categories:
  - Algorithm
tags:
  - Algorithm
date: 2025-12-09 21:56:25
updated: 2025-12-12 20:18:47
toc: true
mathjax: true
description: 
---

##  数值计算库、标准

### *BLAS*、*LAPACK*

-   *BLAS*、*LAPACK*
    -   *Basic Linear Algebra Subroutine*：一系列基本线性代数运算函数的接口标准
        -   向量的线性组合
        -   矩阵乘法
        -   矩阵向量乘法
    -   *Linear Algebra PACKage*：基于 *BLAS* 的科学计算接口规范
        -   线性方程组求解
        -   最小二乘
        -   特征向量求解
        -   矩阵分解
    -   *BLAS*、*LAPACK* 只是接口标准而不是具体实现，即仅为一系列函数签名（声明）
        -   *Netlib* 官方实现：提供源代码，速度较慢，常作为基线版本
            -   *reference BLAS*：*Fortran 77* 实现，最初版本 *BLAS* 实现
            -   *CBLAS*：*C* 版本实现，实际即对 *Fortran BLAS* 的封装，接口与 *Fortran* 略有不同
            -   *LAPACK*：*Fortran 90* 实现
            -   *LAPACKE*：*C* 版本实现，实际挤兑 *Frotran LAPACK* 的封装
                -   *reference BLAS*、*CBLAS* 现在其实已被作为 *LAPACK* 的一部分
        -   *Intel Math Kernel Library*：*Intel* 提供、对 *Intel CPU* 性能最高的版本
            -   文档齐全，推荐参考
        -   *OpenBlas*
        -   *ATLAS*
        -   *XBLAS*：实现 4 精度浮点、混合精度

> - *BLAS* 简介：<https://wuli.wiki/online/BLAS.html>
> - *Netlib BLAS* 官网：<http://www.netlib.org/blas/>
> - *Intel MKL BLAS*：<https://software.intel.com/en-us/mkl-developer-reference-c-blas-and-sparse-blas-routines>

####    *MKL CBLAS* 接口

```c
// y = alpha * a * x + beta * y
void cblas_zgemv(
    const CBLAS_LAYOUT Layout,          // 枚举，行、列优先布局
    const CBLAS_TRANSPOSE trans,        // 枚举，是否预先转置矩阵
    const MKL_INT m,                    // 矩阵行数
    const MKL_INT n,                    // 矩阵列数

    const void *alpha,
    const void *a,                      // 矩阵指针
    const MKL_INT lda,                  // Leading Dimension，矩阵优先维长度（即次维度相邻元素间隔）

    const void *x,                      // 向量指针
    const MKL_INT incx,                 // 步长（向量 x 中元素间隔）

    const void *beta,
    void *y                             // 结果向量指针
    const MKL_INT incy,                 // 步长（向量 y 中元素间隔）
)
```

-   函数名称格式 `cblas_<DTYPE><MATRIX><OPERATION>`
    -   *BLAS* 中包含 4 中数据类型、3 种操作级别，在函数签名中用不同字母代表

| 4 种数据类型 `DTYPE` | 代表字符 |
|----------------------|----------|
| `single` 单精度浮点  | `s`      |
| `double` 双精度浮点  | `d`      |
| `complex`            | `c`      |
| `double complex`     | `z`      |

| 3 种操作级别 `OPERATION` | 代表字符 |
|--------------------------|----------|
| 向量与向量操作           | `vv`     |
| 矩阵与向量操作           | `mv`     |
| 矩阵与矩阵操作           | `mm`     |

| 矩阵类型 | 代表字符 |
|----------|----------|
| 一般矩阵 | `ge`     |
| 对称矩阵 | `sy`     |
| 三角矩阵 | `tr`     |

##  *AI Infra*

> - *AI* 技术栈解析及应用：<https://aii-sdu.github.io/AI-Technology-Stack/Chapter2/2-%E6%8A%80%E6%9C%AF%E6%A0%88%E6%9E%B6%E6%9E%84%E6%A6%82%E8%BF%B0.html>

### *CUDA*

| 层次                         | 内容                                 |
|------------------------------|--------------------------------------|
| *Driver* 硬件驱动            | *GPU* 驱动、*CUDA Driver*            |
| *Runtime* 运行时             | *CUDA Runtime API*                   |
| *Programming Model&Language* | *CUDA C++*                           |
| *Libraries* 库               | *cuBLAS*、*cuDNN*、*CUTLASS*、*NCCL* |
| *Application*                | *PyTorch*、*TensorFlow*              |

-   *Compute Unified Device Architecture*：*Nvdia* 推出、基于 *GPGPU* 并行计算平台、编程模型框架
    -   *CUDA Toolkit*：*GPU* 加速应用开发工具包，包含
        -   *CUDA Runtime* `cudart`
            -   资源管理：`cudaMalloc`、`cudaFree`、`cudaMemcpy`、`cudaHostAlloc`
            -   核函数：`threadIdx`、`blockIdx`
            -   调度、通信：`cudaStreamCreate`、`cudaStreamSynchronize`
        -   *CUDA* 开发工具链
            -   *C/CPP* 编译器 `nvcc`：编译 *CUDA C++*
            -   调试、优化工具
        -   *NVDIA* 实现的 *GPU* 加速闭源库
            -   *cuBLAS*
            -   *cuFFT*
    -   *NVIDA HPC SDK*：高性能计算 *SDK*，包含
        -   *CUDA* 开发工具
            -   *C/CPP* 及 *Fortran* 编译器：`nvcc`、`nvfortran`
        -   *NVDIA* 实现的 *GPU* 加速闭源库：*cuBLAS*、*cuFFT* 等
        -   通信库：*OpenMPI*、*NVSHMEM* 等
    -   *CUDA-X Libraries*：基于 *CUDA* 的库
        -   可能包含在工具包中、独立安装或者兼有

> - *CUDA*：<https://aii-sdu.github.io/AI-Technology-Stack/Chapter3/Chapter3.1/31-cuda.html>

####    *CUDA-X Libraries*

-   *CUDA-X Libraries*：基于 *CUDA* 的库
    -   *CUDA Math Libraries*：大部分包含在 *CUDA Toolkits* 中
        -   *cuBLAS*：*CUDA* 框架下的 *BLAS* 实现
            -   包含 *BLAS* 中矩阵运算算子实现
        -   *cuFFT*
        -   *cuSOLVER*
        -   *cuSPARSE*
    -   *Deep Learning Core Libraries*
        -   *cuDNN*：*CUDA* 框架下的神经网络计算库
            -   包含神经网络中常用的卷积、池化、*GeMM* 算子实现
                -   避免打破 *cuBLAS* 定位、设计理念，故独立于 *cuBLAS* 实现
            -   不包含在 *CUDA Toolkits* 中，需要额外安装
        -   *CUda Templates for Linear Algebra SubroutineS*：*CUDA* 框架中用于实现矩阵运算的模板库
            -   将底层 *CUDA* 指令模块化、封装调度的模板库
                -   通过模板机制将线代运算分解为可重用的模块化组件
                -   支持多精度、架构优化
            -   允许用户根据自定义 *Fused Kernel* 以提升效率
                -   *cuDNN* 对 *LLM* 中 *Transformer* 性能优化一般
                -   *NVDIA* 期望通过开源 *CUTLASS*、借助社区开发高性能算子
        -   *TensorRT*：高性能深度学习 **推理** 工具集
            -   包括用于生产环境的推理编译器、运行时、模型优化器

> - *GPU-Accelerated-Libraries*：<https://developer.nvidia.com/gpu-accelerated-libraries>
> - *NVDIA CUTLASS Documentation*：<https://docs.nvidia.com/cutlass/latest/>
> - *cuBLAS* 产品系列介绍一：<https://zhuanlan.zhihu.com/p/31803812664>
> - 深入解析 *CUTLASS* 的诞生历程、特性和对友商的各大优势：<https://mp.weixin.qq.com/s/-dHq2DOzsEiJQe1LvsRxrQ>
> - *CUTLASS* 介绍和基本使用方法：<https://www.cnblogs.com/maliesa/articles/18773400>

### *ROCm*

| 层次                         | 内容                                 |
|------------------------------|--------------------------------------|
| *Driver* 硬件驱动            | *GPU* 驱动、*ROCm Driver*、*HIP API* |
| *Runtime* 运行时             | *ROCm Runtime API*                   |
| *Programming Model&Language* | *HIP C++*                            |
| *Libraries* 库               | *rocBLAS/hipBLAS*、*hipDNN*、*rccl*  |
| *Application*                | *PyTorch*、*TensorFlow*              |

-   *Radeon Open Computing platforM*：*AMD* 开源、对标 *CUDA* 的计算平台、编程模型框架
    -   *ROCm* 目标是建立可替代 *CUDA*、可移植、高性能的 *GPU* 的计算平台

> - *DeepWiki ROCm/TheRock*：<https://deepwiki.com/ROCm/TheRock/1-overview>
> - *ROCm/HIP*：<https://aii-sdu.github.io/AI-Technology-Stack/Chapter4/Chapter4.1/41-rocm--hip.html>

####    *HIP*

-   *Heterogeneous-computing Interfacef for Portability*：*AMD* 开发的 *C++ Runtime API*、语言
    -   特点
        -   *Source-code Portability* 源码级的移植性：提供与 *CUDA* 高度相似的 *API*、编程模型，允许编译后可在不同 *GPU* 运行
        -   *Migration Path* 迁移工具：帮助 *CUDA* 源码迁移至 *ROCm* 平台
        -   *No Performance Penalty* 性能无损：在 *NVDIA*、*AMD* 平台上均直接调用原生 *CUDA*、*ROCm* 驱动
    -   *HIP* 是通用规范，包括编程模型、*C++* 方言及工具链、*Runtime API* 等
        -   *HIP* 设计基本比照 *CUDA* 复刻（*CUDA* 同时指规范、具体实现）
        -   *HIP* 库、*API* 为规范、*ROCm* 库的封装，*ROCm* 库、*API* 则是在 *AMD* 平台、遵守 *HIP* 规范的具体实现
            | 库               | 说明                                                |
            |------------------|-----------------------------------------------------|
            | *hipBLAS-common* | *BLAS* 库通用头文件、工具                           |
            | *hipBLASLt*      | 针对 *GPU* 优化、扩展 *BLAS* 库                     |
            | *rocBLAS*        | *BLAS* 核心实现，依赖 *hipBLAS-common*、*hipBLASLt* |
            | *hipBLAS*        | *BLAS* 用户接口暴露，依赖 *rocBLAS*                 |

> - *HIP介绍*：<https://zhuanlan.zhihu.com/p/1943693230341326400>

####    *CUDA*、*ROCm/HIP*

| *CUDA*             | *ROCm*                           | 说明              |
|--------------------|----------------------------------|-------------------|
| *CUDA API*         | *HIP API*                        | *C++ Runtime API* |
| *NVCC*             | *HCC*                            | *C++* 编译器      |
| *CUDA-X-Libraries* | *ROC* 库、*HIP* 库               |                   |
| *Thrust*           | *Parallel STL*（*HCC* 原生支持） | 通信              |
| *Profiler*         | *ROCm Profiler*                  |                   |
| *CUDA-GDB*         | *ROCm-GDB*                       | 调试              |
| *nvidia-smi*       | *rocm-smi*                       | 设备管理          |
| *DirectGPU RDMA*   | *ROCm RDMA*                      | *peer2peer*       |
| *TensorRT*         | *Tensile*                        | 推理组件          |
| *CUDA-Docker*      | *ROCm-Docker*                    |                   |


### *Direct ML*

-   *Direct ML*

##  *Triton*

-   *Triton*：用于并行编程的 *DSL*、编译器
    -   提供基于 *Python* 的编程环境，以编写自定义 *GPU* 加速算子

> - *Triton* 教程：<https://triton-lang.cn/main/getting-started/tutorials/index.html>
> - *Triton* 相关工作：<https://triton-lang.cn/main/programming-guide/chapter-2/related-work.html>

