---
title: 
categories:
  - Coding
  - Rust
tags:
  - Coding
  - Rust
date: 2026-01-25 16:18:02
updated: 2026-01-26 11:51:51
toc: true
mathjax: true
description: 
---

##  *Rust Std*

| *Std Crate* | 说明                                                   |
|-------------|--------------------------------------------------------|
| `core`      | 标准库的平台无关、无上游链接、无依赖的基础             |
| `alloc`     | 提供管理堆内存分配的智能指针、集合                     |
| `std`       | *Rust* 生态的共享抽象，包含 `core`、`alloc` 的重新导出 |

> - `crate core`：<https://doc.rust-lang.org/stable/core/index.html>
> - `crate core` 中文：<https://rustwiki.org/zh-CN/core/index.html>
> - `crate alloc`：<https://doc.rust-lang.org/stable/alloc/index.html>
> - `crate alloc` 中文：<https://rustwiki.org/zh-CN/alloc/index.html>
> - `crate std`：<https://doc.rust-lang.org/stable/std/index.html>
> - `crate std` 中文：<https://rustwiki.org/zh-CN/std/index.html>

### 标准库模块

-   标准库说明
    -   `alloc`、`core` 库中内容已在 `std` 中重新导出
        -   仅在 `#![no_std]` 属性启用时需要直接使用

####    基本功能

| *Modules*   | `core` 说明        | `alloc` 说明          | `std` 补充     | 备注  |
|-------------|--------------------|-----------------------|----------------|-------|
| `alloc`     | 内存分配 *API*     | 内存分配 *API*        |                |       |
| `ptr`       | 裸指针内存管理     |                       |                |       |
| `mem`       | 内存控制           |                       |                |       |
| `cmp`       | 比较、排序         |                       |                |       |
| `convert`   | 类型转换           |                       |                |       |
| `borrow`    | `Borrow` 借用      | `Cow` 写克隆          |                |       |
| `any`       | 动态类型、类型反射 |                       |                |       |
| `from`      | `From` 衍生宏      |                       |                | *Exp* |
| `result`    | `Result enum`      |                       |                |       |
| `option`    | `Option enum`      |                       |                |       |
| `marker`    | 原语级特性、类型   |                       |                |       |
| `clone`     | `Clone trait`      |                       |                |       |
| `default`   | `Default trait`    |                       |                |       |
| `primitive` | 原语类型重导出     |                       |                |       |
| `prelude`   | *Core Prelude*     |                       | *Rust Prelude* |       |
| `ops`       | 可重载符号         |                       |                |       |
| `fmt`       | 格式化、打印字符串 | 格式化、打印 `String` |                |       |

####    数据类型

| *Modules*     | `core` 说明                | `alloc` 说明                  | `std` 补充 | 备注  |
|---------------|----------------------------|-------------------------------|------------|-------|
| `array`       | `array` 类型工具           |                               |            |       |
| `char`        | `char` 类型工具            |                               |            |       |
| `f32`         | `f32` 类型常量             |                               |            |       |
| `f64`         | `f64` 类型常量             |                               |            |       |
| `f16`         | `f16` 类型常量             |                               |            | *Exp* |
| `f128`        | `f128` 类型常量            |                               |            | *Exp* |
| `num`         | 内置数值类型特性、函数     |                               |            |       |
| `str`         | 字符串操作                 | `str` 类型工具                |            |       |
| `ascii`       | *ASCII* 字符、串操作       |                               |            |       |
| `bstr`        | `ByteStr` 类型、特性       | `ByteStr`、`ByteString` 实现  |            | *Exp* |
| `slice`       | *Slice* 切片操作、管理     | *Slice* 工具                  |            |       |
| `index`       | *Slice* 切片索引辅助类型   |                               |            | *Exp* |
| `range`       | 替代 `ops` 中 *Range* 类型 |                               |            | *Exp* |
| `cell`        | `Cell<T>` 共享可变容器     |                               |            |       |
| `boxed`       |                            | `Box<T>` 堆内存分配           |            |       |
| `rc`          |                            | `Rc` 单线程引用计数           |            |       |
| `vec`         |                            | `Vec<T>` 连续、变长、堆上数组 |            |       |
| `string`      |                            | *UTF-8* 编码、变长字符串      |            |       |
| `collections` |                            | 集合类型                      |            |       |
| `iter`        | 外部迭代                   |                               |            |       |

####    系统功能

| *Modules*    | `core` 说明            | `alloc` 说明           | `std` 补充         | 备注  |
|--------------|------------------------|------------------------|--------------------|-------|
| `panic`      | *Panic* 支持           |                        |                    |       |
| `error`      | *Error* 处理           |                        |                    |       |
| `io`         | *IO* 功能              |                        |                    | *Exp* |
| `os`         | *OS* 相关功能          |                        |                    | *Exp* |
| `env`        |                        |                        | 进程环境监控、操作 |       |
| `fs`         |                        |                        | 文件系统操作       |       |
| `path`       |                        |                        | 跨平台路径         |       |
| `pin`        | `Pin` 数据固定         |                        |                    |       |
| `sync`       | `Sync` 同步原语        | 线程安全引用计数指针   |                    |       |
| `thread`     |                        |                        | 原生线程           |       |
| `process`    |                        |                        | 进程工具           |       |
| `future`     | `Future` 异步基础      |                        |                    |       |
| `task`       | `Context` 异步任务执行 | 异步任务工具类型、特性 |                    |       |
| `async_iter` | 异步迭代               |                        |                    | *Exp* |
| `net`        | *IP* 通信原语          |                        |                    |       |
| `hash`       | 通用哈希支持           |                        |                    |       |
| `time`       | 时间量化               |                        |                    |       |
| `random`     | 随机数生成             |                        |                    | *Exp* |

####    编译处理

| *Modules*        | `core` 说明               | `alloc` 说明   | `std` 补充         | 备注  |
|------------------|---------------------------|----------------|--------------------|-------|
| `assert_matches` | `assert_matches` 宏       |                |                    | *Exp* |
| `autodiff`       | `autodiff` 宏             |                |                    | *Exp* |
| `constracts`     | 语言保证宏                |                |                    | *Exp* |
| `hint`           | 编译器优化提示            |                |                    |       |
| `arch`           | *SIMD*、*Vendor* 内部函数 |                |                    |       |
| `ffi`            | 平台相关类型（*C* 定义）  | *FFI* 绑定工具 |                    |       |
| `intrinsics`     | 编译器函数                |                |                    | *Exp* |
| `simd`           | 可迁移 *SIMD*             |                |                    | *Exp* |
| `panicking`      | *Core Panic* 支持         |                |                    | *Exp* |
| `pat`            | `pattern_type` 宏导出     |                |                    | *Exp* |
| `profiling`      | 编译器探测标记            |                |                    | *Exp* |
| `ub_checks`      | 不安全函数前提检查        |                |                    | *Exp* |
| `unsafe_binder`  | 不安全绑定                |                |                    | *Exp* |
| `backtrace`      |                           |                | 捕获系统线程堆栈迹 |       |
