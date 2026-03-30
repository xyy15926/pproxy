---
title: 
categories:
  - Coding
  - Rust
tags:
  - Coding
  - Rust
date: 2026-01-25 16:18:02
updated: 2026-03-30 19:57:11
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
        -   `core` 包括最基本类型、函数，不依赖 `libc`、内存分配器、操作系统
        -   `alloc` 在 `core` 基础上增加堆内存分配能力，增加需要全局堆分配器的类型，如 `Vec`、`Box`、`Arc` 等
        -   通常仅在嵌入式 *Rust*、`#![no_std]` 属性启用时需要直接使用

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

### `std` 宏

> - `std::macro`：<https://doc.rust-lang.org/stable/std/index.html#macros>

####    编译、断言宏

| `std` 编译、断言宏                          | 返回值             | 描述                         | 其他                       |
|---------------------------------------------|--------------------|------------------------------|----------------------------|
| `assert!(cond[, fmt, ...])`                 | `()`               | 断言                         |                            |
| `assert_eq!(left, right[, fmt, ...])`       | `()`               | 断言相等                     |                            |
| `assert_ne!(left, right[, fmt, ...])`       | `()`               | 断言不等                     |                            |
| `panic!(fmt[, ...])`                        | `()`               | 运行时 `panic`               |                            |
| `debug!(expr)`                              | `Any`              | 返回值并打印内容至标准错误   |                            |
| `debug_assert!(cond[, msg, fmt])`           | `()`               | 断言                         | 仅限非优化构建             |
| `debug_assert_eq!(left, right[, msg, fmt])` | `()`               | 断言相等                     | 仅限非优化构建             |
| `debug_assert_ne!(left, right[, msg, fmt])` | `()`               | 断言不等                     | 仅限非优化构建             |
| `line!()`                                   | `u32`              | 宏所在行数                   |                            |
| `columns!()`                                | `u32`              | 此宏所在列数                 |                            |
| `file!()`                                   | `&'static str`     | 宏所在文件名                 |                            |
| `module_path()`                             | `&'static str`     | 宏所在模块名                 |                            |
| `todo!([&str])`                             | `!`                | 指示未完成函数               | 函数返回类型总能编译通过   |
| `implemented!([&str])`                      | `!`                | 指示不应使用函数             | 调用时 `panic`             |
| `unreachable!([&str])`                      | `!`                | 指示不应触发代码             | 触发时 `panic`             |
| `compile_error(&str)`                       | `()`               | 触发编译错误                 |                            |
| `env!(env_name)`                            | `&'static str`     | 扩展为编译期环境变量         | 运行时使用 `std::env::var` |
| `option_env!(env_name)`                     | `Option<Any>`      | 尝试获取环境变量值           |                            |
| `cfg!(cfg_flag)`                            | `bool`             | 编译属性是否置位             |                            |
| `include!(file)`                            | `Any`              | 引入其他文件作为可执行、编译 |                            |
| `include_bytes!(file)`                      | `&'static [u8, N]` | 引入其他文件作为字节数组     |                            |
| `include_str!(file)`                        | `&'static str`     | 引入其他文件作为字符串切片   |                            |
| `is_x86_feature_detected(feature)`          | `bool`             | 是否支持特性                 |                            |

####    格式化

| `std` 格式化、快捷宏       | 返回值                | 描述                               | 其他             |
|----------------------------|-----------------------|------------------------------------|------------------|
| `vec![...]`                | `Vec<T>`              | 创建向量                           |                  |
| `thread_local!{...}`       | `()`                  | 创建 *TLS* 实例                    |                  |
| `matches!(val, ptn)`       | `bool`                | 值满足模式匹配                     |                  |
| `stringify!(expr)`         | `&'static str`        | 转换字符串切片                     |                  |
| `concat!(...)`             | `&'static str`        | 拼接为字符串                       |                  |
| `format_args!(fmt[, ...])` | `std::fmt::Arguments` | 创建格式化字符串参数               | 其他格式化宏中介 |
| `print!(fmt[, ...])`       | `()`                  | 标准输出（不换行）                 |                  |
| `println!(fmt[, ...])`     | `()`                  | 标准输出（换行）                   |                  |
| `format!(fmt[, ...])`      | `String`              | 格式化字符串                       |                  |
| `eprint!(fmt[, ...])`      | `()`                  | 标准错误输出（不换行）             |                  |
| `eprintln!(fmt[, ...])`    | `()`                  | 标准错误输出（换行）               |                  |
| `write(buf, fmt[, ...])`   | `()`                  | 向缓冲区写入格式化字符串（不换行） |                  |
| `writeln(buf, fmt[, ...])` | `()`                  | 向缓冲区写入格式化字符串（换行）   |                  |

##  常用 *Traits*

| 常用 *Traits*                         | 说明                       | *Prelude* | 其他        |
|---------------------------------------|----------------------------|-----------|-------------|
| `convert::AsRef<T>`                   | 转换为引用                 | 2018      |             |
| `convert::AsMut<T>`                   | 转换为可变引用             | 2018      |             |
| `convert::From<T>`                    | 从消耗并转换               | 2018      |             |
| `convert::Into<T>`                    | 转换为                     | 2018      |             |
| `convert::TryFrom<T>`                 | 可能失败的 `From`          | 2021      |             |
| `convert::TryInto<T>`                 | 可能失败的 `Into`          | 2021      |             |
| `borrow::Borrow<Borrowed>`            | 借用为                     |           |             |
| `borrow::BorrowMut<Borrowed>`         | 可变借用为                 |           |             |
| `borrow::ToOwned`                     | 转换为所有权类型           | 2018      |             |
| `clone::Clone`                        | 显式拷贝                   | 2018      | `#[derive]` |
| `cmp::PartialEq<Rhs=Self>`            | 不完整等价                 | 2018      | `#[derive]` |
| `cmp::Eq:PartialEq`                   | 等价关系                   | 2018      | `#[derive]` |
| `cmp::PartialOrd<Rhs=Self>`           | 偏序关系                   | 2018      | `#[derive]` |
| `cmp::Ord:Eq + PartialOrd<Rhs=Self>`  | 全序关系                   | 2018      | `#[derive]` |
| `default::Default`                    | 默认值                     | 2018      | `#[derive]` |
| `iter::Iterator`                      | 迭代器                     | 2018      |             |
| `iter::IntoIterator`                  | 可转化为迭代器             | 2018      |             |
| `iter::Extend<A>`                     | 从迭代器扩展               | 2018      |             |
| `iter::FromIterator<A>`               | 从迭代器转化               | 2021      |             |
| `iter::DoubleEndedIterator: Iterator` | 双头迭代器                 | 2018      |             |
| `iter::ExactSizeIterator: Iterator`   | 长度确定迭代器             | 2018      |             |
| `ops::FnOnce<Args>`                   | 仅可调用一次闭包           | 2018      |             |
| `ops::FnMut<Args>: FnOnce<Args>`      | 调用后闭包自身状态改变     | 2018      |             |
| `ops::Fn<Args>: FnMut<Args>`          | 调用后闭包自身状态不变     | 2018      |             |
| `ops::Deref`                          | 不可变场景引用类型强制转换 |           |             |
| `ops::DerefMut`                       | 可变场景引用类型强制转换   |           |             |
| `ops::Drop`                           | 析构                       | 2018      |             |
| `ops::Index`                          | 索引                       |           |             |
| `string::ToString`                    | 转换为 `String`            | 2018      |             |
| `future::Future`                      | 异步任务结果               | 2024      |             |
| `future::IntoFuture`                  | 可转换为异步任务           | 2024      |             |
| `task::Wake`                          | 可唤醒异步任务             |           |             |
| `hash::Hash`                          | 可哈希                     |           |             |
| `any::Any`                            | 模拟动态类型               |           |             |
| `io::Read`                            | 可从中读取字节             |           |             |
| `io::Write`                           | 可向其写入字节             |           |             |

> - `std::prelude`：<https://doc.rust-lang.org/stable/std/prelude/index.html>
> - `std::prelude` 中文：<https://rustwiki.org/zh-CN/std/prelude/index.html>

### `std::convert`

```rust
pub trait AsRef<T: ?Sized> {                // 可转换为 `&T`
    fn as_ref(&self) -> &T;
}

pub trait AsMut<T: ?Sized> {                // 可转换为 `&mut T`
    fn as_mut(&mut self) -> &T;
}

pub trait From<T>: Sized {                  // 可从 `T` 类型转换为当前类型
    fn from(value: T) -> Self;
}

pub trait Into<T>: Sized {                  // 与 `From` 行为逻辑相反，优先实现 `From`
    fn into(self) -> T;
}
```

-   `std::convert` 中特性说明
    -   `convert::AsRef<T>` 特性：可转换为 `&T` 引用类型
        -   建议仅用于廉价、确定的转换
            -   若转换代价高昂、或可能产生错误，可实现 `From`、或自定义返回 `Result<T, E>` 的函数
        -   `AsRef<T>` 常用作 *Trait Bound*，要求泛型参数 `T` 满足 `T: AsRef<U>`
            -   则，`AsRef<T>::as_ref()` 可以用于获取 `&T` 引用类型
            -   则，显式的调用 `.as_ref()` 使得方法可兼容所以可转换为 `&T` 的类型，提高方法灵活性
            -   非泛型场合较少使用，毕竟 `&` 可直接用于获取引用
    -   `convert::From<T>` 特性：可从类型 `T` 转换为当前类型
        -   `From<T>::from()` 消耗原值返回当前类型值
        -   `From` 与 `Into` 动作逻辑相反
            -   一般建议优先实现 `From`，标准库会自动为 `impl From<T>` 实现 `Into`
            -   指定 *Trait Bound* 时，优先使用 `Into`，此时直接实现 `Into` 类型也满足泛型限制

```rust
impl File {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<File> {  // 路径 `path` 可转换为 `&Path` 类型
    }
}
```


> - Rust 的 `Borrow` 和 `AsRef`：让你的代码用起来像呼吸一样自然：<https://zhuanlan.zhihu.com/p/684078465>
> - `std::convert::AsRef` 中文：<https://rustwiki.org/zh-CN/std/convert/trait.AsRef.html>
> - `std::convert::AsMut` 中文：<https://rustwiki.org/zh-CN/std/convert/trait.AsMut.html>
> - `std::convert::Into` 中文：<https://rustwiki.org/zh-CN/std/convert/trait.Into.html>
> - `std::convert::From` 中文：<https://rustwiki.org/zh-CN/std/convert/trait.From.html>

### `std::borrow`

```rust
// ****************** core::borrow.rs
pub trait Borrow<Borrowed: ?Sized> {                        // 可借用为 `Borrowed`，即可获取 `&Borrowed` 类型引用
    fn borrow(&self) -> &Borrowed;
}

pub trait BorrowMut<Borrowed: ?Sized>: Borrow(Borrowed) {   // 可可变借用为 `Borrowed`，即可获取 `&mut Borrowed` 类型引用
    fn borrow_mut(&mut self) -> &mut Borrowed;
}

// ****************** alloc::borrow.rs
pub trait ToOwned {                                         // 可创建可获取 `&Self` 类型引用的所有权类型
    type Owned: Borrow<Self>;
    fn to_owned(&self) -> Self::Owned;
    fn clone_into(&self, target: &mut Self::Owned) {...}
}
```

-   `std::borrow` 中特性说明
    -   `borrow::Borrow<Borrowed>` 特性：可被借用为 `Borrowed` 类型（被 `.borrow` 借用后为 `&Borrowed` 类型）
        -   `Borrow` 特性要求：**`impl Borrow<Borrowed>` 类型与 `Borrowed` 类型的 `Hash`、`Eq`、`Ord` 行为逻辑相同**
            -   此即 `borrow::Borrow` 与 `convert::AsRef` 在设计逻辑、要求上的不同（二者在接口层面完全相同）
            -   `AsRef` 只要求获取给定类型引用，但 `Borrow` 还要求引用类型行为逻辑与原类型相同
            -   并且与 `AsRef` 不同，`Borrow` 为所有类型均有默认实现，即获取自身类型的引用
        -   类似 `AsRef`，`Borrow` 特性常用作 *Trait Bound*，要求泛型参数 `T` 满足 `T: Borrow<Borrowed>`
            -   则，`Borrow<Borrowed>::borrow()` 可以用于获取 `&Borrowed` 引用类型
            -   则，显式的调用 `.borrow()` 使得方法可兼容所以可获取 `&Borrowed` 的类型，提高方法灵活性
            -   非泛型场合较少使用，毕竟 `&` 可直接用于获取引用
    -   `borrow::ToOwned` 特性：可从自身创建（克隆）拥有所有权、满足 `: Borrow<Self>` 的类型

```rust
impl <K, V> HashMap<K, V> {
    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,                                       // 键 `K` 需可被借用为 `Q`，则可用 `Borrow::borrow` 均获取 `&Q` 进行比较
        Q: Hash + Eq + ?Sized
    {
    }
}
```

> - Rust 的 `Borrow` 和 `AsRef`：让你的代码用起来像呼吸一样自然：<https://zhuanlan.zhihu.com/p/684078465>
> - Rust 哪些事至 `Borrow` VS `AsRef`：<https://developer.aliyun.com/article/1478269>
> - `std::borrow::ToOwned` 中文：<https://rustwiki.org/zh-CN/std/borrow/trait.ToOwned.html>
> - `std::borrow::Borrow` 中文：<https://rustwiki.org/zh-CN/std/borrow/trait.Borrow.html>
> - `std::borrow::BorrowMut` 中文：<https://rustwiki.org/zh-CN/std/borrow/trait.BorrowMut.html>

### `std::clone`

```rust
pub trait Clone: Sized {
    fn clone(&self) -> Self;
    fn clone_from(&mut self, source: &Self) {...}
}
```

-   `std::clone` 中特性说明
    -   `clone::Clone`：显式复制
        -   而 `std::marker::Copy` 是隐式的、廉价的按位复制，且不允许被重新实现
        -   若，所有字段均为 `Clone`，则可以与 `#[derive]` 一起使用
            -   即，在每个字段上调用 `Clone::clone`
        -   说明：`Clone` 只是语义上的 “复制”，不代表真的复制数据
            -   `Rc::clone()` 仅增加引用计数

> - `std::clone::Clone`：<https://doc.rust-lang.org/std/clone/trait.Clone.html>

### `std::cmp`

```rust
pub trait PartialEq<Rhs = Self>                                 // 只在部分情况下可比较是否相等
where
    Rhs: ?Sized
{
    fn eq(&self, other: &Rhs) -> bool;                          // `x == y` 是 `x.eq(y)` 的语法糖
    fn ne(&self, other: &Rhs) -> bool { ... }
}

pub trait Eq: PartialEq<Self> {}                                // 要求所有值均可比较是否相等

pub trait PartialOrd<Rhs = Self>: PartialEq<Rhs>
where
    Rhs: ?Sized,
{
    fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;     // 返回可选的比较结果 `Option<Ordering>`
    fn lt(&self, other: &Rhs) -> bool { ... }
    fn le(&self, other: &Rhs) -> bool { ... }
    fn gt(&self, other: &Rhs) -> bool { ... }
    fn ge(&self, other: &Rhs) -> bool { ... }
}

pub trait Ord: Eq + PartialOrd<Self> {
    fn cmp(&self, other: &Self) -> Ordering;
    fn max(self, other: Self) -> Self
        where Self: Sized { ... }
    fn min(self, other: Self) -> Self
        where Self: Sized { ... }
    fn clamp(self, min: Self, max: Self) -> Self
        where Self: Sized + PartialOrd<Self> { ... }
}
```

-   `std::cmp` 特性说明
    -   `cmp::Eq`：等价关系，满足自反性、对称性、传递性的二元关系
        -   自反性 $\forall x \in A, xRx$
        -   对称性（即等价） $\forall x,y \in A, xRy \Rightarrow yRx$
        -   传递性 $\forall x,y,z \in A, (xRy \wedge yRz) \Rigtharrow xRz$
    -   `cmp::PartialEq`：不满足自反性的等价关系
        -   `cmp::Eq` 要求实现类型在所有情况下（所有值之间）均可比较是否相等
            -   否则，仅应实现 `cmp::PartialEq`
            -   `f32` 仅实现 `PartialEq`：`f32::NAN != f32:NAN`
        -   且 `cmp::Eq` 无额外实现，是用户对 `PartialEq` 之外的额外保证
            -   毕竟，编译器无法判断是否所有情况下均可比较是否相等
    -   `cmp:PartialOrd`：偏序关系，满足自反性、反对称、传递性的二元关系
        -   自反性 $\forall x \in A, xRx$
        -   反对称性（即偏序） $\forall x,y \in A, (xRy \wedge yRx) \Rightarrow x = y$
            -   即，若 `x <= y` 且 `y <= x`，则 `x == y`
        -   传递性 $\forall x,y,z \in A,(xRy \wedge yRz) \Rigtharrow xRz$
    -   `cmp::Ord`：全序关系，$\forall x, y \in A$ 总有 $xRy$ 或 $yRx$ 的偏序关系
        -   即，类型在所有情况下（所有值之间）总可比较大小

> - 实践解析丨Rust 内置 trait：`PartialEq` 和 `Eq`：<https://zhuanlan.zhihu.com/p/359178964>
> - 7.2 `Eq` 和 `PartialEq`：<https://course.rs/difficulties/eq.html>
> - 偏序与全序关系：<https://zhuanlan.zhihu.com/p/717078572>
> - `std::cmp::PartialEq`：<https://rustwiki.org/zh-CN/std/cmp/trait.PartialEq.html>
> - `std::cmp::Eq`：<https://rustwiki.org/zh-CN/std/cmp/trait.Eq.html>
> - `std::cmp::PartialOrd`：<https://rustwiki.org/zh-CN/std/cmp/trait.PartialOrd.html>
> - `std::cmp::Ord`：<https://rustwiki.org/zh-CN/std/cmp/trait.Ord.html>

### `std::default`

```rust
pub trait Default: Sized {
    fn default() -> Self;
}
```

> - `std::default::Default`：<https://rustwiki.org/zh-CN/std/default/trait.Default.html>

### `std::iter`

```rust
pub trait Iterator {                                // 迭代器
    type Item;
    fn next(&mut self) -> Option<Self::Item>;       // 仅此方法需实现，其余方法均有默认实现
    ...
}

pub trait DoubleEndedIterator: Iterator {           // 双头迭代器
    fn next_back(&mut self) -> Option<Self::Item>;
    ...
}

pub trait ExactSizeIterator: Iterator {             // 长度确定迭代器
    fn len(&self) -> usize {...}
    fn is_empty(&self) -> bool {...}
}
```

-   `std::iter` 中特性说明
    -   `Extend`、`IntoIterator`、`FromIterator` 均为为集合实现的特性
        -   `Extend`：指可用迭代器内容扩展 `impl Extend` 的集合
        -   `IntoIterator`：可将 `impl IntoIterator` 的集合转换为迭代器
        -   `FromIterator`：可从迭代器创建 `impl From Iterator` 的集合

```rust
pub trait IntoIterator {                            // 可转换为迭代器
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;
    fn into_iter(self) -> Self::IntoIter;
}

pub trait Extend<A> {                               // 可从迭代器中获取元素、扩展自身，即 `Vec` 等容器
    fn extend<T>(&mut self, iter: T)
        where T: IntoIterator<Item = A>;
    fn extend_one(&mut self, item: A) {...}
    fn extend_reserve(&mut self, additional: usize) { ... }
}

pub trait FromIterator<A>: Sized {                  // 可从迭代器转换，即 `Vec`、`HashMap` 等容器
    fn from_iter<T>(iter: T) -> Self
        where T: IntoIterator<Item = A>;
}
```

####    `iter::Iterator`

> - `std::iterator::Iterator`：<https://rustwiki.org/zh-CN/std/iter/trait.Iterator.html>

#####   迭代

| `Iterator` 方法                         | 返回值                                             | 说明                                         | 其他  |
|-----------------------------------------|----------------------------------------------------|----------------------------------------------|-------|
| `next(&mut self)`                       | `Option<Self::Item>`                               |                                              |       |
| `next_chunk<const N: usize>(&mut self)` | `Result<[Self::Item; N], IntoIter<Self::Item, N>>` | 迭代为 `N` 长数组                            | *Exp* |
| `last(self)`                            | `Option<Self::Item>`                               | 返回最后                                     |       |
| `advance_by(&mut self, n: usize)`       | `Result<(), NoneZeroUsize>`                        | 跳过 `n` 个                                  | *Exp* |
| `nth(&mut self, n: usize)`              | `Option<Self::Item>`                               | 返回后第 `n` 个                              |       |
| `find<P>(&mut self, predicate: P)`      | `Option<Self::Item>`                               | `P(&Self::Item) -> bool` 首个 `true` 元素    |       |
| `find_map<B, F>(&mut self, f: F)`       | `Option<B>`                                        | `P(&Self::Item) -> Option<B>` 首个 `Some<B>` |       |
| `position<P>(&mut self, predicate: P)`  | `Option<usize>`                                    | `P(Self::Item) -> bool` 首个 `true` 位置     |       |
| `rposition<P>(&mut self, predicate: P)` | `Option<usize>`                                    | `P(Self::Item) -> bool` 右数首个 `true` 位置 |       |

#####   转换（适配器）

| `Iterator` 方法                                 | 适配器结果            | 适配器迭代返回 `Option<>`            | 说明                                                    | 其他 |
|-------------------------------------------------|-----------------------|--------------------------------------|---------------------------------------------------------|------|
| `map<B, F>(self, f: F)`                         | `Map<Self, F>`        | `B`                                  | `F(Self::Item) -> B` 映射为 `B`                         |      |
| `filter<P>(self, predicate: P)`                 | `Filter<Self, P>`     | `Self::Item`                         | `P(&Self::Item) -> bool` 判断                           |      |
| `filter_map<B, F>(self, f: F)`                  | `FilterMap<Self, F>`  | `B`                                  | `F(Self::Item) -> Option<B>` 映射为 `B`                 |      |
| `enumerate(self)`                               | `Enumerate<Self>`     | `(usize, Self::Item)`                | 返回 `(idx, val)`                                       |      |
| `peekable(self)`                                | `Peekable<Self>`      | `Self::Item`                         | 可 `.peek()` 提前获取下个结果的引用而不消耗             |      |
| `skip(self, n: usize)`                          | `Skip<Self>`          | `Self::Item`                         | 跳过前 `n` 个                                           |      |
| `take(self, n: usize)`                          | `Take<Self>`          | `Self::Item`                         | 截断前 `n` 个                                           |      |
| `skip_while<P>(self, predicate: P)`             | `SkipWhile<Self, P>`  | `Self::Item`                         | `P(&Self::Item) -> bool` 跳过直至首个 `False`           |      |
| `take_while<P>(self, predicate: P)`             | `TakeWhile<Self, P>`  | `Self::Item`                         | `P(&Self::Item) -> bool` 获取直至首个 `False`           |      |
| `map_while<B, P>(self, predicate: P)`           | `MapWhile<Self, P>`   | `B`                                  | `P(Self::Item) -> Option<B>` 映射为 `B` 直至首个 `None` |      |
| `step_by(self, step: usize)`                    | `StepBy<Self>`        | `Self::Item`                         | 间隔返回                                                |      |
| `flatten(self)`                                 | `Flatten<Self>`       | `<Self::Item as IntoIterator>::Item` | 展平                                                    |      |
| `flat_map<U, F>(self, f: F)`                    | `FlatMap<Self, U, F>` | `<U as IntoIterator>::Item`          | 将 `F(Self::Item) -> U` 映射结果 `U` 展平               |      |
| `fuse(self)`                                    | `Fuse<Self>`          | `Self::Item`                         | 获取元素直至首个 `None`                                 |      |
| `copied<'a, T>(self)`                           | `Copied<Self>`        | `Self::Item`                         | 逐元素复制                                              |      |
| `cloned<'a, T>(self)`                           | `Cloned<Self>`        | `Self::Item`                         | 逐元素克隆                                              |      |
| `cycle(self)`                                   | `Cycle<Self>`         | `Self::Item`                         | 循环                                                    |      |
| `scan<St, B, F>(self, initial_state: St, f: F)` | `Scan<Self, St, F>`   | `B`                                  | `F(&mut St, Self::Item) -> Option<B>` 带状态转换        |      |

-   适配器方法封装原迭代器，并再次实现 `Iterator` 以实现延迟执行
    -   `flat_map(f)` 可视为 `map(f).flatten()`
    -   `Peekable<Self>.peek()`、`.peek_mut()` 可以查看迭代器下个元素而不消耗
        -   但注意，首次调用 `peek()`、`peek_mut()` 时，底层迭代器在前进（即有副作用）

#####   特殊转换、应用

| `Iterator` 方法                            | 返回值                                       | 说明                                           | 其他  |
|--------------------------------------------|----------------------------------------------|------------------------------------------------|-------|
| `by_ref(&mut self)`                        | `&mut Self`                                  | 借用迭代器                                     |       |
| `chain<U>(self, other: U)`                 | `Chain<Self, <U as IntoIterator>::IntoIter>` | 链接                                           |       |
| `zip<U>(self, other:U)`                    | `Zip<Self, <U as IntoIterator>::IntoIter>`   | 匹配迭代                                       |       |
| `unzip<A, B, FromA, FromB>(self)`          | `(FromA, FromB)`                             | 迭代分拆                                       |       |
| `intersperse(self, separator: Self::Item)` | `Intersperse<Self>`                          | 插入 `U` 分隔、迭代                            | *Exp* |
| `intersperse_with<G>(self, separator:G)`   | `IntersperseWith<Self, G>`                   | 插入 `G()` 分隔、迭代                          | *Exp* |
| `array_chunks<const N: usize>(self)`       | `ArrayChunk<Self, N>`                        | 按 `N` 长数组迭代                              | *Exp* |
| `inspect<F>(self, f: F)`                   | `Inspect<Self, F>`                           | `F(&Self::Item)` 逐个检视                      |       |
| `partition<B, F>(self, f: F)`              | `(B, B)`                                     | 根据 `F(&Self::Item) -> bool` 分组为两集合 `B` |       |

-   特殊转换、应用的说明
    -   `inspect<F>(self, f: F)` 常被 **插入链式调用中检视元素**
        -   `F(&Self::Item)` 获取迭代的元素的不可变引用、无返回值
        -   即，适配器（安全地）不修改迭代元素 `Self::Item`、仅作查看，可视为适配器版 `for_each(self, f: F)`
    -   `by_ref(&mut self)` 常被用于希望保留迭代器所有权用于后续使用的场合
        -   `by_ref` 实际上即返回迭代器的可变引用
        -   即，`by_ref` 仅是用于简化链式调用、配合 `take` 等获取所有权方法保留所有权

#####   聚集

| `Iterator` 方法                             | 返回值                                | 说明                                                       | 其他  |
|---------------------------------------------|---------------------------------------|------------------------------------------------------------|-------|
| `reduce<F>(self, f: F)`                     | `Option<Self::Item>`                  |                                                            |       |
| `fold<B, F>(self, init: B, f: F)`           | `B`                                   | `F(B, Self::Item) -> B` 聚集                               |       |
| `collect<B>(self)`                          | `B: Default + Extend<Self::Item>`     | 转换为集合 `B`                                             |       |
| `collect_into<E>(self, collection: &mut E)` | `&mut E; where E: Extend<Self::Item>` | `Extend` 至集合 `E`                                        | *Exp* |
| `max(self)`                                 | `Option::<Self::Item>`                | 末个最大元素                                               |       |
| `min(self)`                                 | `Option::<Self::Item>`                | 首个最小元素                                               |       |
| `max_by_key<B, F>(self, f: F)`              | `Option<Self::Item>`                  | 按 `F(&Self::Item) -> B` 作键，末个最大元素                |       |
| `max_by<F>(self, compare:F)`                | `Option<Self::Item>`                  | 按 `F(&Self::Item, &Self::Item) -> Ordering`，末个最大元素 |       |
| `min_by_key<B, F>(self, f: F)`              | `Option<Self::Item>`                  | 按 `F(&Self::Item) -> B` 作键，首个最小元素                |       |
| `min_by<F>(self, compare:F)`                | `Option<Self::Item>`                  | 按 `F(&Self::Item, &Self::Item) -> Ordering`，首个最小元素 |       |
| `rev(self)`                                 | `Rev<Self>`                           | 逆序双头迭代器                                             |       |
| `sum<S>(self)`                              | `S: Sum<Self::Item>`                  | 求和                                                       |       |
| `product<P>(self)`                          | `P: Product<Self::Item>`              | 求积                                                       |       |
| `for_each<F>(self, f: F)`                   | `()`                                  | `P(Self::Item) -> ()` 逐个应用                             |       |

-  聚集方法说明
    -   `for_each<F>(self, f: F)` 方法返回 `()`，可视为特殊聚集方法

#####   `try_`

| `Iterator` 方法                               | 返回值                                                                   | 说明                                         | 其他  |
|-----------------------------------------------|--------------------------------------------------------------------------|----------------------------------------------|-------|
| `try_fold<B, F, R>(&mut self, init: B, f: F)` | `R: Try<Output = B>`                                                     | 解包、有初值 `B` 聚集                        |       |
| `try_for_each<F, R>(&mut self, f: F)`         | `R: Try<Output = ()>`                                                    | 解包、无状态聚集（逐个应用）                 |       |
| `try_collect<B>(self)`                        | `<<Self::Item as Try>::Residual as Residual<B>>::TryType`                | 解包、转换为集合                             | *Exp* |
| `try_reduce<F, R>(&mut self, f: F)`           | `<<R as Try>::Residual as Residual<Option<<R as Try>::Output>>>:TryType` | 解包、无初值聚集                             | *Exp* |
| `try_find<F, R>(&mut self, f: F)`             | `<<R as Try>::Residual as Residual<Option<Self::Item>>>::TryType`        | `F(&Self::Item) -> R` 首个 `true` 元素或错误 | *Exp* |

-   迭代器的 `try_` 类方法通常指 “允许失败” 的方法
    -   “允许失败” 通常指迭代元素包含可 `Option::None`、`Result::Err()`
        -   对 `Option::Some()`、`Result::Ok()` 将解包参与计算
        -   对 `Option::None`、`Result:Err()` 将中断计算，并短路返回 `Err()`、`None` 类似结果

#####   元信息

| `Iterator` 方法                                    | 返回值                   | 说明                                                         | 其他  |
|----------------------------------------------------|--------------------------|--------------------------------------------------------------|-------|
| `size_hint(&self)`                                 | `(usize, Option<usize>)` | 长度下、上界                                                 |       |
| `count(self)`                                      | `usize`                  | 计数                                                         |       |
| `partition_in_place('a, T, P)(self, predicate: P)` | `usize`                  | 根据 `P(&Self::Item = &'a mut T) -> bool` 在位划分双头迭代器 | *Exp* |
| `is_partitioned<P>(self, predicate: P)`            | `bool`                   | 根据 `P(Self::Item) -> bool` 检查是否划分                    | *Exp* |
| `all<F>(&mut self, f: F)`                          | `bool`                   | 检查 `F(Self::Item) -> bool` 是否均满足                      |       |
| `any<F>(&mut self, f: F)`                          | `bool`                   | 检查 `F(Self::Item) -> bool` 是否存在满足                    |       |

-   迭代器元信息说明
    -   `size_hint(&self)` 需返回迭代器剩余长度上下界 `(usize, Option<usize>)`
        -   默认返回总正确的 `(0, None)`（`None` 表示无上界）
        -   `size_hint` 结果不可信，没有强制要求迭代器产生元素数量满足上下界要求
        -   但，若迭代器 `impl ExactSizeIterator`，则上下界必须相同、且准确
            -   `ExactSizeIterator::len()` 方法内部默认即调用 `size_hint` 方法

#####   逐元素比较、排序

| `Iterator` 方法                                | 返回值             | 说明                                                            | 其他  |
|------------------------------------------------|--------------------|-----------------------------------------------------------------|-------|
| `cmp<I>(self, other: I)`                       | `Ordering`         | 字典序比较两迭代器                                              |       |
| `cmp_by<I, F>(self, other: I, cmp: F)`         | `Ordering`         | 按 `F(Self::Item, <I as IntoIterator>::Item) -> Ordering` 比较  | *Exp* |
| `partial_cmp<I>(self, other: I)`               | `Option<Ordering>` | 字典序、偏序比较                                                |       |
| `partial_cmp_by<I, F>(self, other: I, cmp: F)` | `Option<Ordering>` | 按 `F` 字典序、偏序比较                                         | *Exp* |
| `eq<I>(self, other: I)`                        | `bool`             | 相等                                                            |       |
| `eq_by<I, F>(self, other: I, cmp: F)`          | `bool`             | 按 `F` 相等                                                     | *Exp* |
| `ne<I>(self, other: I)`                        | `bool`             | 不相等                                                          |       |
| `lt<I>(self, other: I)`                        | `bool`             | 字典序小于                                                      |       |
| `le<I>(self, other: I)`                        | `bool`             | 字典序不大于                                                    |       |
| `gt<I>(self, other: I)`                        | `bool`             | 字典序大于                                                      |       |
| `ge<I>(self, other: I)`                        | `bool`             | 字典序不小于                                                    |       |
| `is_sorted(self)`                              | `bool`             | 是否已排序                                                      | *Exp* |
| `is_sorted_by<F>(self, compare: F)`            | `bool`             | 是否按 `F(&Self::Item, &Self::Item) -> Option<Ordering>` 已排序 | *Exp* |
| `is_sorted_by_key<K, F>(self, f: F)`           | `boool`            | 是否按 `F(Self::Item) -> K` 作键已排序                          | *Exp* |

### `std::ops`

####    `ops::Fn_`

```rust
pub trait FnOnce<Args>
where
    Args: Tuple,
{
    type Output;
    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

pub trait FnMut<Args>: FnOnce<Args>
where
    Args: Tuple,
{
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

pub trait Fn<Args>: FnMut<Args>                 // `impl Fn` 不变闭包可用于 `FnMut` 可变闭包场合
where
    Args: Tuple,
{
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}
```

####    `ops::Index`

```rust
pub trait Index<Idx: ?Sized> {
    type Output: ?Sized;
    fn index(&self, index: Idx) -> &Self::Output;
}
```

> - `std::ops::Index`：<https://doc.rust-lang.org/stable/std/ops/trait.Index.html>

### `std::any`

```rust
pub trait Any: 'static {
    fn type_id(&self) -> TypeId;                                // 类型唯一ID
}
impl dyn Any {                                                  // 为动态对象实现
    fn is<T: Any>(&self) -> bool { }
    fn downcast_ref<T: Any>(&self) -> Option<&T> { }
    fn downcast_ref<T: Any>(&mut self) -> Option<&mut T> { }
    ...
}
```

-   `std::any` 用于模拟动态类型、类型反射
    -   `any::Any` 核心即为、要求类型配置唯一的 `TypeId`
        -   运行时，根据 `TypeId` 判断值类型、实现类型降级，模拟动态类型
        -   故，动态类型模拟只能在 `impl Any` 上实现

> - `std::any`：<https://doc.rust-lang.org/stable/std/any/index.html>
> - `std::any::Any`：<https://doc.rust-lang.org/stable/std/any/trait.Any.html>

### `std::io`

> - `std::io`：<https://doc.rust-lang.org/stable/std/io/index.html>

####    `io::Read`

| `Read` 方法              | 返回值                   | 描述                                                 | 其他                             |
|--------------------------|--------------------------|------------------------------------------------------|----------------------------------|
| `read(buf)`              | `Result<usize>`          | 从自身读取字节至 `&mut [u8]`、返回字节数             | *Req*、失败返回 `Err`            |
| `read_exact(buf)`        | `Result<()>`             | 读取字节刚好填充 `&mut Vec<u8>`                      |                                  |
| `read_to_end(buf)`       | `Result<usize>`          | 读取全部字节至 `&mut [u8]`                           |                                  |
| `read_to_string(buf)`    | `Result<usize>`          | 读取全部字节至 `&mut String`                         |                                  |
| `read_vectored(buf)`     | `Result<usize>`          | 从自身读取字节至 `&mut [IoSliceMut<'_>]`、返回字节数 | 失败返回 `Err`                   |
| `is_read_vectored()`     | `bool`                   | 是否有高效 `read_vectored` 实现                      | *Exp*                            |
| `by_ref()`               | `&mut Self`              | 创建 `&mut Self`                                     |                                  |
| `bytes()`                | `Bytes<Self>`            | 转换为字节迭代器                                     | 迭代项为 `Result<u8, io::Error>` |
| `chain(next: impl Read)` | `Chain<Self, impl Read>` | 转换为串联 *Reader*                                  |                                  |
| `take(limit)`            | `Take<Self>`             | 转换为最多读取 `limit` 字节                          |                                  |

-   `impl io::Read` 类型即为 *Reader*，可从中读取字节
    -   `Read::read` 方法不为数据阻塞做任何保证
        -   对 `Ok(n)` 返回值，只、需保证 `0 <= n <= buf.len()`
            -   对 `n > 0`，表示缓冲 `buf` 填充 `n` 字节数据
            -   对 `n == 0`，表示数据 “暂时” 全部读取完毕、或缓冲区长度 `buf.len() == 0`
            -   `n < buf.len()` 是合理的，即使数据未读取完毕
        -   `read` 方法不应对 `buf` 有任何假设
            -   即，只应向 `buf` 写入数据而不应读取数据
        -   若 `read` 遇到任何错误，应、将返回错误，此时应保证未读取任何数据
            -   `ErrorKind::Interrupted` 错误表示非致命错误、可重试
    -   其他 `read_` 方法默认实现均通过调用 `read()` 实现
        -   若获取到 `read` 返回的 `Interrupted` 将会重试，否则直接返回错误

> - `std::io::Read`：<https://doc.rust-lang.org/stable/std/io/trait.Read.html>

####    `io::BufRead: Read`

| `BufRead` 方法          | 返回值          | 描述                                                | 其他                                      |
|-------------------------|-----------------|-----------------------------------------------------|-------------------------------------------|
| `fill_buf()`            | `Result<&[u8]>` | 返回内部缓冲区内容                                  | *Req*                                     |
| `consume(amount)`       | `()`            | 标记内部缓冲区字节已读                              | *Req*                                     |
| `has_data_left()`       | `Result<bool>`  | 检查是否还有数据可读                                | *Exp*，可能读取数据至内部缓冲区并 `Error` |
| `read_until(byte, buf)` | `Result<usize>` | 读取数据至 `&mut Vec<u8>` 直至 `byte`、末尾 `<EOF>` |                                           |
| `skip_until(byte)`      | `Result<usize>` | 读取、丢弃数据直至 `byte`、末尾 `<EOF>`             |                                           |
| `read_line(buf)`        | `Result<usize>` | 读取数据直至 `0x0A` 字节追加至 `&mut String`        |                                           |
| `split(byte)`           | `Split<Self>`   | 迭代 `byte` 分割结果                                |                                           |
| `lines()`               | `Line<Self>`    | 迭代 `0x0A` 分割结果                                |                                           |

-   `impl io::BufRead` 是具有内部缓冲的 *Reader*
    -   `impl BufRead` 实现有额外的数据读取方式
        -   按行、指定分隔符迭代读取
    -   `BufRead::fill_buf()` 是低层方法（相较 `Read` 中方法）
        -   与 `consume(amount)` 搭配使用标记

> - `std::io::BufRead`：<https://doc.rust-lang.org/stable/std/io/trait.BufRead.html>

####    `io::Write`

| `Write` 方法                   | 返回值          | 描述                                                     | 其他                 |
|--------------------------------|-----------------|----------------------------------------------------------|----------------------|
| `flush()`                      | `()`            | 显式、强制将自身数据刷入真正目标                         | *Req*                |
| `write(buf)`                   | `Result<usize>` | 从 `&[u8]` 向自身写入、返回字节数                        | *Req*                |
| `write_all(buf)`               | `Reuslt<()>`    | 将 `&[u8]` 中字节全部写入自身                            |                      |
| `writer_vectored(buf)`         | `Result<usize>` | 从 `&[IoSlice<'_>]` 获取字节写入自身、返回字节数         |                      |
| `writer_all_vectored(buf)`     | `Result<usize>` | 从 `&mut [IoSlice<'_>]` 获取字节全部写入自身、返回字节数 | *Exp*                |
| `write_fmt(args)`              | `Result<()>`    | 从 `Arguments<'_>` 向自身写入格式化字符串                | 常被 `write!` 宏替代 |
| `is_write_vectored(&mut self)` | `bool`          | 是否有高效 `write_vectored` 实现                         | *Exp*                |

-   实现 `io::Writer` 类型即为 *Writer*，可向其中写入字节
    -   `Writer::write(buf)` 方法为需实现方法
        -   `write` 方法尝试将所有内容写入自身、不保证等待数据阻塞
        -   对 `Ok(n)` 返回值，只、需保证 `0 <= n <= buf.len()`
            -   对 `n == 0`，表示无法向任何数据中写入、或缓冲区长度 `buf.len() == 0`
            -   对 `n > 0`，表示已从缓冲 `buf` 向自身写入 `n` 字节数据
        -   若 `write` 遇到任何错误，应、将返回错误，此时应保证未写入任何数据
            -   `ErrorKind::Interrupted` 错误表示非致命错误、可重试
    -   其他 `write_` 方法默认实现均通过调用 `write()` 实现
        -   若获取到 `write()` 返回的 `Interrupted` 将会重试，否则直接返回错误
    -   `Writer::write_fmt(args)` 用于对接 `format_args!` 宏，常用 `write!` 宏替代

```rust
let mut buffer = File::create("foo.txt")?;
write!(buffer, "{:.*}", 2, 1.234};                      // 转换为如下
buffer.write_fmt(format_args!("{:.*}", 2, 1.234};
```

> - `std::io::Write`：<https://doc.rust-lang.org/stable/std/io/trait.Write.html>

####    `io::Seek`

| `Seek` 方法             | 返回值        | 描述                                         | 其他                                 |
|-------------------------|---------------|----------------------------------------------|--------------------------------------|
| `seek(pos)`             | `Result<u64>` | 指针移动至 `SeekFrom` 偏移、返回距开头处偏移 | *Req*                                |
| `seek_relative(offset)` | `Result<()>`  | 指针移动至相对当前位置偏移                   | 不返回绝对位置，可能较 `seek` 更高效 |
| `rewind()`              | `Result<()>`  | 指针移动至开头                               |                                      |
| `stream_len()`          | `Result<u64>` | 获取流长                                     |                                      |
| `stream_position()`     | `Result<u64>` | 获取距开头处偏移                             |                                      |

-   实现 `io::Seek` 类型可移动内部指示当前读、写位置的指针
    -   要求类型有固定长度

> - `std::io::Seek`：<https://doc.rust-lang.org/stable/std/io/trait.Seek.html>

### 异步任务

####    `future::Future`

```rust
pub trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

pub trait IntoFuture {
    type Output;
    type IntoFuture: Future<Output = Self::Output>;
    fn into_future(self) -> Self::IntoFuture;
}
```

-   `future::Future` 用于实现具体异步计算任务、状态机
    -   `Future::poll()` 方法尝试推动 `Future` 计算任务（手动定义异步任务、`impl Future`）
        -   若返回、得到 `task::Poll::Ready(T)`，表示任务完成、结果为 `T`
        -   若返回 `task::Poll::Pending`，表示任务尚未就绪，在返回前应
            -   `poll()` 中应 `Context::waker()` 获取、设置 `&Waker`
            -   执行具体异步任务，并在任务完成后通知异步运行时、唤醒任务
    -   此外，常用 `async {}` 异步代码块生成匿名 `impl Future`
        -   其中，常用 `<impl Future>.await` 推动 `Future` 计算任务，而不是直接调用 `Future::poll()`

> - `std::future::Future`：<https://doc.rust-lang.org/stable/std/future/trait.Future.html>
> - `std::future::IntoFuture`：<https://doc.rust-lang.org/stable/std/future/trait.IntoFuture.html>

####    `task::Wake`

```rust
pub trait Wake {
    fn wake(self: Arc<Self>);                   // 仅必须实现此方法
    fn wake_by_ref(self: &Arc<Self>) {}
}
impl<W> From<Arc<W>> for Waker                  // `Wake` 特性用于快速创建 `Waker`
where
    W: Wake + Send + Sync + 'static
```

-   `task::Wake` 用于快速创建 `task::Waker`
    -   `Waker` 实现有 `From<Arc<W: impl Wake>>`，可从通过定义类型 `W: impl Wake` 快速创建

> - `std::task::Wake`：<https://doc.rust-lang.org/stable/std/task/trait.Wake.html>

##  常用类型

| 常用类型                             | 说明                       | *Prelude*  |
|--------------------------------------|----------------------------|------------|
| `option::Option::{self, Some, None}` | 值存在与否                 | 2018       |
| `result::Result::{self, Ok, Err}`    | 结果正确与否               | 2018       |
| `string::String`                     | 堆分配字符串               | 2018       |
| `box::Box<T>`                        | 堆上值包装                 | 2018       |
| `vec::Vec<T>`                        | 变长、堆分配向量           | 2018       |
| `collections::VecDeque<T>`           | 双头向量                   |            |
| `rc::Rc<T>`                          | 单线程引用计数             |            |
| `rc::Weak<T>`                        | 弱引用计数                 |            |
| `cell::Cell<T>`                      | 不可借用的内部可变性       |            |
| `cell::RefCell<T>`                   | 允许借用的内部可变性       |            |
| `cell::Ref<T>`                       | 内部值借用                 |            |
| `cell::RefMut<T>`                    | 内部值可变借用             |            |
| `cell::OnceCell<T>`                  | 初始化时修改一次、可借用   |            |
| `cell::LazyCell<T>`                  | 延迟初始化                 |            |
| `collections::HashMap<K,V>`          | 哈希表                     |            |
| `collections::HashSet<K,V>`          | 哈希集合                   |            |
| `borrow::Cow<'a, B>`                 | 写时克隆智能指针           |            |
| `io::BufReader<R>`                   | 带缓冲 *Reader*            |            |
| `io::BufWriter<W>`                   | 带缓冲 *Writer*            |            |
| `io::Stdin`                          | 标准输入                   |            |
| `io::Stdout`                         | 标准输出                   |            |
| `fs::File`                           | 文件系统文件               |            |
| `net::TcpListener`                   | TCP 监听服务               |            |
| `net::TcpStream`                     | TCP 数据流                 |            |
| `time::Duration`                     | 时间段                     |            |
| `time::Instant`                      | 单增时刻计数               |            |
| `time::SystemTime`                   | 系统时间                   |            |
| `thread::Thread`                     | 线程句柄                   |            |
| `thread::JoinHandle`                 | 线程阻塞句柄               |            |
| `thread::Builder`                    | 线程工厂                   |            |
| `thread::LocalKey<T: 'static>`       | 线程局部变量（静态）       |            |
| `sync::Arc<T>`                       | 原子（线程安全）引用计数   |            |
| `sync::Mutex<T>`                     | 互斥锁                     |            |
| `sync::RwLock`                       | 读写锁                     |            |
| `sync::Barrier`                      | 多线程执行障碍             |            |
| `sync::CondVar`                      | 条件变量                   |            |
| `sync::mpsc`                         | 多生产者、单消费者消息队列 | 模块       |
| `sync::mpmc`                         | 多生产者、多消费者消息队列 | *Exp* 模块 |
| `mem::MaybeUninit<T>`                | 未初始化实例               |            |
| `mem::ManuallyDrop<T>`               | 避免编译器自动销毁实例     |            |
| `ptr::NonNull<T>`                    | 非空、协变 `*mut T`        |            |
| `alloc::Layout`                      | 内存布局                   |            |
| `task::Context`                      | 异步上下文                 |            |
| `task::Waker`                        | 唤醒器                     |            |
| `task::RawWaker`                     | 裸唤醒器（宽指针）         |            |
| `task::RawWakerVTable`               | 唤醒器函数虚表             |

> - `std::prelude`：<https://doc.rust-lang.org/stable/std/prelude/index.html>
> - `std::prelude` 中文：<https://rustwiki.org/zh-CN/std/prelude/index.html>

### `[T; N]` 数组

| `[T; N]` 方法    | 返回值        | 描述                | 其他                  |
|------------------|---------------|---------------------|-----------------------|
| `map(f)`         | `[U; N]`      | 按 `f(T) -> U` 映射 |                       |
| `as_slice()`     | `&[T]`        | 创建共享切片        | 等价于 `&arr[..]`     |
| `as_mut_slice()` | `&[T]`        | 创建可变切片        | 等价于 `&mut arr[..]` |
| `each_ref()`     | `[&T; N]`     | 创建共享引用数组    |                       |
| `each_mut()`     | `[&mut T; N]` | 创建可变引用数组    |                       |

> - Primitive Type `array`：<https://doc.rust-lang.org/stable/std/primitive.array.html>

### `[T]` 切片

-   `&[T]` 是作为数组、`Vec` 引用的强制转换对象，很多方法可直接在数组、`Vec` 上调用

> - Primitive Type `slice`：<https://doc.rust-lang.org/stable/std/primitive.slice.html>
> - `std::vec`：<https://doc.rust-lang.org/stable/std/vec/index.html>

####    `[T]` 元信息、类型转换

| `[T]` 元信息、逻辑      | 返回值  | 描述                                      | 其他 |
|-------------------------|---------|-------------------------------------------|------|
| `len()`                 | `usize` | 长度                                      |      |
| `is_sorted()`           | `bool`  | 是否有序                                  |      |
| `is_sorted_by(compare)` | `bool`  | 是否按 `compare(&T, &T) -> Ordering` 有序 |      |
| `is_sorted_by_key(f)`   | `bool`  | 是否按 `f(&T) -> PartialOrd` 作键有序     |      |
| `is_empty()`            | `bool`  | 是否长为 0                                |      |
| `contains(x)`           | `bool`  | 是否包含                                  |      |
| `starts_with(needle)`   | `bool`  | 是否为前缀                                |      |
| `ends_with(needle)`     | `bool`  | 是否为后缀                                |      |

| `[T]` 类型转换       | 返回值                           | 描述                 | 其他                          |
|----------------------|----------------------------------|----------------------|-------------------------------|
| `as_ptr()`           | `*const T`                       | 切片缓冲指针         |                               |
| `as_mut_ptr()`       | `*const T`                       | 切片缓冲可变指针     |                               |
| `as_ptr_range()`     | `Range<*const T>`                | 缓冲首尾指针         |                               |
| `as_mut_ptr_range()` | `Range<*mut T>`                  | 缓冲首尾指针         |                               |
| `as_array<N>()`      | `Option<&[T;N]>`                 | 创建底层数组引用     | `N` 不等于切片长时返回 `None` |
| `as_mut_array<N>()`  | `Option<&mut [T;N]>`             | 创建底层数组可变引用 | `N` 不等于切片长时返回 `None` |
| `align_to<U>()`      | `(&[T], &[U], &[T])`             | 位重解释为 `[U]`     | 同时返回未对齐首尾            |
| `align_to_mut<U>()`  | `(&mut [T], &mut [U], &mut [T])` | 位重解释位 `[U]`     | 同时返回未对其首尾            |

####    `[T]` 排序、查找

| `[T]` 排序、查找                             | 返回值                         | 描述                                                         | 其他                                      |
|----------------------------------------------|--------------------------------|--------------------------------------------------------------|-------------------------------------------|
| `partition_point(pred)`                      | `usize`                        | 首个 `pred(&T) == false` 划分点                              |                                           |
| `binary_search(x)`                           | `Result<usize, usize>`         | 二分查找                                                     | 返回位置或可插入位置，要求已有序          |
| `binary_search_by(x, f)`                     | `Result<usize, usize>`         | 按 `f(&T) -> Ordering` 比较结果二分查找                      | 返回位置或可插入位置，要求已有序          |
| `binary_search_by_key(b: B, f)`              | `Result<usize, usize>`         | 按 `f(&T) -> B` 做键二分查找                                 | 返回位置或可插入位置，要求按键 `B` 已有序 |
| `sort_unstable()`                            | `()`                           | 在位不稳定排序                                               |                                           |
| `sort_unstable_by(compare)`                  | `()`                           | 在位按 `compare(&T, &T) -> Ordering` 不稳定排序              |                                           |
| `sort_unstable_by_key(f)`                    | `()`                           | 在位按 `f(&T) -> Ord` 作键不稳定排序                         |                                           |
| `select_nth_unstable(index)`                 | `(&mut [T], &mut T, &mut [T])` | 在位划分出第 `index` 大元素                                  |                                           |
| `select_nth_unstable_by(index, compare)`     | `(&mut [T], &mut T, &mut [T])` | 在位按 `compare(&T, &T) -> Ordering` 划分出第 `index` 大元素 |                                           |
| `select_nth_unstable_by_key(index, compare)` | `(&mut [T], &mut T, &mut [T])` | 在位按 `f(&T) -> Ord` 作键划分出第 `index` 大元素            |                                           |

####    `[T]` 在位填充、修改

| `[T]` 在位填充、复制     | 返回值 | 描述                                     | 其他                        |
|--------------------------|--------|------------------------------------------|-----------------------------|
| `fill(value)`            | `()`   | 在位填充值                               |                             |
| `fill_with(f)`           | `()`   | 在位填充 `f()`                           |                             |
| `reverse()`              | `()`   | 在位逆序                                 |                             |
| `copy_from_slice(src)`   | `()`   | 从相同长度 `src` Copy                    | 长度需相等                  |
| `clone_from_slice(src)`  | `()`   | 从相同长度 `src` Clone                   | 长度需相等                  |
| `copy_within(src, dest)` | `()`   | 从自身范围 `src` Copy 至 `dest` 开始位置 |                             |
| `swap(a, b)`             | `()`   | 在位交换两位置元素                       | 两位置元素相等时无动作      |
| `swap_with_slice(other)` | `()`   | 交换内容                                 |                             |
| `swap_unchecked(a, b)`   | `()`   | 在位交换两位置元素                       | *Exp* `unsafe` 需保证不越界 |
| `rotate_left(mid)`       | `()`   | 在位将首 `mid` 个、剩余交换位置          |                             |
| `rotate_right(k)`        | `()`   | 在位将末 `mid` 个、剩余交换位置          |                             |

####    `[T]` 索引

| `[T]` 索引                            | 返回值                                                                | 描述                      | 其他                                    |
|---------------------------------------|-----------------------------------------------------------------------|---------------------------|-----------------------------------------|
| `get(index)`                          | `Option<&<I as SliceIndex<[T]>>::Output>`                             | 共享引用或子切片          | 支持位置、范围                          |
| `get_mut(index)`                      | `Option<&mut <I as SliceIndex<[T]>>::Output>`                         | 可变引用或子切片          | 支持位置、范围                          |
| `get_unchecked(index)`                | `&<I as SliceIndex<[T]>>::Output`                                     | 共享引用或切片            | `unsafe` 需确保不越界                   |
| `get_unchecked_mut(index)`            | `&mut <I as SliceIndex<[T]>>::Output`                                 | 可变引用或切片            | `unsafe` 需确保不越界                   |
| `get_disjoint_unchecked_mut(indices)` | `[&mut <I as SliceIndex<[T]>>::Output;N]`                             | 获取多个引用              | 支持范围返回切片，`unsafe` 需确保不越界 |
| `get_disjoint_mut(indices)`           | `Result<[&mut <I as SliceIndex<[T]>>::Output;N], GetDisjoinMutError>` | 获取多个引用              | 支持范围返回切片，`unsafe` 需确保不越界 |
| `first()`                             | `Option<&T>`                                                          | 首个元素引用              |                                         |
| `first_mut()`                         | `Option<&mut T>`                                                      | 首个元素可变引用          |                                         |
| `first_chunk<N>()`                    | `Option<&[T;N]>`                                                      | 前 `N` 个元素数组引用     |                                         |
| `first_chunk_mut<N>()`                | `Option<&mut [T;N]>`                                                  | 前 `N` 个元素数组可变引用 |                                         |
| `last()`                              | `Option<&T>`                                                          | 末个元素引用              |                                         |
| `last_mut()`                          | `Option<&mut T>`                                                      | 末个元素可变引用          |                                         |
| `last_chunk<N>()`                     | `Option<&[T;N]>`                                                      | 前 `N` 个元素数组引用     |                                         |
| `last_chunk_mut<N>()`                 | `Option<&mut [T;N]>`                                                  | 前 `N` 个元素数组可变引用 |                                         |

####    `[T]` 二分

| `[T]` 二分                    | 返回值                           | 描述                          | 其他                  |
|-------------------------------|----------------------------------|-------------------------------|-----------------------|
| `split_first()`               | `Option<(&T, &[T])>`             | 拆分首个元素、剩余            |                       |
| `split_first_mut()`           | `Option<(&mut T, &mut [T])>`     | 可变拆分首个元素、剩余        |                       |
| `split_first_chunk<N>()`      | `Option<(&[T;N], &[T])>`         | 拆分前 `N` 个元素数组引用     |                       |
| `split_first_chunk_mut<N>()`  | `Option<(&mut [T;N], &mut [T])>` | 拆分前 `N` 个元素数组可变引用 |                       |
| `split_last()`                | `Option<(&T, &[T])>`             | 拆分末个元素、剩余            |                       |
| `split_last_mut()`            | `Option<(&mut T, &mut [T])>`     | 可变拆分末个元素、剩余        |                       |
| `split_last_chunk<N>()`       | `Option<(&[T;N], &[T])>`         | 拆分前 `N` 个元素数组引用     |                       |
| `split_last_chunk_mut<N>()`   | `Option<(&mut [T;N], &mut [T])>` | 拆分前 `N` 个元素数组可变引用 |                       |
| `split_at(mid)`               | `(&[T], &[T])`                   | 二分段                        | 越界 `panic`          |
| `split_at_mut(mid)`           | `(&mut [T], &mut [T])`           | 可变二分段                    | 越界 `panic`          |
| `split_at_unchecked(mid)`     | `(&[T], &[T])`                   | 二分段                        | `unsafe` 需确保不越界 |
| `split_at_mut_unchecked(mid)` | `(&mut [T], &mut [T])`           | 可变二分段                    | `unsafe` 需确保不越界 |
| `split_at_checked(mid)`       | `Option<(&[T], &[T])>`           | 二分段                        |                       |
| `split_at_mut_checked(mid)`   | `Option<(&mut [T], &mut [T])>`   | 可变二分段                    |                       |

####    `[T]` 分段

| `[T]` 分块                      | 返回值                    | 描述                       | 其他                  |
|---------------------------------|---------------------------|----------------------------|-----------------------|
| `as_chunks<N>()`                | `(&[[T;N]], &[T])`        | 分块，单独返回余数         |                       |
| `as_chunks_mut<N>()`            | `(&mut [[T;N], &mut [T])` | 可变分块                   |                       |
| `as_chunks_unchecked<N>()`      | `&[[T;N]]`                | 分块                       | `unsafe` 需保证可整分 |
| `as_chunks_unchecked_mut<N>()`  | `&mut [[T;N]]`            | 分块                       | `unsafe` 需保证可整分 |
| `as_rchunks<N>()`               | `(&[T], &[[T;N]])`        | 尾部开始分块，单独返回余数 |                       |
| `as_rchunks_mut<N>()`           | `(&mut [[T;N], &mut [T])` | 尾部开始可变分块           |                       |
| `as_rchunks_unchecked<N>()`     | `&[[T;N]]`                | 尾部开始分块               | `unsafe` 需保证可整分 |
| `as_rchunks_unchecked_mut<N>()` | `&mut [[T;N]]`            | 尾部开始可变分块           | `unsafe` 需保证可整分 |

####    `[T]` 移除方法

| `[T]` 移除                | 返回值                | 描述                        | 其他                               |
|---------------------------|-----------------------|-----------------------------|------------------------------------|
| `split_off(range)`        | `Option<&'a [T]>`     | 移除并返回 `range` 范围元素 | 越界返回 `None`                    |
| `split_off_mut(range)`    | `Option<&'a mut [T]>` | 移除并返回 `range` 范围元素 | 越界返回 `None`                    |
| `split_off_first()`       | `Option<&'a T>`       | 移除并返回首个元素          | 越界返回 `None`                    |
| `split_off_first_mut()`   | `Option<&'a mut T>`   | 移除并返回首个元素          | 越界返回 `None`                    |
| `split_off_last()`        | `Option<&'a T>`       | 移除并返回末个元素          | 越界返回 `None`                    |
| `split_off_last_mut()`    | `Option<&'a mut T>`   | 移除并返回末个元素          | 越界返回 `None`                    |
| `strip_prefix(prefix)`    | `Option<&[T]>`        | 剔除前缀                    | 前缀不匹配返回 `None`              |
| `strip_suffix(suffix)`    | `Option<&[T]>`        | 剔除后缀                    | 后缀不匹配返回 `None`              |
| `strip_circumfix(suffix)` | `Option<&[T]>`        | 剔除前、后缀                | *Exp*、前后缀任意不匹配返回 `None` |
| `trim_prefix(prefix)`     | `&[T]`                | 剔除前缀                    | *Exp*、前缀不匹配返回原切片        |
| `trim_suffix(suffix)`     | `&[T]`                | 剔除后缀                    | *Exp*、后缀不匹配返回原切片        |

####    `[T]` 迭代适配

| `[T]` 迭代适配                  | 迭代适配器               | 迭代返回 `Option<>` | 描述                     | 其他       |
|---------------------------------|--------------------------|---------------------|--------------------------|------------|
| `iter()`                        | `Iter<'_, T>`            | `&T`                | 迭代引用                 |            |
| `iter_mut()`                    | `IterMut<'_, T>`         | `&mut T`            | 迭代可变引用             |            |
| `windows(size)`                 | `Windows<'_, T>`         | `&[T]`              | 迭代滑动窗口（重叠）     |            |
| `chunks(chunk_size)`            | `Chunks<'_, T>`          | `&[T]`              | 迭代段（不重叠）         |            |
| `chunks_mut(chunk_size)`        | `ChunksMut<'_, T>`       | `&mut [T]`          | 迭代段（不重叠）         |            |
| `chunks_exact(chunk_size)`      | `ChunksExact<'_, T>`     | `&[T]`              | 迭代段（不重叠）         | 余数被保留 |
| `chunks_exact_mut(chunk_size)`  | `ChunksExactMut<'_, T>`  | `&mut [T]`          | 迭代段（不重叠）         | 余数被保留 |
| `rchunks(chunk_size)`           | `RChunks<'_, T>`         | `&[T]`              | 尾部开始迭代段（不重叠） |            |
| `rchunks_mut(chunk_size)`       | `RChunksMut<'_, T>`      | `&mut [T]`          | 尾部开始迭代段（不重叠） |            |
| `rchunks_exact(chunk_size)`     | `RChunksExact<'_, T>`    | `&[T]`              | 尾部开始迭代段（不重叠） | 余数被保留 |
| `rchunks_exact_mut(chunk_size)` | `RChunksExactMut<'_, T>` | `&mut [T]`          | 尾部开始迭代段（不重叠） | 余数被保留 |

| 有序 `[T]` 迭代适配         | 迭代适配器                    | 迭代返回 `Option<>` | 描述                                              | 其他               |
|-----------------------------|-------------------------------|---------------------|---------------------------------------------------|--------------------|
| `chunk_by(pred)`            | `ChunkBy<'_, T, F>`           | `&[T]`              | 在相邻元素 `pred(&T, &T) == false` 中间分段       |                    |
| `chunk_by_mut(pred)`        | `ChunkByMut<'_, T, F>`        | `&mut [T]`          | 在相邻元素 `pred(&T, &T) == false` 中间分段       |                    |
| `split(pred)`               | `Split<'_, T, F>`             | `&[T]`              | 以 `pred(&T) == false` 分段                       | 分段元素被忽略     |
| `split_mut(pred)`           | `SplitMut<'_, T, F>`          | `&mut [T]`          | 以 `pred(&T) == false` 分段                       | 分段元素被忽略     |
| `split_inclusive(pred)`     | `SplitInclusive<'_, T, F>`    | `&[T]`              | 在 `pred(&T) == false` 后分段                     | 分段元素归属前个段 |
| `split_inclusive_mut(pred)` | `SplitInclusiveMut<'_, T, F>` | `&mut [T]`          | 在 `pred(&T) == false` 后分段                     | 分段元素归属前个段 |
| `rsplit(pred)`              | `RSplit<'_, T, F>`            | `&[T]`              | 尾部开始，以 `pred(&T) == false` 分段             | 分段元素被忽略     |
| `rsplit_mut(pred)`          | `RSplitMut<'_, T, F>`         | `&mut [T]`          | 尾部开始，以 `pred(&T) == false` 分段             | 分段元素被忽略     |
| `splitn(n, pred)`           | `SplitN<'_, T, F>`            | `&[T]`              | 以 `pred(&T) == false` 分段、且最长为 `n`         | 分段元素被忽略     |
| `splitn_mut(n, pred)`       | `SplitNMut<'_, T, F>`         | `&mut [T]`          | 以 `pred(&T) == false` 分段、且最长为 `n`         | 分段元素被忽略     |
| `rsplitn(n, pred)`          | `RSplitN<'_, T, F>`           | `&[T]`              | 尾部开始以 `pred(&T) == false` 分段、且最长为 `n` | 分段元素被忽略     |
| `rsplitn_mut(n, pred)`      | `RSplitNMut<'_, T, F>`        | `&mut [T]`          | 尾部开始以 `pred(&T) == false` 分段、且最长为 `n` | 分段元素被忽略     |

####    `[MaybeUninit<T>]` 方法

| `[MaybeUninit<T>]` 方法     | 描述                      | 其他                      |
|-----------------------------|---------------------------|---------------------------|
| `write_copy_of_slice(src)`  | 从长度相同的 `&[T]` Copy  |                           |
| `write_clone_of_slice(src)` | 从长度相同的 `&[T]` Clone |                           |
| `assume_init_drop()`        | 在位丢弃值                | `unsafe` 需保证值已初始化 |
| `assume_init_ref()`         | 创建共享引用 `&[T]`       | `unsafe` 需保证值已初始化 |
| `assume_init_mut()`         | 创建可变引用 `&mut [T]`   | `unsafe` 需保证值已初始化 |

####    `[u8]` 方法

| `[u8]` 方法                   | 描述                      | 其他                                  |
|-------------------------------|---------------------------|---------------------------------------|
| `is_ascii()`                  | 是否在 *ASCII* 编码范围内 |                                       |
| `eq_ignore_ascii_case(other)` | 大小写不敏感相等          |                                       |
| `make_ascii_uppercase()`      | 转为大写                  |                                       |
| `make_ascii_lowercase()`      | 转为小写                  |                                       |
| `escape_ascii()`              | 迭代转义                  |                                       |
| `trim_ascii_start()`          | 移除开头空白              | 空白由 `u8::is_ascii_whitespace` 定义 |
| `trim_ascii_end()`            | 移除结尾空白              | 空白由 `u8::is_ascii_whitespace` 定义 |

### 裸指针

> - Primitive Type `pointer`：<https://doc.rust-lang.org/stable/std/primitive.pointer.html>
> - `std::ptr`：<https://doc.rust-lang.org/stable/std/ptr/index.html>

####    `*const T` 方法
| `*const T` 转换、操作     | 返回值                       | 描述                                      | 其他                                |
|---------------------------|------------------------------|-------------------------------------------|-------------------------------------|
| `is_null()`               | `bool`                       | 是否为空                                  |                                     |
| `cast<U>()`               | `*const U`                   | 转换为 `*const U` 类型                    |                                     |
| `cast_mut<U>()`           | `*mut U`                     | 转换为 `*mut U` 类型                      |                                     |
| `addr()`                  | `usize`                      | 获取地址，同时丢弃起源信息                |                                     |
| `expose_provenance()`     | `usize`                      | 获取地址，同时暴露起源信息                |                                     |
| `with_addr(addr)`         | `*const T`                   | 按 `usize`、自身起源信息创建              |                                     |
| `map_addr(f)`             | `*const T`                   | 按 `f(usize) -> usize` 变换自身地址创建   |                                     |
| `as_ref()`                | `Option<&'a T>`              | 尝试创建引用（值应已初始化）              | `unsafe`、指针为空时返回 `None`     |
| `as_ref_unchecked()`      | `&'a T`                      | 不检查空、是否初始化地创建引用            | *Exp*，`unsafe`                     |
| `as_uninit_ref<'a>(self)` | `Option<&'a MaybeUninit<T>>` | 尝试创建引用（值可未初始化）              | *Exp*，`unsafe`                     |
| `cast_uninit()`           | `*const MaybeUninit<T>`      | 转换为未初始化                            | *Exp*                               |
| `cast_init()`             | `*const T`                   | 转换为初始化                              | *Exp*，`*const MaybeUninit<T>` 类型 |
| `read()`                  | `T`                          | 读取但不移动                              | `unsafe`                            |
| `read_volatite()`         | `T`                          | 易变、可见地读取                          | `unsafe`                            |
| `read_unaligned()`        | `T`                          | 读取，指针可能未对齐                      | `unsafe`                            |
| `copy_to(dest, count)`    | `()`                         | 复制 `usize * size_of::<T>()` 至 `*mut T` | `unsafe`                            |

####    `*const T` 偏移

| `*const T` 偏移                     | 返回值     | 描述                                         | 其他     |
|-------------------------------------|------------|----------------------------------------------|----------|
| `offset(count)`                     | `*const T` | 偏移 `isize * size_of::<T>()`                | `unsafe` |
| `byte_offset(count)`                | `*const T` | 按字节偏移 `isize`                           | `unsafe` |
| `add(count)`                        | `*const T` | 前向偏移 `usize * size_of::<T>()`            | `unsafe` |
| `add_byte(count)`                   | `*const T` | 前向偏移 `usize` 字节                        | `unsafe` |
| `sub(count)`                        | `*const T` | 反向偏移 `usize * size_of::<T>()`            | `unsafe` |
| `sub_byte(count)`                   | `*const T` | 反向偏移 `usize` 字节                        | `unsafe` |
| `wrapping_offset(count)`            | `*const T` | 回环偏移 `isize * size_of::<T>()`            |          |
| `wrapping_byte_offset(count)`       | `*const T` | 回环按字节偏移 `isize`                       |          |
| `wrapping_add(count)`               | `*const T` | 回环前向偏移 `usize * size_of::<T>()`        |          |
| `wrapping_add_byte(count)`          | `*const T` | 回环前向偏移 `usize` 字节                    |          |
| `wrapping_sub(count)`               | `*const T` | 回环反向偏移 `usize * size_of::<T>()`        |          |
| `wrapping_sub_byte(count)`          | `*const T` | 回环反向偏移 `usize` 字节                    |          |
| `offset_from(origin)`               | `isize`    | 与同类型裸指针 `size_of::<T>()` 偏移         | `unsafe` |
| `offset_from_unsigned(origin)`      | `usize`    | 与较小地址同类型裸指针 `size_of::<T>()` 偏移 | `unsafe` |
| `byte_offset_from(origin)`          | `isize`    | 与其他裸指针字节偏移                         | `unsafe` |
| `byte_offset_from_unsigned(origin)` | `usize`    | 与较小地址其他裸指针字节偏移                 | `unsafe` |
| `align_offset(align)`               | `usize`    | 对齐所需移动字节                             |          |
| `is_aligned()`                      | `bool`     | 是否对齐                                     |          |

-   关于 `offset(count)`、`wrapping_offset(count)` 的说明
    -   `offset(count)` 不安全方法，要求用户确保地址不溢出、不越界
        -   原因：*LLVM* 要求地址合法，以便做优化
    -   `wrapping_offset(count)` 不检查地址合法性，是安全的方法
        -   `wrapping_offset` 会在地址偏移越界时回环，但是不保证、要求结果地址合法
            -   `wrappping_offset` 记住有裸指针起源，即裸指针最初所属内存分配、指向内存，以实现越界回环
        -   而，地址合法性要求被延迟到用户解引用时确保

> - What is the difference between `ptr::offset()` and `ptr::wrapping_offset()`：<https://users.rust-lang.org/t/what-is-the-difference-between-ptr-offset-and-ptr-wrapping-offset/68083>

####    `*const [T]` 方法

| `*const [T]` 方法    | 返回值                                  | 描述                       | 其他            |
|----------------------|-----------------------------------------|----------------------------|-----------------|
| `len()`              | `usize`                                 | 长度                       |                 |
| `is_empty()`         | `bool`                                  | 长度为 0                   |                 |
| `as_ptr()`           | `*const T`                              | 转换为指向切片缓冲的裸指针 | *Exp*           |
| `as_array<N>()`      | `Option<*const [T;N]>`                  | 转换为数组裸指针           | *Exp*           |
| `get_uncheck(index)` | `*const <I as SliceIndex<[T]>>::Output` | 索引元素、子切片           | *Exp*、`unsafe` |
| `as_uninit_slice()`  | `Option<&'a [MaybeUninit<T>]>`          | 转换为初始化切片           | `unsafe`        |

####    `*mut T`

| `*mut T` 方法             | 返回值              | 描述                                       | 其他                        |
|---------------------------|---------------------|--------------------------------------------|-----------------------------|
| `write_bytes(val, count)` | `()`                | 将 `count * size_of::<T>()` 字节置为 `val` | `unsafe`                    |
| `write(val)`              | `()`                | 覆盖写入 `T`，且原值不执行析构             | `unsafe`                    |
| `write_volatite(val)`     | `()`                | 易变、可见地覆盖写入，且不执行 `drop`      | `unsafe`                    |
| `write_unaligned(val)`    | `()`                | 覆盖写入，指针可能未对齐                   | `unsafe`                    |
| `replace(src)`            | `T`                 | 替换并返回旧值，且不执行 `drop`            | `unsafe`                    |
| `swap(with)`              | `()`                | 与 `*mut T` 指针交换内容，不执行反初始化   | `unsafe`                    |
| `as_mut()`                | `Option<&'a mut T>` | 转换为可变引用                             | `unsafe`，空指针返回 `None` |

| `*mut [T]` 方法     | 返回值               | 描述                       | 其他  |
|---------------------|----------------------|----------------------------|-------|
| `as_mut_array<N>()` | `Option<*mut [T;N]>` | 转换为数组裸指针           | *Exp* |
| `as_mut_ptr()`      | `*mut T`             | 转换为指向切片缓冲的裸指针 | *Exp* |

####    `ptr::NonNull<T>`

```rust
pub struct NonNull<T: PointeeSized> {
    pointer: *const T,
}
```

-   `ptr::NonNull<T>` 非空、协变 `*mut T`
    -   `NonNull<T>` 在内部封装 `*const T` 实现对 `T` 协变
        -   但，`*mut T` 其实没必要对 `T` 不变，毕竟解引用获取 `T` 总是 `unsafe` 的
    -   `NonNull<T>` 实现方法、特性基本同 `*mut T`
        -   类似的，`NonNull<[T]>` 方法基本同 `*mut [T]`
    -   `*const T`、`Option<NonNull<T>>`、`*mut T` 类型形参 **代表函数不同承诺**
        -   `*const T`：函数承诺不修改值、兼容 `Super<T>` 父类型实参
            -   *Rust* 中父子类型仅与生命周期相关，不修改值显然暗含兼容父类型实参
        -   `Option<NonNull<T>>`：函数可能修改值、但兼容 `Super<T>` 父类型实参
        -   `*mut T`：函数可能修改值、且不兼容 `Super<T>` 父类行实参

| `NonNull<T>` 方法             | 返回值               | 描述                       | 其他                    |
|-------------------------------|----------------------|----------------------------|-------------------------|
| `NonNull::new(ptr)`           | `Option<NonNull<T>>` | 从 `*mut T` 创建           | 空指针返回 `None`       |
| `NonNull::new_unchecked(ptr)` | `NonNull<T>`         | 从 `*mut T` 创建           | `unsafe` 需确保非空指针 |
| `NonNull::from_mut(r)`        | `NonNull<T>`         | 从 `&mut T` 转换           |                         |
| `NonNull::from_ref(r)`        | `NonNull<T>`         | 从 `&T` 转换               |                         |
| `as_ref()`                    | `&'a T`              | 创建共享引用               | `unsafe`                |
| `as_mut()`                    | `&'a mut T`          | 创建可变引用               | `unsafe`                |
| `as_uninit_ref()`             | `&'a T`              | 创建共享引用，值可未初始化 | *Exp*，`unsafe`         |
| `as_uninit_mut()`             | `&'a mut T`          | 创建可变引用，值可未初始化 | *Exp*，`unsafe`         |
| `as_ptr()`                    | `*mut T`             | 获取底层 `*mut T`          |                         |

> - `std::ptr::NonNull`：<https://doc.rust-lang.org/stable/std/ptr/struct.NonNull.html>
> - Variance and `NonNull` in Rust：<https://zhuanlan.zhihu.com/p/42756635>

####    `std::ptr` 函数

| `std::ptr` 指针操作                      | 返回值       | 描述                                    | 其他                                  |
|------------------------------------------|--------------|-----------------------------------------|---------------------------------------|
| `addr_eq<T, U>(p, q)`                    | `bool`       | 裸指针地址相等                          | 忽略指针元信息，对同类型瘦指针同 `eq` |
| `eq<T>(a, b)`                            | `bool`       | 同类裸指针相等                          |                                       |
| `fn_addr_eq<T, U>(f, g)`                 | `bool`       | 函数指针地址相等                        |                                       |
| `from_mut<T>(r)`                         | `*mut T`     | 将可变引用转换为可变裸指针              |                                       |
| `from_ref<T>(r)`                         | `*const T`   | 将共享引用转换位共享裸指针              |                                       |
| `dangling<T>()`                          | `*const T`   | 创建悬垂、非空共享裸指针                |                                       |
| `dangling_mut<T>()`                      | `*mut T`     | 创建悬垂、非空可变裸指针                |                                       |
| `null<T>()`                              | `*const T`   | 创建空共享裸指针                        |                                       |
| `null_mut<T>()`                          | `*mut T`     | 创建空共享裸指针                        |                                       |
| `slice_from_raw_parts<T>(data, len)`     | `*const [T]` | 从裸指针创建切片裸指针                  | 创建安全但使用不安全                  |
| `slice_from_raw_parts_mut<T>(data, len)` | `*mut [T]`   | 从裸指针创建切片裸指针                  | 创建安全但使用不安全                  |
| `with_exposed_provenance<T>(addr)`       | `*const T`   | 将 `usize` 地址结合溯源信息转换为裸指针 |                                       |
| `with_exposed_provenance_mut<T>(addr)`   | `*mut T`     | 将 `usize` 地址结合溯源信息转换为裸指针 |                                       |
| `without_provenance<T>(addr)`            | `*const T`   | 将 `usize` 地址转换为裸指针             | 等价于 `ptr::null().with_addr()`      |
| `without_provenance_mut<T>(addr)`        | `*mut T`     | 将 `usize` 地址转换为裸指针             |                                       |

| `std::ptr` 指针指向值操作                 | 返回值 | 描述                               | 其他                                          |
|-------------------------------------------|--------|------------------------------------|-----------------------------------------------|
| `read<T>(src)`                            | `T`    | 读取、原值不受影响                 | `unsafe` 需确保指针已初始化、已对齐、适合读取 |
| `read_unaligned<T>(src)`                  | `T`    | 读取、原值不受影响                 | `unsafe` 需确保指针已初始化、适合读取         |
| `read_volatile<T>(src)`                   | `T`    | 易变读取                           | `unsafe` 需确保指针已初始化、已对齐、适合读取 |
| `write<T>(dest, src)`                     | `()`   | 覆盖写入、不析构原值               | `unsafe` 需确保指针已对齐、适合写入           |
| `write_unaligned<T>(dest, src)`           | `()`   | 覆盖写入、不析构原值               | `unsafe` 需确保指针适合写入                   |
| `write_bytes<T>(dst, val, count)`         | `()`   | 覆盖写入 `count` 个 `val`          | `unsafe` 需确保指针已对齐、适合写入           |
| `write_volatile<T>(dst, src)`             | `()`   | 易变写入、不知析构原值             | `unsafe`                                      |
| `replace<T>(dest, src)`                   | `T`    | 替换裸指针指向值、返回原值         | `unsafe` 需确保指针已初始化、已对齐、适合读写 |
| `swap<T>(x, y)`                           | `()`   | 交换裸指针指向值                   | `unsafe` 需确保指针已初始化、已对齐、适合读写 |
| `swap_nooverlapping<T>(x, y)`             | `()`   | 交换裸指针指向值、值不可重叠       | `unsafe` 需确保指针已初始化、已对齐、适合读写 |
| `copy<T>(src, dst, count)`                | `bool` | 复制 `size_of<T>() * count` 字节数 | `unsafe` 需确保读、写、对齐有效               |
| `copy_nonoverlapping<T>(src, dst, count)` | `bool` | 复制，但不允许有重叠               | `unsafe` 需确保读、写、对齐有效               |
| `drop_in_place<T>(to_drop)`               | `()`   | 析构指针指向的值                   | `unsafe` 需确保值有效、对齐                   |
| `hash<T, S>(hashee, into)`                | `()`   | 将裸指针写入 `impl Hasher`         |                                               |

> - `std::ptr` 函数：<https://doc.rust-lang.org/stable/std/ptr/index.html#functions>


### 内存管理

####    `mem::ManuallyDrop<T>`

| `ManuallyDrop<T>` 方法           | 返回值            | 描述                       | 其他                            |
|----------------------------------|-------------------|----------------------------|---------------------------------|
| `ManuallyDrop::new(value)`       | `MannallyDrop<T>` | 封装 `T` 值                |                                 |
| `ManuallyDrop::into_inner(slot)` | `T`               | 解包获取内部值             |                                 |
| `ManuallyDrop::take(slot)`       | `T`               | 从 `&mut T` 解包获取内部值 | `unsafe` 需确保 `self` 不再使用 |
| `ManuallyDrop::drop(slot)`       | `()`              | 通过 `&mut T` 丢弃内部值   | `unsafe` 需确保 `self` 不再使用 |

-   `ManuallyDrop<T>` 禁止编译器自动析构的、零成本包装器
    -   `ManuallyDrop<T>` 实现有 `DerefMut`，故其方法均为关联函数、内部值方法可直接在其上应用

> - `std::mem::ManuallyDrop`：<https://doc.rust-lang.org/stable/std/mem/struct.ManuallyDrop.html>
> - How `std::mem::ManuallyDrop` work?：<https://users.rust-lang.org/t/how-std-manuallydrop-work/69939>

####    `mem::MaybeUninit<T>`

| `MaybeUninit<T>` 方法                      | 返回值                   | 描述                         | 其他                                      |
|--------------------------------------------|--------------------------|------------------------------|-------------------------------------------|
| `MaybeUninit::new(val)`                    | `MaybeUninit<T>`         | 按指定值创建                 |                                           |
| `MaybeUninit::uninit()`                    | `MaybeUninit<T>`         | 按未初始化状态创建           |                                           |
| `MabyeUninit::zeroed()`                    | `MaybeUninit<T>`         | 内存填 0 创建                |                                           |
| `write(val)`                               | `&mut T`                 | 写入值、原值不执行析构       | 同 `*mut T` 裸指针 `write` 方法           |
| `as_ptr()`                                 | `*const T`               | 获取内部值常量裸指针         |                                           |
| `as_mut_ptr()`                             | `*const T`               | 获取内部值可变裸指针         |                                           |
| `assume_init()`                            | `T`                      | 解包获取内部值               | `unsafe` 需确保内部值已初始化             |
| `assume_init_read()`                       | `T`                      | 读取内部值创建副本           | `unsafe` 需确保内部值已初始化、可创建副本 |
| `assume_init_drop()`                       | `()`                     | 在位析构内部值               | `unsafe` 需确保内部值已初始化             |
| `assume_init_ref()`                        | `&T`                     | 获取内部值共享引用           | `unsafe` 需确保内部值已初始化             |
| `assume_init_mut()`                        | `&mut T`                 | 获取内部值可变引用           | `unsafe` 需确保内部值已初始化             |
| `MaybeUninit::array_assume_init<N>(array)` | `[T;N]`                  | 解包转换为数组               | *Exp*，`unsafe` 需确保数组元素均已初始化  |
| `as_bytes()`                               | `&[MaybeUninit<u8>]`     | 创建未初始化字节切片共享引用 | *Exp*                                     |
| `as_bytes_mut()`                           | `&mut [MaybeUninit<u8>]` | 创建未初始化字节切片可变引用 | *Exp*                                     |

-   `mem::MaybeUninit<T>` 创建、处理未初始化值的包装器
    -   编译器将不对 `MaybeUninit<T>` 实例做适用于已初始化 `T` 值的假设、优化
    -   `MaybeUninit<T>` 主要通过 `write(val)` 方法写入数据、再 `assume_`、`mem::transmute` 得到目标类型
        -   存储函数结果的外层指针：将函数将结果直接写入传入 `MaybeUninit<T>` 实参
        -   逐元素初始化大数组
        -   结构体类型逐字段初始化
    -   `MaybeUninit<T>` 保证大小、对齐、*ABI* 与 `T` 一致

| `MaybeUninit<[T;N]>` 方法 | 返回值               | 描述         | 其他  |
|---------------------------|----------------------|--------------|-------|
| `transpose()`             | `[MaybeUninit<T>;N]` | 未初始化内置 | *Exp* |

| `Box<MaybeUninit<T>, A>` 方法 | 返回值      | 描述                   | 其他                          |
|-------------------------------|-------------|------------------------|-------------------------------|
| `assume_init()`               | `Box<T, A>` | 解包获取内部值         | `unsafe` 需确保内部值已初始化 |
| `write(boxed)`                | `Box<T, A>` | 设置值、原值不执行析构 |                               |

> - `std::mem::MaybeUninit`：<https://doc.rust-lang.org/stable/std/mem/union.MaybeUninit.html>

####    `alloc::Layout`

| `Layout` 方法                                    | 返回值                                 | 描述                           | 其他                        |
|--------------------------------------------------|----------------------------------------|--------------------------------|-----------------------------|
| `Layout::from_size_align(size, align)`           | `Result<Layout, LayoutError>`          | 按大小、对齐创建               | 不满足布局要求返回 `Err`    |
| `Layout::from_size_align_unchecked(size, align)` | `Result<Layout, LayoutError>`          | 按大小、对齐创建               | `unsafe` 需确保满足布局要求 |
| `Layout::new<T>(t)`                              | `Layout`                               | 创建满足类型要求布局           |                             |
| `Layout::array<T>(n)`                            | `Result<Layout, LayoutError>`          | 创建满足 `[T; n]` 要求布局     |                             |
| `size()`                                         | `usize`                                | 布局大小所需的内存块最小字节数 |                             |
| `align()`                                        | `usize`                                | 布局对齐所需的内存块最小字节数 |                             |
| `align_to(align)`                                | `Result<(Layout, usize), LayoutError>` | 按对齐调整布局                 |                             |
| `pad_to_align()`                                 | `Layout`                               | 按调整存储大小至对齐整数倍     |                             |
| `extend(next)`                                   | `Result<Layout, usize>`                | 扩展布局                       | 返回新布局、后个布局偏移    |

-   `alloc::Layout` 描述内存块布局
    -   布局通过两字段描述
        -   `align` 对齐：寻址最小单位（即地址起始位置必须为 `align` 整数倍），必须为 2 的幂次
        -   `size` 存储大小：存储全部数据所需字节数
            -   因为，类型值在内存中起始地址必须为 `align` 的整数倍
            -   故，单个类型值在内存真正所需字节数需在真实 `size` 基础上补齐 `pad`

> - `struct::alloc::Layout`：<https://doc.rust-lang.org/stable/std/alloc/struct.Layout.html>
> - 浅谈 Rust 程序内存布局：<https://rustcc.cn/article?id=98adb067-30c8-4ce9-a4df-bfa5b6122c2e>

####    `std::mem` 函数

| `std::mem` 函数                  | 返回值            | 描述                                | 其他                       |
|----------------------------------|-------------------|-------------------------------------|----------------------------|
| `align_of<T>()`                  | `usize`           | *ABI* 所需的最小对齐                | 可能比推荐对齐小           |
| `align_of_val<T>(val)`           | `usize`           | *ABI* 所需的最小对齐                |                            |
| `size_of<T>()`                   | `usize`           | 类型大小                            |                            |
| `size_of_val<T>(val)`            | `usize`           | 类型大小                            | 支持 `!Sized` 类型         |
| `discriminant<T>(v)`             | `Discriminant<T>` | 枚举变体的唯一标识                  |                            |
| `drop<T>(_x)`                    | `()`              | 销毁                                |                            |
| `copy<T>(x)`                     | `T`               | 按位复制                            | *Exp*                      |
| `forget<T>(t)`                   | `()`              | 置位手动析构                        |                            |
| `forget_usized<T>(t)`            | `()`              | 直接 `!Sized` 的手动析构            | *Exp*                      |
| `needs_drop<T>()`                | `bool`            | 是否有必要销毁                      | 仅是优化提示               |
| `replace(dest, src)`             | `T`               | 替换、返回原值                      |                            |
| `swap(x, y)`                     | `()`              | 交换两值                            |                            |
| `take(dest)`                     | `T`               | 替换为 `Default::default`、返回原值 | `T: Default`               |
| `transmute<Src, Dst>(src)`       | `Dst`             | 位重解释                            | `unsafe` 需确保重解释合法  |
| `transmute_copy<Src, Dest>(src)` | `Dst`             | 位重解释后复制                      | `unsafe` 需确保重解释合法  |
| `zerod<T>()`                     | `T`               | 位全置 0                            | `unsafe` 需确保全 0 值合法 |

> - `std::mem` 函数：<https://doc.rust-lang.org/stable/std/mem/index.html#functions>

### `Option`、`Result`

| 实现特性       | `Option` | `Result` |
|----------------|----------|----------|
| `Clone`        | *Yes*    | *Yes*    |
| `CloneFromCell`| *Yes*    | *Yes*    |
| `Copy`         | *Yes*    | *Yes*    |
| `Debug`        | *Yes*    | *Yes*    |
| `Default`      | *Yes*    | *Yes*    |
| `Eq`           | *Yes*    | *Yes*    |
| `PartialEq`    | *Yes*    | *Yes*    |
| `Ord`          | *Yes*    | *Yes*    |
| `PartialOrd`   | *Yes*    | *Yes*    |
| `Sum`          | *Yes*    | *Yes*    |
| `Product`      | *Yes*    | *Yes*    |
| `Hash`         | *Yes*    | *Yes*    |
| `From`         | *Yes*    | *Yes*    |
| `FromIterator` | *Yes*    | *Yes*    |
| `IntoIterator` | *Yes*    | *Yes*    |

> - `std::option`：<https://doc.rust-lang.org/stable/std/option/index.html>
> - `std::option::Option`：<https://doc.rust-lang.org/stable/std/option/enum.Option.html>
> - `std::result`：<https://doc.rust-lang.org/stable/std/result/index.html>
> - `std::result::Result`：<https://doc.rust-lang.org/stable/std/result/enum.Result.html>

####    `is_` 判断方法

| `Option<T>`      | `Result<T,E>`   | 描述                                 |
|------------------|-----------------|--------------------------------------|
| `is_some()`      | `is_ok()`       | 是否为 `Some<T>`、`Ok<T>`            |
| `is_none()`      | `is_err()`      | 是否为 `None`、`Err<E>`              |
| `is_some_and(f)` | `is_ok_and(f)`  | `Some<T>` 且 `f(T)`                  |
| `is_none_or(f)`  | `is_err_and(f)` | `None` 或 `f(T)`、`Err<E>` 且 `f(E)` |

####    `&`、`&mut` 转换方法

| `&self` 方法     | `Option<T>`                                    | `Result<T,E>`                                       |
|------------------|------------------------------------------------|-----------------------------------------------------|
| `as_ref()`       | `&Option<T>` 转 `Option<&T>`                   | `&Result<T, E>` 转 `Result<&T, &E>`                 |
| `as_mut()`       | `&mut Option<T>` 转 `Option<&mut T>`           | `&mut Result<T, E>` 转 `Result<&mut T, &mut E>`     |
| `as_deref()`     | `&Option<T>` 转 `Option<&T::Traget>`           | `&Result<T, E>` 转 `Result<&T::Target, &E>`         |
| `as_deref_mut()` | `&mut Option<T>` 转 `Option<&mut T::Target>`   | `&Result<T, E>` 转 `Result<&mut T::Target, &mut E>` |
| `as_pin_ref()`   | `Pin<&Option<T>>` 转 `Option<Pin<&T>>`         |                                                     |
| `as_pin_mut()`   | `Pin<&mut Option<T>>` 转 `Option<Pin<&mut T>>` |                                                     |
| `as_slice()`     | `&Option<T>` 转 `&[T]`                         |                                                     |
| `as_mut_slice()` | `&mut Option<T>` 转 `&mut [T]`                 |                                                     |

####    解包取值方法

| 正常解包取值方法          | `Option::None`、`Result::Err<E>` |
|---------------------------|----------------------------------|
| `expect(error_msg)`       | `panic` 并打印自定义错误信息     |
| `unwrap()`                | `panic`                          |
| `unwrap_or_default()`     | 当前类型默认值                   |
| `unwrap_or(default)`      | 自定义默认值                     |
| `unwrap_or_else(closure)` | 闭包（无参）返回值               |
| `unwrap_or_unchecked()`   | 产生 *UB*                        |

| `Err<E>` 解包取错误方法 | `Ok<T>`                  |
|-------------------------|--------------------------|
| `expect_err(error_msg)` | `panic` 并打印自定义信息 |
| `unwrap_err()`          | `panic`                  |
| `unwrap_err_default()`  | 产生 *UB*                |

-   解包取值说明
    -   `?` 算符：解包 `Some<T>`、`Ok<T>`、或直接返回 `None`、`Err<E>`

####    `Option`、`Result` 互转

| `Option`、`Result` 互转 | 描述                                 |
|-------------------------|--------------------------------------|
| `Option::ok_or(err)`    | `Option<T>` 转 `Result<T, err>`      |
| `Option::ok_or_else(f)` | `Option<T>` 转 `Result<T, f()>`      |
| `Result::err()`         | `Result<T, E>` 转 `Option<E>`        |
| `Result::ok()`          | `Result<T, E>` 转 `Option<T>`        |
| `Option::transpose()`   | `Option<Result>` 转 `Result<Option>` |
| `Result::transpose()`   | `Result<Option>` 转 `Option<Result>` |

####    `Result` 转换

| `Result` 方法            | `Ok(t)` 转          | `Err(e)` 转         |
|--------------------------|---------------------|---------------------|
| `map(f)`                 | `Ok(f(t))`          | 不变                |
| `inspect(f)`             | 不变，`f(t)` 仅检查 | 不变                |
| `map_err(f)`             | 不变                | `f(e) -> F`         |
| `inspect_err(f)`         | 不变                | 不变，`f(e)` 仅检查 |
| `map_or(f, defe)`        | `f(t)`              | `defe`              |
| `map_or_else(fok, ferr)` | `fok(t)`            | `ferr(e)`           |

####    `Option` 转换

| `Option` 方法               | `Some(t)` 转                            | `None` 转 |
|-----------------------------|-----------------------------------------|-----------|
| `filter(f)`                 | 若 `f(t) == false`，`Some(t)` 转 `None` | `None`    |
| `flatten()`                 | `Some<Option>` 转 `Option`              | `None`    |
| `inspect(f)`                | 不变，仅 `f(t)` 检查                    | 不变      |
| `map(f)`                    | `Some(f(t))`                            | `None`    |
| `map_or(f, def)`            | `f(t)`                                  | `def`     |
| `map_or_else(fsome, fnone)` | `fsome(t)`                              | `fnone()` |
| `zip(other)`                | `Some(t, o)`                            | `None`    |
| `zip_with(other, f)`        | `Some(f(s,o))`                          | `None`    |

####    逻辑运算方法

| 逻辑运算方法  | `Option<T>`                        | `Result<T, E>`                        |
|---------------|------------------------------------|---------------------------------------|
| `and(rhs)`    | `rhs: Option<U>` 可为不同类型      | `rhs: Result<U, E>` 可为不同类型      |
| `or(rhs)`     | `rhs: Option<T>` 须为相同类型      | `rhs: Result<T, F>` 可为不同类型      |
| `xor(rhs)`    | `rhs: Option<T>` 须为相同类型      | 不支持                                |
| `and_then(f)` | `f(T) -> Option<U>` 可返回不同类型 | `f(T) -> Result<U, E>` 可返回不同类型 |
| `or_else(f)`  | `f(T) -> Option<T>` 须返回相同类型 | `f(T) -> Result<T, F>` 可返回不同类型 |

-   逻辑运算方法说明
    -   逻辑运算方法的基本特点
        -   `Some<T>`、`Ok<T>` 被视为真值
        -   逻辑运算均为短路求值
            -   返回最后运算结果（除 `xor` 运算，优先返回 `Some<T>`）
            -   作为参数的闭包可能不执行运算
        -   故，部分方法的参数可为不同类型

### `box::Box`

| `Box` 方法                   | 描述                                     | 其他                                     |
|------------------------------|------------------------------------------|------------------------------------------|
| `downcast()`                 | 从 `Box<dyn>` 降级为具体类型             |                                          |
| `downcast_unchecked()`       | 从 `Box<dyn>` 降级为具体类型             | *Exp*、`unsafe` 需用户保证类型一致       |
| `Box::new(x)`                | 创建 `Box<T>`                            |                                          |
| `Box::try_new(x)`            | 尝试创建，返回 `Result<Box, AllocError>` | *Exp*                                    |
| `Box::new_uninit()`          | 创建未初始化 `Box<MaybeUninit<T>>`       |                                          |
| `Box::new_zeroed()`          | 创建 0 初始化 `Box<MaybeUninit<T>>`      |                                          |
| `Box::new_uninit_slice(len)` | 创建未初始化 `Box<[MaybeUninit<T>]>`     |                                          |
| `Box::new_zeroed_slice(len)` | 创建 0 初始化 `Box<[MaybeUninit<T>]>`    |                                          |
| `Box::write(boxed, value)`   | 向 `MaybeUninit<T>` 写值、并转为 `T`     | 避免与 `MaybeUninit::write` 方法重复     |
| `assume_init()`              | `Box` 中 `MaybeUninit<T>` 视为、转为 `T` | `unsafe` 需保证已初始化                  |
| `Box::from_raw(raw)`         | 从 `*mut T` 创建 `Box<T>`                | `unsafe` 需保证内存布局与 `Box` 要求一致 |
| `Box::into_raw(b)`           | 消耗 `Box`、返回 `*mut T`                | 消费 `Box` 但不丢弃值                    |
| `Box::leak<'a>(b)`           | 消耗 `Box`、返回 `&'a mut T`             | 后续由用户管理内存                       |
| `Box::pin(x)`                | 创建 `Pin<Box<T>>`                       |                                          |
| `Box::into_pin(boxed)`       | 转为 `Pin<Box<T>>`                       |                                          |

-   `Box`
    -   为避免与内部类型方法冲突，大部分函数均为实现为关联函数而不是方法
        -   但注意，`Box` 实现大量特性，特性中包含大量方法
    -   `Box::leak` 消耗 `Box` 并返回 `&'a mut T` 可变引用
        -   此时，需由用户自行负责内存管理、手动丢弃值
        -   常用于获取 `&'static mut T` 在程序运行期间都需存在的数据

```rust
// ************************** Box::write
let big_box = Box::<[usize; 1024]>::new_init();
big_box[0].write(1);                                        // 自动解引用后调用 `MaybeUninit::write` 方法，不会初始化
let mut arr = [0; 1024];
big_box = Box::write(big_box, arr);                         // `Box::write` 关联函数整体，内部将初始化
```

> - `std::box`：<https://doc.rust-lang.org/stable/std/boxed/index.html>
> - `std::box::Box`：<https://doc.rust-lang.org/stable/std/boxed/struct.Box.html>
> - Rust - `Box::leak`：<https://juejin.cn/post/7490967893779464207>

### `Rc` 引用计数

-   `rc::Rc<T>` 单线程引用计数：提供堆上值的共享所有权
    -   堆上值在所有指向其的 `Rc<T>` 指针销毁后被丢弃
    -   类似共享引用，可能无法获取 `Rc<T>` 内部值的可变引用
        -   `Rc<T>::get_mut()` 在存在其他引用计数时返回 `None`
        -   `Rc<T>::make_mut()` 在存在其他引用计数时 Clone、并返回副本可变引用
    -   `Rc<T>` 实现的一些特性说明
        -   `!Send`：多个线程可能竞争 `clone`、修改引用计数
        -   `Clone`：`Rc::clone()` 创建新的指向堆上相同位置（值）的 `Rc<T>` 指针
        -   `Deref`：`T` 方法可直接在 `Rc<T>` 上调用（故，`Rc<T>` 大部分方法都是关联函数）
-   `rc::Weak<T>` 单线程弱引用计数：无所有权的引用计数
    -   `Weak<T>` 不拥有的值的所有权、不影响值丢弃，也不保证值有效
        -   `Weak<T>` 主要用于避免 `Rc<T>` 循环引用导致的值无法被丢弃
    -   `Rc<T>`、`Weak<T>` 可通过 `Weak::upgrade()`、`Rc::downgrade()` 相互转换
        -   `Weak::upgrade()` 可能失败，在值被丢弃时将返回 `None`

> - `std::rc`：<https://doc.rust-lang.org/stable/std/rc/index.html>
> - `std::rc::Rc`：<https://doc.rust-lang.org/stable/std/rc/struct.Rc.html>
> - `std::rc::Weak`：<https://doc.rust-lang.org/stable/std/rc/struct.Weak.html>

####    `Rc` 创建、转换

| `Rc` 创建                         | 返回值                   | 描述                                                          | 其他                                    |
|-----------------------------------|--------------------------|---------------------------------------------------------------|-----------------------------------------|
| `downcast()`                      | `Rc<T>`                  | 从 `Box<dyn>` 降级为具体类型                                  |                                         |
| `downcast_unchecked()`            | `Rc<T>`                  | 从 `Box<dyn>` 降级为具体类型                                  | *Exp*、`unsafe` 需用户保证类型一致      |
| `Rc::new(value)`                  | `Rc<T>`                  | 创建 `Rc<T>`                                                  |                                         |
| `Rc::new_cyclic(data_fn)`         | `Rc<T>`                  | 通过 `date_fn(&Weak<T>) -> T` 创建包含拥有自身 `Weak` 的 `Rc` |                                         |
| `Rc::try_new(value)`              | `Result<Rc, AllocError>` | 尝试创建，返回 `Result<Rc, AllocError>`                       | *Exp*                                   |
| `Rc::new_uninit()`                | `Rc<MaybeUninit<T>>`     | 创建未初始化 `Rc<MaybeUninit<T>>`                             |                                         |
| `Rc::new_zeroed()`                | `Rc<MaybeUninit<T>>`     | 创建 0 初始化 `Rc<MaybeUninit<T>>`                            |                                         |
| `Rc::new_uninit_slice(len)`       | `Rc<MaybeUninit<T>>`     | 创建未初始化 `Rc<[MaybeUninit<T>]>`                           |                                         |
| `Rc::new_zeroed_slice(len)`       | `Rc<MaybeUninit<T>>`     | 创建 0 初始化 `Rc<[MaybeUninit<T>]>`                          |                                         |
| `assume_init()`                   | `Rc<T>`                  | `Rc` 中 `MaybeUninit<T>` 视为、转为 `T`                       | `unsafe` 需保证已初始化                 |
| `Rc::from_raw(ptr)`               | `Rc<T>`                  | 从 `*const T` 创建 `Rc<T>`                                    | `unsafe` 需保证内存布局与 `Rc` 要求一致 |

####    `Rc` 元信息

| `Rc` 元信息                       | 返回值  | 描述                                | 其他                                |
|-----------------------------------|---------|-------------------------------------|-------------------------------------|
| `Rc::increment_strong_count(ptr)` | `()`    | 将 `ptr` 指向的 `Rc` 强引用计数加 1 | `unsafe` 需保证 `ptr` 指向有效 `Rc` |
| `Rc::decrement_strong_count(ptr)` | `()`    | 将 `ptr` 指向的 `Rc` 强引用计数减 1 | `unsafe` 需保证 `ptr` 指向有效 `Rc` |
| `Rc::weak_count(this)`            | `usize` | 获取弱引用计数                      |                                     |
| `Rc::strong_count(this)`          | `usize` | 获取强引用计数                      |                                     |

####    `Rc` 转换

| `Rc` 转换                   | 返回值           | 描述                                                         | 其他                 |
|-----------------------------|------------------|--------------------------------------------------------------|----------------------|
| `Rc::downgrade(this)`       | `Weak<T>`        | 从 `Rc<T>` 创建 `Weak<T>`                                    |                      |
| `Rc::into_raw(b)`           | `*const T`       | 转换为 `*const T`                                            | 消费 `Rc` 但不丢弃值 |
| `Rc::into_inner(this)`      | `Option<T>`      | 仅有单个强引用时获取内部值，返回 `Option<T>`                 |                      |
| `Rc::as_ptr(this)`          | `*cosnt T`       | 创建指向 `Rc` 中数据的 `*const T`                            |                      |
| `Rc::ptr_eq(this, other)`   | `bool`           | 比较两 `Rc` 是否指向相同数据                                 |                      |
| `Rc::pin(value)`            | `Pin<Rc<T>>`     | 创建 `Pin<Rc<T>>`                                            |                      |
| `Rc::get_mut(this)`         | `Option<&mut T>` | 创建 `Option<&mut T>`，有其他 `Rc`、`Weak` 时返回 `None`     |                      |
| `Rc::make_mut(this)`        | `&mut T`         | 创建 `&mut T`，有其他 `Rc` 时 `clone` 值、仅有 `Weak` 时断联 |                      |
| `Rc::unwrap_or_clone(this)` | `T`              | 获取或克隆 `T`，有其他 `Rc` 时 `clone` 值、否则直接获取      |                      |

####    `Weak` 方法

| `Weak` 方法                 | 返回值     | 描述                           | 其他                                      |
|-----------------------------|------------|--------------------------------|-------------------------------------------|
| `Weak::new(value)`          | `Weak<T>`  | 创建 `Weak<T>`                 |                                           |
| `Weak::upgrade(this)`       | `Rc<T>`    | 从 `Weak<T>` 创建 `Rc<T>`      |                                           |
| `Weak::from_raw(ptr)`       | `Weak<T>`  | 从 `*const T` 创建 `Weak<T>`   | `unsafe` 需保证内存布局与 `Weak` 要求一致 |
| `Weak::into_raw(b)`         | `*const T` | 转换为指向数据的 `*const T`    | 弱引用计数不变                            |
| `Weak::as_ptr(this)`        | `*const T` | 创建指向数据的 `*const T`      |                                           |
| `Weak::ptr_eq(this, other)` | `bool`     | 比较两 `Weak` 是否指向相同数据 |                                           |
| `Weak::weak_count(this)`    | `usize`    | 获取弱引用计数                 |                                           |
| `Weak::strong_count(this)`  | `usize`    | 获取强引用计数                 |                                           |

> - `std::rc::Weak`：<https://doc.rust-lang.org/stable/std/rc/struct.Weak.html>

####    `sync::Arc`、`sync::Weak`

-   `sync::Arc<T>` 原子化、线程安全版引用计数
    -   `Arc<T>` 的行为、方法同 `rc::Rc<T>`
    -   但，`Arc<T>` 内部引用计数使用原子操作，线程安全但开销更大
        -   故，`Arc<T>` 是 `Send`、`Sync`（在 `T` 是 `Send`、`Sync` 时）
    -   同样的，`sync::Weak<T>` 是 `Send`、`Sync` 版本的 `rc::Weak<T>`

> - `std::sync::Arc`：<https://doc.rust-lang.org/stable/std/sync/struct.Arc.html>
> - `std::sync::Weak`：<https://doc.rust-lang.org/stable/std/sync/struct.Weak.html>

### `vec::Vec`、`collections::VecDeque`

-   关于 `Vec<T>`、`VecDeque<T>` 方法的说明
    -   `Vec<T>` 实现有 `DerefMut<Target=[T]>`，故 `[T]` 切片方法均可直接在 `Vec` 上直接调用
    -   `VecDeque<T>` 是双头向量，操作基本同 `Vec`，但
        -   `VecDeque` 部分方法拆分为 `_front`、`_back` 两类
        -   `VecDeque` 循环利用连续内存块，`as_slices()` 等方法返回两个切片
        -   `VecDeque` 未实现 `Deref`，无法解引用强制类型转换接受数组方法

> - `std::vec`：<https://doc.rust-lang.org/stable/std/vec/index.html>
> - `std::vec::Vec`：<https://doc.rust-lang.org/stable/std/vec/struct.Vec.html>
> - `std::collections::VecDeque`：<https://doc.rust-lang.org/stable/std/collections/struct.VecDeque.html>

####    `Vec` 容量长度

| `Vec` 容量、长度                            | 返回值                             | 描述                                                    | 其他                            |
|---------------------------------------------|------------------------------------|---------------------------------------------------------|---------------------------------|
| `Vec::new()`                                | `Vec<T>`                           | 创建空 `Vec<T>`                                         |                                 |
| `Vec::with_capacity(capcity)`               | `Vec<T>`                           | 创建容量至少为 `capcity` 的空 `Vec<T>`                  |                                 |
| `Vec::from_raw_parts(ptr, length, capcity)` | `Vec<T>`                           | 从 `*mut T` 创建 `Vec<T>`                               | `unsafe` 需保证指针地址符合要求 |
| `into_raw_parts()`                          | `(*mut T, usize, usize)`           | 将 `Vec<T>` 拆解为 `*mut T`、长度、容量                 |                                 |
| `reserve(additional)`                       | `()`                               | 保留至少 `additional` 容量                              | 重分配可能分配更多内存          |
| `reserve_exact(additional)`                 | `()`                               | 保留至少 `additional` 容量                              | 重分配则精确分配内存            |
| `try_reserve(additional)`                   | `Result<(), TryReserveError>`      | 尝试预分配容量，返回 `Result<Vec, TryReserveError>`     |                                 |
| `try_reserve_exact(additional)`             | `result<(), TryReserveExactError>` | 精准尝试预分配容量，返回 `Result<Vec, TryReserveError>` |                                 |
| `shrink_to_fit()`                           | `()`                               | 尽可能缩小容量                                          |                                 |
| `shrink_to(min_capcity)`                    | `()`                               | 缩小容量、但不小于阈值                                  |                                 |
| `capacity()`                                | `usize`                            | 返回容量                                                |                                 |
| `len()`                                     | `usize`                            | 获取长度                                                |                                 |
| `is_empty()`                                | `bool`                             | 是否为空                                                |                                 |

####    `Vec` 转换

| `Vec` 创建、转换     | 返回值        | 描述                            | 其他               |
|----------------------|---------------|---------------------------------|--------------------|
| `into_boxed_slice()` | `Box<T>`      | 转换为 `Box<[T]>`               |                    |
| `leak<'a>()`         | `&'a mut [T]` | 转换为可变切片 `&'a mut [T]`    | 后续由用户管理内存 |
| `as_slice()`         | `&[T]`        | 创建切片 `&[T]`                 |                    |
| `as_mut_slice()`     | `&mut [T]`    | 创建可变切片 `&mut [T]`         |                    |
| `as_ptr()`           | `*const T`    | 创建指向缓冲的 `*const T`       |                    |
| `as_mut_ptr()`       | `*mut T`      | 创建指向缓冲的 `*mut T`         |                    |
| `into_flattened()`   | `Vec<T>`      | 将 `Vec<[T;N]>` 展平为 `Vec<T>` |                    |

####    `Vec` 插入、移除

| `Vec` 插入、移除           | 返回值      | 描述                                  | 其他                                      |
|----------------------------|-------------|---------------------------------------|-------------------------------------------|
| `insert(index, element)`   | `()`        | 插入元素                              |                                           |
| `push(value)`              | `()`        | 末尾插入                              |                                           |
| `pop()`                    | `T`         | 末尾移除                              |                                           |
| `pop_if(predicate)`        | `Option<T>` | 尝试末尾移除                          |                                           |
| `append(other)`            | `()`        | 将 `&mut Vec` 中元素移入              |                                           |
| `extend_from_slice(other)` | `()`        | 将 `&[T]` 中元素克隆、移入            |                                           |
| `extend_from_within(src)`  | `()`        | 将自身 `src` 范围中元素克隆、移入     |                                           |
| `clear()`                  | `()`        | 清空                                  |                                           |
| `remove(index)`            | `()`        | 移除指定位置元素                      |                                           |
| `swap_remove(index)`       | `T`         | 移除并返回元素，用最后元素替代        | *O(1)* 时间操作                           |
| `set_len(new_len)`         | `()`        | 强制设置长度                          | `unsafe` 需确保容量支持、将丢弃值已初始化 |
| `truncate(len)`            | `()`        | 安全截断                              |                                           |
| `resize_with(new_len,f)`   | `()`        | 在位延长或截断，新增值用 `f()` 补充   |                                           |
| `resize(new_len, value)`   | `()`        | 在位延长或截断，新增值用 `value` 填充 |                                           |

####    `Vec` 逐元素操作

| `Vec` 逐元素操作              | 返回值                                         | 描述                                              | 其他             |
|-------------------------------|------------------------------------------------|---------------------------------------------------|------------------|
| `retain(f)`                   | `()`                                           | 仅保留 `f(&T) == true` 元素                       |                  |
| `retain_mut(f)`               | `()`                                           | 仅保留 `f(&mut T) == true` 元素                   | 可同时在位修改值 |
| `dedup_by_key(key)`           | `()`                                           | 剔除 `key(&mut T)` 重复元素，仅保留首个           |                  |
| `dedup()`                     | `()`                                           | 根据 `PartialEq` 结果移除相邻重复值               |                  |
| `dedup_by(same_bucket)`       | `()`                                           | 根据 `same_bucket(&mut T, &mut T)` 剔除相邻重复值 |                  |
| `split_off(at)`               | `Vec<T>`                                       | 在 `at` 处拆分，返回包含 `at` 之后元素的 `Vec`    |                  |
| `sparse_capacity_mut()`       | `&mut [MaybeUninit<T>]`                        | 将空闲容量返回为 `&mut [MaybeUninit<T>]`          |                  |
| `drain(range)`                | `Drain<'_, T, A>`                              | 双头迭代移除 `range` 范围内元素                   |                  |
| `splice(range, replace_with)` | `Splice<'_, <I as IntoIterator>::IntoIter, A>` | 迭代替代、返回范围元素                            | 长度可不同       |
| `extract_if(range, filter)`   | `Extract<'_, T, F, A>`                         | 迭代移除、返回范围内 `filter(T) == true`          |                  |

####    `VecDeque` 额外方法

| `VecDeque` 额外方法        | 返回值                 | 描述                                              | 其他                          |
|----------------------------|------------------------|---------------------------------------------------|-------------------------------|
| `front()`                  | `Option<&T>`           | 首个元素共享引用                                  | 空则返回 `None`               |
| `front_mut()`              | `Option<&mut T>`       | 首个元素可变引用                                  | 空则返回 `None`               |
| `back()`                   | `Option<&T>`           | 末个元素共享引用                                  | 空则返回 `None`               |
| `back_mut()`               | `Option<&mut T>`       | 末个元素可变引用                                  | 空则返回 `None`               |
| `pop_front()`              | `Option<T>`            | 移除、并返回首个元素                              | 空则返回 `None`               |
| `pop_front_if(pred)`       | `Option<T>`            | 若首个元素 `pred(&mut T) == true`，则移除、并返回 | 空、或 `false` 则返回 `None`  |
| `pop_back()`               | `Option<T>`            | 移除、并返回末个元素                              | 空则返回 `None`               |
| `pop_back_if(pred)`        | `Option<T>`            | 若末个元素 `pred(&mut T) == true`，则移除、并返回 | 空、或 `false` 则返回 `None`  |
| `push_front(T)`            | `()`                   | 队首插入元素                                      |                               |
| `push_front_mut(T)`        | `&mut T`               | 队首插入元素并返回可变引用                        | *Exp*                         |
| `push_back(T)`             | `()`                   | 队末插入元素                                      |                               |
| `push_back_mut(T)`         | `&mut T`               | 队末插入元素并返回可变引用                        | *Exp*                         |
| `prepend(other)`           | `()`                   | 向队首 **整体** 插入迭代元素                      | *Exp*、兼容可转为双头迭代类型 |
| `extend_front(other)`      | `()`                   | 向队首 **迭代** 插入迭代元素                      | *Exp*、插入元素逆序           |
| `swap_remove_front(index)` | `T`                    | 移除并返回元素，用队首元素替代                    | *O(1)* 时间操作               |
| `swap_remove_back(index)`  | `T`                    | 移除并返回元素，用队末元素替代                    | *O(1)* 时间操作               |
| `make_contiguous()`        | `&mut [T]`             | 内存连续排列元素                                  |                               |
| `rotate_left(mid)`         | `()`                   | 在位将首 `mid` 个、剩余交换位置                   |                               |
| `rotate_right(k)`          | `()`                   | 在位将末 `mid` 个、剩余交换位置                   |                               |
| `as_slices()`              | `(&[T], &[T])`         | 创建切片 `&[T]`                                   |                               |
| `as_mut_slices()`          | `(&mut [T], &mut [T])` | 创建可变切片 `&mut [T]`                           |                               |

### `string::String`

> - `std::string::String`：<https://doc.rust-lang.org/stable/std/string/struct.String.html>

####    `String` 容量、长度

| `String` 容量、长度                             | 返回值                              | 描述                       | 其他                    |
|-------------------------------------------------|-------------------------------------|----------------------------|-------------------------|
| `String::new()`                                 | `String`                            | 创建空字符串               |                         |
| `String::with_capacity()`                       | `String`                            | 创建指定容量字符串         |                         |
| `String::from_utf8(vec)`                        | `Result<String, FromUtf8Error>`     | 从 `Vec<u8>` 创建字符串    |                         |
| `String::from_utf8_lossy(v)`                    | `Cow<'_, str>`                      | 从 `&[u8]` 创建字符串      | 无效序列替换为 `U+FFFD` |
| `String::from_utf16(vec)`                       | `Result<String, FromUtf16Error>`    | 从 `Vec<u16>` 创建字符串   |                         |
| `String::from_utf16_lossy(v)`                   | `String`                            | 从 `&[u16]` 创建字符串     | 无效序列替换为 `U+FFFD` |
| `String::from_raw_parts(buf, length, capacity)` | `String`                            | 从指针、长度、容量创建     | `unsafe` 需确保实参有效 |
| `into_raw_parts()`                              | `(*mut u8, usize, unsize)`          | 转换为指针、长度、容量     |                         |
| `String::from_utf8_unchecked(vec)`              | `String`                            | 从 `Vec<u8>` 创建字符串    | `unsafe` 需保证编码正确 |
| `len()`                                         | `usize`                             | 字节数量                   |                         |
| `is_empty()`                                    | `bool`                              | 是否为空                   |                         |
| `capcity()`                                     | `usize`                             | 容量                       |                         |
| `reserve(additional)`                           | `()`                                | 保留至少 `additional` 容量 | 重分配可能分配更多内存  |
| `reserve_exact(additional)`                     | `()`                                | 保留至少 `additional` 容量 | 重分配则精确分配内存    |
| `try_reserve(additional)`                       | `Result<(), TryReserveError>`       | 保留至少 `additional` 容量 | 重分配可能分配更多内存  |
| `try_reserve_exact(additional)`                 | `Result<(), TryReserverExactError>` | 保留至少 `additional` 容量 | 重分配则精确分配内存    |
| `shrink_to_fit()`                               | `()`                                | 缩小容量                   |                         |
| `shrink_to(min_capacity)`                       | `()`                                | 缩小容量、但不小于阈值     |                         |

-   说明
    -   `U+FFFD` *Replacemnent Character*

####    `String` 内容

| `String` 内容             | 返回值         | 描述                          | 其他                       |
|---------------------------|----------------|-------------------------------|----------------------------|
| `clear()`                 | `()`           | 清空                          |                            |
| `push(ch)`                | `()`           | 追加 `char`                   |                            |
| `push_str(string)`        | `()`           | 追加                          |                            |
| `pop()`                   | `Option<char>` | 弹出末尾字符                  |                            |
| `remove(idx)`             | `char`         | 移除指定位置字符              |                            |
| `truncate(new_len)`       | `()`           | 截断                          |                            |
| `extend_from_within(src)` | `()`           | 从范围 `src` 复制、追加到末尾 |                            |
| `drain(range)`            | `Drain<'_>`    | 迭代移除 `range` 范围内容     |                            |
| `retain(f)`               | `()`           | 仅保留 `f(char) == true` 字符 |                            |
| `insert(idx, ch)`         | `()`           | 指定位置插入字符              |                            |
| `insert_str(idx, string)` | `()`           | 指定位置插入 `&str`           |                            |
| `split_off(at)`           | `String`       | 在 `at` 分段                  | 返回包含 `at` 后内容字符串 |

####    `String` 转换、创建

| `String` 转换、创建                  | 返回值                     | 描述                                 | 其他                              |
|--------------------------------------|----------------------------|--------------------------------------|-----------------------------------|
| `into_bytes()`                       | `Vec<u8>`                  | 转换为字节向量                       |                                   |
| `as_str()`                           | `&str`                     | 转换为字符串共享切片                 |                                   |
| `as_mut_str()`                       | `&mut str`                 | 转换为字符串可变切片                 |                                   |
| `as_bytes()`                         | `&[u8]`                    | 创建字节切片                         |                                   |
| `as_mut_vec()`                       | `&mut Vec<u8>`             | 创建 `&mut Vec<u8>` 字节向量可变引用 | `unsafe` 需保证后续修改为有效编码 |
| `replace_range(range, replace_with)` | `()`                       | 移除、并替换范围内容                 |                                   |
| `into_boxed_str()`                   | `Box<str>`                 | 转换为 `Box<str>`                    |                                   |
| `leak()`                             | `&'a mut str`              | 转换为可变引用                       | 后续由用户管理内存                |

### `Cell`s

| 单线程 `Cell`s   | 线程安全         | 描述       |
|------------------|------------------|------------|
| `RefCell<T>`     | `Mutex<T>`       | 可变共享   |
| `OnceCell<T>`    | `OnceLock<T>`    | 初始化可变 |
| `LazyCell<T, F>` | `LazyLock<T, F>` | 延迟初始化 |

-   `Cell`s 用于实现内部可变性：即 `Cell`s 变量自身不可变，但其内部值可变，即作为 **被声明为不可变变量的可变成员**
    -   `Cell<T>` 通过值移入、移出实现内部可变性
        -   不可借用内部值，只能通过 `Copy`、移出方式获取变量
            -   对 `T: Copy`，可通过 `get` 方法获取内部值副本
            -   否则，只能通过 `replace`、`take`、`into_inner` 通过移出内部值的方式获取内部值
        -   即，`Cell` 一般用于包装 `Copy` 类型值
    -   `RefCell<T>` 在运行时检查引用规则（借助生命周期）实现动态可变借用，进而实现内部可变性
        -   `RefCell<T>` 在运行时检查借用规则
            -   若，运行时不满足借用规则则 `panic`
            -   注意：`RefCell::borrow_mut(&self)` 是不可变借用，同时的多次借用并未违反编译器的借用规则
        -   `RefCell` 典型用法
            -   `Rc<RefCell<T>>` 配合实现共享内部可变性
    -   `OnceCell<T>` 只允许对未初始化状态在初始化时修改内部值
        -   `OnceCell<T>` 可视为 `Cell`、`RefCell` 的混合体
            -   可获取内部值的共享引用、可变引用
            -   同时无需、没有运行时借用检查：`OnceCell::get_mut(&mut self)` 由编译器负责检查
        -   即，`OnceCell<T>` 常用于仅需初始化一次、后续值不变的场合
    -   `LazyCell<T, F>` 包装初始化闭包 `F` 实现延迟计算
        -   可视为 `OnceCell<T>` 的延迟计算版
            -   可获取内部值的共享引用、可变引用
            -   同时无需、没有运行时借用检查：`OnceCell::get_mut(&mut self)` 由编译器负责检查
    -   其他说明
        -   `Cell<T>`、`RefCell<T>`、`OnceCell<T>` 并未实现 `Deref`，不是智能指针
        -   `Ref<T>`、`RefMut<T>`、`LazyCell<T, F>` 才实现有 `Deref`
            -   故，其内函数均为关联函数

> - `std::cell`：<https://doc.rust-lang.org/stable/std/cell/index.html>
> - 4.4.5 `Cell` 与 `RefCell` 内部可变性：<https://course.rs/advance/smart-pointer/cell-refcell.html>

####    `Cell` 方法

| `Cell` 转换、创建     | 返回值          | 描述                                           | 其他                        |
|-----------------------|-----------------|------------------------------------------------|-----------------------------|
| `Cell::new()`         | `Cell<T>`       | 创建                                           |                             |
| `Cell::from_mut(t)`   | `&Cell<T>`      | 从 `&mut T` 可变引用创建                       |                             |
| `set(val)`            | `()`            | 设置内部值                                     | 即 `replace` 且 `drop` 旧值 |
| `get()`               | `T`             | 创建内部值副本                                 | `T: Copy`                   |
| `replace(val)`        | `T`             | 替换并返回内部值                               |                             |
| `take()`              | `T`             | 获取内部值并替换为 `Default::default()` 默认值 | `T: Default`                |
| `into_inner()`        | `T`             | 消耗、解包为内部值                             |                             |
| `update(f)`           | `()`            | 根据 `f(T) -> T` 更新内部值                    | `T: Copy`                   |
| `swap(other)`         | `()`            | 与 `&Cell<T>` 交换内部值                       |                             |
| `as_ptr()`            | `*mut T`        | 创建内部值可变裸指针                           |                             |
| `get_mut()`           | `&mut T`        | 创建内部值可变引用                             |                             |
| `as_slice_of_cells()` | `&[Cell<T>]`    | 从 `Cell<[T]>` 创建 `&[Cell<T>]`               |                             |
| `as_array_of_cells()` | `&[Cell<T>; N]` | 从 `Cell<[T; N]>` 创建 `&[Cell<T>; N]`         |                             |

> - `std::cell::Cell`：<https://doc.rust-lang.org/stable/std/cell/struct.Cell.html>

####    `RefCell` 方法

| `RefCell` 转换、创建     | 返回值                               | 描述                                    | 其他                                              |
|--------------------------|--------------------------------------|-----------------------------------------|---------------------------------------------------|
| `RefCell::new()`         | `RefCell<T>`                         | 创建                                    |                                                   |
| `swap(other)`            | `()`                                 | 与 `&RefCell<T>` 交换内部值             |                                                   |
| `replace(val)`           | `T`                                  | 替换并返回内部值                        |                                                   |
| `replace_with(f)`        | `T`                                  | 按 `f(&mut T) -> T` 替换并返回内部值    |                                                   |
| `into_inner()`           | `T`                                  | 解包为内部值                            |                                                   |
| `as_ptr()`               | `*mut T`                             | 创建内部值可变裸指针                    |                                                   |
| `get_mut()`              | `&mut T`                             | 创建内部值可变引用                      |                                                   |
| `take()`                 | `T`                                  | 获取内部值并替换为 `Default::default()` | `T: Default`                                      |
| `borrow()`               | `Ref<'_, T>`                         | 不可变借用内部值                        | 若已被可变借用则 `panic`                          |
| `try_borrow()`           | `Result<Ref<'_, T>, BorrowError>`    | 尝试不可变借用内部值                    |                                                   |
| `borrow_mut()`           | `RefMut<'_, T>`                      | 可变借用内部值                          | 若已被借用则 `panic`                              |
| `try_borrow_mut()`       | `Result<RefMut<'_, T>, BorrowError>` | 尝试可变借用内部值                      |                                                   |
| `try_borrow_unguarded()` | `Result<&T, BorrowError>`            | 尝试不可变借用内部值                    | 若已被可变借用则 `panic`，`unsafe` 未修改借用标志 |

> - `std::cell:RefCell`：<https://doc.rust-lang.org/stable/std/cell/struct.RefCell.html>

#####   `Ref` 关联函数

| `Ref<'b, T>` 关联函数      | 返回值                           | 描述                                | 其他                     |
|----------------------------|----------------------------------|-------------------------------------|--------------------------|
| `Ref::clone(orig)`         | `Ref<'b, T>`                     | 克隆                                |                          |
| `Ref::map(orig, f)`        | `Ref<'b, U>`                     | 根据 `f(&T) -> &U` 映射             |                          |
| `Ref::filter_map(orig, f)` | `Result<Ref<'b, U>, Ref<'b, T>>` | 根据 `f(&T) -> Option<&U>` 尝试映射 | 映射为 `None` 则保持不变 |
| `Ref::map_split(orig, f)`  | `(Ref<'b, U>, Ref<'b, V>)`       | 根据 `f(&T) -> (&U, &V)` 拆分       |                          |

> - `std::cell:Ref`：<https://doc.rust-lang.org/stable/std/cell/struct.Ref.html>

#####   `RefMut` 关联函数

| `RefMut<'b, T>` 关联函数      | 返回值                                 | 描述                                        | 其他                     |
|-------------------------------|----------------------------------------|---------------------------------------------|--------------------------|
| `RefMut::map(orig, f)`        | `RefMut<'b, U>`                        | 根据 `f(&mut T) -> &mut U` 映射             |                          |
| `RefMut::filter_map(orig, f)` | `Result<RefMut<'b, U>, RefMut<'b, T>>` | 根据 `f(&mut T) -> Option<&mut U>` 尝试映射 | 映射为 `None` 则保持不变 |
| `RefMut::map_split(orig, f)`  | `(RefMut<'b, U>, RefMut<'b, V>)`       | 根据 `f(&mut T) -> (&mut U, &mut V)` 拆分   |                          |

> - `std::cell:RefMut`：<https://doc.rust-lang.org/stable/std/cell/struct.RefMut.html>

####    `OnceCell` 方法

| `OnceCell` 转换、创建 | 返回值           | 描述                                               | 其他                        |
|-----------------------|------------------|----------------------------------------------------|-----------------------------|
| `OnceCell::new()`     | `OnceCell<T>`    | 创建                                               |                             |
| `set(val)`            | `Result<(), T>`  | 设置内部值                                         | 若已初始化则返回 `Err(val)` |
| `get()`               | `Option<&T>`     | 创建内部值共享引用                                 | 未初始化则返回 `None`       |
| `get_mut()`           | `Option<&mut T>` | 创建内部值可变引用                                 | 未初始化则返回 `None`       |
| `get_or_init(f)`      | `&T`             | 获取内部值共享引用、未初始化则按 `f() -> T` 设置值 |                             |
| `get_mut_or_init(f)`  | `&mut T`         | 获取内部值可变引用、未初始化则设置值               | *Exp*                       |
| `into_inner()`        | `Option<T>`      | 消耗、解包为内部值                                 | 未初始化则返回 `None`       |
| `take()`              | `T`              | 获取内部值并置为未初始化状态                       | 若未初始化则返回 `None`     |

> - `std::cell:OnceCell`：<https://doc.rust-lang.org/stable/std/cell/struct.OnceCell.html>
> - `std::sync::OnceLock`：<https://doc.rust-lang.org/stable/std/sync/struct.OnceLock.html>

####    `LazyCell`、`LazyLock`

| `LazyCell` 转换、创建        | 返回值           | 描述                                 | 其他                    |
|------------------------------|------------------|--------------------------------------|-------------------------|
| `LazyCell::new(f)`           | `LazyCell<T>`    | 包装 `f() -> T` 初始化函数           | 但不执行计算            |
| `LazyCell::into_inner(this)` | `Result<T, F>`   | 消耗、解包为内部值                   | 未初始化则返回 `Err(f)` |
| `LazyCell::force(this)`      | `&T`             | 执行 `f() -> T` 计算、并返回共享引用 | 等价于 `Deref`          |
| `LazyCell::force_mut(this)`  | `&mut T`         | 执行 `f() -> T` 计算、并返回可变引用 |                         |
| `LazyCell::get_mut(this)`    | `Option<&mut T>` | 获取内部值可变引用                   | 未初始化则返回 `None`   |
| `LazyCell::get(this)`        | `Option<&T>`     | 获取内部值共享引用                   | 未初始化则返回 `None`   |

> - `std::cell::LazyCell`：<https://doc.rust-lang.org/stable/std/cell/struct.LazyCell.html>
> - `std::sync::LazyLock`：<https://doc.rust-lang.org/stable/std/sync/struct.LazyLock.html>

### `HashMap`、`HashSet`

-   `collections::HashMap` 说明
    -   `HashMap` 中哈希算法默认为 *SipHash 1-3*
        -   使用 4 次探查、*SIMD* 查找
        -   适合中等数量场合
        -   对 *HashDos* 具有抵御能力
    -   `HashSet` 可视为是值为 `()` 的 `HashMap`

> - `std::collections`：<https://doc.rust-lang.org/stable/std/collections/index.html>
> - `std::collections::HashMap`：<https://doc.rust-lang.org/stable/std/collections/struct.HashMap.html>
> - `std::collections::HashSet`：<https://doc.rust-lang.org/stable/std/collections/struct.HashSet.html>

####    `HashMap` 创建、元信息

| `HashMap` 创建、元信息方法                                | 返回值                        | 描述                                                | 其他                   |
|-----------------------------------------------------------|-------------------------------|-----------------------------------------------------|------------------------|
| `HashMap::new()`                                          | `HashMap<K, V, RandomState>`  | 创建                                                |                        |
| `HashMap::with_capacity(capacity)`                        | `HashMap<K, V, RandomState>`  | 指定最小容量创建                                    |                        |
| `HashMap::with_hasher(hasher_builder)`                    | `HashMap<K, V, S>`            | 指定 `hasher_builder` 计算键创建                    |                        |
| `HashMap::with_capacity_hasher(capacity, hasher_builder)` | `HashMap<K, V, S>`            | 指定容量、 `hasher_builder` 计算键创建              |                        |
| `capacity()`                                              | `usize`                       | 获取容量                                            |                        |
| `len()`                                                   | `usize`                       | 获取长度                                            |                        |
| `is_empty()`                                              | `bool`                        | 是否为空                                            |                        |
| `clear()`                                                 | `()`                          | 清空                                                |                        |
| `hasher()`                                                | `&S`                          | 获取 `BuildHasher`                                  |                        |
| `reserve(additional)`                                     | `()`                          | 保留至少 `additional` 容量                          | 重分配可能分配更多内存 |
| `try_reserve(additional)`                                 | `Result<(), TryReserveError>` | 尝试预分配容量，返回 `Result<Vec, TryReserveError>` |                        |
| `shrink_to_fit()`                                         | `()`                          | 尽可能缩小容量                                      |                        |
| `shrink_to(min_capcity)`                                  | `()`                          | 缩小容量、但不小于阈值                              |                        |

####    `HashMap` 删改

| `HashMap` 方法                   | 返回值                | 描述                                | 其他                                       |
|----------------------------------|-----------------------|-------------------------------------|--------------------------------------------|
| `entry(key)`                     | `Entry<'_, K, V, A>`  | 获取 Hash 表项                      | 可用于在位操作                             |
| `get(k)`                         | `Option<&V>`          | 获取值共享引用                      | 实参可为键的任意借用类型 `Q; K: Borrow<Q>` |
| `get_mut(k)`                     | `Option<&mut V>`      | 获取值可变引用                      |                                            |
| `get_key_value(k)`               | `Option<(&K, &V)>`    | 获取键、值共享引用                  |                                            |
| `get_disjoint_mut(ks)`           | `[Option<&mut V>; N]` | 获取值可变引用数组                  | 存在 “重复” 键则 `panic`                   |
| `get_disjoint_unchecked_mut(ks)` | `[Option<&mut V>; N]` | 获取值可变引用数组                  | `unsafe` 存在重复键导致 *UB*               |
| `contains_key(k)`                | `bool`                | 是否包含键                          |                                            |
| `insert(k, v)`                   | `Option<V>`           | 插入、或更新键                      | 键已存在则更新并返回 `Some(old)`           |
| `remove(k)`                      | `Option<V>`           | 移除键、并返回旧值                  | 键不存在则返回 `None`                      |
| `remove_entry(k)`                | `Option<(K, V)>`      | 移除键、并返回旧键值对              | 键不存在则返回 `None`                      |
| `retain(f)`                      | `()`                  | 仅保留 `f(&K, &mut V) == true` 键值 |                                            |

#####   `collections::hash_map::Entry` 方法

```rust
pub enum Entry<'a, K: 'a, V: 'a, A: Allocator = Global> {
    Occupied(OccupiedEntry<'a, K, V, A>),                   // 非空项
    Vacant(VacantEntry<'a, K, V, A>),                       // 空表项
}
```

| `Entry` 方法                  | 返回值                       | 描述                                               | 其他           |
|-------------------------------|------------------------------|----------------------------------------------------|----------------|
| `key()`                       | `&K`                         | 创建键共享引用                                     |                |
| `or_default()`                | `&'a mut V`                  | 空项则填充 `Default::default`                      |                |
| `or_insert(default)`          | `&'a mut V`                  | 空项则填充值，总返回值可变引用                     |                |
| `or_insert_with(default)`     | `&'a mut V`                  | 空项则按 `default() -> V` 填充，总返回值可变引用   |                |
| `or_insert_with_key(default)` | `&'a mut V`                  | 空项则按 `default(&K) -> V` 填充，总返回值可变引用 |                |
| `insert_entry(value)`         | `OccupiedEntry<'a, K, V, A>` | 插入值                                             |                |
| `and_modify(f)`               | `Self`                       | 并修改非空项值                                     | 空项则原样返回 |

> - `std::collections::hash_map::Entry`：<https://doc.rust-lang.org/stable/std/collections/hash_map/enum.Entry.html>

####    `HashMap` 迭代适配

| `HashMap` 迭代适配 | 返回适配器                  | 迭代 `Option<>`      | 描述                                     | 其他                         |
|--------------------|-----------------------------|----------------------|------------------------------------------|------------------------------|
| `keys()`           | `Keys<'_, K, V>`            | `&'a K`              | 创建迭代键共享引用                       |                              |
| `into_keys()`      | `IntoKeys<K, V, A>`         | `K`                  | 转换为迭代键                             |                              |
| `values()`         | `Values<'_, K, V>`          | `&'a V`              | 创建迭代值共享引用                       | 空 *bucket* 也会被访问       |
| `values_mut()`     | `ValuesMut<'_, K, V>`       | `&mut 'a V`          | 创建迭代值可变引用                       | 空 *bucket* 也会被访问       |
| `into_values()`    | `IntoValues<K, V, A>`       | `V`                  | 转换为迭代值                             |                              |
| `iter()`           | `Iter<'_, K, V>`            | `(&'a K, &'a V)`     | 迭代键、值共享引用                       |                              |
| `iter_mut()`       | `IterMut<'_, K, V>`         | `(&'a K, &'a mut V)` | 迭代键共享引用、值可变引用               |                              |
| `drain()`          | `Drain<'_, K, V, A>`        | `(K, V)`             | 转换迭代键、值                           | 总是清空、但保留已分配空间   |
| `extract_if(pred)` | `ExtractIf<'_, K, V, F, A>` | `(K, V)`             | 迭代移除 `pred(&K, &mut V) == true` 键值 | 对应 `retain` 不返回移除结果 |

### `borrow::Cow`

```rust
pub enum Cow<'a, B: 'a + ToOwned + ?Sized> {
    Borrowed(&'a B),                                // 引用变体
    Owned(<B as ToOwned>::Owned),                   // 所有权变体，写时复制
}
```

-   `borrow::Cow<'a, B>` 写时复制的智能指针
    -   `Cow` 实现有 `Deref`、`AsRef`
    -   `Cow` 常用于延迟 Clone、仅写时复制，减少开销
        -   `Cow::Borrow` 先封装不可变引用
        -   仅在需要可变引用、获取所有权时 Clone 并转换为 `Cow::Owned`

| `Cow<'a, B>` 方法     | 返回值                       | 描述                   | 其他                   |
|-----------------------|------------------------------|------------------------|------------------------|
| `to_mut()`            | `&mut <B as ToOwned>::Owned` | 创建所有权类型可变引用 | 未拥有所有权则先 Clone |
| `into_owned()`        | `<B as ToOwned>::Owned`      | 解包获取内部所有权     | 未拥有所有权则先 Clone |
| `Cow::is_borrowed(c)` | `bool`                       | 是否未借用             | *Exp*                  |
| `Cow::is_owned(c)`    | `bool`                       | 是否拥有数据           | *Exp*                  |

### `std::io` 中工具类型

| `io` 工具类型                | 描述                                                   |
|------------------------------|--------------------------------------------------------|
| `io::Result<T>`              | `Result<T, Error>` 别名                                |
| `io::IoSliceMut<'a>`         | `&mut [u8]` 包装器、*ABI* 二进制兼容 `iovec`、`WSABUF` |
| `io::BufReader<R>`           | 带缓冲 *Reader* 封装                                   |
| `io::BufWriter<R>`           | 带缓冲 *Writer* 封装                                   |
| `io::SeekFrom`               | `Seek` 位置枚举值                                      |
| `io::SeekFrom::Start(u64)`   | `Seek` 起始 + `u64`                                    |
| `io::SeekFrom::End(i64)`     | `Seek` 末尾 + `i64`                                    |
| `io::SeekFrom::Current(i64)` | `Seek` 当前位置 + `i64`                                |
| `io::Stdin`                  | 标准输入                                               |
| `io::Stdout`                 | 标准输出                                               |
| `io::Stderr`                 | 错误输出                                               |
| `io::Sink`                   | 空输出                                                 |

####    `io::IoSliceMut`、`io::IoSlice`

| `IoSliceMut` 方法                     | 返回值         | 描述                                                    | 其他                   |
|---------------------------------------|----------------|---------------------------------------------------------|------------------------|
| `IoSliceMut::new(buf)`                | `IoSlice<'a>`  | 从 `&'a mut [u8]` 创建                                  |                        |
| `advance(n: usize)`                   | `()`           | 移动内部指针                                            | 跳过部分被丢弃         |
| `IoSliceMut::advance_slices(bufs, n)` | `()`           | 将 `&mut &mut [IoSlice<'a>]` 视为串联整体、移动内部指针 | 跳过部分被丢弃         |
| `into_slice()`                        | `&'a mut [u8]` | 转换为字节切片                                          | *Exp*、仅 `IoSliceMut` |
| `as_slice()`                          | `&'a [u8]`     | 转换为字节切片                                          | *Exp*、仅 `IoSlice`    |

-   `IoSliceMut`、`IoSlice` 语义上是 `&[u8]` 的包装器，保证 *ABI* 二进制兼容 *Unix* 平台 `iovec`、*Win* 平台 `WSABUF`
    -   二者均实现 `Deref<target = [u8]`（`IoSliceMut` 额外实现 `DerefMut`），支持切片方法应用
        -   `IoSliceMut.deref()` 仅获取剩余未被消耗的字节切片

> - `std::io::IoSlice`：<https://doc.rust-lang.org/stable/std/io/struct.IoSlice.html>
> - `std::io::IoSliceMut`：<https://doc.rust-lang.org/stable/std/io/struct.IoSliceMut.html>

####    `io::BufReader`

| `BufReader<R>` 方法                         | 返回值          | 描述                                   | 其他                        |
|---------------------------------------------|-----------------|----------------------------------------|-----------------------------|
| `BufReader::new(inner)`                     | `BufReader<R>`  | 从 *Reader* `impl Read` 创建           |                             |
| `BufReader::with_capacity(capacity, inner)` | `BufReader<R>`  | 指定容量、从 *Reader* `impl Read` 创建 |                             |
| `peek(n)`                                   | `Result<&[u8]>` | 前看、返回 `n` 字节                    | *Exp*、可能因到达末尾而更短 |
| `get_ref()`                                 | `&R`            | 获取内部 *Reader* 共享引用             |                             |
| `get_mut()`                                 | `&mut R`        | 获取内部 *Reader* 可变引用             |                             |
| `buffer()`                                  | `&[u8]`         | 获取内部缓冲区共享引用                 |                             |
| `capacity()`                                | `usize`         | 获取容量                               |                             |
| `into_inner()`                              | `R`             | 解包转为内部 *Reader*                  |                             |
| `seek_relative(offset)`                     | `Result<()>`    | 移动指针                               |                             |

-   `io::BufReader<R>` 为 *Reader* `R: impl Read` 添加缓冲区（的封装）
    -   `BufReader<R>` 实现有 `Read`、`BufRead`、`Seek`
    -   `BufReader<R>` 可以提升 *Reader* 在少量、重复读取行为的效率
        -   减少直接对诸如 `TcpStream` 之类读取带来的系统调用开销

```rust
pub struct BufferReader<R: ?Sized> {
    buf: Buffer,                                    // 内部缓冲区
    inner: R,                                       // 内部 Reader
}
pub struct Buffer {
    buf: Box<[MaybeUninit<u8>];
    pos: usize,
    filled: usize,
    initialized: usize,
}
```

> - `std::io::BufReader`：<https://doc.rust-lang.org/stable/std/io/struct.BufReader.html>

####    `io::BufWriter`

| `BufWriter<W>` 方法                         | 返回值                                 | 描述                                    | 其他 |
|---------------------------------------------|----------------------------------------|-----------------------------------------|------|
| `BufWriter::new(inner)`                     | `BufWriter<W>`                         | 从 *Writer* `impl Write` 创建           |      |
| `BufWriter::with_capacity(capacity, inner)` | `BufWriter<W>`                         | 指定容量、从 *Writer* `impl Write` 创建 |      |
| `get_ref()`                                 | `&W`                                   | 获取内部 *Writer* 共享引用              |      |
| `get_mut()`                                 | `&mut W`                               | 获取内部 *Writer* 可变引用              |      |
| `buffer()`                                  | `&[u8]`                                | 获取内部缓冲区共享引用                  |      |
| `capacity()`                                | `usize`                                | 获取容量                                |      |
| `into_inner()`                              | `W`                                    | 解包转为内部 *Writer*                   |      |
| `into_parts()`                              | `(W, Result<Vec<u8>>, WriterPanicked)` | 解包为三元组                            |      |
| `seek_relative(offset)`                     | `Result<()>`                           | 移动指针                                |      |

-   `io::BufWriter<W>` 为 *Writer* `W: impl Write` 添加缓冲区
    -   `BufWriter<R>` 实现有 `Write`、`Seek`
    -   `BufWriter<W>` 可以提升 *Writer* 在少量、重复读取行为的效率
        -   减少直接对诸如 `TcpStream` 之类写入带来的系统调用开销
    -   确保在 `BufWriter<W>` 被丢弃前调用 `flush()`
        -   析构过程会尝试将缓冲区数据刷入真正目标，但析构过程中的错误会被忽略

```rust
pub struct BufWriter<W: ?Sized + Write> {
    buf: Vec<u8>,                                   // 内部缓冲区
    panicked: bool,                                 // 写入 panic 标记
    inner: W,                                       // 内部 Writer
}
```

> - `std::io::BufWriter`：<https://doc.rust-lang.org/stable/std/io/struct.BufWriter.html>

####    `io::Stdin`、`io::Stdout`、`io::Sink`

| `io::Stdin` 方法 | 返回值                     | 描述                                         | 其他 |
|------------------|----------------------------|----------------------------------------------|------|
| `lock()`         | `StdinLock<'static>`       | 锁定句柄                                     |      |
| `read_line(buf)` | `Result<usize>`            | 读取一行输入追加至 `&mut String`、返回字节数 |      |
| `lines()`        | `Line<StdinLock<'static>>` | 转换为按行迭代                               |      |

| `io::Stdout` 方法 | 返回值                | 描述     | 其他 |
|-------------------|-----------------------|----------|------|
| `lock()`          | `StdoutLock<'static>` | 锁定句柄 |      |

-   标准输入、标准输出、错误输出
    -   `io::Stdin` 为进程标准输入流的句柄，即对进程输入全局缓冲的共享引用
        -   `Stdin` 实现有 `Read`
        -   `Stdin` 通过 `io::stdin()` 函数创建、返回
    -   `io::StdinLock` 为已上锁的 `Stdin` 句柄
        -   `StdinLock` 实现有 `Read`、`ReadBuf`
    -   `io::Stdout`、`io::Stderr` 类似
    -   `io::Sink` 空输出 *Writer*
        -   `Sink` 通过 `io::sink()` 创建

> - `std::io::Stdin`：<https://doc.rust-lang.org/stable/std/io/struct.Stdin.html>
> - `std::io::StdinLock`：<https://doc.rust-lang.org/stable/std/io/struct.StdinLock.html>
> - `std::io::Stdout`：<https://doc.rust-lang.org/stable/std/io/struct.Stdout.html>
> - `std::io::StdoutLock`：<https://doc.rust-lang.org/stable/std/io/struct.StdoutLock.html>
> - `std::io::Sink`：<https://doc.rust-lang.org/stable/std/io/struct.Sink.html>

### `fs::File`

-   `fs::File` 文件系统文件访问入口
    -   `File` 实现有 `Read`、`Write`、`Seek`
        -   在少量、频繁读写时，可用 `BufReader`、`BufWriter` 包装以提升效率
    -   文件在 `File` 变量离开作用域、执行 `Drop::drop` 时自动关闭
        -   但，`Drop` 中错误会被忽略
        -   可通过显式调用 `File::sync_all()` 以处理可能出现的错误

> - `std::fs`：<https://doc.rust-lang.org/stable/std/fs/index.html>
> - `std::fs::File`：<https://doc.rust-lang.org/stable/std/fs/struct.File.html>

####    `File` 创建、元信息

| `fs::File` 创建、元信息     | 返回值                     | 描述                                      | 其他                        |
|-----------------------------|----------------------------|-------------------------------------------|-----------------------------|
| `File::open(path)`          | `Result<File>`             | 只读打开 `AsRef<Path>` 指定的文件         |                             |
| `File::open_buffered(path)` | `Result<BufReader<File>>`  | 只读、带缓冲打开 `AsRef<Path>` 指定的文件 |                             |
| `File::create(path)`        | `Result<File>`             | 只写打开 `AsRef<Path>` 指定的文件         | 已有文件将被清空            |
| `File::create(path)`        | `Result<<BufWriter<File>>` | 只写、带缓冲打开 `AsRef<Path>` 指定的文件 | 已有文件将被清空            |
| `File::create_new(path)`    | `Result<File>`             | 只写打开 `AsRef<Path>` 指定的文件         | 已有文件将返回 `Error`      |
| `File::options()`           | `OpenOptions`              | 创建文件打开选项                          | 等同于 `OpenOptions::new()` |
| `set_len(size)`             | `Result<()>`               | 截断、扩展文件大小至 `size`               |                             |
| `metadata()`                | `Result<Metadata>`         | 获取文件元信息                            |                             |
| `set_permissions(perm)`     | `Result<()>`               | 按 `Permissions` 设置权限                 |                             |
| `set_times(times)`          | `Result<()>`               | 按 `FileTimes` 设置时间戳                 |                             |
| `set_modified(times)`       | `Result<()>`               | 按 `SystemTime` 设置修改时间              |                             |

####    `File` 读写辅助

| `fs::File` 读写     | 返回值                     | 描述                           | 其他 |
|---------------------|----------------------------|--------------------------------|------|
| `sync_all()`        | `Result<()>`               | 尝试同步文件内容、元信息至磁盘 |      |
| `sync_data()`       | `Result<()>`               | 尝试同步文件内容至磁盘         |      |
| `lock()`            | `Result<()>`               | 阻塞、直至获取文件互斥锁       |      |
| `lock_shared()`     | `Result<()>`               | 阻塞、直至获取文件非互斥锁     |      |
| `try_lock()`        | `Result<(), TryLockError>` | 非阻塞的尝试获取文件互斥锁     |      |
| `try_lock_shared()` | `Result<(), TryLockError>` | 非阻塞的尝试获取文件非互斥锁   |      |
| `unlock()`          | `Result<()>`               | 释放文件所有锁                 |      |
| `try_clone()`       | `Result<File>`             | 尝试克隆                       |      |

####    `fs::OpenOptions` 方法

| `fs::OpenOptions` 方法 | 返回值         | 描述                           | 其他                            |
|------------------------|----------------|--------------------------------|---------------------------------|
| `OpenOptions::new()`   | `OpenOptions`  | 创建                           |                                 |
| `read(read)`           | `&mut Self`    | 按 `bool` 设置读标志           |                                 |
| `write(write)`         | `&mut Self`    | 按 `bool` 设置写标志           |                                 |
| `append(append)`       | `&mut Self`    | 按 `bool` 设置追加标志         |                                 |
| `truncate(truncate)`   | `&mut Self`    | 按 `bool` 设置清空标志         |                                 |
| `create(append)`       | `&mut Self`    | 按 `bool` 设置创建、或打开标志 | 要求 `write`、`append` 之一置位 |
| `create_new(append)`   | `&mut Self`    | 按 `bool` 设置创建标志         | 后续若文件已存在则报错          |
| `open(path)`           | `Result<File>` | 按已指定标志打开文件           |                                 |

> - `std::fs::OpenOptions`：<https://doc.rust-lang.org/stable/std/fs/struct.OpenOptions.html>

### `std::net` 工具类型

| `net` 工具类型、特性                | 描述                               |
|-------------------------------------|------------------------------------|
| `net::TcpStream`                    | TCP 数据流                         |
| `net::TcpListener`                  | Socket 监听服务                    |
| `net::UdpSocket`                    | UDP Socket                         |
| `net::Shutdown`                     | `Enum`：关闭模式                   |
| `net::Shutdown::Read`               | `Shutdown` 变体：关闭读            |
| `net::Shutdown::Write`              | `Shutdown` 变体：关闭写            |
| `net::Shutdown::Both`               | `Shutdown` 变体：关闭读写          |
| `net::Ipv4Addr`                     | IPv4 地址                          |
| `net::Ipv6Addr`                     | IPv6 地址                          |
| `net::IpAddr`                       | `Enum`：IP 地址                    |
| `net::IpAddr::V4(Ipv4Addr)`         | `IpAddr` 变体                      |
| `net::IpAddr::V6(Ipv6Addr)`         | `IpAddr` 变体                      |
| `net::SocketAddrV4`                 | IPv4 Socket 地址                   |
| `net::SocketAddrV6`                 | IPv6 Socket 地址                   |
| `net::SocketAddr`                   | `Enum`：*Socket* 地址（IP + 端口） |
| `net::SocketAddr::V4(SocketAddrV4)` | `SocketAddr` 变体                  |
| `net::SocketAddr::V6(SocketAddrV6)` | `SocketAddr` 变体                  |
| `net::ToSocketAddrs`                | `Trait`：可转换迭代 `SocketAddr`   |

> - `std::net`：<https://doc.rust-lang.org/stable/std/net/index.html>

####    `net::TcpStream`

| `TcpStream` 方法                            | 返回值                     | 描述                                             | 其他                |
|---------------------------------------------|----------------------------|--------------------------------------------------|---------------------|
| `TcpStream::connect(addr)`                  | `Result<TcpStream>`        | 连接 `impl ToSocketAddrs` 指定的远程主机         |                     |
| `TcpStream::connect_timeout(addr, timeout)` | `Result<TcpStream>`        | 指定 `Duration` 超时、连接单个 `&SocketAddr`     |                     |
| `peer_addr()`                               | `Result<SocketAddr>`       | 获取远程主机 Socket 地址                         |                     |
| `local_addr()`                              | `Result<SocketAddr>`       | 获取本地 Socket 地址                             |                     |
| `set_read_timeout(dur)`                     | `Result<()>`               | 按 `Option<Duration>` 设置超时                   | `None` 表示永久阻塞 |
| `set_write_timeout(dur)`                    | `Result<()>`               | 按 `Option<Duration>` 设置超时                   | `None` 表示永久阻塞 |
| `read_timeout()`                            | `Result<Option<Duration>>` | 获取超时配置                                     |                     |
| `write_timeout()`                           | `Result<Option<Duration>>` | 获取超时配置                                     |                     |
| `set_nodelay(nodelay)`                      | `Result<()>`               | 按 `bool` 设置 Socket `TCP_NODELAY` 选项         |                     |
| `nodelay()`                                 | `Result<bool>`             | 获取 Socket `TCP_NODELAY` 选项                   |                     |
| `set_ttl(ttl)`                              | `Result<()>`               | 按 `u32` 设置 Socket `IP_TTL` 选项               |                     |
| `ttl()`                                     | `Result<u32>`              | 获取 Socket `IP_TTL` 选项                        |                     |
| `take_error()`                              | `Result<Option<Error>>`    | 获取 Socket `SO_ERROR` 选项值                    |                     |
| `set_nonblocking(nonblocking)`              | `Result<()>`               | 按 `bool` 配置是否阻塞模式                       |                     |
| `try_clone()`                               | `Result<TcpStream>`        | 创建不独立、绑定同一 Socket 副本                 | 配置对所有副本生效  |
| `peek(buf)`                                 | `Result<usize>`            | 读取、但不移除队列数据至 `&mut [u8]`，返回字节数 |                     |
| `shutdown(how)`                             | `Result<()>`               | 按 `Shutdown` 关闭读、写                         |                     |

-   `net::TcpStream` 即本地、远程 Socket 之间的 TCP 数据流
    -   `TcpStream` 实现有 `Read`、`Write`，即可向其写入、从中读取字节
    -   `TcpStream` 可由 `TcpStream::connect` 主动连接远程主机、`TcpListener::accept()` 接受远程主机连接请求
        -   远程连接在 `TcpStream` 值被丢弃、或显式调用 `TcpStream::shutdown()` 时关闭

> - `std::net::TcpStream`：<https://doc.rust-lang.org/stable/std/net/struct.TcpStream.html>

####    `net::TcpListener`

| `TcpListener` 方法             | 返回值                            | 描述                                    | 其他                           |
|--------------------------------|-----------------------------------|-----------------------------------------|--------------------------------|
| `TcpListener::bind(addr)`      | `Result<TcpListener>`             | 创建监听、并绑定至 `impl ToSocketAddrs` | 多个地址则尝试直至首个可行地址 |
| `local_addr()`                 | `Result<SocketAddr>`              | 获取绑定的本地 Socket 地址              |                                |
| `set_ttl(ttl)`                 | `Result<()>`                      | 按 `u32` 设置 Socket `IP_TTL` 选项      |                                |
| `ttl()`                        | `Result<u32>`                     | 获取 Socket `IP_TTL` 选项               |                                |
| `take_error()`                 | `Result<Option<Error>>`           | 获取 Socket `SO_ERROR` 选项值           |                                |
| `set_nonblocking(nonblocking)` | `Result<()>`                      | 按 `bool` 配置是否阻塞模式              |                                |
| `try_clone()`                  | `Result<TcpListener>`             | 创建不独立、绑定同一 Socket 副本        | 配置对所有副本生效             |
| `accept()`                     | `Result<(TcpStream, SocketAddr)>` | 阻塞、直至建立连接                      |                                |
| `incoming()`                   | `Incoming<'_>`                    | 迭代建立的连接                          | 迭代返回 `TcpStream`           |

-   `net::TcpListener` 即 Socket 服务器，监听绑定的本地 Socket 连接请求

> - `std::net::TcpListener`：<https://doc.rust-lang.org/stable/std/net/struct.TcpListener.html>

####    `net::UdpSocket`

| `UdpSocket` 方法                           | 返回值                        | 描述                                         | 其他                           |
|--------------------------------------------|-------------------------------|----------------------------------------------|--------------------------------|
| `UdpSocket::bind(addr)`                    | `Result<UdpSocket>`           | 创建 UDP Socket、绑定至 `impl ToSocketAddrs` | 多个地址则尝试直至首个可行地址 |
| `recv_from(buf)`                           | `Result<(usize, SocketAddr)>` | 接收单个数据包                               |                                |
| `send_to(buf, addr)`                       | `Result<usize>`               | 向指定 `impl ToSocketAddrs` 发送数据         | 仅向首个地址发送               |
| `peek_from(buf)`                           | `Result<(usize, SocketAddr)>` | 接收、但不从队列移除单个数据包               |                                |
| `connect(addr)`                            | `Result<()>`                  | 连接至 `impl ToSocketAddrs`                  | 多个地址则尝试直至首个可行地址 |
| `send(buf)`                                | `Result<usize>`               | 将 `&[u8]` 发送至已连接地址                  |                                |
| `recv(buf)`                                | `Result<usize>`               | 从已连接地址获取单个数据包至 `&mut [u8]`     |                                |
| `join_multicast_v4(multiaddr, interface)`  | `Result<()>`                  | 加入 IPv4 多播组                             |                                |
| `join_multicast_v6(multiaddr, interface)`  | `Result<()>`                  | 加入 IPv6 多播组                             |                                |
| `leave_multicast_v4(multiaddr, interface)` | `Result<()>`                  | 离开 IPv4 多播组                             |                                |
| `leave_multicast_v6(multiaddr, interface)` | `Result<()>`                  | 离开 IPv4 多播组                             |                                |
| `take_error()`                             | `Result<Option<Error>>`       | 获取 Socket `SO_ERROR` 选项值                |                                |

-   `net::UdpSocket` 即 UDP Socket
    -   UDP 是无连接协议
        -   `UdpSocket` 可通过绑定的 Socoket 向任意其他 Socket 地址发送、接受数据
        -   另外，可通过 `UdpSocket::connect(addr)` 设置地址，之后可通过 `send`、`recv` 方法默认从该地址获取的数据

| `UdpSocket` 信息、配置                     | 返回值                     | 描述                               | 其他                |
|--------------------------------------------|----------------------------|------------------------------------|---------------------|
| `peer_addr()`                              | `Result<SocketAddr>`       | 获取连接的远程 Socket 地址         |                     |
| `local_addr()`                             | `Result<SocketAddr>`       | 获取本地 Socket 地址               |                     |
| `try_clone()`                              | `Result<UdpSocket>`        | 创建不独立、绑定同一 Socket 副本   | 配置对所有副本生效  |
| `set_read_timeout(dur)`                    | `Result<()>`               | 按 `Option<Duration>` 设置超时     | `None` 表示永久阻塞 |
| `set_write_timeout(dur)`                   | `Result<()>`               | 按 `Option<Duration>` 设置超时     | `None` 表示永久阻塞 |
| `read_timeout()`                           | `Result<Option<Duration>>` | 获取超时配置                       |                     |
| `write_timeout()`                          | `Result<Option<Duration>>` | 获取超时配置                       |                     |
| `set_broadcast(broadcast)`                 | `Result<()>`               | 按 `bool` 配置 `SO_BROADCAST`      |                     |
| `broadcast()`                              | `Result<bool>`             | 获取 `SO_BROADCAST` 配置           |                     |
| `set_multicast_loop_v4(multicast_loop_v4)` | `Result<()>`               | 按 `bool` 配置 `IP_MULTICAST_LOOP` |                     |
| `multicast_loop_v4()`                      | `Result<bool>`             | 获取 `IP_MULTICAST_LOOP` 配置      |                     |
| `set_multicast_ttl_v4(multicast_ttl_v4)`   | `Result<()>`               | 按 `u32` 配置 `IP_MULTICAST_LOOP`  |                     |
| `multicast_ttl_v4()`                       | `Result<u32>`              | 获取 `IP_MULTICAST_LOOP` 配置      |                     |
| `set_multicast_loop_v6(multicast_loop_v6)` | `Result<()>`               | 按 `bool` 配置 `IP_MULTICAST_LOOP` |                     |
| `multicast_loop_v6()`                      | `Result<bool>`             | 获取 `IP_MULTICAST_LOOP` 配置      |                     |
| `set_ttl(ttl)`                             | `Result<()>`               | 按 `u32` 设置 Socket `IP_TTL` 选项 |                     |
| `ttl()`                                    | `Result<u32>`              | 获取 Socket `IP_TTL` 选项          |                     |
| `set_nonblocking(nonblocking)`             | `Result<()>`               | 按 `bool` 配置是否阻塞模式         |                     |

> - `std::net::UdpSocket`：<https://doc.rust-lang.org/stable/std/net/struct.UdpSocket.html>

### `std::time`

> - `std::time`：<https://doc.rust-lang.org/stable/std/time/index.html>

####    `time::Duration`

| `Duration` 创建方法                 | 返回值                                    | 描述                        | 其他                   |
|-------------------------------------|-------------------------------------------|-----------------------------|------------------------|
| `Duration::ZERO`                    | `Duration`                                | 0 时间段                    | 关联常量               |
| `Duration::MAX`                     | `Duration`                                | 最大时间段                  | 关联常量               |
| `Duration::new(secs, nanos)`        | `Duration`                                | 按 `u64` 秒、`u32` 纳秒创建 |                        |
| `Duration::from_secs(secs)`         | `Duration`                                | 按 `u64` 秒创建             |                        |
| `Duration::from_millis(millis)`     | `Duration`                                | 按 `u64` 毫秒创建           |                        |
| `Duration::from_micros(micros)`     | `Duration`                                | 按 `u64` 微秒创建           |                        |
| `Duration::from_nanos(nanos)`       | `Duration`                                | 按 `u64` 纳秒创建           |                        |
| `Duration::from_nanos_u128(nanos)`  | `Duration`                                | 按 `u128` 纳秒创建          |                        |
| `Duration::from_hours(hours)`       | `Duration`                                | 按 `u64` 时创建             |                        |
| `Duration::from_mins(mins)`         | `Duration`                                | 按 `u64` 分创建             |                        |
| `Duration::from_secs_f64(secs)`     | `Duration`                                | 按 `f64` 秒创建             | 溢出、负值 `panic`     |
| `Duration::from_secs_f32(secs)`     | `Duration`                                | 按 `f32` 秒创建             | 溢出、负值 `panic`     |
| `Duration::try_from_secs_f64(secs)` | `Result<Duration, TryFromFloatSecsError>` | 按 `f64` 秒创建             | 溢出、负值返回 `Error` |
| `Duration::try_from_secs_f32(secs)` | `Result<Duration, TryFromFloatSecsError>` | 按 `f32` 秒创建             | 溢出、负值返回 `Error` |

-   `time::Duration` 表示一段时间
    -   `time::Duration` 实现有四则运算、在位四则运算、`Sum`
        -   行为接近无符号整型

| `Duration` 计算、转换   | 返回值             | 描述               | 其他                            |
|-------------------------|--------------------|--------------------|---------------------------------|
| `is_zero()`             | `bool`             | 是否为 0           |                                 |
| `as_secs()`             | `u64`              | 整秒数             |                                 |
| `as_millis()`           | `u128`             | 整毫秒数           |                                 |
| `as_micros()`           | `u128`             | 整微秒数           |                                 |
| `as_nanos()`            | `u128`             | 整纳秒数           |                                 |
| `subsec_millies()`      | `u32`              | 排除整秒后整毫秒数 |                                 |
| `subsec_micros()`       | `u32`              | 排除整秒后整微秒数 |                                 |
| `subsec_nanos()`        | `u32`              | 排除整秒后整纳秒数 |                                 |
| `abs_diff(other)`       | `Duration`         | 绝对值差           |                                 |
| `checked_add(rhs)`      | `Option<Duration>` | 带检查的相加       | 溢出返回 `None`                 |
| `saturating_add(rhs)`   | `Duration`         | 带检查的相加       | 溢出返回 `Duration::MAX`        |
| `checked_sub(rhs)`      | `Option<Duration>` | 带检查的相减       | 溢出、负值返回 `None`           |
| `saturating_subrhs)`    | `Duration`         | 带检查的相减       | 溢出、负值返回 `Duration::ZERO` |
| `checked_mul(rhs)`      | `Option<Duration>` | 带检查乘 `u32`     | 溢出返回 `None`                 |
| `saturating_mul(rhs)`   | `Duration`         | 带检查乘 `u32`     | 溢出返回 `Duration::MAX`        |
| `checked_div(rhs)`      | `Option<Duration>` | 带检查除以 `u32`   | `rhs == 0` 返回 `None`          |
| `as_secs_f64()`         | `f64`              | 浮点秒数           |                                 |
| `as_secs_f32()`         | `f32`              | 浮点秒数           |                                 |
| `mul_f64(rhs)`          | `Duration`         | 乘 `f64` 倍        | 溢出、负值 `panic`              |
| `mul_f32(rhs)`          | `Duration`         | 乘 `f32` 倍        | 溢出、负值 `panic`              |
| `div_f64(rhs)`          | `Duration`         | 除以 `f64`         | 溢出、负值 `panic`              |
| `div_f32(rhs)`          | `Duration`         | 除以 `f32`         | 溢出、负值 `panic`              |
| `div_duration_f64(rhs)` | `f64`              | 两 `Duration` 比值 |                                 |
| `div_duration_f32(rhs)` | `f32`              | 两 `Duration` 比值 |                                 |

> - `std::time::Duration`：<https://doc.rust-lang.org/stable/std/time/struct.Duration.html>

####    `time::Instant`

| `Instant` 方法                       | 返回值             | 描述                  | 其他                                   |
|--------------------------------------|--------------------|-----------------------|----------------------------------------|
| `Instant::now()`                     | `Instant`          | 现在                  |                                        |
| `duration_since(earlier)`            | `Duration`         | 与更早 `Instant` 差距 | 更晚 `Instant` 则返回 `Duration::Zero` |
| `checked_duration_since(earlier)`    | `Option<Duration>` | 与更早 `Instant` 差距 | 更晚 `Instant` 则返回 `None`           |
| `saturating_duration_since(earlier)` | `Duration`         | 与更早 `Instant` 差距 | 更晚 `Instant` 则返回 `Duration::Zero` |
| `elapsed()`                          | `Duration`         | 与现在差距            | 若晚于现在则返回 `Duration::Zero`      |
| `checked_add(duration)`              | `Option<Instant>`  | 后移时间              | 溢出返回 `None`                        |
| `checked_sub(duration)`              | `Option<Instant>`  | 前移时间              | 溢出返回 `None`                        |

-   `time::Instant` 是单增的时钟计数
    -   `Instant` 保证单增、但不保证稳定
        -   即计数增加的真实时间间隔可能不同
    -   `Instant` 只有相对值有比较意义，绝对值没有意义
        -   即，无法根据 `Instant` 获取秒数之类的绝对时间
        -   仅与 `time::Duration` 配合使用时才有意义
    -   `Instant` 的实际大小与平台相关

> - `std::time::Instant`：<https://doc.rust-lang.org/stable/std/time/struct.Instant.html>

####    `time::SystemTime`

| `SystemTime` 方法         | 返回值                              | 描述                              | 其他                   |
|---------------------------|-------------------------------------|-----------------------------------|------------------------|
| `SystemTime::UNIX_EPOCH`  | `SystemTime`                        | `1970-01-01 00:00:00 UTC`、0 时刻 | 常量                   |
| `SystemTime::MAX`         | `SystemTime`                        | 能表示的最大时刻                  | *Exp*、常量            |
| `SystemTime::MIN`         | `SystemTime`                        | 能表示的最小时刻                  | *Exp*、常量            |
| `SystemTime::now()`       | `SystemTime`                        | 当前时刻                          |                        |
| `duration_since(earlier)` | `Result<Duration, SystemTimeError>` | 与更早 `SystemTime` 差距          | 更晚时间则返回 `Err`   |
| `elapsed()`               | `Result<Duration, SystemTimeError>` | 与现在差距                        | 若晚于现在则返回 `Err` |
| `checked_add(duration)`   | `Option<SystemTime>`                | 后移时间                          | 溢出返回 `None`        |
| `checked_sub(duration)`   | `Option<SystemTime>`                | 前移时间                          | 溢出返回 `None`        |

> - `std::time::SystemTime`：<https://doc.rust-lang.org/stable/std/time/struct.SystemTime.html>

### `std::thread`

> - `std::thread`：<https://doc.rust-lang.org/stable/std/thread/index.html>

####    `thread::JoinHandle`

| `JoinHandle` 方法 | 返回值      | 描述                     | 其他                          |
|-------------------|-------------|--------------------------|-------------------------------|
| `thread()`        | `&Thread`   | 获取内部线程句柄         |                               |
| `is_finished()`   | `bool`      | 检查线程是否执行完毕     |                               |
| `join()`          | `Result<T>` | 阻塞直至关联线程执行完毕 | 关联线程 `panic` 则返回 `Err` |

-   `thread::JoinHandle` 代表可阻塞等待 `join` 线程执行的权限
    -   `JoinHandle` 丢弃时将与关联线程断联
        -   此时，无法触达线程、阻塞等待线程
    -   `JoinHandle` 常通过 `thread::spawn()`、`Builder::spawn()` 创建
    -   `JoinHandle` 未实现 `Clone`：`join` 线程的权限必须唯一（平台限制）

> - `std::thread::JoinHandle`：<https://doc.rust-lang.org/stable/std/thread/struct.JoinHandle.html>

####    `thread::Thread`

| `Thread` 方法           | 返回值         | 描述                      | 其他                               |
|-------------------------|----------------|---------------------------|------------------------------------|
| `unpark()`              | `()`           | 原子地让线程 *Token* 可用 | 配合 `thread::park()` 阻塞、唤醒   |
| `id()`                  | `ThreadId`     | 获取线程 Id               |                                    |
| `name()`                | `Option<&str>` | 获取线程名                |                                    |
| `into_raw()`            | `*const ()`    | 转换为 `*const ()`        | *Exp*                              |
| `Thread::from_raw(ptr)` | `Thread`       | 从 `*const ()` 转换为线程 | *Exp*、`unsafe` 需确保指向内存有效 |

-   `thread::Thread` 线程句柄
    -   `Thread` 对象常通过 `JoinHandle::thread()`、`thread::current()` 方式获取

> - `std::thread::Thread`：<https://doc.rust-lang.org/stable/std/thread/struct.Thread.html>
> - `std::thread::park`：<https://doc.rust-lang.org/stable/std/thread/fn.park.html>
> - Rust 多线程的高效等待术：`park()` 与 `unpark()` 信号通信实战：<https://paxonqiao.com/rust-thread-parking/>

####    `thread::Scope`、`thread::ScopedJoined`

| `Scope` 方法 | 返回值                  | 描述           | 其他 |
|--------------|-------------------------|----------------|------|
| `spawn(f)`   | `ScopedJoinHandle<'scope, T>` | 创建新局域线程 |      |

-   `thread::Scope` 创建局域线程的线程作用域
    -   `thread::Scope` 通过 `thread::scope(f)` 函数创建
        -   线程作用域内，通过 `Scope::spawn(f)` 创建 `thread::ScopedJoinedHandle<'scope, T>` 局域线程句柄
        -   `thread::ScopedJoinedHandle<'scope, T>` 局域线程句柄方法同 `thread::JinedHandle<T>`
    -   在创建 `Scope` 实例的 `thread::scope(f)` 函数返回前，未手动 `join` 的线程将被自动 `join`
        -   即，`Scope` 中创建局域线程保证在线程作用域结束前 `join`
        -   故，`Scope` 中局域线程可借用非 `'static` 数据
            -   只要，局域线程借用的数据的生命周期长于局域线程生命周期 `'env: 'scope`

```rust
pub fn scope<'env, F, T>(f: F) -> T
where
    f: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T {}      // HRTB，对所有生命周期 `'scope`

let mut a vec![1, 2, 3];
let mut x = 0;
let scope_ret = thread::scope(|s| {
    s.spawn(|| { println("Scoped thread 1"); });
    s.spawn(|| { println("Scoped thread 2"); x += a[0] + a[2]; });
    println!("Thread scope");
    1
});
a.push(4);                                                          // Scope 结束后可访问借用变量
```

> - `std::thread::Scope`：<https://doc.rust-lang.org/stable/std/thread/struct.Scope.html>
> - `std::thread::scope`：<https://doc.rust-lang.org/stable/std/thread/fn.scope.html>

####    `thread::Builder`

| `Builder` 方法     | 返回值                  | 描述                                             | 其他                                          |
|--------------------|-------------------------|--------------------------------------------------|-----------------------------------------------|
| `Builder::new()`   | `Builder`               | 按基线配置创建                                   |                                               |
| `name(name)`       | `Builder`               | 按 `String` 配置线程名称                         | 当前线程名称仅在 `panic` 信息中               |
| `stack_size(size)` | `Builder`               | 按 `u32` 配置线程栈字节数                        |                                               |
| `spawn(f)`         | `Result<JoinHandle<T>>` | 消耗，创建执行 `f() -> T` 的线程                 |                                               |
| `spawn_uncheck(f)` | `Result<JoinHandle<T>>` | 消耗，不检查生命周期地创建执行 `f() -> T` 的线程 | `unsafe` 需确保线程存活不超过闭包内、返回引用 |

-   `thread::Builder` 即线程工厂，用于配置新创建线程属性

> - `std::thread::Builder`：<https://doc.rust-lang.org/stable/std/thread/struct.Builder.html>

####    `thread::LocalKey`

| `LocalKey<T>` 方法 | 返回值                   | 描述                                 | 其他                         |
|--------------------|--------------------------|--------------------------------------|------------------------------|
| `with(f)`          | `R`                      | 按 `f(&T) -> R` 处理并返回           | 基础方法                     |
| `try_with(f)`      | `Result<R, AccessError>` | 按 `f(&T) -> R` 处理线程局部值并返回 | 若已销毁则返回 `AccessError` |

| `LocalKey<Cell<T>>` 方法 | 返回值 | 描述                                               | 其他                            |
|--------------------------|--------|----------------------------------------------------|---------------------------------|
| `set(value)`             | `()`   | 设置 `Cell<T>` 内部值                              | 不执行 `thread_local!` 内初始化 |
| `get()`                  | `T`    | Copy、返回 `Cell<T>` 内部值                        | `T: Copy`                       |
| `take()`                 | `T`    | 移出 `Cell<T>` 内部值，替换为 `Default::default()` | `T: Default`                    |
| `replace(value)`         | `T`    | 替换 `Cell<T>` 内部值                              |                                 |
| `update(f)`              | `T`    | 按 `f(T) -> T` 更新 `Cell<T>` 内部值               | `T: Copy`                       |

| `LocalKey<RefCell<T>>` 方法 | 返回值 | 描述                                                  | 其他                            |
|-----------------------------|--------|-------------------------------------------------------|---------------------------------|
| `with_borrow(f)`            | `R`    | 按 `f(&T) -> R` 处理并返回                            |                                 |
| `with_borrow_mut(f)`        | `R`    | 按 `f(&mut T) -> R` 处理并返回                        |                                 |
| `set(value)`                | `()`   | 设置 `RefCell<T>` 内部值                              | 不执行 `thread_local!` 内初始化 |
| `take()`                    | `T`    | 移出 `RefCell<T>` 内部值，替换为 `Default::default()` | `T: Default`                    |
| `replace(value)`            | `T`    | 替换 `RefCell<T>` 内部值                              |                                 |

-   `thread::LocalKey<T: 'static>`：拥有内部所有权的、静态生命周期的 *Thread Local Storage* 
    -   `LocalKey<T>` 代表各线程独立拥有的 **静态变量**
        -   即，在需要静态变量、且静态变量为各线程拥有独立副本的场合
            -   如：线程全局信息需多函数访问等
            -   否则，直接在线程内创建局部变量即可
        -   `LocalKey<T>` 有静态生命周期，在线程结束时才被销毁
            -   在 `<T as Drop>::drop` 中访问其他 *TLS* 可能 `panic`（*TLS* 销毁顺序不确定）
        -   `LocalKey<T>` 仅能获取 `&T` 内部数据的共享引用，故一般配合 `Cell`s 类型实现可变访问
            -   且，`LocalKey<RefCell<T>>` 实现有专门的方法穿透至 `Cell` 底层值
    -   `LocalKey<T>` 一般通过 `thread_local!` 创建
        -   `LocalKey<T>` 延迟初始化，仅在首次需要初始化值 `with(f)` 时才执行初始化
            -   可用 `const {}` 块作为初始化值避免延迟初始化，提升执行效率
            -   或，`LocalKey::set(value)` 跳过 `thread_local!` 中默认初始化

```rust
thread_local! {
    pub static FOO: Cell<u32> = const { Cell::new(1) };                 // 不延迟初始化
    static BAR: RefCell<Vec<f32>> = RefCell::new(vec![1.0, 2.0]);
    static NO_PANIC: RefCell<Vec<i32>> = panic!("!");                   // 初始化被后续 `set` 跳过
}
NO_PANIC.set(vec![1, 2, 3]);
```

> - `std::thread::LocalKey`：<https://doc.rust-lang.org/stable/std/thread/struct.LocalKey.html>
> - `std::thread_local`：<https://doc.rust-lang.org/stable/std/macro.thread_local.html>

####    `std::thread` 函数

| `std::thread` 函数        | 返回值                   | 描述                                 | 其他                     |
|---------------------------|--------------------------|--------------------------------------|--------------------------|
| `available_parallelism()` | `Result<NonZero<usize>>` | 估计的并行数量                       |                          |
| `current()`               | `Thread`                 | 获取当前线程句柄                     |                          |
| `panicking()`             | `bool`                   | 当前线程是否因 `panic` 展开          | 常用用于 `Drop::drop` 中 |
| `park()`                  | `bool`                   | 阻塞直至当前线程获取 *Token*         |                          |
| `park_timeout(dur)`       | `bool`                   | 阻塞直至当前线程获取 *Token*、或超时 |                          |
| `sleep(dur)`              | `()`                     | 睡眠至少 `dur` 时长                  |                          |
| `yield_now()`             | `()`                     | 主动挂起自身                         | 协作式放弃剩余时间片     |
| `spawn(f)`                | `JoinHandle<T>`          | 创建新线程                           |                          |
| `scope(f)`                | `T`                      | 创建线程作用域                       |                          |

-   `std::thread` 函数
    -   `thread::yield_now()` 将主动挂起自身，由内核调度其他线程使用剩余时间片
        -   但，此时可能没有其他 *Ready* 线程，导致时间片被浪费
        -   故，应尽量使用 `mpsc::channel`、`Condvar`、`Mutex` 同步机制
    -   `thread::park()`、`Thread::unpark()` 配合作为线程底层阻塞机制
        -   `thread::park()` 阻塞线程也可能被虚假唤醒，注意检查条件
        -   `Thread::unpark()` 调用将同步至 `thread::park()`
            -   即，`unpark()` 之前的内存操作对消耗 *Token*、从 `park()` 返回的线程可见
            -   从原子操作内存顺序角度，即 `unpark()` 执行 *Release* 操作、`park()` 执行 *Acquire* 操作

> - `std::thread` 函数：<https://doc.rust-lang.org/stable/std/thread/index.html#functions>
> - `std::thread::park`：<https://doc.rust-lang.org/stable/std/thread/fn.park.html>
> - Rust 多线程的高效等待术：`park()` 与 `unpark()` 信号通信实战：<https://paxonqiao.com/rust-thread-parking/>

### 线程共享、同步、锁

####    `sync::Mutex` 互斥锁

| `Mutex<T>` 方法  | 返回值                             | 描述                                   | 其他                            |
|------------------|------------------------------------|----------------------------------------|---------------------------------|
| `Mutex::new(t)`  | `Mutex<T>`                         | 创建互斥锁                             |                                 |
| `lock()`         | `LockResult<MutexGuard<'_, T>>`    | 阻塞当前线程直至获取锁                 | `MutexGuard` 离开作用域时锁释放 |
| `try_lock()`     | `TryLockResult<MutexGuard<'_, T>>` | 尝试获取锁，若当前无法获取则返回 `Err` |                                 |
| `get_mut()`      | `LockResult<&mut T>`               | 创建内部值可变引用                     |                                 |
| `into_inner()`   | `LockResult<T>`                    | 解包获取内部值                         |                                 |
| `get_cloned()`   | `Result<T, PoisonError<()>>`       | Clone 内部值                           | *Exp*                           |
| `set(value)`     | `Result<(), PoisonError<T>>`       | 设置内部值                             | *Exp*                           |
| `replace(value)` | `LockResult<T>`                    | 替换、返回内部值                       | *Exp*                           |
| `is_poisioned()` | `bool`                             | 锁是否中毒                             |                                 |
| `clear_poison()` | `()`                               | 清除锁中毒                             |                                 |
| `data_ptr()`     | `*mut T`                           | 创建内部值可变裸指针                   |                                 |

-   `sync::Mutex<T>` 互斥锁 *Mutual Exclusion*：只允许单个线程持有锁
    -   `Mutex<T>` 内部维护有 `Futex`（即 `Atomic<u32>`、`AtomicU32`）作为真正的锁
        -   内部 `Atomic<u32>` 取值即代表锁状态
        -   `Mutex<T>::lock()` 将自旋（`loop` 循环检查）以抢占式的加锁
    -   *Poisoning* 锁中毒：持有锁的线程 `panic` 将导致锁中毒
        -   中毒锁将无法通过方法正常获取、修改内部值，方法转而返回 `PoisionError<T>`
        -   但，锁中毒仅是建议性的，仍可以通过 `PoisionError::into_error()`、`get_mut()` 等访问获取、访问底层值
        -   注意：部分场合 `panic` 可能被跳过，故锁未中毒不代表持有锁的线程未 `panic`
    -   `sync::Mutex<T>` 通过内部 `Futex` 实现同时包含自旋、通知系统挂起
        -   首次尝试为内部加锁时，会 `while` 循环、自旋 100 次尝试直接加锁
        -   若加锁失败，则进入循环
            -   通知操作系统挂起、等待系统通知内部 `Fetux` 状态改变
            -   自旋 100 次尝试加锁
    -   一些说明
        -   `Mutex<T>` 实现 `Send`、`Sync`
        -   多线程共享 `Mutex<T>`：通过 `Arc<Mutex<T>>` 创建多个副本 `move` 进各线程
-   `sync::MutexGuard<'a, T>` 按 *RAII* 实现的作用域互斥锁：离开作用域时锁被释放
    -   `MutexGuard<'a, T>` 实现有 `DerefMut`，`&T`、`&mut T` 的方法可直接应用在 `MutexGuard` 上

```rust
pub type LockResult<T> = Result<T, PoisonError<T>>;
pub type TryLockResult<Gaurd> = Result<Gaurd, TryLockError<Guard>>;

let lock = Arc::new(Mutex::new(0_u32));
let lock2 = Arc::clone(&lock);                  // 多个线程间共享副本
```

> - `std::sync::Mutex`：<https://doc.rust-lang.org/stable/std/sync/struct.Mutex.html>
> - `std::sync::MutexGuard`：<https://doc.rust-lang.org/stable/std/sync/struct.MutexGuard.html>
> - `std::sync::PoisonError`：<https://doc.rust-lang.org/stable/std/sync/struct.PoisonError.html>
> - `std::sys::sync::mutex::futex.rs` 源码：<https://doc.rust-lang.org/stable/src/std/sys/sync/mutex/futex.rs.html>
> - `std::sys::pal::unix::futex.rs` 源码：<https://doc.rust-lang.org/stable/src/std/sys/pal/unix/futex.rs.html>
> - Rust 并发编程番外篇：`Mutex` 内部实现：<https://colobu.com/2023/11/05/inside-rust-mutex/>

####    `sync::RwLock` 读写锁

| `RwLock<T>` 方法 | 返回值                                   | 描述                                     | 其他                                |
|------------------|------------------------------------------|------------------------------------------|-------------------------------------|
| `RwLock::new(t)` | `RwLock<T>`                              | 创建读写锁                               |                                     |
| `read()`         | `LockResult<RwLockReadGuard<'_, T>>`     | 阻塞当前线程直至获取读锁                 | `RwLockReadLock` 离开作用域时锁释放 |
| `try_read()`     | `TryLockResult<RwLockReadGuard<'_, T>>`  | 尝试获取读锁，若当前无法获取则返回 `Err` |                                     |
| `write()`        | `LockResult<RwLockWriteGuard<'_, T>>`    | 阻塞当前线程直至获取写锁                 | `RwLockReadLock` 离开作用域时锁释放 |
| `try_write()`    | `TryLockResult<RwLockWriteGuard<'_, T>>` | 尝试获取写锁，若当前无法获取则返回 `Err` |                                     |
| `get_mut()`      | `LockResult<&mut T>`                     | 创建内部值可变引用                       |                                     |
| `into_inner()`   | `LockResult<T>`                          | 解包获取内部值                           |                                     |
| `get_cloned()`   | `Result<T, PoisonError<()>>`             | Clone 内部值                             | *Exp*                               |
| `set(value)`     | `Result<(), PoisonError<T>>`             | 设置内部值                               | *Exp*                               |
| `replace(value)` | `LockResult<T>`                          | 替换、返回内部值                         | *Exp*                               |
| `is_poisioned()` | `bool`                                   | 锁是否中毒                               |                                     |
| `clear_poison()` | `()`                                     | 清除锁中毒                               |                                     |
| `data_ptr()`     | `*mut T`                                 | 创建内部值可变裸指针                     |                                     |

-   `sync::RwLock` 读写锁：允许复数个线程持有读锁、或单个线程持有写锁
    -   持有读锁仅能读取底层数据，持有写锁仅能修改底层数据
    -   `RwLock` 不保证读写读写优先顺序，具体实现依赖与操作系统环境
    -   `RwLock` 同样会中毒，但仅会在持有写锁的线程 `panic` 时中毒
-   `sync::RwLockReadLock<'rwlock, T>` 按 *RAII* 实现的作用域读锁：离开作用域时锁被释放
    -   `RwLockReadGaurd<'rlwock, T>` 实现有 `Deref`，`&T` 的方法可直接应用在 `RwLockReadGuard` 上
-   `sync::RwLockWriteLock<'rwlock, T>` 按 *RAII* 实现的作用域写锁：离开作用域时锁被释放
    -   `RwLockWriteGaurd<'rwlock, T>` 实现有 `DerefMut`，`&mut T` 的方法可直接应用在 `RwLockWriteGuard` 上
    -   `RwLockWriteGaurd<'rwlock, T>::downgrade()` 可转换为对应的 `RwLockReadGuard`

> - `std::sync::RwLock`：<https://doc.rust-lang.org/stable/std/sync/struct.RwLock.html>
> - `std::sync::RwLockReadGuard`：<https://doc.rust-lang.org/stable/std/sync/struct.RwLockReadGuard.html>
> - `std::sync::RwLockWriteGuard`：<https://doc.rust-lang.org/stable/std/sync/struct.RwLockWriteGuard.html>

####    `sync::Condvar` 条件量 

| `Condvar` 方法                              | 返回值                                            | 描述                                                           | 其他 |
|---------------------------------------------|---------------------------------------------------|----------------------------------------------------------------|------|
| `Condvar::new()`                            | `Condvar`                                         | 创建条件量                                                     |      |
| `wait(guard)`                               | `LockResult<MutexGuard<'a, T>>`                   | 暂时释放 `MutexGaurd<T>`、阻塞线程直到收到通知再次获取锁       |      |
| `wait_while(guard, condition)`              | `LockResult<MutexGuard<'a, T>>`                   | 检查 `condition(&mut T)`，`true` 则 `wait` 直至满足条件        |      |
| `wait_timeout(guard, dur)`                  | `LockResult<(MutexGuard<'a, T>, WaitTimeResult)>` | 暂时释放 `MutexGaurd<T>`、阻塞线程直到收到通知或超时再次获取锁 |      |
| `wait_timeout_while(guard, dur, condition)` | `LockResult<(MutexGuard<'a, T>, WaitTimeResult)>` | 检查 `condition(&mut T)`，`true` 则 `wait` 直至满足条件或超时  |      |
| `notify_one()`                              | `()`                                              | 通知一个被此条件量阻塞的线程                                   |      |
| `notify_all()`                              | `()`                                              | 通知全部被此条件量阻塞的线程                                   |      |

-   `sync::Condvar` 条件量：阻塞线程直至收到信号、满足某个条件
    -   `Condvar` 需与 `MutexGuard<T>` 成对使用
        -   `MutexGaurd<T>`（中 `T`）作为 `Condvar` 需等待的条件
        -   `Condvar` 配合复数个 `MutexGuard<T>` 可能导致运行时 `panic`
    -   为避免线程被虚假唤醒，在条件未满足情况下继续执行
        -   可、应在 `while` 循环中 `Condvar::wait(guard)` 并手动检查条件
        -   或，使用 `Condvar::wait_while(guard, condition)`
        -   注意：`Condvar::wait_while()`、`Condvar::wait_time_while()` 被调用时将先检查条件
            -   若满足条件，线程将继续执行，而不会释放 `MutexGaurd<T>`、阻塞
    -   `Condvar` 同样维护内部 `Futex`，但实现中没有类似 `Mutex` 的自旋机制
        -   不满足条件时，`Condvar` 直接通知操作系统挂起、等待系统通知内部 `Futex` 状态改变

> - `std::sync::CondVar`：<https://doc.rust-lang.org/stable/std/sync/struct.RwLock.html>
> - `std::sys::sync::condvar::futex.rs` 源码：<https://doc.rust-lang.org/stable/src/std/sys/sync/condvar/futex.rs.html>
> - Rust 并发加速器：用 `Condvar` 实现线程间“精确握手”与高效等待：<https://paxonqiao.com/rust-condvar/>

####    `sync::Barrier` 障碍

| `Barrier` 方法    | 返回值              | 描述                            | 其他 |
|-------------------|---------------------|---------------------------------|------|
| `Barrier::new(n)` | `Barrier`           | 创建可阻塞 `usize` 数线程的障碍 |      |
| `wait()`          | `BarrierWaitResult` | 阻塞直至所有线程到达            |      |

-   `sync::Barrier` 障碍用于同步复数个线程计算行为
    -   `Barrier` 可复用：所有线程到达障碍处，`Barrier` 可复用

> - `std::sync::Barrier`：<https://doc.rust-lang.org/stable/std/sync/struct.RwLock.html>

####    `sync::mpsc`

| `mpsc` 函数           | 返回值                         | 描述               | 其他 |
|-----------------------|--------------------------------|--------------------|------|
| `channel()`           | `(Sender<T>, Receiver<T>)`     | 异步通道           |      |
| `sync_channel(bound)` | `(SyncSender<T>, Receiver<T>)` | 指定容量的同步通道 |      |

-   `sync::mpsc` *Multi-Producer, Single-Comsumer* 多生产者、单消费者的 *FIFO* 队列通信原语
    -   队列通信、通道
        -   `mpsc::Sender`、`mpsc::SyncSender` 发送者、生产者：发送数据值至通道
            -   发送数据按其发送顺序在通道中有序
            -   `Sender`、`SyncSender` 可 `Clone` 创建多个生产者，即多生产者
        -   `mpsc::Receiver` 接收者、消费者：从通道获取数据
            -   `Receiver` 不可 `Clone`，即单消费者
    -   异步（通信队列）通道
        -   由 `mpsc::channel()` 创建（返回发送者、接收者）
        -   异步通道容量无限，`Sender::send()` 总不会阻塞所属线程
        -   而无法从通道中获取数据时，`Receiver::recv()` 将阻塞所属线程
    -   同步（通信队列）通道
        -   由 `mpsc::channel(bound)` 创建（返回发送者、接收者）
        -   同步通道包含缓冲区、容量有限，其中信息将排队
            -   缓冲区满时，`Sender::send()` 将阻塞所在线程
            -   特别的，容量 `0` 时，`send` 将在 `recv` 接收之后返回
        -   同样的，无法从通道中获取数据时，`Receiver::recv()` 将阻塞所属线程

| `mpsc::Sender` 方法 | 返回值                     | 描述     | 其他 |
|---------------------|----------------------------|----------|------|
| `send(t)`           | `Result<(), SendError<T>>` | 发送数据 |      |

| `mpsc::SyncSender` 方法 | 返回值                        | 描述     | 其他                     |
|-------------------------|-------------------------------|----------|--------------------------|
| `send(t)`               | `Result<(), SendError<T>>`    | 发送数据 | 通道断开时返回 `Err`     |
| `try_send(t)`           | `Result<(), TrySendError<T>>` | 不阻塞的发送数据 | 通道断开、慢时返回 `Err` |

| `mpsc::Receiver` 方法   | 返回值                        | 描述                 | 其他                   |
|-------------------------|-------------------------------|----------------------|------------------------|
| `recv()`                | `Result<T, RecvError>`        | 接收数据             |                        |
| `try_recv()`            | `Result<T, TryRecvError>`     | 不阻塞的接收数据     | 通道为空时返回 `Err`   |
| `recv_timeout(timeout)` | `Result<T, RecvTimeoutError>` | 指定超时、接收数据   |                        |
| `iter()`                | `Iter<'_, T>`                 | 迭代返回数据         | 通道断开时结束迭代     |
| `try_iter()`            | `TryIter<'_, T>`              | 不阻塞的迭代返回数据 | 通道断开、空时结束迭代 |

> - `std::sync::mpsc`：<https://doc.rust-lang.org/stable/std/sync/mpsc/index.html>

####    `sync::atomic` 原子类型

```rust
pub type Atomic<T> = <T as AtomicPrimitive>::AtomicInner;
```

-   原子类型包含一系列用于线程间共享内存通信的基本类型，用作其他并发类型的底座
    -   原子类型包括一系列 `AutomicBool`、`AotomicIsize`、`AutomicU16` 等，以及 `Atomic<T>` 别名
        -   不同原子类型方法类似
        -   但因系统架构原因，部分原子类型、原子操作方法可能不可用
    -   原子类型实现 `Sync` 可在多线程间安全共享
        -   考虑到仅 `Atomic::get_mut()` 需可变借用，包括 `store`、`swap` 在内的方法均只需共享借用
        -   `move Arc<Atomic<>>`、或直接借用均可在多线程间同步、通信

#####   `sync::AtomicIsize`

| `AtomicIsize` 方法                 | 返回值               | 描述                     | 其他               |
|------------------------------------|----------------------|--------------------------|--------------------|
| `AtomicIsize::new(v)`              | `AtomicIsize`        | 创建                     |                    |
| `AtomicIsize::from_ptr(ptr)`       | `&'a AtomicIsize`    | 从 `*mut isize` 创建     |                    |
| `AtomicIsize::from_mut(v)`         | `&mut AtomicIsize`   | 创建原子访问             | *Exp*              |
| `AtomicIsize::get_mut_slice(this)` | `&mut [isize]`       | 创建切片元素的非原子访问 | *Exp*              |
| `AtomicIsize::from_mut_slice(v)`   | `&mut [AtomicIsize]` | 创建切片元素的原子访问   | *Exp*              |
| `into_inner(self)`                 | `isize`              | 转换为 `isize`           |                    |
| `get_mut()`                        | `&mut isize`         | 创建底层值可变引用       | 仅此方法需可变借用 |
| `as_ptr()`                         | `*mut isize`         | 创建可变裸指针           |                    |

-   `AtomicIsize` 方法说明
    -   `AtomicIsize` 方法往往需要指定内存顺序约束
        -   `load` 方法只读，不允许 `Release`、`AcqRel` 约束
        -   `store` 方法只写，不允许 `Acquire`、`AcqRel` 约束
        -   `fetch_`、`swap` 方法同时包含读写，不同内存顺序约束有不同同步结果
        -   `compare_exchange` 是 *Compare-And-Swap* 操作，需要指定不同情况的内存顺序约束
            -   事实上，方法内部会直接根据 `success`、`failure` 综合确定整体的约束
            -   毕竟，比较成功与否总在读取之后，`success`、`failure` 约束均会描述读约束
            -   同样的，`failure` 不允许 `Release`、`AcqRel` 约束
            -   `compare_exchange_weak` 可能有虚假失败，即原值与 `current` 一致也不更新，以提升效率
        -   `fetch_update` 类似 `compare_exchange` 区分更新成功 `set_order`、失败 `fetch_order`

| `AtomicIsize` 原子操作                                  | 返回值                 | 描述                                          | 其他                     |
|---------------------------------------------------------|------------------------|-----------------------------------------------|--------------------------|
| `load(order)`                                           | `isize`                | 按 `Ordering` 约束载入                        |                          |
| `store(val, order)`                                     | `isize`                | 按 `Ordering` 约束存储                        |                          |
| `swap(val, order)`                                      | `isize`                | 按 `Ordering` 约束交换（读取、存储）          |                          |
| `fetch_add(val, order)`                                 | `isize`                | 加、返回原值                                  |                          |
| `fetch_sub(val, order)`                                 | `isize`                | 减、返回原值                                  |                          |
| `fetch_add(val, order)`                                 | `isize`                | 按位与、返回原值                              |                          |
| `fetch_nand(val, order)`                                | `isize`                | 按位与否、返回原值                            |                          |
| `fetch_or(val, order)`                                  | `isize`                | 按位和、返回原值                              |                          |
| `fetch_xor(val, order)`                                 | `isize`                | 按位异或                                      |                          |
| `fetch_max(val, order)`                                 | `isize`                | 较大者                                        |                          |
| `fetch_max(val, order)`                                 | `isize`                | 较小者                                        |                          |
| `fetch_update(set_order, fetch_order, f)`               | `Result<isize, isize>` | 按 `FnMut(isize) -> Option<isize>` 更新       | 可能执行多次但只生效一次 |
| `compare_exchange(current, new, success, failure)`      | `Result<isize, isize>` | 现值与 `current` 相同则存储 `new`，总返回旧值 |                          |
| `compare_exchange_weak(current, new, success, failure)` | `Result<isize, isize>` | 现值与 `current` 相同则存储 `new`，总返回旧值 |  相同也可能不更新值       |

> - `std::sync::atomic`：<https://doc.rust-lang.org/stable/std/sync/atomic/index.html>
> - `std::sync::atomic::AutomicIsize`：<https://doc.rust-lang.org/stable/std/sync/atomic/struct.AtomicIsize.html#method.store>

### 异步任务

####    `task::Context`、`task::Waker`

| `task::Context<'a>` 方法     | 返回值        | 描述         | 其他 |
|------------------------------|---------------|--------------|------|
| `Context::from_waker(waker)` | `Context<'a>` | 从唤醒器创建 |      |
| `waker()`                    | `&'a Waker`   | 获取唤醒器   |      |

| `task::Waker` 方法         | 返回值                    | 描述                       | 其他                                |
|----------------------------|---------------------------|----------------------------|-------------------------------------|
| `waker()`                  | `()`                      | 消耗、唤醒关联任务         |                                     |
| `waker_by_ref()`           | `()`                      | 唤醒关联任务               |                                     |
| `will_wake(other)`         | `()`                      | 是否两个唤醒器关联同一任务 |                                     |
| `data()`                   | `*const ()`               | 获取数据指针               |                                     |
| `vtable()`                 | `&'static RawWakerVTable` | 获取虚表指针               |                                     |
| `Waker::new(data, vtable)` | `Waker`                   | 创建唤醒器                 | `unsafe` 需确保虚表适配、数据合法   |
| `Waker::from_raw(waker)`   | `Waker`                   | 从 `RawWaker` 创建         | `unsafe` 需确保 `RawWaker` 遵守协议 |
| `Waker::noop()`            | `&'statci Waker`          | 返回无操作 `&Waker`        |                                     |

-   `std::task` 异步任务模块中仅包含用于规范的公共类型、特性，异步运行时、异步任务等依赖额外框架
    -   `task::RawWakerVTable` 指示 `RawWaker` 行为的虚函数指针表
    -   `task::RawWaker` 包含数据裸指针、虚表指针的 “宽指针”
    -   `task::Waker` 唤醒器：任务完成时，通知异步执行器任务已完成、可恢复执行
        -   “通知异步执行器” 常见即：将当前任务加入就绪队列（等待恢复执行）
        -   特别的 `Waker` 实现有 `From<Arc<W: impl Wake>>`，可从通过定义类型 `W: impl Wake` 快速创建
    -   `task::Context` 异步上下文：作为 `Future::poll()` 实参被传递
        -   当前，仅用于获取 `&Waker`

> - `std::task`：<https://doc.rust-lang.org/stable/std/task/index.html>

####    `task::Poll`

```rust
pub enum Poll<T> {
    Ready(T),
    Pending,
}
```

| `task::Poll<T>` 方法 | 返回值    | 描述                              | 其他 |
|----------------------|-----------|-----------------------------------|------|
| `map(f)`             | `Poll<U>` | 对 `Ready(T)` 按 `f(T) -> U` 映射 |      |
| `is_ready()`         | `bool`    | 是否为 `Ready(T)`                 |      |
| `is_pending()`       | `bool`    | 是否为 `Pending`                  |      |

-    `task::Poll` 指示当前任务结果就绪、或等待被异步执行器调度唤醒
    -   `task::Poll` 常由 `Future::poll()` 方法返回

| `task::Poll<Result<T, E>>` 方法 | 返回值               | 描述                                  | 其他 |
|---------------------------------|----------------------|---------------------------------------|------|
| `map_ok(f)`                     | `Poll<Result<U, E>>` | 对 `Read(Ok(T))` 按 `f(T) -> U` 映射  |      |
| `map_err(f)`                    | `Poll<Result<T, U>>` | 对 `Read(Err(E))` 按 `f(E) -> U` 映射 |      |

| `task::Poll<Option<Result<T, E>>>` 方法 | 返回值                       | 描述                                        | 其他 |
|-----------------------------------------|------------------------------|---------------------------------------------|------|
| `map_ok(f)`                             | `Poll<Option<Result<U, E>>>` | 对 `Read(Some(Ok(T)))` 按 `f(T) -> U` 映射  |      |
| `map_err(f)`                            | `Poll<Option<Result<T, U>>>` | 对 `Read(Some(Err(E)))` 按 `f(E) -> U` 映射 |      |

> - `std::task::Poll`：<https://doc.rust-lang.org/stable/std/task/enum.Poll.html>

####    `std::future` 函数

| `std::future` 函数         | 返回值       | 描述                                                     | 其他 |
|----------------------------|--------------|----------------------------------------------------------|------|
| `future::pending<T>()`     | `Pending<T>` | 创建永远在计算的 `impl Future`                           |      |
| `future::ready<T>()`       | `Ready<T>`   | 创建已就绪的 `impl Future`                               |      |
| `future::poll_fn<T, T>(f)` | `PollFn<F>`  | 将 `f(&mut Context<'_>) -> Poll<T>` 转换为 `impl Future` |      |

> - `std::future` 函数：<https://doc.rust-lang.org/stable/std/future/index.html#functions>
