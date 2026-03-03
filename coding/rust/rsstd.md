---
title: 
categories:
  - Coding
  - Rust
tags:
  - Coding
  - Rust
date: 2026-01-25 16:18:02
updated: 2026-03-02 17:01:22
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

##  常用 *Traits*

| 常用 *Traits*                         | 说明                   | *Prelude* | 其他        |
|---------------------------------------|------------------------|-----------|-------------|
| `convert::AsRef<T>`                   | 转换为引用             | 2018      |             |
| `convert::AsMut<T>`                   | 转换为可变引用         | 2018      |             |
| `convert::From<T>`                    | 从消耗并转换           | 2018      |             |
| `convert::Into<T>`                    | 转换为                 | 2018      |             |
| `convert::TryFrom<T>`                 | 可能失败的 `From`      | 2021      |             |
| `convert::TryInto<T>`                 | 可能失败的 `Into`      | 2021      |             |
| `borrow::Borrow<Borrowed>`            | 借用为                 |           |             |
| `borrow::BorrowMut<Borrowed>`         | 可变借用为             |           |             |
| `borrow::ToOwned`                     | 转换为所有权值         | 2018      |             |
| `clone::Clone`                        | 显式拷贝               | 2018      | `#[derive]` |
| `cmp::PartialEq<Rhs=Self>`            | 不完整等价             | 2018      | `#[derive]` |
| `cmp::Eq:PartialEq`                   | 等价关系               | 2018      | `#[derive]` |
| `cmp::PartialOrd<Rhs=Self>`           | 偏序关系               | 2018      | `#[derive]` |
| `cmp::Ord:Eq + PartialOrd<Rhs=Self>`  | 全序关系               | 2018      | `#[derive]` |
| `default::Default`                    | 默认值                 | 2018      | `#[derive]` |
| `iter::Iterator`                      | 迭代器                 | 2018      |             |
| `iter::IntoIterator`                  | 可转化为迭代器         | 2018      |             |
| `iter::Extend<A>`                     | 从迭代器扩展           | 2018      |             |
| `iter::FromIterator<A>`               | 从迭代器转化           | 2021      |             |
| `iter::DoubleEndedIterator: Iterator` | 双头迭代器             | 2018      |             |
| `iter::ExactSizeIterator: Iterator`   | 长度确定迭代器         | 2018      |             |
| `ops::FnOnce<Args>`                   | 仅可调用一次闭包       | 2018      |             |
| `ops::FnMut<Args>: FnOnce<Args>`      | 调用后闭包自身状态改变 | 2018      |             |
| `ops::Fn<Args>: FnMut<Args>`          | 调用后闭包自身状态不变 | 2018      |             |
| `ops::Deref`                          | 引用类型强制转换       | 2018      |             |

> - `std::prelude`：<https://rustwiki.org/zh-CN/std/prelude/index.html>

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

#####   （适配器）转换

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





