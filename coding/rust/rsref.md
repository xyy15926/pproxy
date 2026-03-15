---
title: Rust Ref
categories:
  - Coding
  - Rust
tags:
  - Coding
  - Rust
date: 2025-12-22 10:38:52
updated: 2026-03-14 22:26:42
toc: true
mathjax: true
description: 
---

##  变量、语法

### 变量

```rust
let x: u32 = 5;                     // 不可变变量，显示指明类型
let mut y = 4;                      // 可变变量，编译器自动推断类型
let later;                          // 仅声明，延迟初始化
y = y + 1;
{
    let x = x + 2;                  // 变量遮蔽
    let x = "five";                 // 变量遮蔽可以赋予不同类型
    later = 5;
}
assert_eq!(x, 5);                   // 变量遮蔽直到再次被遮蔽、作用域结束为止
const SECS_IN_A_DAY = 3600 * 24;    // 常量
assert_eq!(later, 5);               // `later` 作用域到此为止
```

-   *Rust* 中变量默认不可变、拥有绑定在其上的值的所有权
    -   `let` 关键字声明变量、为变量（初始化）赋值
        -   可变性：变量默认不可变，不可变变量不可二次赋值
            -   `mut` 关键字指定 **变量可变**（**可变性是变量而不是值的属性**）
            -   **可变性是针对变量的，不是针对值的**
            -   不可变变量值赋值（移动、复制）给可变变量后，值可改变
        -   类型：编译器会尝试推断变量类型，大部分情况下无法手动 `:` 显式指明变量类型
            -   可用 `_` 表示 “不关心” （部分）类型、泛型、声明周期，由编译器推断
        -   遮蔽：同名变量可用 `let` 遮蔽、再次赋值，再次赋值类型可不同
        -   `let` 声明变量和初始化可以分开，可用于调整变量作用域
    -   `const` 关键字声明常量
        -   常量必须注解值类型
        -   常量可在任何作用域中声明，包括全局作用域
        -   常量在程序整个运行期间、在其声明的作用域内有效
        -   常量只能设置为常量表达式、字面量

> - 3.1 变量与可变性：<https://www.rust-book-cn.com/ch03-01-variables-and-mutability.html>

####    静态（全局）变量

```rust
static HALLO: &str = "Hello, world";                // 静态引用变量生命周期总是 `'static`
static mut COUNTER: u32 = 0;
unsafe {
    println!("{}", *(&raw const COUNTER));          // 通过裸指针访问可变静态变量
}
```

-   全局变量、静态变量 `static`：值在内存中有固定地址的变量
    -   静态变量内存地址固定
        -   静态引用变量只能存储具有 `'static` 声明周期的引用
            -   即，编译器总是可以推断此静态引用变量的生命周期，无需显式注解
        -   而，常量 `const` 虽然在程序整个生命周期有效
            -   但，编译器会尽可能将常量内联在代码中
            -   即，同一常量可能对应不同值，即值（数据）被多次复制、内存地址不同
    -   静态变量在整个程序全局只有一个，可能被多个线程访问，必须为 `Sync` 类型
        -   对不可变的静态变量的访问是安全的
        -   而，对可变的静态变量的访问、修改是不安全的，存在竞争风险
            -   编译器不允许创建对可变静态变量的引用，只能通过显式创建裸指针、并解引用访问（包括引用被隐式创建时）

> - 20.1 不安全的 Rust - 访问或修改可变的静态变量：<https://www.rust-book-cn.com/ch20-01-unsafe-rust.html>
> - Rust 中 `const` 常量与 `static` 变量：<https://zhuanlan.zhihu.com/p/110107918>

### 数据类型

####    *Primitive Types*

| 标量数据类型     | 描述                                                    |
|------------------|---------------------------------------------------------|
| `i8`、`u8`       | 整形：8bit 有符号、无符号                               |
| `i16`、`u16`     | 整形：16bit 有符号、无符号                              |
| `i32`、`u32`     | 整形：32bit 有符号（编译器推断默认）、无符号            |
| `i64`、`u64`     | 整形：64bit 有符号、无符号                              |
| `i128`、`u128`   | 整形：128bit 有符号、无符号                             |
| `isize`、`usize` | 整形：*arch* 平台相关有符号、无符号，常用作集合类型索引 |
| `f32`            | 浮点：32bit 浮点                                        |
| `f64`            | 浮点：64bit 浮点（编译器推断默认浮点）                  |
| `bool`           | 布尔：取 `true`、`false`                                |
| `char`           | *Unicode* 字符标量（*utf-8* 编码）                      |

-   数据类型
    -   整形、浮点类型字面量支持
        -   前缀：指定字面量进制
            -   `0x` 十六进制
            -   `0o` 八进制
            -   `0b` 二进制
            -   `b''` 字节：默认 `u8` 类型
        -   后缀：指定数据类型
        -   字面量内可用 `_` 分隔、提升可读性
    -   元组 `(...)` ：固定顺序、多种类型组合而成的复合类型
        -   数据类型 `(TYPE1, ...)`：按顺序给出的成员类型
        -   成员访问：通过模式匹配解构元组、索引访问获取元组成员
        -   `()` 单元：没有任何值的元组
            -   值、类型均为 `()`
            -   表示空值、空返回类型：不返回值表达式隐式返回 *单元* `()`
    -   数组 `[T; N]`：固定长度、单一类型
        -   数据类型 `[TYPE; LEN]`：成员类型、长度

```rust
let tup3: (i32, f64, u8) = (500, 6.4, 1);   // 显式指定元组类型
let (x, y, z) = tup3;                       // 模式匹配解构元组
let x = tup3.0;                             // 索引直接访问

let arr: [i33; 5] = [1, 2, 3, 4, 5];        // 显式指定数组类型
let fst = arr[0];                           // 索引直接访问
let arr = [1; 5];                           // 设置所有值相同
```

> - 3.2 数据类型：<https://www.rust-book-cn.com/ch03-02-data-types.html>
> - Primitive Types：<https://rustwiki.org/zh-CN/std/#primitives>

#####   切片

```rust
let pointer_size = size_of::<&u8>();
let slice_size = size_of::<&[u8]>();
assert_eq!（2 * pointer_size, slice_size);  // 切片类型大小是引用类型的 2 倍
```

-   *Slice* 切片 `[T]`：连续元素序列（即内存块）、动态大小的视图
    -   切片一般通过指针类型使用
        -   `&[T]` 共享切片
        -   `&mut [T]` 可变切片
        -   `Box<[T]>` *Boxed* 切片
    -   注意
        -   切片 `&[T]`、`&mut [T]` 包含地址、长度 2 字段，是普通引用类型大小的 2 倍
        -   `str` 内存表示与 `[u8]` 相同，但 *Rust* 额外要求其为有效的 *utf8* 序列

> - 10.1.7 Type System - Types - Slice Types：<https://doc.rust-lang.org/stable/reference/types/slice.html>
> - 10.1.3 Type System - Types - Textual Types：<https://doc.rust-lang.org/stable/reference/types/textual.html>
> - Primitive Type `slice`：<https://doc.rust-lang.org/stable/std/primitive.slice.html>
> - 4.3 切片类型：<https://www.rust-book-cn.com/ch04-03-slices.html>
> - 7.1 切片和切片引用：<https://course.rs/difficulties/slice.html>

####    标准库集合

| 集合类型                          | 说明               |
|-----------------------------------|--------------------|
| `Vec<T>`                          | （可变）向量       |
| `String`                          | *UTF-8* 编码字符串 |
| `std::collections::HashMap<K, V>` | 哈希映射           |

> - 8.1 使用向量存储值列表：<https://www.rust-book-cn.com/ch08-01-vectors.html>
> - 8.2 使用字符串存储 *UTF-8* 编码的文本：<https://www.rust-book-cn.com/ch08-02-strings.html>
> - 8.3 在哈希映射中存储键即及其关联值：<https://www.rust-book-cn.com/ch08-03-hash-maps.html>

####    高级类型

```rust
type Thunk = Box<dyn Fn() + Send + 'static>;    // 声明类型别名以减少重复
type Result<T> = std::result::Result<T, std::io::Error>;
```

-   高级类型
    -   *NewType* 模式：使用元组结构体包装已实现类型，并在新类型上实现已定义 `trait`
        -   通过 *NewType* 模式包装已有类型以绕过孤儿规则
            -   包装类型在编译器将被省略，没有性能损失
        -   但，*NewType* 包装类型没有原类型方法，可
            -   手动实现所需方法，在方法内委托给 `self.0`
            -   为包装类型实现 `Deref trait`
    -   `type` 声明类型别名
        -   主要用于为冗长类型创建别名，减少重复
    -   `!` *Never 类型*：代表永远不会返回的函数的返回值类型
        -   `!` 类型可以被转换为其他任何类型
        -   故，若线程闭包永不返回值，则 `thread::JobHandle<T>` 中泛型参数 `T` 取任意类型均可编译通过
    -   动态大小类型
        -   `Sized trait`：标记特性，指示类型大小在编译时已知
            -   *Rust* 自动为所有在编译时大小已知的类型添加 `Sized` 标记
            -   *Rust* 隐式、默认为每个泛型参数添加了 `Sized` 约束
        -   `?Sized trait`：标记特性，指示类型大小在编译时已知或未知

> - 20.2 高级特性：<https://www.rust-book-cn.com/ch20-02-advanced-traits.html>
> - 20.3 高级类型：<https://www.rust-book-cn.com/ch20-03-advanced-types.html>

###    类型转换

> - 1.5 理解类型转换：<https://yingang.github.io/effective-rust-cn/chapter_1/item5-casts.html>

####    *Type Coercions* 强制转换

```rust
trait Trait {}
fn foo<X: Trait>(t: X) {}
impl<'a> Trait for &'a i32 {}
fn main() {
    let t: &mut i32 = &mut 0;
    foo(t);                                 // `t: &mut i32` 不会强制转换为 `&i32`
}
```

| 起始类型         | 强制转换目标       | 说明                                                |
|------------------|--------------------|-----------------------------------------------------|
| `T`              | `U`                | （生命周期）子类型 `T:U`                            |
| `&mut T`         | `&T`               |                                                     |
| `*mut T`         | `*const T`         |                                                     |
| `&T`             | `*const T`         |                                                     |
| `&mut T`         | `*mut T`           |                                                     |
| `&T`、`&mut T`   | `&U`               | `T` 实现有 `Deref<Target=U>`                        |
| `&mut T`         | `&mut U`           | `T` 实现有 `DerefMut<Target=U>`                     |
| 函数项           | `fn` 指针          |                                                     |
| 不捕获环境的闭包 | `fn` 指针          |                                                     |
| `!`              | 任何类型           |                                                     |
| 定长类型指针     | 对应非定长类型指针 | 指针类型包括 `&`、`&mut`、`*const`、`*mut`、`Box<>` |

-   *Coercions* （类型）强制转换：在某些上下文中，类型可以被隐式的转换为其他类型
    -   强制转换通常是 **类型弱化**，主要是引用、生命周期的变化
        -   “类型弱化” 一般仅是简化过多显式类型转换
        -   强制转换的传递性目前并未完全支持：`T_1` 强制转换为 `T_2`，`T_2` 强制转换为 `T_3`，则 `T_1` 强制转换为 `T_3`
    -   但，强制转换不会 **被应用以满足 *Trait Bound* **（除方法中 `self` 自动类型转换）
        -   即，若函数参数类型限制为泛型 `&T`，则 `&mut T` 类型实参不会强制转换为 `&T`
    -   强制转换几乎发生在所需可能需要类型转换的地方
        -   `let` 赋值语句、`static` 声明、`const` 声明
        -   结构体、`union`、枚举变体初始化
        -   函数参数、返回值
        -   数组、元组、`()` 包裹的表达式、`{}` 代码块中子表达式
    -   *Unsized Coercions* 非定长类型（的指针）强制转换
        -   标准库借助 `T: Unsize<U>` 标记、实现定长类型 `T` 转换为非定长类型 `U`

| 定长类型指针       | 对应非定长类型指针 | 指针类型包括 `&`、`&mut`、`*const`、`*mut`、`Box<>` |
|--------------------|--------------------|-----------------------------------------------------|
| `&[T;n]` 数组      | `&[T]` 切片        | 定长类型指针转换为非定长指针：切片                  |
| `T: U`             | `dyn U`            | 定长类型指针转换为非定长指针：*Trait Bound*         |
| `dyn {T: U}`       | `dyn U`            | 定长类型指针转换为非定长指针：*Super Trait*         |
| 末尾字段类型为以上 | 末尾字段为以上     |                                                     |

> - 4.1 强制转换：<https://doc.rust-lang.net.cn/nomicon/coercions.html>
> - 4.1 Coercions：<https://doc.rust-lang.org/nomicon/coercions.html>
> - 10.7 Type system - Type coercions：<https://doc.rust-lang.org/stable/reference/type-coercions.html>
> - 10.7 Type system - Type coercions - Coercions sites：<https://doc.rust-lang.org/stable/reference/type-coercions.html#r-coerce.site>
> - 10.7 类型转换 - 可强制转换的类型：<https://doc.rust-lang.net.cn/reference/type-coercions.html#coercion-types>
> - `impl Trait for Type` 和 `impl Trait for &Type` 是什么关系？：<https://rustcc.cn/article?id=d993e943-64df-4252-9467-155b2a43a9d5>

####    *Type Cast* 类型转换

-   *Cast* （类型）转换 `EXPR as TYPE`
    -   类型转换是强制转换超集，所有强制转换都可以通过转换显式完成
    -   （类型）转换不是 `unsafe` 的（通常不违反内存安全）
        -   转换很危险但不会执行失败，但可能出现 “难以理解” 情况
        -   转换往往围绕 *Raw Pointer* 裸指针、原始数据类型进行
    -   注意事项
        -   转换原始切片时不会自动调整长度：`*const [u16] as *const [u8]` 得到切片包含一半内存
        -   转换不可传递：`e as U1 as U2` 有效不保证 `e as U2` 有效

> - 4.3 转换：<https://doc.rust-lang.net.cn/nomicon/casts.html>
> - 4.3 Cast：<https://doc.rust-lang.org/nomicon/casts.html>
> - 8.2.4 运算符表达式 - 类型转换表达式：<https://doc.rust-lang.net.cn/reference/expressions/operator-expr.html#type-cast-expressions>

####   `transmute`

-   `std::mem::transmute<T,U>`：位重解释
    -   接受 `T` 类型值重新解释为 `U` 类型值
        -   仅要求 `T`、`U` 类型值大小相同
        -   将带来很多未定义行为
            -   创建无效状态的实例将导致无法预测的将结果
            -   未指定返回类型时，返回值类型不确定
            -   共享引用 `&` 转换为可变引用 `&mut` 是未定义行为
            -   未显式提供生命周期时，重解释的引用产生无界生命周期
            -   不同的复合类型之间内存布局可能不同，字段将被错误解释
    -   `std::mem::transmute_copy<T,U>`：从 `&T` 中复制 `size_of<U>` 字节并解释为 `U`

> - 4.4 Transmute：<https://doc.rust-lang.net.cn/nomicon/transmutes.html>
> - 4.4 Transmute：<https://doc.rust-lang.org/nomicon/transmutes.html>

### 表达式、函数、控制流

| 关键字                    | 描述                              |
|---------------------------|-----------------------------------|
| `fn (...) -> RETURN {}`   | 函数定义                          |
| `if COND {} else {}`      | 条件                              |
| `loop {... break ...}`    | 无限循环（只能 `break` 内部打破） |
| `while COND {}`           | 条件循环                          |
| `for ELE in CONTAINER {}` | 遍历循环                          |

-   重点说明
    -   `if`、`while`、`for` 后表达式无 `()` 引导
    -   表达式、语句
        -   表达式：计算并返回值，可作为语句的一部分、后跟 `;` 转换为语句
        -   语句：执行操作但不返回值
    -   `fn` 函数返回值
        -   需在 `->` 后声明返回值类型，否则返回 `()` 单元
        -   默认返回函数体最后表达式，可用 `return` 提前返回指定值

> - 3.5 控制流：<https://www.rust-book-cn.com/ch03-03-how-functions-work.html>
> - 3.3 函数：<https://www.rust-book-cn.com/ch03-03-how-functions-work.html>
> - 18 常量求值：<https://doc.rust-lang.net.cn/stable/reference/const_eval.html>

### 结构体 `struct`

```rust
struct User {
    pub(super) active: bool,                // `pub(super)` 仅对父模块可见
    username: String,
}

impl User {
    pub fn new(username: String) {          // 关联函数，首个参数非 `self`
        User{
            active: true,
            username,
        }
    }
    pub print(&self) {                      // `&self` 是 `self: &Self` 的简写
        println!("{}", self.username);
    }
}

struct Color(i32, i32, i32);                // 元组结构体

fn main() {
    let username = String::from("Rust");
    let user1 = User {
        active: true,                       // 显式键值对指定
        username,                           // 同名自动初始化
    }
    let user1 = User {
        active: true,
        ..user1                             // 从其他实例更新字段，会移动数据
    }

    let rgb = Color(255, 255, 255);
    let Color(r, g, b) = rgb;               // 带类型的模式匹配解构
}
```

-   结构体：将相关的值打包、命名、形成有意义组合的自定义数据类型
    -   **结构体成员可变性与结构体变量、引用保持一致**
    -   特殊结构体
        -   *元组结构体*：字段没有关联名称、仅有类型的结构体
            -   适合需将元组命名以区别与其他元组的场合
            -   元组结构体行为类似元组访问、（带类型）模式匹配解构
        -   *单元结构体*：无字段结构体，零存储开销
            -   适合实例无需存储数据，仅关心类型行为、需实现某 `trait` 的场合
            -   单元结构体给出名称即 “实例化”、无需 `{}`
    -   结构体类型、枚举类型通过 `impl` 块（可分多块）定义关联函数
        -   *Method* 方法：首个形参为 `self` 类实例自身的关联函数
            -   虽然，可通过完全限定语法调用方法、或首个参数为 `Self` 类型但非 `self` 名称的关联函数
            -   但，编译其对方法中 `self` 有特殊的类型转换机制，对完全限定语法不适用
        -   狭义关联函数：非方法的关联函数

> - 2.7 方法 Method：<https://course.rs/basic/method.html>
> - 5.1 定义和实例化结构体：<https://www.rust-book-cn.com/ch05-01-defining-structs.html>
> - 2.4.3 结构体：<https://course.rs/basic/compound-type/struct.html>

####    `self` 类型转换

| `self` 类型        | 说明                 |
|--------------------|----------------------|
| `self: Self`       | 当前类型为 `self`    |
| `self: &Self`      | 可简写为 `&self`     |
| `self: &mut Self`  | 可简写为 `&mut self` |
| `self: Rc<Self>`   |                      |
| `self: Box<Self>`  |                      |
| `self: Arc<Self>`  |                      |
| `self: Pin<&Self>` |                      |

-   方法中 `self` 参数可以为多种类型（有限的）
    -   `.` 运算符（调用方法）会自动引用、解引用、强制转换直至调用者、方法签名类型匹配
        -   引用除 `&`、`&mut` 外，还 **包装进智能指针**
        -   解引用也包括智能指针解引用 `Deref`（甚至支持多层嵌套）
        -   强制转换更多指转换为 `trait` 对象、调用 `trait` 中方法

> - 4.2 点运算符：<https://doc.rust-lang.net.cn/nomicon/dot-operator.html>
> - How is it even possible? self type’s is not Self but Pin<&mut Self>：<https://users.rust-lang.org/t/how-is-it-even-possible-self-types-is-not-self-but-pin-mut-self/49683>
> - 6.15 The Rust Reference - Associated items - Method：<https://doc.rust-lang.org/stable/reference/items/associated-items.html#methods>

####    内存布局

```rust
struct A {
    a: i16,
    _pad1: [u8,2];                      // 可能布局：填充以对齐至 4B
    b: u64,
}
struct B {                              // 不保证 `A`、`B` 类型值内存中字段顺序一致
    a: i16,
    _pad1: [u8,2];
    b: u64,
}

enum Foo{
    A(u32),
    B,
}
struct FooRepr{                         // 1. `Foo` 枚举的可能布局，也可能
    data: u32,
    tag: u8,                            // 2. 根据 `tag` 字段确定 `data` 字段含义
}
```

-   内存布局
    -   *Alignment* 对齐：若某类型对齐为 `n`，则存储其值的有效地址必须为 `n` 的整数倍
        -   类型大小必须是对齐的整数倍
            -   确保该类型数组总可通过偏移类型大小的整数倍来索引
        -   基本类型通常与其大小对齐
            -   但与平台相关，*X86* 平台上 `u64`、`f64` 通常对齐到 4B
        -   复合结构类型内各字段需对齐，且按字段中对齐最大者对齐
            -   为此，*Rust* 会在必要时 *padding* 以将字段对齐
    -   除数组外，*Rust* 不保证字段顺序、完整性
        -   仅，保证同一类型的不同值中各字段顺序一致
        -   对结构体类型，内存中字段顺序实际可能与定义顺序不一致，可避免空间浪费
        -   对枚举类型，*Rust* 可能会执行 *空指针优化*，删除标志字段

> - 2.1 `repr(Rust)` 内存布局：<https://doc.rust-lang.net.cn/nomicon/repr-rust.html>

####    类型大小

```rust
struct Nothing;                         // ZST
struct LotsOfNothing {                  // ZST
    foo: Nothing,
    qux: (),
    baz: [u8; 0],
}
```

-   特殊大小类型
    -   *Dynamic Sized Type* 动态大小类型 `!Sized`：没有静态已知大小、对齐方式的类型
        -   *DST* 只能存储在指针后，且指向 *DST* 指针为包含两个 `usize` 字段的 “宽指针”
            -   值地址：指向对象值
            -   *Metadata* 元数据：值的补充信息
                -   对切片，为切片长度
                -   对 *Trait* 对象，为类型的 *vtable* 虚表地址
        -   两种主要的 *DST*
            -   *Trait* 对象 `dyn TRAIT`
            -   切片 `[T]`、`str`
            -   结构体可在最后字段存储单个 *DST* 并成为 *DST*
    -   *Zero Size Type* 零大小类型：不占用空间的类型
        -   *Rust* 会针对 *ZST* 做优化，对产生、存储 *ZST* 的操作会可简化为无操作
        -   主要的 *ZST*
            -   单元元组
            -   单元结构体
    -   空类型：无法实例化的类型
        -   空类型只能讨论类型，无法讨论值，主要用于体现类型值的 “不可达性”
            -   指向空类型的裸指针有效，但是解引用是未定义行为
        -   *Rust* 同样会对空类型做优化

> - 2.2 特殊大小类型：<https://doc.rust-lang.net.cn/nomicon/exotic-sizes.html>
> - *Rust* 虚表布局规则介绍：<https://zhuanlan.zhihu.com/p/680849759>

####    内存初始化

```rust
use std::mem::{self, MaybeUninit};

const SIZE: usize = 10;

let x = {
    let mut x: [MaybeUninit<Box<u32>>; SIZE] = unsafe {
        MaybeUninit::uninit().assume_init()                 // 获取内存，但不初始化
    };
    for i in 0..SIZE {
        x[i] = MaybeUninit::new(Box::new(i as u32));        // 动态赋值初始化
        // x[i].write(i);                                   // 或直接 `.write` 按位写入
    }
    unsafe { mem::transmute::<_, [Box<u32>; SIZE]>(x) }     // 重解释类型
};
```

-   内存初始化：将内存设置为合理、有意义的值，并置位初始化标志
    -   *Rust* 中栈内变量在显式赋值前未初始化，且不允许在被初始化前被使用（读取、赋值）
        -   对 `Copy` 标记变量，初始化后不会变为未初始化状态（遮蔽不视作同一变量）
        -   非 `Copy` 标记（实现 `Drop`）类型变量值被移出后，**变量被反初始化**
            -   反初始化：**实质上仅需要复位初始化标志**
            -   即，仅需要逻辑上变为未初始化，实际值无需调整
            -   此时，可变变量可被赋值、重新初始化
    -   如上，对实现 `Drop` 类型变量，可能在初始化、未初始化之间变化
        -   而对已初始化变量被丢弃（赋值、离开作用域）时，*Rust* 需调用析构器释放（变量原值）资源
            -   部分情况下，变量的初始化状态可静态确定，但不总是这样
        -   故，*Rust* 会在运行时维护 *Drop Flag* 销毁标志确定变量是否初始化、是否需要析构
            -   在变量被初始化、反初始化时，销毁标志被切换
            -   但注意，**通过解引用赋值时，引用值总是无条件被析构**
                -   因为一般的，引用总是需要原值已初始化才可被创建
                -   但，某些情况下引用可在未初始化的情况下被创建，此时通过解引用赋值将导致 `panic`
    -   但，在需要动态初始化数组时，可通过 `unsafe` 的 `mem::MaybeUninit` 延迟初始化，避免原生初始化的额外开销
        -   `MaybeUninit<T>` 值被丢弃时不触发 `Drop` 析构
            -   `MaybeUninit<T>` 可在内存未初始化的情况下被重赋值
                -   否则，未初始化析构将导致 `panic`
            -   但注意，此时通过 `MaybeUninit<T>.as_mut_ptr()` 创建 `T` 的可变引用，借此重新赋值将丢弃未初始化值、导致 `panic`
        -   `MaybeUninit<T>` 与 `T` 在内存布局中完全一致
            -   `MaybeUninit<T>` 被初始化后可通过 `mem::transmute` 直接重解释、转换为 `T` 类型供后续使用
            -   但注意，`CONTAINER<MaybeUninit<T>>` 与 `CONTAINER<T>` 不一定一致

> - 5.1 受检查的未初始化内存：<https://doc.rust-lang.net.cn/nomicon/checked-uninit.html>
> - 5.1 Checked：<https://doc.rust-lang.org/nomicon/checked-uninit.html>
> - 5.3 未经检查的未初始化内存：<https://doc.rust-lang.net.cn/nomicon/unchecked-uninit.html>
> - 5.3 Unchecked：<https://doc.rust-lang.org/nomicon/unchecked-uninit.html>
> - Rust：`MaybeUninit`（避免初始化带来的性能损失）：<https://zhuanlan.zhihu.com/p/1889248158661473931>

### 枚举、模式匹配

####    枚举类型

```rust
enum Message {
    Quit,                           // 无值枚举变体
    Move { x: i32, y: i32 },        // 枚举变体带命名字段，类似结构体
    Write(String),                  // 枚举变体带单值
    ChangeColor(i32, i32, i32),     // 枚举变体带元组值
}

fn main() {
    let m1 = Message::Quit;         // 枚举变体在其标识符下命名空间化
    let m2 = Message::Move{x: 2, y: 2};
    let m2 = Message::Write(String::from("Write"));
    let m2 = Message::ChangeColor(255, 255, 255);
}
```

-   枚举：标识值属于给定集合的方式
    -   *Variant* 枚举变体：是枚举类型的取值类型（而结构体字段是结构体的成员、组成部分）
        -   枚举变体可以关联不同类型、数量数据
        -   枚举变体在枚举名称（标识符）下命名空间化，且可以独立导入
    -   *Field-less* 枚举：所有枚举变体都没有关联数据的枚举

> - 6.1 定义枚举：<https://www.rust-book-cn.com/ch06-01-defining-an-enum.html>

####    模式

| 模式                | 说明                                               |
|---------------------|----------------------------------------------------|
| 字面量              |                                                    |
| 命名变量            |                                                    |
| `\|`                | 分隔多个模式                                       |
| `..=`               | 范围，支持整数、`char`                             |
| 结构体、枚举解构    | `Message::ChangeColor(Color::RGB(r, g, b))`        |
| `_`                 | 忽略整体、部分值                                   |
| `..`                | 忽略剩余部分                                       |
| `PTN if ...`        | 匹配守卫作为匹配额外条件，避免匹配命名变量时的遮蔽 |
| `FIELD : VAR @ PTN` | 绑定匹配值至新变量名 `VAR`                         |

-   模式、模式匹配、可反驳性
    -   模式由以下嵌套、组合构成
        -   字面量
        -   解构数组、枚举、结构体或元组
        -   变量
        -   通配符
        -   占位符
    -   *Irrefutable* 不可反驳的：能匹配任何可能传递的值的模式
        -   `match`、`let`、`for` 只能接受不可反驳的模式
            -   值不匹配时程序无法工作，无法编译
    -   *Refutable* 可反驳的：对某些之会匹配失败的模式
        -   `if let`、`while let`、`let ... else` 接受可反驳的模式
            -   且，编译器将警告不可反驳的模式
    -   `ref`、`ref mut` 关键字：匹配变量（所有者），**但不消耗、而是创建** 匹配的共享、可变引用
        -   仅用于模式匹配中模式，包括 `match`、`while`、赋值、函数等场合
            -   即，`ref` 表示创建引用
            -   而，`&` 在模式中表示引用类型，是类型的部分

| 模式中 `ref`、`&` | 说明                           |
|-------------------|--------------------------------|
| `&PTN`            | 匹配引用，`PTN` 为共享引用     |
| `&mut PTN`        | 匹配可变引用，`PTN` 为可变引用 |
| `ref PTN`         | 匹配所有者，`PTN` 为共享引用   |
| `ref mut PTN`     | 匹配所有者，`PTN` 为可变引用   |

```rust
let tup2 = (10, 20);
let (ref x, ref y) = tup2;              // `x`、`y` 是 `tup2` 中元素的不可变引用

fn main() {
    let vec = vec![1, 2, 3];
    for ref x in &vec {
        println!("x: {}", *x);          // `x` 是 `vec` 中元素的不可变引用
    }
}
```

> - 19 模式与匹配：<https://www.rust-book-cn.com/ch19-00-patterns.html>
> - 19.2 可反驳性：模式是否可能匹配失败：<https://www.rust-book-cn.com/ch19-02-refutability.html>
> - 19.3 模式语法：<https://www.rust-book-cn.com/ch19-03-pattern-syntax.html>
> - 一文搞懂 Rust `ref` 关键字用法：<https://zhuanlan.zhihu.com/p/27723074770>
> - Rust 中 `*`、`&`、`mut`、`&mut`、`ref`、`ref mut` 的用法和区别：<https://www.cnblogs.com/risheng/p/18323252>

####    模式匹配类型

| 模式匹配                      | 说明                              |
|-------------------------------|-----------------------------------|
| `match VAL {PTN => ...}`      | 穷尽模式匹配                      |
| `if let PTN = VAL {} else {}` | 简化匹配，忽略剩余模式            |
| `while let PTN = VAL {}`      | 循环匹配，匹配失败则中断循环      |
| `let PTN = VAL`               | （模式匹配解构）赋值              |
| `let PTN = VAL else {}`       | **带失败处理的** 模式匹配解构赋值 |
| `for PTN in ITER {}`          | 遍历赋值匹配                      |
| `fn FNMAME(PTN) {}`           | 函数传参模式匹配                  |

-   模式匹配类型
    -   `match`：基本的、穷尽的模式匹配
        -   分支匹配是短路的，前序分支匹配成功即中断剩余匹配
        -   分支可以将变量绑定到与模式匹配的值，以从枚举变体中获取值
            -   绑定变量作用域仅限于 `match`、`if let` 中分支对应代码块
        -   分支模式必须是穷尽所有可能性
            -   `_` 匹配所有模式且不绑定值的特殊模式
    -   `if let`：牺牲 `match` 穷尽检查的简化模式匹配
        -   绑定变量作用域仅限于 `if let` 中分支代码块
    -   `while let`：类似 `if let`，匹配则继续循环，否则中断
    -   `let` 赋值语句：也是模式匹配
    -   `let else`：是带失败处理的模式匹配解构赋值
        -   可视为 `let VAR = if let PTN = VAL {} else {}` 的简化
    -   函数传参：模式匹配解构实参以匹配形参

> - 6.2 `match` 控制流构造器：<https://www.rust-book-cn.com/ch06-02-match.html>
> - 6.3 使用 `if let` 和 `let else` 实现简洁的控制流：<https://www.rust-book-cn.com/ch06-03-if-let.html>
> - 19.1 模式的所有使用场景：<https://www.rust-book-cn.com/ch19-01-all-the-places-for-patterns.html>

####    空 `Option`、异常 `Result`

```rust
enum Option<T> {
    None,               // 空
    Some(T),            // 非空，及关联值
}
enum Result<T, E> {
    Ok(T),              // 成功，及关联值
    Err(E),             // 失败，及关联错误
}
```

-   `Option<T>`：指示某个值可能为空的枚举类型
    -   类型检查：`Option<T>`、`T` 类型不同
        -   编译器不允许直接使用 `Option<T>` 值
        -   代码中必须显式处理空值 `None` 情况
    -   空值替换：将 `Some(VAL).take(&mut self)` 将获取值 `Some(VAL)` 并在位置为 `None`
-   `Result<T>`：指示某个行为可能成功（返回正常结果）、或者失败（返回错误）
    -   处理可恢复错误

| `Option<T>`、`Result<T, E>` 方法     | `None`、`Err<E>`               | `Some(T)`、`Ok(T)` |
|--------------------------------------|--------------------------------|--------------------|
| `.expect(error_msg)`                 | `panic` 并打印自定义错误信息   | 解包取值           |
| `.unwrap()`                          | `panic`                        | 解包取值           |
| `.unwrap_or(default)`                | 自定义默认值                   | 解包取值           |
| `.unwrap_or_else(closure)`           | 闭包（无参）返回值             | 解包取值           |
| `.unwrap_or_default()`               | 当前类型默认值                 | 解包取值           |
| `.map(closure)`                      | `None`                         | 作为参数执行闭包   |
| `.map_or(default, closure)`          | 默认值                         | 作为参数执行闭包   |
| `.map_or_else(NEclosure, SOclosure)` | 执行无参闭包                   | 作为参数执行闭包   |
| `Option.ok_or(error_msg)`            | 错误信息映射为 `Err`           | 包装为 `Ok`        |
| `Option.ok_or_else(closure)`         | 闭包结果映射为 `Err`           | 包装为 `Ok`        |
| `Result.ok()`                        | `None`                         | 包装为 `Some(T)`   |
| `Result.err()`                       | `Some(E)`                      | `None`             |
| `?` 算符                             | **函数直接返回** `None`、`Err` | 解包取值           |

> - 6.1 定义枚举：<https://www.rust-book-cn.com/ch06-01-defining-an-enum.html>
> - 9.2 使用 `Result` 处理可恢复的错误：<https://www.rust-book-cn.com/ch09-02-recoverable-errors-with-result.html>
> - Rust 错误处理：`Option` 和 `Result` 的使用总结：<https://zhuanlan.zhihu.com/p/668022700>

###    `const fn` 编译期优化

```rust
// ******************************* 常量函数
const fn full_featured<T, const N: usize>(              // 常量函数，将在编译期执行
    arr: [T; N]
) -> [T; N]
where
    T: ~const Clone + ~const Destruct + Copy,           // 类型 `T` 可在编译期克隆、析构
    [T; N]: Sized,
{
    arr.clone()
}

// ****************************** 常量特性  #TODO
pub const trait Borrow<Borrowed: ?Sized> {
    fn borrow(&self) -> &Borrowed;
}
```

-   `const fn` 常量函数：允许在 `const` 上下文（常量表达式、块等）中调用的函数
    -   常量函数将在编译期由编译器解释（执行）
        -   函数体须只能使用常量表达式
        -   参数、返回值类型必须在编译时可确定
        -   只能创建共享（不可变）引用
    -   即，常量函数返回值是常量
        -   则，常量函数被其他函数封装（调用）后，**仅在编译时执行一次、得到单个实例**
    -   `~const`、`[const]`：指示 **在 `const` 上下文可用** 的修饰符
        -   `const fn` 将在编译期执行，则要求参数的方法满足可在编译期执行
        -   则，可用 `~const` 修饰 *Trait Bound* 指示 `trait` 中方法可在编译期执行

> - 2.8.1 泛型 Generics：<https://course.rs/basic/trait/generic.html>
> - *const_destruct*：<https://doc.rust-lang.org/unstable-book/language-features/const-destruct.html>
> - Make trait methods callable in const contexts：<https://github.com/rust-lang/rfcs/pull/3762>
> - *Std Core* 源码 - `core::borrow.rs`：<https://doc.rust-lang.org/stable/src/core/borrow.rs.html>

##  多态

### *Trait* 特性、公共行为

```rust
trait Summary {
    type TypeNoDefault;                         // 关联类型
    const CONST_NO_DEFAULT: i32;                // 无默认值常量
    const CONST_WITH_DEFAULT: i32 = 99;         // 带默认值常量
    fn sum_user(&self) -> String;               // 方法仅签名（无默认实现）
    fn summarize(&self) -> String {             // 带默认实现的方法
        format!("Read more from {}", self.sum_user())
    }
}
impl Summary for User {                         // 为结构体实现 Trait
    const CONST_NO_DEFAULT = 80;
    fn sum_user(&self) -> String {
        self.username.clone()
    }
}
impl Summary for Message {                      // 为枚举类型实现 Trait
    const CONST_NO_DEFAULT = 80;
    fn sum_user(&self) -> String {
        format!("{:?}", &self)
    }
}

impl Summary for &User {                        // 为结构体引用类型 `&User` 实现 Trait
    const CONST_NO_DEFAULT = 80;
    fn sum_user(&self) -> String {
        (*self).username.clone()                // 此时，`Self` 为 `&User`、`self` 为 `&&User` 类型
    }
}
```

-   `trait` 特性：描述类型可以实现的抽象接口（组合），以定义实现某些目的所需一组的行为
    -   `trait` 特性可以包含以下 3 种 *关联项*
        -   函数 `fn`：方法、关联函数
            -   `trait` 中函数不可为 `const` 函数：*Trait* 对象 `dyn TRAIT` 的动态分派与 `const fn` 编译时计算冲突
        -   类型 `type`：关联类型
            -   在 `trait` 中定义的类型占位符
                -   不能指定默认类型，仅可在实现 `trait` 时指定为具体类型
                -   主要用于关联函数签名中参数、返回值类型
            -   相较于 `trait` 中泛型参数编译时多态，**关联类型只能指定一次**、无需注解类型
        -   常量 `const`：关联常量
    -   为类型实现特性时，特性中无默认值函数、常量需要具体实现
    -   `trait` 中方法调用前，需要将单独将 `trait` 引入作用域（不仅需要引入实现其的 `struct`、`enum`）
        -   `trait` 中关联项总是公开的（`trait` 就是对外暴露的接口）
        -   即，`trait` 与具体类型是松耦合的，**类型可根据需要仅 “拥有” 部分特性**
            -   此即 `trait` 与继承的差异（继承中子类型的无法根据需要舍弃父类行）
    -   完全限定语法：消除（来自多个 `trait`、`struct`）同名方法之间的歧异
        -   `INST.METHOD()`：优先调用直接在 `<INST>` 所属的具体类型上实现的方法
        -   `TRAIT::METHOD(&INST)` 完全限定语法：类似调用 `trait` 关联函数，显式指定调用 `trait` 中方法
        -   `<STRUCT as TRAIT>::REL_FN()` 关联函数的完全限定语法：显式指定调用 `trait` 关联函数的实现

> - 10.2 Traits：定义共享行为：<https://www.rust-book-cn.com/ch10-02-traits.html>
> - 20.2 高级特性：<https://www.rust-book-cn.com/ch20-02-advanced-traits.html>
> - 6.11 Traits：<https://doc.rust-lang.net.cn/stable/reference/items/traits.html>
> - 6.11 Traits：<https://rustwiki.org/zh-CN/reference/items/traits.html>

####    *Orphan Rule* 孤儿规则

-   *Trait* 实现规则、特例
    -   *Orphan Rule* 孤儿规则：只能在 `trait` 或某一类型是本地 *Crate* 的情况下可以实现 `trait`
        -   孤儿规则用于
            -   确保代码不被从外部破坏、影响下游
            -   避免同一类型、`trait` 实现多次实现的冲突
        -   具体的，对 `impl<P1..=Pn> Trait<T1..=Tn> for T0`，满足两条件之一即合法
            -   `Trait` 在本地 *Crate* 定义
            -   `T0..=Tn` 中存在类型 `Ti` 在本地 *Crate* 定义、且 `T0..Ti` 中无 `P1..=Pn`
        -   `#[fundamental]` 类型：允许在定义 *Crate* 外、违反孤儿规则的情况下实现 `trait`
            -   `#[fundamental]` 注解的类型包括 `Box`、`&T`、`&mut T`、`Pin` 等
    -   *Blanket Implementation* 通用实现：为任意满足 *Trait Bound* 的类型实现 `trait`
        -   即，形如 `impl<T: TraitBoud> SomeTrait for T`、`&T` 的实现
        -   只能在定义 `trait` 的 *Crate* 执行 *Blanket Implementation*

> - 6.12 Items - Implementations - Orphan Rules：<https://doc.rust-lang.org/reference/items/implementations.html#r-items.impl.trait.orphan-rule>
> - Rust的Blanket Implements(通用实现)：<https://segmentfault.com/a/1190000037510636>

####    运算符重载

```rust
use std::ops::Add;

type Add<Rhs=Self> {                            // 标准库中 `Add trait` 定义，泛型参数有默认类型
    type Output;                                // 关联类型，实现时需指定具体类型
    fn add(self, rhs: Rhs) -> Self::Output;
}

impl<i32> Add for Point<i32> {                  // 需通过实现 `Add trait` 重载运算符
    type Output = Point<i32>;                   // 指定关联类型为特性类型
    fn add(self, rhs: Point) -> Point {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}
```

-   运算符重载：*Rust* 中只能重载有对应 `trait` 的运算符
    -   对 `=`、`?`、`&&`、`||` 等无对应 `trait` 的运算符，则无法重载
    -   对自定义类型，显式的实现 `std::ops`、`std::cmp` 中特性即可实现运算符重载
    -   对原生类型，`std::ops`、`std::cmp` 中 `trait` 实现就是对应运算符
        -   所以，不能将运算符视为是对应 `trait` 中方法的语法糖
            -   编译器仅对有重载运算符的类型，在执行运算符时调用对应方法
        -   原生类型、运算符 `trait` 均在标准库中定义，则无法重载原生类型的运算符（孤儿规则）
        -   即，从语法层面可保证原生类型的运算符行为不变

> - 20.2 高级特性：<https://www.rust-book-cn.com/ch20-02-advanced-traits.html>
> - 深入 `std::ops` 与 `std::cmp`：Rust 运算符重载的 Trait 契约与泛型设计（深度解析）：<https://blog.csdn.net/AtomicVerse/article/details/154286827>
> - Rust运算符重载：让类型更灵活优雅：<https://juejin.cn/post/7360647839448285221>
> - Rust 运算符重载：开启高效编程之旅：<https://juejin.cn/post/7442711650387951631>
> - `std::ops`：<https://doc.rust-lang.org/std/ops/index.html>
> - `std::ops::arith::add_impl!` 源码：<https://rustwiki.org/zh-CN/src/core/ops/arith.rs.html>


###   *Generic* 泛型

```rust
use std::ops::Add;

fn first<T>(list: &[T]) -> &T {                 // 函数定义中泛型
    &list.0
}
enum Option<T> {                                // 枚举定义中泛型
    Some(T),
    None,
}
struct Point<T=i32> {                           // 结构体定义中泛型，泛型参数带默认值
    x: T,
    y: T,
}
impl<T> Point<T> {                              // 方法定义中泛型
    fn x(&self) -> &T {
        &self.x
    }
}
pub trait Iterator<T> {                         // `trait` 中定义泛型
    fn next(&mut self) -> Option<T>;
}
```

-   *Generic* 泛型：类型未确定、可代表不同类型的编程模型
    -   *Rust* 在编译时对泛型代码单态化，保证泛型参数没有额外运行时成本
        -   单态化：编译时使用具体类型将泛型代码填充为特定类型的代码
        -   即类似，手动为每个类型复制一套代码
    -   `::<T>` *Turbo Fish*：显式指定泛型参数的具体类型
        -   一般仅用于编译器无法推测泛型参数的场合
    -   泛型参数可以指定默认类型（以简化使用）
        -   无法推断、且不指定泛型参数时，将使用默认类型单态化

> - 10.2 Traits：定义共享行为：<https://www.rust-book-cn.com/ch10-02-traits.html>
> - 2.8.1 泛型 Generics：<https://course.rs/basic/trait/generic.html>
> - 常见的 Rust Lifetime 误解：<https://zhuanlan.zhihu.com/p/165976086>
> - Common Rust Lifetime Misconceptions：<https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md>
> - 带你揭秘rust中参数默认类型：<https://rustcc.cn/article?id=065aa7e9-e8ec-47e3-94ba-9aa6dc44b5b7>

####    泛型实现 `trait`

| 泛型标识 | 说明               | 案例                                |
|----------|--------------------|-------------------------------------|
| `T`      | 包括所有类型       | `i32`、`&i32`、`&mut i32`、`& &i32` |
| `&T`     | 仅包括共享引用类型 | `&i32`、`& &mut i32`、`&&i32`       |
| `&mut T` | 仅包括可变引用类型 | `&mut i32`、`&mut & i32`            |

-   3 种泛型标识 `T`、`&T`、`&mut T` 含义不同
    -   泛型标识 `T` 包括所有类型，包括所有权类型、共享引用 `&T`、可变引用 `&mut T`
    -   不能为同一类型类型重复实现同一方法，对泛型 `T`
        -   故不能同时为 `T` 与 `&T`、或 `T` 与 `&mut T` 实现同一 `trait`
        -   但可为 `&T` 与 `&mut T` 实现同一 `trait`
    -   注意，若 `T` 为具体类型，`T`、`&T`、`&mut T` 是三种不同类型，可以同时实现同一 `trait`
        -   此时，编译器将优先调用 `self` 形参类型完全相同的方法，再尝试解引用、类型转换
        -   为 `T`、`&T`、`&mut T` 等分别实现同一 `trait` 主要用于使得，形参带泛型 *Trait Bound* 的函数同时支持所有者、引用作为实参
            -   因为，强制类型转换不会为满足 *Trait Bound* 而应用

```rust
trait Trait {}
impl<T> Trait for T {}
impl<T> Trait for &T {}                         // 编译失败
impl<T> Trait for &mut T {}                     // 编译失败

trait Trait {}
impl<T> Trait for &T {}                         // 成功
impl<T> Trait for &mut T {}                     // 成功

// ***************** Borrow<T> 本身包含泛型参数，实际没有为类型多次实现
impl<T> Borrow<T> for T {}
impl<T> Borrow<T> for &T {}                     // 成功
impl<T> Borrow<T> for &mut T {}                 // 成功
```

> - `impl Trait for Type` 和 `impl Trait for &Type` 是什么关系？：<https://rustcc.cn/article?id=d993e943-64df-4252-9467-155b2a43a9d5>
> - `std::borrow::Borrow` 中文：<https://rustwiki.org/zh-CN/std/borrow/trait.Borrow.html>

####   `const` 泛型

```rust
fn display_array<T: std::fmt::Debug, const N: usize>(arr: [T; N]) {     // `const N` 常量泛型
    println!("{:?}", arr);
}

fn something<T>(val: T)
where
    Assert<{ core::mem::size_of::<T>() < 768 }>: IsTrue,                // `const` 表达式作 bound
{
}
fn main() {
    something([0u8; 0]);
    something([0u8; 512]);
    something([0u8; 1024]);                                             // 编译错误，数组长度超限
}
pub enum Assert<const CHECK: bool> {
    //
}
pub trait IsTrue {
    //
}
impl IsTrue for Assert<true> {
    //
}
```

-   `const N: TYPE` 常量泛型：针对值的泛型，对常量值的抽象

> - 2.8.1 泛型 Generics：<https://course.rs/basic/trait/generic.html>

### *Trait Bound*、*Super Trait*

```rust
use std::fmt::Display; 

// *********************** 函数 Trait Bound
fn out_sum<T: Summary + Display>(x: &T) {       // Trait Bound，`+` 分隔多个
    println!("{}", x.summarize());
}
fn out_sum(x: &(impl Summary + Display)) {      // Trait Bound 简写语法糖
    println!("{}", x.summarize());
}
fn out_sum<T>(x: &T) -> ()
where
    T: Summary + Display,                       // `where` 子句格式的 Trait Bound
{
    println!("{}", x.summarize());
}

// *********************** 
impl<T: Display> Point<T> {                     // Trait Bound：限制为特定类型实现方法
    fn display(&self) {}
}
impl<T: Display> Summary for T {}               // Trait Bound：限制为特定类型实现 `trait`

// *********************** Super Trait
trait Summary: Display {}                       // Super Trait：实现 `Summary` 前需实现 `Display`
```

-   *Trait Bound*：泛型单态化时，要求类型必须实现某些 `trait`
    -   函数返回类型可为泛型参数，但 **函数实际返回的类型必须一致**
        -   因为，泛型通过编译时单态化实现
        -   若确实需要返回不同类型，应用动态类型 `Box<dyn ...>`
    -   *Super Trait*：限制为某类型实现该 `trait` 前，该类型需实现其他 `trait`

> - 10.1 泛型数据类型：<https://www.rust-book-cn.com/ch10-01-syntax.html>
> - 20.2 高级特性：<https://www.rust-book-cn.com/ch20-02-advanced-traits.html>

####    *Mark Trait* 标记特性

| *Mark Trait* | 说明                               | 对应     |
|--------------|------------------------------------|----------|
| `Copy`       | 只需按位复制即可复制值的类型       | `Clone`  |
| `Send`       | 可以跨线程边界传输的类型           | `!Send`  |
| `Sized`      | 编译时大小已知为常量的类型         | `?Sized` |
| `Sync`       | 可在线程间安全共享引用的类型       | `!Sync`  |
| `Unpin`      | 无需固定、可安全在内存间移动的类型 | `!Unpin` |

-   *Mark Trait* 标记特性：仅用于 *Trait Bound*，即用于编译器检查某类型是否满足要求
    -   标记特性没有类似方法的关联项，**实现本身即开发对类型的保证**
        -   正确的实现即类型具备标记特性应有的特征
        -   错误的实现可能导致未定义的行为
    -   *Auto Trait* 自动特性：自动为所有类型实现、除非类型自身或包含显式的否定实现的标记特性
    -   *Negtive Trait* 负面特性 `!`：标记类型不具有某特性，如 `!Send`、`!Sync` 
        -   **`!` 标记特性会 “传染”**：若某类型具备 `!` 标记特性，则包含该类型字段的类型也有 `!` 特性
        -   负面特性不可以作为 *Trait Bound*
    -   标记特性说明
        -   `Send`、`Sync`、`Unpin` 标记特性成对、互补
            -   标记特性自身为基本类型、大部分标准库类型实现，并为满足条件的类型自动派生
            -   `!Send`、`!Sync`、`!Unpin` 特性则与对应特性相反，实现即表示不满足对应标记特性的要求
        -   `Sized` 所有类型参数均默认带有 `Sized` 限制，需显式 `?Sized` 解除限制
        -   `Copy` 与 `Clone` 成对

> - `std::marker` 中文：<https://rustwiki.org/zh-CN/std/marker/index.html>
> - `src/core/marker` 源码：<https://doc.rust-lang.org/src/core/marker.rs.html>
> - 特殊类型和 `trait`：<https://rustwiki.org/zh-CN/reference/special-types-and-traits.html>
> - *Auto Traits*：<https://doc.rust-lang.org/beta/unstable-book/language-features/auto-traits.html>

### *Trait Object*

```Rust
pub struct Page{
    pub components: Vec<Box<dyn Summary>>,      // 保存在向量中的 Trait 对象，其中元素类型可不同
}

impl Page {
    pub fn run(&self) {
        for comp in self.components.iter() {
            comp.summarize();                   // 运行时根据内部指针调用方法
        }
    }
}
```

-   `dyn TRAIT` *Trait 对象*：代表编译时无法确定、但均实现某 `trait` 的运行时才能确定的类型
    -   *Trait 对象* 仅用于对通用行为进行抽象
    -   原始类型被擦除，编译器无法预知具体类型、需要调用的方法，只能在运行时动态分派
        -   内存中存储指向 *Trait* 对象值、该类型 *vtable* 虚表的宽指针
        -   在运行时通过内部指针确定需调用的方法，有运行时成本

> - 18.2 使用允许不同类型值的 Trait 对象：<https://www.rust-book-cn.com/ch18-02-trait-objects.html>
> - 2.2 特殊大小类型：<https://doc.rust-lang.net.cn/nomicon/exotic-sizes.html>
> - *Rust* 虚表布局规则介绍：<https://zhuanlan.zhihu.com/p/680849759>

####    *vtable* 虚表

```
pub trait Grand {
    fn grand_fun1(&self);
    fn grand_fun2(&self);
}

pub trait Parent : Grand {
    fn parent_fun1(&self);
    fn parent_fun2(&self);
}

pub trait Trait : Parent {
    fn fun(&self);
}
// *************************** 单线 *Super Trait* 的 `dyn Trait`
+-------------------------------+
| fn drop_in_place(*mut T)      |               // 析构函数指针
+-------------------------------+
| size of T                     |               // 类型大小
+-------------------------------+
| align of T                    |               // 类型内存对齐
+-------------------------------+
| fn <T as Grand>::grand_fun1   |               // 顺次为 Super Trait 关联项
+-------------------------------+
| fn <T as Grand>::grand_fun2   |
+-------------------------------+
| fn <T as Parent>::parent_fun1 |
+-------------------------------+
| fn <T as Parent>::parent_fun2 |
+-------------------------------+
| fn <T as Trait>::fun          |               // Trait 自身关联项
+-------------------------------+
```

-   虚表内存布局
    -   虚表 *Header* 总为 3 个 `usize` 结构
        -   `drop_in_place` 析构函数指针
        -   `size` 类型大小
        -   `align` 类型内存对齐
            -   `size`、`align` 两结构共同构成 `alloc::Layout` 结构，指示内存分配、释放方案
    -   *Header* 后即为 *Trait* 中关联函数指针
        -   对无 *Super Trait* 的 *Trait*，虚表内存布局与 *Trait* 中定义的关联项一致
        -   对单线 *Super Trait* 的 *Trait*，虚表以 *Super Trait* 中关联项开头
            -   此时，虚表截断即为 *Super Trait* 对应 *Trait Object* 虚表
            -   即，此时将 `dyn Trait` 类型向上转换为 `dyn Parent` 类型时，无需修改 `dyn Trait` 类型宽指针的 *metadata*（虚表指针）
        -   对带分支 *Super Trait* 的 *Trait*，虚表
            -   此时，虚表截断部不总为 *Super Trait* 对应 *Trait Object* 虚表
            -   即，此时类型向上转换，可能需要修改 `dyn Trait` 类型宽指针的 *metadata*（虚表指针）

```rust
pub trait Base {
    fn base_fun1(&self);
    fn base_fun2(&self);
}

pub trait Left : Base {
    fn left_fun1(&self);
    fn left_fun2(&self);
}

pub trait Right : Base {
    fn right_fun1(&self);
    fn right_fun2(&self);
}

pub trait Trait : Left + Right {
    fn fun(&self);
}
// *************************** 带分支 *Super Trait* 的 `dyn Trait`
+-----------------------------+
| fn drop_in_place(*mut T)    |
+-----------------------------+
| size of T                   |
+-----------------------------+
| align of T                  |
+-----------------------------+
| fn <T as Base>::base_fun1   |
+-----------------------------+
| fn <T as Base>::base_fun2   |
+-----------------------------+
| fn <T as Left>::left_fun1   |
+-----------------------------+
| fn <T as Left>::left_fun2   |
+-----------------------------+
| fn <T as Right>::right_fun1 |
+-----------------------------+
| fn <T as Right>::right_fun2 |
+-----------------------------+
| ptr to <T as Right>::vtable |
+-----------------------------+
| fn <T as Trait>::fun        |
+-----------------------------+
```

> - *Rust* 虚表布局规则介绍：<https://zhuanlan.zhihu.com/p/680849759>
> - Vtable format to support dyn upcasting coercion：<https://rust-lang.github.io/dyn-upcasting-coercion-initiative/design-discussions/vtable-layout.html>

####    泛型实现 *Trait* 对象

```rust
// ******************************* core::task::wake
pub struct RawWaker {                           // 代表 *Trait* 对象的宽指针
    data: *const (),                            // 值指针，单元裸指针 `*const ()` 擦除具体类型
    vtable: &'static RawWakerVTable,            // 虚表指针，对不同类型指向不同虚表结构体
}
impl RawWaker {
    pub const fn new(                           // 常量函数，仅编译时执行一次
        data: *const (),
        vtable: &'static RawWakerVTable,
    ) -> RawWaker {
        RawWaker{ data, vtable }
    }
}
pub struct RawWakerVTable {                     // 虚表结构体，包含 4 个函数指针（省略 `::new` 方法）
    clone: unsafe fn(*const ()) -> RawWaker,
    wake: unsafe fn(*const ()),
    wake_by_ref: unsafe fn(*const ()),
    drop: unsafe fn(*const ()),
}
impl RawWakerTable {
    pub const fn new (                          // 常量函数，被 `waker_vtable` 调用仅在编译时（为各类型）执行一次
        clone::unsafe fn(*const ()) -> RawWaker,
        wake: unsafe fn(*const ()),
        wake_by_ref: unsafe fn(*const ()),
        drop: unsafe fn(*const ())
    ) -> Self {
        Self { clone, wake, wake_by_ref, drop }
    }
}
pub struct Waker {
    waker: RawWaker,
}
impl Unpin for Waker {}
unsafe impl Send for Waker {}                   // 要求实现者保证标记特性
unsafe impl Sync for Waker {}
impl Waker {
    pub fn wake(self) {
        let this = ManuallyDrop::New(self);     // `self` 将在 `RawWakerTable.wake` 中被释放
        unsafe { (this.waker.vtable.wake)(this.waker.data) };
    }
    pub fn wake_by_ref(&self) {
        unsafe { (this.waker.vtable.wake_by_ref)(self.waker.data) };
    }
}

// ****************************** futures_task::arc_wake
pub trait ArcWake: Send + Sync {                // 实现 `ArcWake` 的类型将根据类型动态分派方法
    fn wake(self: Arc<Self>) {                  // 方法与虚表类 `RawWakerVTable` 同名，类型的具体实现将被调用（即动态分派）
        Self::wake_by_ref(&self)
    }
    fn wake_by_ref(arc_self: &Arc<Self>);
}

// ****************************** futures_task::waker
pub(super) fn waker_vtable<W: ArcWake + 'static>() -> &'static RawWakerTable {
    &RawWakerVTable::new(                       // 编译时创建虚表，各 `W` 单态化类型所有实例共用一个虚表
        clone_arc_raw::<W>,
        wake_arc_raw::<W>,                      // 下仅列出 `wake`、`wake_by_ref` 函数实现
        wake_by_ref_arc_raw::<W>,
        drop_arc_raw::<W>,
    )
}
unsafe fn wake_arc_raw<T: ArcWake + 'static>(data: *const ()) {
    let arc: Arc<T> = unsafe { Arc::from_raw(data.cast::<T>()) };   // 将单元类型转换为具体类型
    ArcWake::wake(arc);                         // 封装 `trait ArcWake` 的关联函数 `ArcWake::wake`
}
unsafe fn wake_by_ref_arc_raw<T: ArcWake + 'static>(data: *const ()) {
    let arc = mem::ManuallyDrop::new(unsafe { Arc::<T>::from_raw(data.cast::<T>()) });
    ArcWake::wake_by_ref(arc);                  // 封装 `trait ArcWake` 的关联函数 `ArcWake::wake_by_ref`
}
pub fn waker<W>(wake: Arc<W>) -> Waker
where
    W: ArcWake + 'static,
{
    let ptr = Arc::into_raw(wake).cast::<()>();
    unsafe { Waker::from_raw(RawWaker::new(ptr, waker_vtable::<W>())) }
}
```

> - `core::task::wake`：<https://doc.rust-lang.org/src/core/task/wake.rs.html>
> - `futures_task::waker`：<https://docs.rs/futures-task/latest/src/futures_task/waker.rs.html>
> - `futures_task::arc_wake`：<https://docs.rs/futures-task/latest/src/futures_task/arc_wake.rs.html>

##  所有权、指针（引用）、生命周期

### 所有权、`Copy`、`Clone`

-   所有权：指定 *Rust* 程序 **管理内存** 的一组规则
    -   即，变量拥有值实质上指 **变量被丢弃时，值所在内存资源被释放**
    -   所有权规则
        -   每个值在某个时点都有、且仅有唯一所有者（变量）
        -   所有者（变量）超出作用域时，值被丢弃
    -   “赋值” 时，对 `impl Copy` 类型，*Rust* 执行复制，否则 **执行移动**
        -   *Move* 移动：所有权转移，将原变量绑定值移动给（赋值给）新变量
            -   即，浅拷贝的同时，无效原变量（变量从逻辑上变成未初始化）
            -   移动可能发生在赋值、函数传参时
        -   *Copy* 复制：原变量对值的所有权保留，并逐位复制绑定给新变量

> - 4.1 什么是所有权：<https://www.rust-book-cn.com/ch04-01-what-is-ownership.html>
> - 5.1 受检查的初始化内存：<https://doc.rust-lang.net.cn/nomicon/checked-uninit.html>
> - 5.1 Checked ：<https://doc.rust-lang.org/nomicon/checked-uninit.html>
> - Rust 在什么情况下，必须手写 `Drop`，释放资源？：<https://www.zhihu.com/question/653741156/answer/3475999527>

####    `Copy`、`Clone`

```rust
pub trait Copy: Clone {}                                    // `Clone` 是 `Copy` 的 *Super Trait*

pub trait Clone: Sized {
    fn clone(&self) -> Self;
    fn clone_from(&mut self, source: &Self) {...}
}
```

-   `marker::Copy`：指示类型逐位复制即可得到副本的标记特性
    -   “赋值” 时，对 `impl Copy` 类型，*Rust* 执行复制，否则 **执行移动**
        -   `Copy` 仅能对所有成员均 `impl Copy` 的类型实现
        -   否则，编译器将报错
            -   `impl Drop`、“拥有” 堆上内存
            -   `&mut` 可变引用
    -   `Copy`、`Clone` 均可通过 `#derive[]` 为自定义类型实现
        -   `Copy` 是隐式的、不可重载的逐位复制，是标记特性
        -   `Clone` 必须显式调用、可自定义 “克隆” 语义的特性

> - `std::marker::Copy`：<https://doc.rust-lang.org/std/marker/trait.Copy.html>

####    `marker::Unpin`

```rust
let mut p = Box::new(String::from("hello"));
println!("{:p}"), &*p);                             // 堆上地址
let a: String = *p;
println!("{:p}"), &a);                              // 赋值后必然移动到栈上，地址变化
```

-   `marker::Unpin`：指示类型值 **可在内存中移动位置** 的标记特性
    -   *Rust* 中所有值都 **能、会** 在内存中移动（编译器无限制）
        -   编译器可 **任意将值在移动中内存位置、且无地址变动通知**
        -   说明：*内存移动* 与 *所有权移动* 不同
            -   内存移动：指值内存地址的变动
            -   *Move* 所有权移动：语义上的移动，可伴随内存移动（也可能被编译器优化）
            -   *Copy* 拷贝：语义上对应所有权移动，同样伴随内存移动
        -   对具体的变量，内存移动可通过变量（值所有者，包括智能指针）、可变引用 `&mut` 实现
            -   显然，变量在转移所有权时可能触发内存移动
            -   `&mut` 可被 `mem::replace`、`mem::swap` 等用于内存移动
    -   大部分类型 **可** 在内存中移动而不影响程序运行，即 `impl Unpin`
    -   但，*Self Referential* 自引用等类型的内存移动可能影响程序运行，即 `impl !Unpin`
        -   标准库中 `impl !Unpin` 类型
            -   `std::marker::PhantomPinned`：用于标记自定义 `!Unpin` 类型（`!Unpin` 可传染）
            -   `impl Future` 异步线程块对象
        -   为此，需要 `pin::Pin<P<T>>` 包装、**防止编译器值移动值**
            -   即 *Pinning Guarantee* 固定保证：值在被丢弃前内存地址不会改变

> - `std::pin`：<https://doc.rust-lang.org/std/pin/index.html>
> - Rust 的 `Pin` 与 `Unpin`：<https://folyd.com/blog/rust-pin-unpin/>
> - Rust `Pin` 进阶：<https://folyd.com/blog/rust-pin-advanced/>、<https://juejin.cn/post/7064473476173660190>
> - Rust 自引用结构、`Pin` 和 `Unpin`：<https://zhuanlan.zhihu.com/p/600784379>

#####   `pin::Pin`

```rust
pub struct Pin<Ptr> {
    pointer: Ptr,
}
```

-   `std::pin::Pin<P<T>>`：固定（智能）**指针指向值** 内存位置、避免值移动导致 `!Unpin` 值失效的包装器
    -   `pin::Pin<P<T>>` 仅仅通过 **限制对 `T: !Unpin` 获取 `&mut T`、`P<T>`**  避免内存移动（**不涉及编译器特殊处理**）
        -   限制指，可获取 `&mut T`、`P<T>` 方法对 `T: !Unpin` 均被标记为 `unsafe` 不安全
            -   `unsafe` 方法即 `Pin` 中一系列 `_unchecked` **不检查 `Unpin` 的** 方法
            -   即，若类型 `T: !Unpin`，则需 `unsafe {}` 块才能获取 `P<T>`、`&mut`
            -   即在值不可内存移动、但有可能移动时，要求用户保证无内存移动
        -   同时，`Pin` 中包装 `&mut T`、`P<T>`，确保在 `Pin` “生命周期” 内，无法绕过 `Pin` 移动值
            -   `&'a mut T`：根据引用规则，在生命周周期 `'a` 范围内无法创建引用
            -   `P<T>`：智能指针拥有数据，由智能指针限制值 `T` 移动
            -   其他说明
                -   `Pin` 的 “生命周期” 指内部引用 `&'a mut T` 的生命周期 `'a`，可能早于 `Pin` 被丢弃
                -   部分智能指针 `Rc<T>`、`RefCell<T>` 在运行时才检查引用规则
    -   对 `T: Unpin` 则应、可使用对应安全方法，与 `P<T>` 无区别
    -   说明
        -   `P<T>` 指 `impl Deref` 的智能指针、拥有 `T` 值，获取 `P<T>` 显然可以移动值
        -   `pin::Pin` 在 `T: Unpin` 时实现有 `Unpin`、`DerefMut`，是智能指针

| `Pin<P<T>>` 方法                     | `T: Unpin` 额外 | 说明                 | 其他                        |
|--------------------------------------|-----------------|----------------------|-----------------------------|
| `new_unchecked(pointer: P)`          | `new()`         | 创建 `Pin`           | 关联函数                    |
| `into_inner_unchecked(pin: Pin<P>)`  | `into_inner()`  | 解包为 `P<T>`        | 关联函数                    |
| `get_unchecked_mut(self)`            | `get_mut()`     | 获取 `&mut T`        | 仅 `P<T>` 为 `&mut T`       |
| `get_ref(self)`                      | 不区分          | 获取 `&T`            | 仅 `P<T>` 为 `&T`           |
| `as_ref(&self)`                      | 不区分          | 转换为 `Pin<&T>`     |                             |
| `as_mut(&mut self)`                  | 不区分          | 转换为 `Pin<&mut T>` | 仅 `T: DerefMut`            |
| `as_deref_mut(self: Pin<&mut Self>)` | 不区分          | 解包为 `Pin<&mut T>` | 对嵌套 `Pin<&mut Pin<P<T>>` |

> - `std::pin::Pin`：<https://doc.rust-lang.org/std/pin/struct.Pin.html>
> - Rust 的 `Pin` 与 `Unpin`：<https://folyd.com/blog/rust-pin-unpin/>
> - Rust `Pin` 进阶：<https://folyd.com/blog/rust-pin-advanced/>、<https://juejin.cn/post/7064473476173660190>
> - Rust 自引用结构、`Pin` 和 `Unpin`：<https://zhuanlan.zhihu.com/p/600784379>
> - 定海神针 `Pin`、`Unpin`：<https://course.rs/advance/async/pin-unpin.html>
> - Rust 的 Pin 机制：<https://www.cnblogs.com/RioTian/p/18135131>
> - 17.5 深入研究 `Async` 的特性：<https://www.rust-book-cn.com/ch17-05-traits-for-async.html>
> - 25.4 协程 - Rust 异步的基石 - `Pin` 精准的固定：<https://uxiew.github.io/rust_study/%E7%AC%AC%2025%20%E7%AB%A0%20%E5%8D%8F%E7%A8%8B/25.4%20%E5%8D%8F%E7%A8%8B.html>

#####   *pinning* 的说明

```rust
pub marco pin($value:expr $(,)?) {
    {
        super let mut pinned = $value;
        unsafe { $crate::pin::Pin::new_unchecked($mut pinned) }
    }
}
```

-   注意：`Pin<P<T>>` 协议是保证 `T:!Unpin` 值 **在被丢弃前不会发生内存移动**
    -   即实务中，**值无内存移动应是针对值整个存续期间的要求**
        -   毕竟，`T:!Unpin` 即表示任何的内存移动将导致 *Undefined Behavior*
        -   事实上，更多场合是 **函数要求实参在多次调用间无内存移动**
            -   `Future::poll(self: Pin<&mut Self>, cx: &mut Context<'_>)`
    -   但，编译器 **仅能保证在 `Pin<P<T>>` “生命周期” 内不发生内存移动**
        -   即，编译器无法确保在 `Pin<P<T>>` 的 “生命周期” 之外不发生内存移动
        -   则，**在可绕过 `Pin` 访问值的情况下，总可能安全的移动 `T:!Unpin` 值**
            -   故，`Pin::new_unchecked` 构造 `Pin` 是 `unsafe` 的
    -   因此，安全地 *pinning* `T:!Unpin` 核心即确保 **`Pin` 丢弃后值被丢弃、或无法访问**
        -   `Pin<P<T>>` 作为参数由外部传递：实质上是将 `unsafe` 外移
            -   且参数不应该包含其他 `P<T>`，否则依然可能在函数内部移动值
        -   `Box::pin()`、`pin::pin!()` 获取所有权：`Pin` 丢弃时值被丢弃
            -   `Box::pin()` 将值固定在堆上，由 `Box` 唯一拥有
            -   `pin::pin!()` 利用 `super let` 获取值所有权并延迟丢弃

```rust
struct SelfRef {                                            // 存在自引用 `!Unpin` 类型
    phantom: std::mark::PhantomPinned;
}
impl SelfRef {
    fn init() { }
    fn call_pinned(self: &Pin<&mut self>);                  // 签名中 `Pin` 即要求 `self` 内存固定
}

let mut s = SelfRef::init();
{ unsafe { pinn::new_unchecked(&mut s) }.call_pinned(); }
let mut news = s;                                           // 两次 `Pin` 需求之间可能移动值
{ unsafe { pinn::new_unchecked(&mut s) }.call_pinned(); }   // 值已移动，方法签名对内存固定要求未被满足
```

> - Why it could be UB that `Pin::new_unchecked` on an `&'a mut T` has dropped?：<https://users.rust-lang.org/t/why-it-could-be-ub-that-pin-new-unchecked-on-an-a-mut-t-has-dropped/90725>
> - `std::pin::Pin`：<https://doc.rust-lang.org/std/pin/struct.Pin.html>
> - Rust `Pin` 进阶：<https://folyd.com/blog/rust-pin-advanced/>、<https://juejin.cn/post/7064473476173660190>

### 引用

####    指针、别名

-   *Pointer* 指针：**包含内存地址的变量的通用概念，其中地址指向、“引用” 其他数据**
    -   *Rust* 中最常见指针类型即引用 `&`
-   *Alias* 别名：变量、指针 “引用” 的内存区域重叠时则 “存在别名”（重叠者 “互为别名”）
    -   *Rust* 中变量为值所有者，指针即为引用、智能指针等
    -   编译器可通过分析别名执行（其他语言不支持）优化
        -   *Rust* 引用规则限制了对值的写操作（可变变量、可变引用唯一）
        -   而，写操作是优化的主要障碍

> - 4.2 引用与借用：<https://www.rust-book-cn.kcom/ch04-02-references-and-borrowing.html>
> - 3.2 Aliasing：<https://doc.rust-lang.org/nomicon/aliasing.html>
> - 3.2 别名：<https://doc.rust-lang.net.cn/nomicon/aliasing.html>

####    引用、切片

-   *Reference* 引用（类型） `&`：可用于访问其包含（指向）地址上存储的数据的指针
    -   引用仅仅 **借用其指向的（所引用的变量绑定的）值**
        -   两种引用
            -   *Shared Reference* 共享、不可变引用 `&`
            -   *Mutable Reference* 可变引用 `&mut`
        -   *Borrow* 借用：创建引用的行为
            -   或许，**特指创建可变引用，此时所有者变量不可用**
        -   引用没有其他特殊功能，也没有额外开销
    -   结构体字段可以分别独立创建可变引用，不违反引用规则
        -   但，不可为数组切片、单独元素安全的创建多个可变引用

```rust
// ************************* 结构体字段可以分别创建可变引用
struct Foo {
    a: i32,
    b: i32,
    c: i32,
}
let mut x = Foo {a: 0, b: 0, c: 0};
let a = &mut x.a;
let b = &mut x.b;
let c = &x.c;                                   // 编译通过

// ************************* 数组不可创建元素、切片的多个可变引用
let mut x = [1, 2, 3];
let a = &mut x[0];
let b = &mut x[1];                              // 数组 `x` 创建多个可变引用，编译失败
println!("{} {}", a, b);

// ************************* `&mut [..]` 可变的切片类型方法，返回两个可变切片
pub fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
    let len = self.len();
    let ptr = self.as_mut_ptr();
    unsafe {                                    // 只能 `unsafe` 的创建两个可变切片
        assert!(mid <= len);
        (from_raw_parts_mut(ptr, mid),
         from_raw_parts_mut(ptr.add(mid), len - mid))
    }
}
```

> - 4.2 引用与借用：<https://www.rust-book-cn.kcom/ch04-02-references-and-borrowing.html>
> - 3.11 拆分借用：<https://doc.rust-lang.net.cn/nomicon/borrow-splitting.html>
> - 3.11 Splitting Borrows：<https://doc.rust-lang.org/nomicon/borrow-splitting.html>

####    引用规则、检查

-   引用遵守以下 2 条规则：别名规则（编译器 **通过生命周期检查引用是否遵守规则**）
    -   引用的生命周期不可超过其借用对象：确保值有效性
        -   保证引用在其生命周期内指向的特定类型值有效
        -   对值所有者，值仅随所有者清理而被释放，无需考虑值被清理而所有者仍存在
        -   但对引用（变量），**其借用值是否仍存在并不确定**
            -   即，引用作用域、指向值的作用域不一定满足要求
            -   需要额外检查，防止 *Dangling Reference* 悬垂引用
    -   可变引用不能有别名：避免数据竞争
        -   即，任意时间点，**值** 只有一个 **可用的可变引用（或所有者）**、或多个不可变引用（及所有者）
        -   变量只可存在一个可变引用 `&mut`、或多个不可变引用 `&`
            -   对不可变变量，不可创建可变引用
            -   对可变变量
                -   **可变引用 `&mut` 借用归还前，所有者不可被使用**
                -   **不可变引用 `&` 借用归还前，所有者不可用于修改值、被重新赋值**
        -   多数时候可变引用 `&mut` 在函数、方法调用时被创建，作用域局限在函数体内
            -   即，除非显式手动创建，`&mut` 可变引用不会与所有者作用域重叠

> - 3.1 引用：<https://doc.rust-lang.net.cn/nomicon/references.html>
> - 3.2 别名：<https://doc.rust-lang.net.cn/nomicon/aliasing.html>
> - 3.1 References：<https://doc.rust-lang.org/nomicon/references.html>
> - 3.2 Aliasing：<https://doc.rust-lang.org/nomicon/aliasing.html>
> - `std::ptr` - Pointer to Reference Conversion：<https://doc.rust-lang.org/stable/std/ptr/index.html#pointer-to-reference-conversion>

#####   运行时检查

```rust
let mut a = Rc::new(1);
let b = Rc::clone(&a);
let c = Rc::get_mut(&mut a).unwrap();           // 可编译通过，但运行时会 panic
let d = b;
```

-   注意，编译器的引用检查 **局限在单个函数内部的 `&`、`&mut` 引用类型（操作符）**（显式借用、方法中 `self`）
    -   即，通过检查 `&`、`&mut` 引用类型的生命周期冲突确定是否违反引用规则
        -   但，检查 **仅限于包含引用（或包含的引用结构）的类型（只有引用类型有生命周期）**
        -   智能指针无生命周期参数，编译器无法检查
    -   即，若单个函数内借用满足 “值有最多单一可变引用（所有者）”，则编译通过
        -   对指令，**编译器逐（调用栈）层内检查借用**，递归的确保整体满足要求
        -   对数据又有，结构体中成员可变性与所有者（引用）可变性一致
            -   即正常情况下，无法修改不可变所有者、引用的成员值
            -   除非，通过 `unsafe` 方式从不可变获取可变，实现不可变外层的内部可变性
    -   此时，需要智能指针 **自行实现运行时单一可变引用检查**
        -   `RefCell<T>.borrow_mut(&self)` 多次调用、创建、返回多个 `RefMut<'a, T>` 可正常通过编译
            -   解糖为 `borrow_mut(self: &'a Self) -> RefMut<'a, T>`，其中 `self` 参数为自身的不可变引用
            -   故，多个 `RefMut<'a, T>` 可共享借用 `RefCell`，不违反编译器的引用规则（也同时限制对 `RefCell<T>` 的可变借用）
            -   但，运行时检查共享的 `RefMut<'a, T>.borrow: &Cell<Counter>`、存在多个有效的可变引用时将 `panic`
        -   `Rc::get_mut(this: &mut self)` 与多个 `Rc::clone(&self)` 共存可正常通过编译
            -   `Rc::clone(&self)` 返回的 `Rc<T>` 中无引用（**无生命周期参数**）

> - RefCell, why borrowing is checked only in runtime：<https://users.rust-lang.org/t/refcell-why-borrowing-is-checked-only-in-runtime/52721/5>
> - Rust 的源码阅读：`Cell/RefCell` 与内部可变性：<https://zhuanlan.zhihu.com/p/384388192>

### 生命周期

```rust
fn out_sum<'a, T>: Summary>(x: &'a T, y: &'a T) -> &'a T{   // 函数中泛型、生命周期标注
    x
}
struct Except<'a> {                                         // 结构体生命周期标注
    part: &'a str,
}
```

-   生命周期：代码中由编译器确定的、引用 **须** 保持有效的具名区域（某作用域）
    -   简单情况下，引用的生命周期与其作用域 **重合**（但不是同一个东西）
    -   但，若将 **引用传递（移动）至外部作用域，编译器会为引用推断出更大的生命周期**
        -   生命周期实际上是，编译器推断的、**引用所指向值须有效的最小范围**
        -   所以，若引用被传递值外部作用域，则值须保证有效的范围将扩大（大于引用本身作用域）
    -   生命周期是 **特殊的泛型、是（引用）类型的一部分**
        -   **类型相同的引用包括生命周期相同**
        -   只是，在函数内部，**通常无需显式命名（标注）涉及的生命周期**
            -   编译器拥有函数内部变量、引用的所有信息，可自动推断引用最优（最小范围）生命周期
        -   若需在函数内标注生命周期，需在函数签名中先标注、声明
    -   关于生命周期的其他说明
        -   因为存在变量遮蔽，引用的生命周期可能是不连续的（包含空洞）
        -   实际上，每个 `let` 语句创建变量都引入一个作用域，即 “候选生命周期”
        -   但，引用的 **生命周期只有终止位置有意义，起始位置其实无意义**
            -   引用必须声明后使用，则引用声明前的区域也可认为是其生命周期的一部分
            -   即，引用的生命周期总是可视为 **起始于代码块开头**
            -   此结论可从 `&mut (&'a T)` 可变引用对 `&'a T` 不便宜构造案例验证

```rust
// ****************** 引用 `&x` 生命周期被扩大
let x = 0;
let z;
let y = &x;
z = y;

'a: {                                               // 不合法的命名作用域标记，仅作分析
    let x: i32 = 0;
    'b: {
        let z: &'b i32;
        'c: {
            let y: &'b i32 = &'b x;                 // `&x` 的生命周期被扩大至 `'b`，否则为 `'c`
            z = y;
        }
    }
}
```

> - 15.4 生命周期：<https://doc.rust-lang.org/stable/rust-by-example/zh/scope/lifetime.html>

####    生命周期标注

```rust
// ********************** 编译失败
fn lifetime_fail<'a, 'b:'a>() {                 // 函数签名中要求 `'b:'a`
    let hello: &'a str = "hello";
    let world = String::from("world");
    let rw: &'b str = &world;                   // 但此处生命周期标注表明无法满足 `'b:'a`
}
```

-   生命周期标注 `'`：描述多个引用的生命周期之间的关系的 **泛型参数**
    -   生命周期标注可视为 **生命周期 “形参”**
        -   编译器将选择合适的生命周期（实参）替代生命周期标注
        -   函数的生命周期标注对引用的实际生命周期没有影响
    -   函数内部 **仅能使用在函数签名中声明** 的生命周期标注、`'static` 静态生命周期
        -   此时，引用的生命周期被显式指定、并被编译器检查
        -   同时，**函数签名中生命周期范围均大于函数内部引用生命周期**（外部传入引用，显然）
        -   实际上，一般无需在函数体内显式标注生命周期
    -   跨越函数边界（调用函数时），编译器无法获取外部函数全部信息，需要额外的生命周期标注
        -   **函数签名中的生命周期标注（之间的关系）是函数对参数、返回值中引用的限制、要求**
        -   毕竟，生命周期是引用类型的一部分，函数对参数、返回值类型应有限制，并体现、暴露在函数签名中
            -   对函数自身，返回值中引用的生命周期需满足函数签名要求
            -   对调用者，参数、返回值中引用也应满足函数签名要求
        -   同样的，生命周期支持 *Subtyping and Variance* 子类型、变异（类型对直接生命周期总是协变）
            -   在不考虑被调用函数签名（变异前）情况下
                -   实参中引用的生命周期 `'b` 只需是函数签名中形参的生命周期 `'a` 的子类型 `'b:'a` 即可，不必完全一致
                -   可视为，函数签名中 **生命周期形参 `'a` 被置为（泛型单态化）包含 `'a` 的各引用生命周期交集**
                -   即常见描述的，引用的实际生命周期（变异前）不小于函数签名中生命周期即可
            -   **当然引用的实际生命周期时变异后生命周期：参数、返回值中引用的生命周期并集**
        -   即，函数签名中生命周期实际上应、用于体现 **参数、返回值之间有效性的依赖关系**

> - 10.3 使用生命周期验证引用：<https://www.rust-book-cn.com/ch10-03-lifetime-syntax.html>
> - 2.10 认识生命周期：<https://course.rs/basic/lifetime.html>
> - 4.1.1 深入生命周期：<https://course.rs/advance/lifetime/advance.html>

#####   生命周期标注省略

```rust
impl<'a> Reader for BufReader<'a> {}            // 生命周期 `'a` 未被使用
impl Reader for BufReader<'_> {}                // 则可被省略为匿名生命周期 `'_`

struct Ref<'a, T:'a> {
    r: &'a T
}
struct Ref<'a, T> {                             // 省略泛型参数的生命周期约束
    r: &'a T
}
```

-   （函数签名）生命周期标注省略规则：编译器在没有显式标注情况下根据以下 3 条件规则推断函数签名中所有引用的生命周期
    -   每个参数分配单独生命周期参数
    -   若只有 1 个生命周期参数，改生命周期被分给所有输出作为生命周期参数
    -   若有 `&self`、`&mut self` 作为参数（即方法），则 `self` 的生命周期参数被分配给所有输出作为生命周期参数
-   其他生命周期标注特殊情况
    -   `'_` *Anonymous Lifetime* 匿名生命周期：类型签名中包含生命周期，但是在 `impl` 块、`fn` 中未使用，可用 `'_` 替代
        -   匿名生命周期是指生命周期可 **由编译器自行推断、故可用 `'_` 占位**
        -   即，依然被编译器检查、替换为实际生命周期
        -   若，匿名生命周期无法被正确推动、替换，则依然无法编译通过
    -   `struct` 声明中，对泛型的生命周期约束可以被省略

> - 10.3 使用生命周期验证引用：<https://www.rust-book-cn.com/ch10-03-lifetime-syntax.html>
> - 2.10 认识生命周期：<https://course.rs/basic/lifetime.html>
> - 4.1.1 深入生命周期：<https://course.rs/advance/lifetime/advance.html>

#####   生命周期约束

```rust
struct DoubleRef<'a, 'b:'a, T> {                    // 生命周期比较限制，`'b` 须长于 `'a`，即 `'b>='a`
    r: &'a T,
    s: &'b T,
}

struct Closure<F> {
    data: (u8, u16),
    func: F,
}

impl<F> Closure<F>
    where for<'a> F: Fn(&'a (u8, u16)) -> &'a u8,   // 对任意生命周期 `'a`，泛型 `F` 均应满足约束
{
    fn call(&self) -> &u8 {
        (self.func)(&self.data)
    }
}

fn do_it(data: &(u8, u16)) -> &u8 { &data.0 }

fn main() {
    let clo = Closure { data: (0, 1), func: do_it };
    println!("{}", clo.call());
}
```

-   *Lifetime Bound* 生命周期约束
    -   `'b:'a` 生周周期子类型：指示不同生命周期范围大小，生命周期 `'b` 须大于生命周期 `'a`
    -   `T:'a` 泛型生命周期约束：泛型 `T` **中引用的生命周期** 须长于生命周期 `'a`
    -   *Higher-Ranked Trait Bounds* 高阶 *trait* 约束 `for<'a>`：对任意生命周期都成立
        -   在生命周期 `'a` 定义前，要求泛型满足对所有生命周期都满足某约束
        -   主要用于泛型参数需满足 `Fn` 约束、且 `Fn` 约束需标注生命周期

> - 3.7 高阶 *trait* 约束：<https://doc.rust-lang.net.cn/nomicon/hrtb.html>
> - Rust 中高阶 trait 边界（HRTB）中的生命周期：<https://www.cnblogs.com/jzssuanfa/p/19373681>

#####   特殊生命周期 `'static`、无界

-   特殊生命周期
    -   `'static` 静态生命周期：指示引用可在整个程序期间存活
        -   静态生命周期超过单个函数，可跨越线程
            -   字符串字面量、特征对象拥有 `'static` 生命周期
            -   常用于作泛型约束 `T: 'static`：泛型 `T` 中引用只能为静态生命周期
                -   实际上，`T: 'static` 约束可直接视为 **要求 `T` 不包含（为）引用**
        -   静态声明周期似乎是 “确定类型”，不是 “泛型”
    -   *Unbound Lifetime* 无界生命周期：凭空产生的引用的生命周期
        -   无界生命周期会根据上下文需要任意扩展，甚至比 `'static` “更泛用”
            -   无界生命周期可以转换为 `'static`
        -   无界生命周期来源
            -   对函数，任何不来源于输入参数的输出生命周期都是无界的
            -   对结构体，未在内部字段中引用使用的生命周期（但是无法编译通过）
            -   最常见的来源是对解引用的裸指针取引用 `&*(s: *const)`

> - 4.1.2 `&'static` 和 `T: 'static`：<https://course.rs/advance/lifetime/static.html>
> - 4.1.1 深入生命周期：<https://course.rs/advance/lifetime/advance.html>
> - 3.7 无界生命周期：<https://doc.rust-lang.net.cn/nomicon/unbounded-lifetimes.html>
> - 3.7 Unbounded Lifetimes：<https://doc.rust-lang.org/nomicon/unbounded-lifetimes.html>
> - 常见的 Rust Lifetime 误解：<https://zhuanlan.zhihu.com/p/165976086>
> - Common Rust Lifetime Misconceptions：<https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md>

####    子类型和型变（变异性）

```rust
fn assign<T>(input: &mut T, val: T) {
    *input = val;
}

// ************************ 编译失败
fn main() {
    let mut hello: &'static str = "hello";          // 1-pre. `&mut hello` 是 `&mut &'static str` 类型
    {
        let world = String::from("world");          // 1-pre. `&world` 是 `&'world String` 类型，即泛型 `T` 单态化结果
        assign(&mut hello, &world);                 // 1. `&mut T` 对 `T` 不变，`&mut hello` 不可降级为 `&mut &'world str`
                                                    //     即无法满足 `assign` 签名中两个 `T` 类型完全一致
    }                                               // 3. 即使此处无 `{}` 分隔作用域，实际无悬垂，也无法编译
    println!("{hello}");                            // 2. 否则，可能出现悬垂指针
}

// ************************ 编译通过
fn main() {
    let world = String::from("world");
    let refh = &mut (&world[..]);                   // 4-pre. `refh: &mut &'world str`
    {
        let hello: &'static str = "hello";
        assign(refh, hello);                        // 4. `&T` 对 `T` 协变，`hello: &'static str` 可降级为 `&'world str`
    }
    println!("{refh}");
}

// ************************ 编译通过
fn main() {
    let hello = String::from("hello");
    let mut rref: &str = &hello;
    let world = String::from("world");
    assign(&mut rref, &world);                      // 5. `&world` 生命周期实际上与 `rref` 一致，并不更小，否则编译失败
    println!("{rref}");
}
```

-   *Subtyping* 子类型：类型 `Sub` 是 `Super` 的子类型，若 `Super` 的 “要求” 完全可由 `Sub` 满足
    -   *Rust* 中没有继承机制，子类型仅局限于生命周期的型变（变异性）
        -   对生命周期 `'b:'a`，**生命周期 `'b` 是生命周期 `'a` 的子类型**
    -   子类型可被 “降级” 为父类型，用于需要父类型的场合
        -   常见场合即赋值、函数调用
        -   因，类型 `&'a T` 对 `'a` 协变，若 `'b:'a` 则有 `&'b T: &'a T`
        -   故，即使函数签名中各引用参数的生命周期标注相同，引用实参可具有不同生命周期
-   *Variance* 型变、变异性：描述 “组合泛型” `F<Sub>`、`F<Super>` 之间的父子类型关系
    -   即，描述类型 `F` 与、组成 `F` 的类型的父子类型关系的相关性
        -   *Covariance* 协变性：若 `F` 是协变的，则 `F<Sub>` 是 `F<Super>` 的子类型 `F<Sub>:F<Super>`
        -   *Contravariance* 逆变性：若 `F` 是逆变的，则 `F<Super>` 是 `F<Sub>` 的子类型 `F<Super>:F<Sub>`
        -   *Invariant* 不变：若 `F` 是不变的，则 `F<Sub>`、`F<Super>` 之间不存在父子关系
    -   内置类型的型变
        -   “组合泛型” 对组成其的生命周期总是协变的
        -   “组合泛型” 对组成其一般泛型 `T` 其实是 **讨论 `T` 本身为引用、或包含引用** 的生命周期
            -   不可变 `&T`、`*const T`、`Box<T>`、`Vec<T>` 对其中引用泛型 `T` 协变
            -   **（内部）可变 `&mut T`、`*mut T`、`UnsafeCell<T>` 对其中泛型 `T` 不可变**
                -   考虑 `T` 对应父子类型 `&'long TT: &'short TT`
                -   若 `&mut &'long TT` 可被降级为 `&mut &'short TT`
                -   则在调用外部函数中，可被解引用后修改为 `&'short TT` 类型，而调用者不知情，可能出现悬垂引用
            -   `fn(T) -> U` 对引用参数 `T` 逆变、对返回值 `U` 协变
                -   将被传递给函数 `fn` 的参数需要为 `T` 的子类型，则若参数要求放松为 `T` 的父类型依然可行
    -   对包含（生命周期）泛型 `'a`、`T` 的自定义类型（分别独立考虑）
        -   若所有成员对 `'a`、`T` 均协变、逆变，则自定义类型对 `'a`、`T` 协变、逆变
        -   否则，自定义类型对 `'a`、`T` 不变

| “组合泛型” `F`  | `'a` | `T`      | `U`  |
|-----------------|------|----------|------|
| `&'a T`         | 协变 | 协变     |      |
| `&'a mut T`     | 协变 | **不变** |      |
| `*const T`      |      | 协变     |      |
| `*mut T`        |      | **不变** |      |
| `[T]`、`[T; n]` |      | 协变     |      |
| `fn(T) -> U`    |      | 协变     | 逆变 |
| `Box<T>`        |      | 协变     |      |
| `UnsafaCell<T>` |      | 不变     |      |
| `Vec<T>`        |      | 协变     |      |
| `Cell<T>`       |      | 不变     |      |
| `RefCell<T>`    |      | 不变     |      |


> - 3.8 子类型和协变性：<https://doc.rust-lang.net.cn/nomicon/subtyping.html>
> - 3.8 Subtyping and Variance：<https://doc.rust-lang.org/nomicon/subtyping.html>
> - 10.5 子类型和变异性：<https://doc.rust-lang.net.cn/reference/subtyping.html#variance>
> - 10.5 Subtyping and Variance://doc.rust-lang.org/reference/subtyping.html#variance>
> - Rust：深入理解 `PhantomData`：<https://zhuanlan.zhihu.com/p/533695108>

####    函数生命周期说明

```rust
// ****************** 引用传参导致
fn do_nothing(aref: &mut i32) -> &str {
    "hello, world!"                                     // 显然返回引用 `&str` 与 `aref` 无关，但编译失败
}

fn main() {
    let mut a = 0;
    let b = do_nothing(&mut a);                         // 2. 导致函数实参 `&mut a` 生命周期需扩展至函数结尾
    println!("{}", a);                                  // 3. 导致此处，可变引用存在别名
    println!("{}", b);                                  // 1. 引用 `b` 的生命周期需扩展至此
}

fn do_nothing(aref: &mut i32) -> &'static str {         // 返回值类型 `&'static str` 也可编译通过
    "hello, world!"
}
fn do_nothing<'a, 'b>(aref: &'a mut i32) -> &'b str {   // 分配不同生命周期（无界生命周期）也可编译通过
    "hello, world!"
}
```

-   在函数内部，通常无需显式命名（标注）涉及的生命周期
    -   编译器拥有函数内部变量、引用的所有信息，可自动推断引用最优（最小范围）生命周期，并被检查
        -   函数体自身局部：检查引用是否遵守 2 条引用规则
            -   比较可变引用的生命周期是否重叠即可
        -   被调用暴露：检查引用是否满足函数自身签名中生命周期标注
        -   调用外部函数：检查引用是否满足被调用函数签名 **类型（中生命周期）要求**（生命周期是引用类型的一部分）
            -   生命周期是特殊的泛型，函数签名中对引用的生命周期限制只能是 **多个引用的生命周期是否一致、是否满足父子类型（范围大小）**
                -   引用生命周期 **将尝试变异以满足函数签名中要求**
                -   引用生命周期的变异是 **真正改变引用的生命周期，不是仅为满足函数签名的暂时改变**
            -   对一致性要求，对若函数签名中引用对生命周期协变、逆变，则总可以找到满足签名的生命周期
                -   若协变，即各引用的生命周期并集
                -   即，**各引用的生命周期被调整为初始生命周期的最大者**（生命周期的起始位置无意义）
                -   故实际上，仅在引用对某生命周期不变、且不匹配时无法满足签名要求
                -   考虑到引用的变异性，一般 **仅可变引用内的引用的生命周期无法匹配**
        -   若以上校验不通过，即表示引用规则被违反
    -   注意，除显式命名引用外，**表达式中的匿名引用** 的生命周期同样被用于检查
        -   其中较易忽略的即，方法中实例自身的引用（尤其是可变引用）
        -   匿名引用作用域看似 “仅限于表达式所在语句”
            -   但，被调用函数（签名）对实参生命周期的要求
            -   匿名引用的实际生命周期往往被 **变异**、扩大化
    -   但，编译器生命周期的推断依然较为粗粒度
        -   尤其是，函数签名中生命周周期标注省略规则会将函数签名中引用分配相同生命周期
        -   将导致，无关的引用被推断为生命周期相关，进而导致生命周期扩大，违反引用规则
            -   当然在良好的实践中，函数返回的引用应与某个入参引用相关
            -   即，返回引用依赖参数引用
            -   则，参数引用理应扩大生命周期确保返回引用有效
        -   即，合理的生命周期标注应能解决问题
    -   另外，闭包与函数功能上类似，但闭包与宿主函数关系密切，可能捕获宿主函数中引用
        -   即，闭包的中引用的生命周期分散在宿主函数体内，难以分析
        -   故，**编译器难以推断闭包中引用的生命周期**，部分场景下无法替代函数

```rust
// ****************** 方法中实例隐式的可变引用导致编译失败
#[derive(Debug)]
struct Foo;

impl Foo {
    fn mutate_and_share(&mut self) -> &Self {
        &*self
    }
    fn share(&self) {}
}
fn main() {
    let mut foo = Foo;
    let loan = foo.mutate_and_share();                      // 方法中隐式的可变引用 `&mut foo`
    foo.share();
    println!("{:?}", loan);
}

impl Foo {
    fn mutate_and_share<'a>(&'a mut self) -> &'a Self {     // 按规则自动推断生命周期
        &'a *self
    }
    fn share<'a>(&'a self) {}
}
fn main() {
    'b: {
        let mut foo: Foo = Foo;
        'c: {
            let loan: &'c Foo = Foo::mutate_and_share::<'c>(&'c mut foo);
            'd: {
                Foo::share::<'d>(&'d foo);
            }
            println!("{:?}", loan);
        }
    }
}
```

> - 2.10 认识生命周期：<https://course.rs/basic/lifetime.html>
> - 4.1.1 深入生命周期：<https://course.rs/advance/lifetime/advance.html>
> - 3.3 生命周期：<https://doc.rust-lang.net.cn/nomicon/lifetimes.html>
> - 3.3 Lifetimes：<https://doc.rust-lang.org/nomicon/lifetimes.html>
> - 3.4 生命周期的限制：<https://doc.rust-lang.net.cn/nomicon/lifetime-mismatch.html>
> - 3.4 Limits of Lifetimes：<https://doc.rust-lang.org/nomicon/lifetime-mismatch.html>

### 智能指针

-   智能指针：行为类似指针、拥有额外元数据和功能的数据结构
    -   与引用（通常意义上的简单指针）相比
        -   智能指针 **大部分情况下拥有指向的数据**，而引用只是借用数据
        -   智能指针实现有 `Drop`、`Deref` 特性
            -   `Deref` 特性：自定义解引用运算符 `*`，使得智能指针实例可类似引用被使用
            -   `Drop` 特性：自定义智能指针实例超出作用域时行为
    -   某种意义上，`String`、`Vec` 也是智能指针
        -   `String` 类型拥有 `str`（包含指向堆上 `str` 的引用）

> - 15 智能指针：<https://www.rust-book-cn.com/ch15-00-smart-pointers.html>

####    `Box`、`Rc`、`Cell`

| 智能指针                    | 说明                                                                       | 线程安全版             |
|-----------------------------|----------------------------------------------------------------------------|------------------------|
| `alloc::boxed::Box<T>`      | **有所有权** 的堆内存分配、封装，对齐非同类数据                            |                        |
| `alloc::rc::Rc<T>`          | 引用计数器，共享数据                                                       | `std::sync::Arc<T>`    |
| `alloc::rc::Weak<T>`        | 弱引用计数，由 `Rc<T>::downgrade()` 避免引用循环                           |                        |
| `core::cell::Cell<T>`       | 不可借用的内部可变性，适合包装 `Copy` 类型数据、实现作为结构体成员独立可变 | `std::sync::Mutex<T>`  |
| `core::cell::RefCell<T>`    | 可借用的内部可变性，配合 `Rc<T>` 共享可变数据                              | `std::sync::RwLock<T>` |
| `core::cell::Ref<T>`        | 值的不可变 “引用”，`RefCell<T>.borrow()` 返回                              |                        |
| `core::cell::RefMut<T>`     | 值的可变 “引用”，`RefCell<T>.borrow_mut()` 返回                            |                        |
| `core::cell::UnsafeCell<T>` | 通过共享引用从不可变获取可变，`Cell`、`RefCell` 内核心                     |                        |

> - 15.1 使用 `Box<T>` 指向堆上的数据：<https://www.rust-book-cn.com/ch15-01-box.html>
> - 15.4 `Rc<T>`，引用计数智能指针：<https://www.rust-book-cn.com/ch15-01-box.html>
> - 15.5 `RefCell<T>` 和内部可变性模式：<https://www.rust-book-cn.com/ch15-05-interior-mutability.html>
> - 15.6 引用循环可能导致内存泄漏：<https://www.rust-book-cn.com/ch15-06-reference-cycles.html>
> - 16.3 共享状态并发：<https://www.rust-book-cn.com/ch16-03-shared-state.html>
> - 聊聊 Rust 的 `Cell` 和 `RefCell`：<https://zhuanlan.zhihu.com/p/659426524>
> - Rust 的源码阅读：`Cell/RefCell` 与内部可变性：<https://zhuanlan.zhihu.com/p/384388192>

####    `Deref` 解引用、解引用强制转换

```rust
std::ops::Deref;

pub const trait Deref: PointeeSized {
    type Target: ?Sized;
    fn deref(&self) -> &Self::Target;
}
impl<T: ?Sized> const Deref for &T {                    // 仅方便泛型编程实现
    type Target = T;
    #[rustc_diagnostic_item = "noop_method_deref"]      // noop 方法，调用无意义
    fn deref(&self) -> &T {
        self                                            // `&&T` 同样自动解引用为 `&T`，不影响
                                                        // 之前版本源码似乎为 `*self`
    }
}
impl<T: ?Sized> const !DerefMut for &T {}               // `DerefMut` 也存在对应 `!DerefMut`
impl<T: ?Sized> const Deref for &mut T {
    type Target = T;
    #[rustc_diagnostic_item = "noop_method_deref"]      // noop 方法，调用无意义
    fn deref(&self) -> &T {
        self
    }
}

pub const trait DerefMut: [const] Deref + PointeeSized {
    fn deref_mut(&mut self) -> &mut Self::Target;
}
```

-   `ops::Deref`：重载解引用运算符 `*` 行为，使得智能指针实例可类似引用被使用
    -   `*` （显式）解引用：“获取” 引用、智能指针指向的值
    -   对非引用类型 `&T`、`&mut T`，`*<impl Deref>` 将被 **自动转换** 为 `*Deref::deref(&<impl Deref>)`
        -   **显式解引用不会触发链式转换**
            -   即，若 `<impl Deref>.deref()` 依然返回 `<impl Deref>`，不会被继续转换
        -   类似的，在需要可变性的场合 `*<impl DerefMut>` 将被 **自动转换** 为 `*DerefMut::deref_mut`
    -   对 `&T`、`&mut T`、`Box<T>` 类型，**解引用操作为编译器特殊处理**，不涉及 `Deref`
        -   `impl Deref for &T、&mut T` 为所有引用类型实现 `Deref` 仅为方便 **泛型编程、多重引用归一化**
            -   泛型编程：使得 `Deref<Target=T>` 包含 `&T`、`&mut T`
            -   多重引用归一化：多重引用如 `&&T`、`&&&Box<T>` 等可被隐式转换为 `&T`
        -   同样的，`&mut T` 自动（强制）转换为 `&T` 也是编译器的特殊处理
    -   注意：`Deref::deref` 更应被理解为 **“解包” 而不是 “解引用”**（但在解引用时触发）
        -   `Deref::deref` 方法返回引用，是 “指针到指针的转换”
            -   并不是类似 `*` 真正从 “指针获取值”
            -   同时，避免所有权转移
        -   而是获取 “等价” 的引用（再由 `*` 真正 “解引用获取值”）

> - `std::ops::Deref`：<https://doc.rust-lang.org/std/ops/trait.Deref.html>
> - `std::ops::DerefMut`：<https://doc.rust-lang.org/std/ops/trait.DerefMut.html>
> - 4.4.2 `Deref` 解引用：<https://course.rs/advance/smart-pointer/deref.html>
> - 15.2 Treat Smart Pointers Like Regular References：<https://doc.rust-lang.org/stable/book/ch15-02-deref.html>
> - Rust为什么对所有引用类型（`&T`）实现 `Deref` 特征？：<https://www.zhihu.com/question/573398534/answer/2810145394>
> - 15.2 使用 `Deref` 将智能指针当作常规引用处理：<https://www.rust-book-cn.com/ch15-02-deref.html> 翻译有错

#####   *Deref Coercion* 解引用强制转换

```rust
let a = &&&&1_i32;
let p: &i32 = a;                                        // `let` 语句中的自动解引用

let a = &&&&Box::new(Box::new(1_i32));
let p: &i32 = a;                                        // `Box` 可被解引用，可印证为一般的解引用强制转换，不是对 `&&T` 特殊处理
// let p: Box<i32> = a;                                 // 编译失败，`Box<i32>` 不是引用类型，不支持解引用强制转换
```

-   **（链式）解引用 强制转换**：在 *Type Coercion* 类型转换场合，将 `&<impl Deref>` 引用类型（链式）转换为目标引用类型
    -   *Rust* 将自动、隐式、链式 **`Deref::deref` + 解引用**，直至得到目前引用类型
        -   函数参数、`let` 赋值语句等场合中变量类型已确定，故可自动解引用以匹配
        -   链式指 `*Deref::deref(P)` 整体链式，不是链式调用 `Deref::deref(P)` 后解引用 `*` 一次
            -   即，可以视为是 **隐式添加数量合适 `*`、并应用**
            -   对 `impl Deref` 将被转换为 `*Deref::deref()`
            -   对 `&&&&T` 即 *多重引用归一化*（当然，`Deref.deref(&T)` 返回 `&T` 与上条逻辑一致）
    -   注意：解引用强制转换仅是 *Type Coarcion* 类型强制转换的一种
        -   但，应该是类型强制转换中传递性（链式）支持较好的？
        -   *Rust* 在实参类型 `T: Deref` 特性实现、与形参类型 `U` 为以下情况时执行解引用强制转换
            -   `&T` 需转换为 `&U`：`T: Deref<Target=U>`
            -   `&mut T` 需转换为 `&mut U`：`T: DerefMut<Target=U>`
            -   `&mut T` 需转换为 `&U`：`T: Deref<Target=U>`
-   方法中 `self` 参数更为特殊：*Rust* 调用方法时会执行 **自动引用和解引用** 以匹配形参（多了自动引用）
    -   即，类似自动添加 `&`、`&mut`、`*`
    -   也因此，*Rust* 中没有 `->` 操作符
    -   也因此，`impl Deref` 类型不应与 `Target` 类型有重复方法

> - `std::ops::Deref`：<https://doc.rust-lang.org/std/ops/trait.Deref.html>
> - 4.4.2 `Deref` 解引用：<https://course.rs/advance/smart-pointer/deref.html>
> - 15.2 Treat Smart Pointers Like Regular References：<https://doc.rust-lang.org/stable/book/ch15-02-deref.html>
> - 15.2 使用 `Deref` 将智能指针当作常规引用处理：<https://www.rust-book-cn.com/ch15-02-deref.html> 翻译有错
> - 5.3 方法语法：<https://www.rust-book-cn.com/ch05-03-method-syntax.html>
> - 10.7 Type system - Type coercions：<https://doc.rust-lang.org/stable/reference/type-coercions.html>
> - 10.7 类型转换 - 可强制转换的类型：<https://doc.rust-lang.net.cn/reference/type-coercions.html>
> - 10.7 Type system - Type coercions - site.block：<https://doc.rust-lang.org/stable/reference/type-coercions.html#r-coerce.site.block>
> - 10.7 Type system - Type coercions - site.let：<https://doc.rust-lang.org/stable/reference/type-coercions.html#r-coerce.site.let>

#####   `impl Deref for Box`

```rust
impl<T:?Sized, A: Allocator> Deref for Box<T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        &**self                                             // 逻辑上 `*self` 即为 `Box` 类型，再次 `*` 将死循环
    }
}
impl<T:?Sized, A: Allocator> DerefMut for Box<T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        &mut **self
    }
}

let mut p = Box::new(String::from("hello"));
drop(*p);                                                   // `*p` 可从 `Box` 中移出数据
*p = String::from("World");                                 // 且可通过 `*p` 再次向 `Box` 中赋值
```

-   关于 `impl Deref for Box` 实现的说明
    -   `Box` 被设计为 **可解包以拥有封装的数据**，且解包被设计为解引用 `*`
        -   即，**可通过 `*(Box<T>)` 获取、并移动指向的堆上数据**
        -   而，`DerefMut::deref_mut` 总是返回可变引用 `&mut T`
            -   可变引用 `&mut T` 无法被用作获取数据所有权
            -   即，不可能满足 `*(Box<T>)` 获取所有权
        -   故，`impl Deref for Box<T>` 需要编译器特殊处理，获取、转移所有权
    -   事实上，`String`、`Vec<T>` 等智能指针均 “拥有” 指向数据的所有权（但 `Rc`、`RefCell` 是共享）
        -   智能指针对通过 “改变、移出内部部分数据” 体现对数据的所有权
        -   `Vec<T>` 等智能指针改变、移出数据的方法（`Vec::pop` 等）与解引用 `*` 是分离的
        -   而，`Box` 是拥有、指向单个值的简单封装，改变、移出数据的方法被设计为解引用 `*`

> - 向大佬请教 `Box` 的 `Deref` 实现问题：<https://2020conf.rustcc.cn/article?id=75f9e6ee-a410-43f5-8b26-9ba9817bb203>
> - Box vs other Deref implementation：<https://doc.rust-lang.org/reference/types/closure.html?highlight=deref#box-vs-other-deref-implementations>
> - 从源码看Box智能指针 - 评论区：<https://zhuanlan.zhihu.com/p/1889784767618211951>
> - `src::alloc::boxed` 源码：<https://doc.rust-lang.org/src/alloc/boxed.rs.html>

####    `PhantomData`

```Rust
use std::core::marker::PhantomData;
use std::ptr::NonNull;

// ************************** “传染” 否定标记给宿主类型
type PhantomNotSync = PhantomData<Cell<u8>>;
type PhantomNotSendNorSync = PhantomData<&'static Cell<u8>>;
struct CannotBeSharedAcrossThreads {
    _zst_not_sync_mark: PhantomNotSync;             // “传染” `!Sync` 给宿主类型
}
unsafe impl Sync for PhantomNotSend {}              // 也可以手动实现标记特性，隔离 “传染”

// ************************** 关联生命周期参数限制型变
struct OutRef<'a, T: 'a> {
    // ptr: *mut T,                                 // 1. 目标是非空的 `*mut T`
    ptr: NonNull<T>,                                // 2. 故使用 `NonNull<T>`，但对 `T` 协变
    _marker: PhantomData<&'a mut T>,                // 3. 若无此字段，`OutRef` 整体对 `T` 协变
}

// *************************** 指示类型逻辑上拥有 `T` 类型值（RFC 1238 后事实上无需）
struct Vec<T> {
    data: *const T,
    len: usize,
    cap: usize,
    _owns_T: PhantomData<T>,                        // 指示 `Vec` 丢弃时同时负责清理 `T` 类型值
}
```

-   `PhantomData<T>`：逻辑上关联宿主类型与泛型（或具体类型）、生命周期 `T`，
    -   即，指示宿主类型 **逻辑上应拥有** `T` 类型值，用于
        -   将特性类型的标记特性 “传染” 给宿主类型
            -   **泛型参数的标记特性会 “传染” 给包含其的类型**
        -   为宿主类型添加额外的泛型参数、生命周期参数作为限制（供、要求编译器检查）
            -   *Rust* 不允许存在未显式使用的泛型参数（即不允许 *Unbounded Lifetime* 无界生命周期）
        -   对所有权类型 `T`，指示宿主类型丢弃时会负责清理 `T` 类型值
            -   虽然，宿主类型变量在内存空间上不直接拥有 `T` 占用的内存资源
            -   但，宿主类型变量被丢弃时，某个 `T` 类型值（堆上）内存资源将被释放
    -   说明
        -   `PhantomData<T>` 是 *ZST* 类型，不占用空间、零开销
        -   `PhantomData<T>` 对 `T` 协变，可忽略其对外部类型、泛型 `T` 之间变异性传导的影响

> - 3.10 幻影数据：<https://doc.rust-lang.net.cn/nomicon/phantom-data.html>
> - 3.10 PhantomData：<https://doc.rust-lang.org/nomicon/phantom-data.html>
> - Rust：`PhantomData`、`#may_dangle` 和Drop Check 真真假假：<https://zhuanlan.zhihu.com/p/383004091>
> - Rust：深入理解 `PhantomData`：<https://zhuanlan.zhihu.com/p/533695108>
> - Looking for a deeper understanding of PhantomdData：<https://users.rust-lang.org/t/looking-for-a-deeper-understanding-of-phantomdata/32477/4>
> - Rust：`PhantomData` 与 `dropck` 解惑：<https://zhuanlan.zhihu.com/p/533541500>

### 裸指针

```rust
let mut num = 5;
let r1: *const i32 = &raw const num;                    // 创建不可变裸指针，无需在 `unsafe` 块中
let r2: *mut i32 = &raw mut num;                        // 创建可变裸指针
let r3 = r1 as *const u32;                              // 裸指针类型强制转换
```

-   裸指针（类型） `*const T`（不可变）、`*mut T`（可变）：“不安全” 的指针（类型）
    -   `&raw` 裸借用（操作符）用于创建裸指针
    -   与引用、智能指针不同，裸指针
        -   允许忽略借用规则：可同时拥有不可变、可变指针，或多个可变指针指向相同位置
        -   不保证指针有效：指针指向位置可能无效
        -   允许为空
        -   不实现任何自动清理

> - 20.1 不安全的 Rust - 解引用裸指针：<https://www.rust-book-cn.com/ch20-01-unsafe-rust.html>

####    *Safety*

-   *Safety* 裸指针的安全性
    -   *Validity* 裸指针有效性依赖具体访问：操作（读、写）、操作范围（内存字节），且裸指针有效性的精确规则未确定
        -   若操作范围为 0，所有指针均有效，包括空指针
        -   空指针永远无效（除操作范围为 0 场景）
        -   指针可解引用：起始于指针、特性类型内存范围在指针所属的内存分配范围内
            -   可解引用是指针有效的必要、但不充分条件
    -   *Alignment* 裸指针对齐：`*const T` 类型需要对齐至 `align_of::<T>()`
        -   大部分函数要求参数正确对齐，即使操作范围为 0
    -   *Pointer to Reference Conversion* 裸指针转换为引用
        -   指针须对齐、非空、可解引用
        -   指针须指向有效值
        -   指针解引用须满足引用（别名）规则
    -   *Allocation* 内存分配：可寻址、支持指针算术运算的内存子集
        -   每个变量视为单独的内存分配，包括
            -   堆内存分配
            -   栈内存分配
            -   静态量
            -   常量
        -   内存分配包括：起始地址、大小、内存地址集合
            -   0 大小内存分配也需要有起始地址
            -   不同的内存分配起始地址可以相同
            -   当前内存分配都是连续的，但不保证之后不改变

> - `std::ptr`：<https://doc.rust-lang.org/stable/std/ptr/index.html>

####    *Provenance*

-   *Provenance* 起源信息：指示裸指针起源的内存分配（即内存访问权限）
    -   指针不是简单的整形值、地址，语义上应包含
        -   指向地址：可用 `usize` 表示
        -   起源信息：对内存的访问权限
            -   **起源信息可为空，但是此时指针无权访问任何内存**
    -   起源信息的具体结构未明确，但包含 3 部分
        -   *Spatial* 空间权限：指针允许访问的内存地址集合
        -   *Temporal* 时间权限：指针允许访问内存地址的时间段
        -   *Mutability* 可变性：指针返回内存地址的读、写权限
    -   内存分配创建时包含有唯一、原初指针，即调用 *Alloc APIs* 返回的指针
        -   原初指针起源信息即限制
            -   空间权限：内存分配的内存范围
            -   时间先去：内存分配的生命周期
        -   基于原初指针通过偏移、借用、类型转换的衍生指针将继承原初指针的起源信息
            -   指针的权限无法通过操作扩展
    -   起源信息影响、判断 *Undefined Behavior*

> - `std::ptr`：<https://doc.rust-lang.org/stable/std/ptr/index.html>

####    `NonNull`、`Unique`

| 裸指针封装                | 说明                            |
|---------------------------|---------------------------------|
| `ptr::NonNull<T: ?Sized>` | 非空、协变 `*mut T`             |
| `ptr::Unique<T: ?Szied>`  | （指针、数据）唯一 `NonNull<T>` |

-   裸指针封装类型
    -   `*mut T` 可变裸指针对 `T` 不变，在作为类型成员时兼容性较差
    -   `ptr::NonNull<T>` 则封装 `*const T` 以实现对 `T` 协变
        -   并，在 `NonNull::as_ptr()` 中 `transmute` 为 `*mut T` 并返回获取可变性
            -   解引用裸指针总是 `unsafe`，即需要用户确保指针有效

```rust
pub struct NonNull<T: PointeeSized> {
    pointer: *const T,                                  // 故 `NonNull` 对 `T` 协变
}
pub struct Unique<T: PointeeSized> {
    pointer: NonNull<T>,
    _maker: PhantomData<T>,                             // 指示逻辑上拥有 `T` 
}

pub struct Box<T:?Sized, A:Allocator=Global>(Unqiue<T>, A);
```

> - 9.1 布局：<https://doc.rust-lang.net.cn/nomicon/vec/vec-layout.html>
> - `ptr::NonNull`：<https://doc.rust-lang.org/std/ptr/struct.NonNull.html>
> - `ptr::Unique`（无文档）：<https://doc.rust-lang.org/src/core/ptr/unique.rs.html>
> - `alloc::boxed`：<<https://doc.rust-lang.org/src/alloc/boxed.rs.html>
> - Rust 中 `Unique`、`NonNull` 的用法：<https://www.duidaima.com/Group/Topic/Rust/14535>

### *Destructor*

| *Destructor* 析构、内存清理                                    | 描述                |
|----------------------------------------------------------------|---------------------|
| `fn core::mem::drop<T>(_x:T)`                                  | 丢弃 **变量所有值** |
| `fn unsafe const core::ptr::drop_in_place<T>(to_drop: *mut T)` | 丢弃 **指针指向值** |
| `fn core::ops::Drop::drop(&mut self)`                          | 释放申请的堆内存    |
| `struct core::mem::ManuallyDrop<T>`                            | 手动丢弃封装        |
| `fn const core::mem::forget<T>(t: T)`                          | 手动丢弃            |

-   *Destructor* 析构（器）
    -   以下场合，**已初始化变量、临时变量** 执行析构：值被丢弃、内存被释放
        -   已初始化变量、临时变量离开作用域 *Drop Scope*
            -   **变量、临时变量按声明顺序逆序清理**
        -   已初始化的左值（即已初始化可变变量被重新赋值时，原值被丢弃）
            -   若仅部分初始化，则仅初始化部分字段被丢弃
    -   类型 `T` 值的析构过程包括
        -   若 `T: Drop`，调用 `<T as core::ops::Drop>::drop`（清理堆内存）
            -   故，不允许手动显式调用 `Drop::drop` 方法，否则可能导致 *Double Free*
        -   *Drop Glue*：递归的对类型 `T` 包含字段析构
            -   结构体、元组、枚举变体、数组：字段按声明顺序丢弃（与 *C/C++* 相反）
            -   闭包捕获的变量丢弃顺序不固定
            -   *Trait 对象* 执行底层类型析构
            -   其他类型不执行额外操作

> - Destructors：<https://doc.rust-lang.org/stable/reference/destructors.html>
> - `std::ops::Drop`：<https://doc.rust-lang.org/std/ops/trait.Drop.html>

####    `Drop`

```rust
use std::ops::Drop;

struct CustomSmartPointer{
    data: String,
}

impl Drop for CustomSmartPointer{
    fn drop(&mut self) {
    }
}
```

-   `core::ops::Drop`：自定义值离开作用域时需运行的代码
    -   `Drop::drop` 方法仅应释放 **类型自身申请的堆上内存**
        -   变量自身（栈上内存）：将在执行析构时、在 `Drop::drop` 之后被自动释放
        -   变量成员：递归的执行析构时，释放其申请堆上内存、自身栈上内存
        -   引用成员：不拥有值，无需释放内存、或仅释放存储地址的栈上内存
    -   故，一般仅在类型包含裸指针、与外部资源交互时需 `impl Drop`
        -   对裸指针，调用 `core::ptr::drop_in_place<T>` 函数释放指针指向内存
        -   对外部资源，调用外部资源提供的 *API* 释放内存
    -   `Drop` 与 `Copy` 标记特性不兼容
        -   `impl Drop` 将自动移除 `Copy` 标记特性

> - `std::ops::Drop`：<https://doc.rust-lang.org/std/ops/trait.Drop.html>
> - 15.3 使用 `Drop` 特性在清理时运行代码：<https://www.rust-book-cn.com/ch15-03-drop.html>
> - Rust 在什么情况下，必须手写Drop，释放资源？：<https://www.zhihu.com/question/653741156/answer/3475999527>

#####   *Sound Generic Drop*

```rust
// *************************** `#[may_dangle]` 指示可能出现悬垂引用
#![feature(dropck_eyepatch)]

struct Inspector<'a>(&'a u8, &'static str);

unsafe impl<#[may_dangle] 'a> Drop for Inspector<'a> {      // 跳过对 `'a` 的检查
    fn drop(&mut self) {
        println!("Inspector(_, {}) knows when *not* to inspect.", self.1);
    }
}

struct World<'a> {
    days: Box<u8>,
    inspector: Option<Inspector<'a>>,
}

fn main() {
    let mut world = World {
        inspector: None,
        days: Box::new(1),
    };
    world.inspector = Some(Inspector(&world.days, "gadget"));
}
```

-   *Sound Generic Drop* 健全泛型丢弃：（泛型）类型中泛型参数存活时间必须 **严格超过（包含）** （泛型）类型值（否则编译失败）
    -   即，若类型包含的泛型参数为引用，引用（泛型参数）的生命周期必须严格超过（包含）该（泛型）类型
    -   对无自引用类型
        -   考虑到编译器严格按声明顺序清理变量，则泛型类型、引用（泛型参数）的生命周期不严格相等
        -   即，满足一般的生命周期约束即可
    -   对存在自引用的类型
        -   若类型未实现 `Drop Trait`，编译器将严格按顺序清理变量
            -   对合理的字段顺序，编译器接受泛型参数存活时间严格超过类型值
            -   类型中字段存在自引用可编译通过
        -   若类型实现 `Drop Trait`
            -   编译期器将无法判断 `drop` 方法中成员清理顺序
            -   类型中字段存在自引用时总是编译失败
    -   `#[may_dangle] 'a` 标记泛型参数（或生命周期参数 `'a`）：可能出现悬垂引用，但是保证不会访问悬垂引用
        -   编译器将忽略被标记的泛型参数是否严格超过类型值

> - 3.9 丢弃检查：<https://doc.rust-lang.net.cn/nomicon/dropck.html>
> - 3.9 Drop Check：<https://doc.rust-lang.org/nomicon/dropck.html>
> - Rust: `PhantomData`，`#may_dangle` 和Drop Check 真真假假：<https://zhuanlan.zhihu.com/p/383004091>

####    `drop`、`drop_in_place`

```rust
// ************************** `core::mem::drop`
pub const fn drop<T>(_x: T)
where T: [const] Destruct,
{                                               // 只是获取了实参所有权、但不做任何事
}                                               // 实参离开 `drop` 作用域时被析构、丢弃

pub const unsafe fn drop_in_place<T: PointeeSized>(to_drop: *mut T)
where
    T: [const] Destruct,
{                                               // 函数体不重要，将被编译器替换为真正的 drop glue
    usafe { drop_in_place(to_drop) }
}
```

-   `drop`、`drop_in_place` 分别为所有者变量、可变裸指针执行（递归的、完整的）析构
    -   `drop_in_place` 可用于丢弃 `!Sized` 类型值
        -   `!Sized` 类型值无法读取至栈上、被获取所有权，即无法被正常 `drop` 丢弃
        -   `drop_in_place` 实际将由编译器替换为完整的 *Drop Glue* 丢弃流程

> - `core::mem::drop`：<https://doc.rust-lang.org/core/mem/fn.drop.html>
> - `core::ptr::drop_in_place`：<https://doc.rust-lang.org/stable/core/ptr/fn.drop_in_place.html>
> - `core::ptr::drop_in_place` 中文：<https://rustwiki.org/zh-CN/core/ptr/fn.drop_in_place.html>
> - `core::ptr::drop_in_place` 源码：<https://doc.rust-lang.org/stable/src/core/ptr/mod.rs.htm>

####    `ManuallyDrop`、`forget`

```rust 
use std::mem:{ManuallyDrop, forget};

// ************************** `core::mem::ManuallyDrop`
#[lang = "manaully_drop"]                       // 指示编译器不自动析构
pub struct MannuallyDrop<T: ?Sized> {
    value: T,                                   // 零开销
}
impl<T> ManuallyDrop<T> {
    pub const fn new(value: T) -> MannullyDrop<T> {
        ManuallyDrop { value }
    }
}
impl<T:?Sized> ManuallyDrop<T> {
    pub const unsafe fn drop(slot: &mut ManuallyDrop<T>)
    where T: [const] Destruct,                  // 必须手动析构
    {
        unsafe { ptr::drop_in_place(&mut slot.value) }
    }
}

// ************************** `core::mem::forget`
pub const fn forget<T>(t: T) {
    let _ = ManuallyDrop::new(t);               // 获取实参所有权、`ManuallyDrop` 封装禁止自动析构
}
```

-   `mem::ManuallyDrop` 封装的变量离开作用域时不会自动析构

> - `core::mem::ManuallyDrop`：<https://doc.rust-lang.org/stable/core/mem/struct.ManuallyDrop.html>
> - `core::mem::forget`：<https://doc.rust-lang.org/stable/core/mem/fn.forget.html>
> - How `std::mem::MannuallyDrop` work?：<https://users.rust-lang.org/t/how-std-manuallydrop-work/69939>

### `unsafe`

```rust
let mut num = 5;
let r1 = &raw const num;                        // 创建裸指针是安全的
let r2 = &raw mut num;
unsafe {                                        // `unsafe` 块
    println!("r1 is: {}", *r1);                 // 解引用裸指针才是 `unsafe` 的
    println!("r2 is: {}", *r2);
}

unsafe fn dangerous() {                         // `unsafe` 函数，标记函数调用需要满足一定条件
    unsafe {                                    // `unsafe` 函数体中依然需要 `unsafe`
    }
}
unsafe extern "C" {                             // 不安全 `extern` 外部函数接口声明块
    fn abs(input: i32) -> i32;                  // `unsafe extern` 中每个项都是隐式 `unsafe` 的
    safe fn neg(input: i32) -> i32;             // 可显式标记为 `safe` 向编译器做出保证
}
unsafe {
    dangerous();
    abs(-3);
}
neg(3);                                         // 显式标记 `safe` 的函数无需在 `unsafe` 块中调用

static mut COUNTER: i32 = 0;
unsafe {                                        // 静态可变变量的访问、修改不安全
    COUNTER += 1;
    println!("{}", *(&raw const COUNTER));
}

unsafe trait Foo{}                              // 不安全 `trait`
unsafe impl Foo for Bar {}
```

-   `unsafe` 块
    -   `unsafe` 块的原因、用途
        -   编译器的静态分析本质上是保守的
            -   即使代码没有问题，编译器没有足够信息确信时，依然会拒绝代码
        -   硬件本质上 “不安全”，存在需要不安全操作完成的任务
    -   `unsafe` 块中允许的不安全操作，借用检查器等其他安全检查依然适用
        -   解引用 `*const`、`*mut` 裸指针
            -   创建裸指针是安全的
            -   但，访问裸指针指向的值可能不安全
        -   调用 `unsafe fn` 不安全函数、方法
            -   `unsafe fn` 不安全函数：表示函数有需要遵守的要求，但是编译器无法确保已经满足
                -   仅，**函数调用需要满足要求，且此要求需要调用者自行保证时**，添加 `unsafe` 标记提示编译器
                    -   不是包含 `unsafe` 块的函数就需要标记为 `unsafe fn`
                    -   若函数已经完成安全抽象，则无需 `unsafe` 标记
                -   `unsafe fn` 定义的不安全函数中执行不安全操作依然需要使用 `unsafe` 块
            -   `unsafe extern` 外部函数接口：外部函数未执行 *Rust* 保证
                -   `unsafe extern` 块中项隐式均为 `unsafe`，需要在 `unsafe` 块中使用
                -   但，可以显示标记为 `safe`，由用户确保、向编译器保证安全
        -   访问、修改可变的静态变量
        -   实现不安全 `trait`
            -   存在不安全的方法时 `trait` 即为不安全的
        -   访问 `union` 实例字段
            -   *Rust* 无法保证存储在 `union` 实例中的数据类型
            -   主要用于与 *C* 中的 `union` 类型进行交互

##  语言特性

### 函数式

####    闭包

```rust
let oclosure = |x| x;                               // 闭包定义，
let out = oclosure(5);                              // 闭包参数、返回值类型确定，不可换类型调用

let val = vec![1, 2, 3];
let immut_ = || println!("{val:?}");                // 捕获不可变引用
let mut mut_ = || val.push(1);                      // 捕获可变引用，此闭包需声明为 `mut` 才可调用
mut_();                                             // 调用前，闭包持有可变引用，不能使用 `val`
let move_ = move || {val.push(1); val}              // `move` 获取所有权
```

-   闭包：可保存在变量中、或作为参数传递给其他函数的匿名函数（可类似函数通过变量名、括号调用）
    -   每个闭包类型都是独特、不同类型
        -   闭包无需类似 `fn` 注解参数、返回值类型
            -   闭包不会暴露给用户
            -   闭包通常较短、应用于少数上下文中：编译器可以自动推断参数、返回值类型
        -   但闭包参数、返回值类型是确定的，故注意
            -   闭包的参数、返回值类型时隐式推断出后唯一、固定
            -   同一闭包不可先后使用不同类型值调用
    -   闭包可以从定义其的作用域中捕获值
        -   编译器 **根据闭包主体** 确定捕获方式
            -   捕获不可变引用、可变引用：编译器根据闭包中是否需要可变引用判断
            -   获取所有权：`move` 关键字显式指定（`Copy` 值仅复制）
        -   对闭包而言，捕获的可变引用类似静态变量，可在多次重入间维护状态
    -   闭包将被编译为实现以下一个或多个特性 `trait` 的值，由闭包捕获、处理值方式决定
        -   `FnOnce<Args>` 只调用一次：所有闭包至少实现此特性
        -   `FnMut<Args>: FnOnce<Args>` 调用后闭包可能改变：可能改变捕获值、但不获取所有权
        -   `Fn<Args>: FnMut<Args>` 调用后闭包不改变：不改变捕获值、且不获取所有权
            -   **故、目标** `impl Fn` 可用于所有需要 `FnMut` 类型的场合
            -   显然，“不允许闭包变化” 相较 “闭包调用后可变” 更严格

> - 13.1 闭包：可以捕获环境的匿名函数：<https://www.rust-book-cn.com/ch13-01-closures.html>
> - 闭包是什么：<https://www.cnblogs.com/dream397/p/14190206.html>

####    高级函数

```rust
fn add_one(x: i32) -> i32 {
    x + 1
}
fn do_twice(f: fn(i32) -> i32, arg: i32) -> i32 {           // 函数指针作为参数
    f(arg) + f(arg)
}
assert_eq!(do_twice(add_one, 5), 12);
let clv = vec![Box::new(|x| x + 1), Box::new(|x| x + 2)];   // 闭包特征对象作为向量元素
```

-   高级函数：函数、闭包作为参数、返回值的函数
    -   `fn` 函数指针（类型）：函数定义将被转换为 `fn` 类型值
        -   函数指针实现了全部闭包 `FnOnce`、`FnMut`、`Fn` 特性
            -   除非特意避免闭包作为参数，一般使用泛型、闭包特征做出参数以兼容函数、闭包
        -   枚举变体（的名称）也会被初始化为函数（指针）
    -   每个闭包类型都不同，且没有具体类型
        -   单个闭包作为普通值使用（作为参数、返回值）时，可用泛型参数、闭包特征实现
        -   但，多个闭包需作为同类值集合时，只能使用特征对象 `Box<dyn Fn(i32) -> u32>`

> - 20.4 高级函数和闭包：<https://www.rust-book-cn.com/ch20-04-advanced-functions-and-closures.html>

####    迭代器

```rust
pub trait Iterator {                                // 标准库中定义
    type Item;
    fn next(&mut self) -> Option<Self::Item>;       // 仅 `next` 方法需实现
}
```

-   迭代器：负责遍历每个项目，并确定何时结束
    -   迭代器是惰性的，在调用消耗迭代器的方法前，不会产生效果
        -   消耗适配器：调用 `.next()` 的方法
        -   迭代器适配器：改变原迭代器某些方面是生成不同迭代器
    -   迭代器效率与 `for` 遍历循环效率基本一致
        -   事实上，编译器会将 `for` 循环编译成与 `Iterator trait` 等效的代码

| `Iterator` 迭代器方法      | 描述             |
|----------------------------|------------------|
| `.next()`                  |                  |
| `.sum()`                   |                  |
| `.map(|x| ...)`            | 映射             |
| `.filter(|x| -> bool ...)` | 过滤             |
| `.collect()`               | 收集结果值至集合 |

| 集合转换为迭代器 | 描述                     |
|------------------|--------------------------|
| `.iter()`        | 遍历获取元素不可变引用   |
| `.iter_mut()`    | 遍历获取元素可变引用     |
| `.into_iter()`   | 遍历获取元素（消耗集合） |

> - 13.2 使用迭代器处理一系列项：<https://www.rust-book-cn.com/ch13-02-iterators.html>
> - 13.4 比较性能：循环与迭代器：<https://www.rust-book-cn.com/ch13-04-performance.html>

### 宏

| 注册注解                        | 宏注册语法               | 宏使用语法           | 说明                               |
|---------------------------------|--------------------------|----------------------|------------------------------------|
| `#[macro_export]`               | `macro_rules! <MACRO>{}` | `<MACRO>!()`         | 声明式宏，类似函数调用             |
| `#[proc_macro]`                 | `pub fn <MACRO>(){}`     | `<MACRO>!()`         | 函数过程宏，类似函数调用           |
| `#[proc_macro_attribute]`       | `pub fn <MACRO>(){}`     | `#[<MACRO>()]`       | 属性过程宏，任何项的自定义属性     |
| `#[proc_macro_derive(<MACRO>)]` | `pub fn <ANYNAME>(){}`   | `#[derive(<MACRO>)]` | 派生宏，给结构体、枚举类型添加代码 |

-   宏：生成其他代码的代码，元编程
    -   宏类似函数用于减少重复逻辑，但更复杂
        -   宏可以接受可变数量的参数
        -   宏在编译前已经展开，可用于编译时在给定类型上实现 `trait`，而函数在运行时调用
        -   在文件中调用宏前，必须先在文件中定义、引入宏，而函数可在任意位置定义
    -   `macro_rules!` 声明式宏：类似 `match` 表达式的代码，将源代码字面值与模式匹配、替换为对应代码
    -   过程式宏：类似函数，接受 `TokenStream` 代码（的抽象语法树）作为输入、解析、执行逻辑，并生成代码作为输出
        -   *Derive* 派生宏 `#derive[MACRO]`：常用于在结构体、枚举类型上添加的代码
        -   *Attribute-like* 属性式宏 `#[MACRO]`：定义可用于任何项的自定义属性
        -   *Function-like* 函数式宏 `MACRO!()`：类似函数调用，但操作作为参数传递的标记

> - 20.5 宏：<https://www.rust-book-cn.com/ch20-05-macros.html>

####    声明式宏

```rust
#[macro_export]                                 // 宏导出注解，定义宏的 crate 引入时，宏即可用
macro_rules! add {
    ($a: expr) => { $a };
    ($a: expr, $b: expr) => { { $a + $b } };    // 内层 `{}` 也是输出
    ($a: expr, $($b:tt)*) => {
        { $a + add!($($b)*) }                   // 宏递归调用
    };
}
```

-   声明式宏：类似模式匹配，可以有多个分支，宏根据匹配的参数展开至不同代码
    -   每个分支可以有多个参数
        -   `()`、`[]`、`,` 等符号都是模式（匹配）的一部分
        -   参数以 `$` 开头、后跟 `:` 指示代码字面量中 *Token* 类型
        -   支持可变数量参数
            -   `*` 单独、后缀：零个或更多 *Token*
            -   `+` 单独、后缀：一个以上 *Token*
            -   `$(...)*`、`$(...)+`：为可变数量参数指定名称

| *Token* 类型 | 说明                          |
|--------------|-------------------------------|
| `item`       | 项：函数、结构体、模块等      |
| `block`      | 块：`{}` 包裹的语句块、表达式 |
| `stmt`       | 语句                          |
| `pat`        | 模式                          |
| `expr`       | 表达式                        |
| `ty`         | 类型                          |
| `ident`      | 标识符                        |
| `path`       | 路径                          |
| `meta`       | 元数据项：`#[...]`、`#![...]` |
| `tt`         | 词法树                        |
| `vis`        | 可能为空的可见性限定词：`pub` |

> - Rust 宏：教程与示例（一）：<https://zhuanlan.zhihu.com/p/353421021>

####    过程式宏

```rust
use proc_macro::TokenStream;                    // 过程式宏输入、输出类型

#[proc_macro]                                   // 注册为函数宏
pub fn macro_add_one(_input: TokenStream) {
    "fn add_one(x: i32) -> i32 { x + 1 }"
        .parse()
        .unwrap()
}
macro_add_one!();                               // 调用函数宏，编译期间展开（其他 Crate 中）

#[proc_macro_attribute]                         // 注册为属性宏
pub fn trace_vars(_metadata: TokenStream, input: TokenStream) -> TokenStream {
}
#[trace_vars(a)]                                // 调用属性宏（其他 Crate 中）
fn do_something() {
    let mut a = 9;
    a = 6;
}

#[proc_macro_derive(Builder)]                   // 注册名为 `Builder` 的派生宏
pub fn builder_derive(input: TokenStream) -> TokenStream {
}
#[derive(Builder)]                              // 使用名为 `Builder` 的派生宏
struct SOME_STRUCT {
}
```

-   过程式宏：类似函数，接受 `proc_macro::TokenStream` 代码（的抽象语法树）作为输入、解析、执行逻辑，并生成代码作为输出
    -   过程式宏只能定义在单独的 `proc-macro` *Crate* 中
        -   `Cargo.toml` 需配置 `lib.proc_macro = true` 以声明 *Crate* 中将创建过程宏
        -   且 *Crate* 内不能使用内部定义的宏
            -   即，过程式宏基本只能单独创建新项目实现
            -   故，也无法在同一源码文件中进行单元测试，只能进行集成测试
        -   必要依赖 *Crate*
            -   `proc_macro`：编译器自带 *API*，从源码文件中读取、操作源码
            -   `syn`：将源码转换为抽象语法树
            -   `quote`：将抽象语法树转换为源码
    -   *Derive* 派生宏 `#derive[MACRO]`：常用于在结构体、枚举类型上添加的代码
    -   *Attribute-like* 属性式宏 `#[MACRO]`：定义可用于任何项的自定义属性
    -   *Function-like* 函数式宏 `MACRO!()`：类似函数调用，但操作作为参数传递的标记

```toml
# Cargo.toml
[lib]
proc_macro = true

[dependencies]
syn = "2.0"
quote = "1.0"
```

> - Rust 宏：教程与示例（二）：<https://zhuanlan.zhihu.com/p/356427780>
> - 厌倦样板代码？用 Rust 过程宏自动生成！：<https://segmentfault.com/a/1190000047100635>

### 多线程、并行

-   *Rust* 标准库使用 `1:1` 线程实现模型

####    `Send`、`Sync`

-   *Rust* 通过 `Send`、`Sync` 标记特征管理线程数据安全
    -   `std::marker::Send trait`：标记类型（值）的所有权可以在线程间转移
    -   `std::marker::Sync trait`：标记类型（值）可以在多个线程之间（安全）共享（多线程访问引用）
        -   类型 `T` 是 `Sync` 的，当且仅当 `&T` 是 `Send` 的
    -   大部分类型都是 `Send`、`Sync` 的
        -   除裸指针外原始类型都 `Send`、`Sync`
        -   如果类型完全由 `Send`、`Sync` 组成，则该类型自动派生 `Send`、`Sync`
        -   特别的，对闭包、函数、`async` 代码块，其中所有项都 `Send` 则 `Send`
            -   闭包、函数、`async` 代码块将被编译为 `impl Future` 等对应类型实例
            -   其中项应被作为实例中成员，即包含 `!Send` 值，则整体 `!Send`
            -   `thread::spawn`、`tokio::spawn` 等函数涉及将闭包等传递给其他线程（执行），故要求参数整体 `Send`，即闭包中各项均 `Send`
    -   主要的 `!Send`、`!Sync` 例外
        -   裸指针 `*mut`、`*const` 既不 `Send`、也不 `Sync`
        -   `UnsafeCell<T>`、`Cell<T>`、`RefCell<T>` 不 `Sync`
            -   则，`&UnsafeCell<T>` 不 `Send`
        -   `Rc<T>` 既不 `Send`、也不 `Sync`
            -   若 `Rc<T>` 的多个副本在多个线程共享，多个线程可能同时克隆、更新引用计数
            -   应换用 *Atom Rc* `std::sync::Arc`
    -   从数据竞争角度，`Send`、`Sync` 的核心是类型是否具有内部可变性
        -   *Rust* 的所有权机制、引用规则在很大程度上避免了数据竞争
        -   若类型具有内部可变性，则可能存在跨线程的数据竞争，则不 `Send`、`Sync`，需要加锁避免数据竞争
            -   `Rc<T>` 内部引用计数可变
            -   `UnsafeCell<T>`、`Cell<T>`、`RefCell<T>` 泛型参数值可变

> - 8.2 `Send` 和 `Sync`：<https://doc.rust-lang.net.cn/nomicon/send-and-sync.html>
> - 16.1 使用线程同时运行代码：<https://www.rust-book-cn.com/ch16-01-threads.html>
> - 16.2 使用消息在线程间传输数据：<https://www.rust-book-cn.com/ch16-02-message-passing.html>
> - 如何理解 `Sync` 和 `Send`：<https://hexilee.me/2019/05/05/how-to-understand-sync-and-send-in-rust/>
> - Function `std::thread::spawn`：<https://rustwiki.org/zh-CN//std/thread/fn.spawn.html>

####    `park()`、`unpark()`

-   *Rust* 标准库使用 *Parking* 模型提供线程自身的阻塞、信号唤醒
    -   模型设计：线程行为类似自旋锁、`Condvar` 条件量
        -   每个 *Thread Handle* 线程（句柄）关联一个 *Token* 令牌
            -   初始状态时，令牌不存在
        -   `thread::park()` 将阻塞调用此函数线程（即当前线程），直至 *Token* 可用
            -   `thread::park_timeout(dur)` 类似，但额外指定最大阻塞时间
            -   线程从阻塞返回后将消耗 *Token*
            -   线程也可能被虚假唤醒，但不消耗 *Token*
        -   `thread::Thread::unpark()` 原子地将线程 *Token* 置为可用
            -   注意：*Token* 可被未 `park()` 阻塞线程持有
            -   则，`Thread::unpark()` 之后 `park()` 将立刻返回
    -   `Thread::unpark()` 与 `park()` 调用之间有同步机制
        -   即，`unpark()` 调用前的所有内存操作对（消耗 *Token* 从 `park()` 返回的）线程是可见的
        -   即，线程 `park()`、`unpark()` 应成对、有序

> - `std::thread::park`：<https://doc.rust-lang.org/stable/std/thread/fn.park.html>
> - `std::thread`：<https://doc.rust-lang.org/stable/std/thread/index.html>
> - Rust 多线程的高效等待术：`park()` 与 `unpark()` 信号通信实战：<https://paxonqiao.com/rust-thread-parking/>
> - Rust 并发加速器：用 `Condvar` 实现线程间“精确握手”与高效等待：<https://paxonqiao.com/rust-condvar/>
> - 为什么会有虚假唤醒？：<https://cloud.tencent.com/developer/article/2493152>

### 协程

####    `Coroutine` 协程

```rust
pub enum CoroutineState<Y, R> {                         // 协程输出值
    Yielded(Y),                                         // 协程 `yield` 值
    Complete(R),                                        // 协程 `return` 返回值
}
trait Coroutine<R=()> {
    type Yield;
    type Return;
    fn resume(self: Pin<&mut Self>, arg: R)             // 向协程传递值、从挂起处继续执行
        -> CoroutineState<Self::Yield, Self::Return>;
}
```

-   *Coroutine* 协程：实现 `yield` 机制，可 **暂停、恢复、并在其间传递数据** 的计算单元
    -   协程是用户态、协作式的多任务机制，可视为是轻量级的（内核）线程
        -   用户态：轻量、不受内核调度
        -   协作式：非抢占式、由任务主动暂停执行让出资源
    -   协程本质上是可重入、传递数据的双向通道
        -   `yield` 关键字将值向外传递给调用者
            -   `yield` 必须在 `#[coroutine] || {}` 或 `gen {}` 中
        -   `.resume()` 方法从调用者获取值向内传给协程，协程从上次 `yield` 挂起处继续执行
    -   编译器自动将包含 `yield` 的闭包编译为 `impl Coroutine` 的状态机
        -   状态机为匿名枚举类型（或枚举类型的零开销 `struct` 封装）
            -   状态机各状态即为枚举变体
                -   `yield` 点即状态机内部状态保存点、协程挂起点
                -   即，`yield` 点对应状态机各个（中间）状态
            -   编译器将协程 **恢复执行** 所需的变量（跨越 `yield` 的变量）置为状态机的成员变量进行维护
                -   `.resume` 恢复执行时，匹配状态机状态
                -   即，确定分支、恢复变量值，实现重入、状态记忆
        -   协程内部引用存在自引用问题，需要 `Pin` 住（协程依然为实验性功能的核心原因）
            -   协程在首次执行前自引用指针、指针尚未初始化，可以移动
            -   但在协程 `.resume` 恢复执行之后，协程移动可能导致自引用指针指向未定义值
            -   故 `.resume` 方法签名中要求 `self: Pin<&mut Self>` 保证协程被 `Pin`、不可移动、内存地址不变

```rust
// *********** 协程至少需要 2024 nightly 工具链
#![feature(stmt_expr_attributes, coroutines, coroutine_trait)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn main() {
    let mut g = #[coroutine] || {           // 闭包将被编译为协程，生成 Fib 数列
        let mut curr: u64 = 1;
        let mut next: u64 = 1;
        loop {
            let new_next = curr.checked_add(next);
            if let Some(new_next) = new_next {
                curr = next;
                next = new_next;
                yield curr;                 // `yield` 挂起、输出值
            } else {
                return;
            }
        }
    };

    let mut pinned = Pin::new(&mut g);      // 协程包含自引用，必须 `Pin`

    loop {
        match Coroutine::resume(pinned.as_mut(), ()) {
            CoroutineState::Yielded(v) => println!("{}", v),
            CoroutineState::Complete(_) => break,
        }
    }
}
```

> - 25.1 协程 - 基础简介：<https://uxiew.github.io/rust_study/%E7%AC%AC%2025%20%E7%AB%A0%20%E5%8D%8F%E7%A8%8B/25.1%20%E5%9F%BA%E7%A1%80%E7%AE%80%E4%BB%8B.html>
> - 25.4 协程 - Rust 异步的基石：<https://uxiew.github.io/rust_study/%E7%AC%AC%2025%20%E7%AB%A0%20%E5%8D%8F%E7%A8%8B/25.4%20%E5%8D%8F%E7%A8%8B.html>
> - 25.6 `Coroutine` 与 `Future`：<https://uxiew.github.io/rust_study/%E7%AC%AC%2025%20%E7%AB%A0%20%E5%8D%8F%E7%A8%8B/25.6%20Coroutine%20%E4%B8%8E%20Future.html>
> - *Coroutines*：<https://doc.rust-lang.org/nightly/unstable-book/language-features/coroutines.html>
> - `Coroutine`：<https://doc.rust-lang.org/stable/std/ops/trait.Coroutine.html>
> - *Std Core* 源码 - `coroutine`：<https://doc.rust-lang.org/src/core/ops/coroutine.rs.html>
> - *Rustc HIR* 源码 - `CoroutineSource`、`CoroutineKind`：<https://gitcode.com/GitHub_Trending/ru/rust/blob/master/compiler/rustc_hir/src/hir.rs>


####    *Generator* 生成器

-   *Generator* 生成器：专门用于惰性的、按需生成值序列的协程
    -   生成器是不接受输入、无返回值的特殊协程
        -   只利用协程 `yield` 向外传值的能力
    -   在 *Rust* 中定义、特化为生成 *Iterator* 迭代器的快捷方式
        -   `gen {}` *gen_blocks* 特性：构造生成器的语法糖

```rust
#![feature(gen_blocks)]

fn fib() -> impl Iterator<Item = usize> {
    gen {                                   // *gen block* 返回 `impl Iterator` 的匿名类型
        let mut n1 = 1;
        let mut n2 = 1;
        let mut n3 = 2;
        yield n1;
        yield n2;
        loop {
            yield n3;
            n1 = n2;
            n2 = n3;
            n3 = n1 + n2;
        }
    }
}

fn main() {
    println!("{:?}", fib().take(10).collect::<Vec<usize>>());
}
```

> - 25.1 协程 - 基础简介：<https://uxiew.github.io/rust_study/%E7%AC%AC%2025%20%E7%AB%A0%20%E5%8D%8F%E7%A8%8B/25.1%20%E5%9F%BA%E7%A1%80%E7%AE%80%E4%BB%8B.html>
> - *Rustc HIR* 源码 - `CoroutineSource`、`CoroutineKind`：<https://gitcode.com/GitHub_Trending/ru/rust/blob/master/compiler/rustc_hir/src/hir.rs>

### 异步任务

####    `Future` 异步任务

```rust
use std::pin::Pin;
use std::future::Future;
use std::task::{Context, Pool};

enum Poll<T> {                                  // 异步操作结果，`Future.poll` 返回类型
    Ready(T),                                   // 异步操作已完成，返回结果为 `T`
    Pending,                                    // 异步操作还未完成，需等待
}

pub trait Future {                              // `future::Future` 标准库实现
    type Output;                                // 返回值类型
    fn poll(                                    // 被异步执行器调用，以确定执行状态
        self: Pin<&mut Self>,                   // 实例应为 `!unpin`
        cx: &mut Context<'_>                    // 执行上下文
    ) -> Poll(Self::Output);
}
```

-   `future::Future` 异步任务：实现 `await` 机制（可挂起、等待依赖完成）、无 `yield` 值（或 `yield ()`）的协程
    -   `impl Future` 语义上指：现在可能还没准备好，但未来将准备好的值 *Future*、*Task*、*Promise*
        -   `async {}` 用于创建异步代码块，并由编译器负责转换为匿名 `Future` 对象
            -   但对复杂逻辑，可能仍需手动实现 `impl Future`
            -   `impl Future` 被调度时可能线程之间移动，故需要 `Send + !Unpin + 'static`
                -   `Future` 默认 `Unpin`，需显式 `Pin`
        -   `<impl Future>.await` 表示（外部 `impl Future`）等待当前 `impl Future` 完成
            -   `impl Future` 是延迟执行执行的，仅在 `.await` 处开始执行
                -   或 `futures::join!`、`futures::try_join!` 等封装 `.await` 的类似宏
            -   `<impl Future>.await` 将被编译为 `loop` 循环两个动作
                -   先 `<impl Future>.poll()` 尝试获取结果
                -   **随后 `yield` 转移控制权**
            -   即，`.await` 指明 **异步任务之间的依赖** 关系，逻辑上可视为 “调用-被调用”
                -   对线性、顺序的任务依赖，`.await` 等待异步任务与普通函数调用无差别，总是顺序阻塞、执行
                -   对网状、可并行的任务依赖，`.await` 转移控制权给异步运行时，由运行时调度并行执行、减少无效等待
    -   `impl Future` 仅利用协程的状态保存（挂起、恢复）能力，可视为状态机
        -   状态（枚举变体）即对应异步代码块 **初始、终止、`await` 阻塞** 等不同执行阶段
            -   状态（机)（枚举变体）需维护代码块状态，以允许重入、恢复执行
        -   `Future::poll()` 即匹配状态、执行对应分支、返回任务状态 `Poll`
-   `future::Poll` 为 `Future::poll` 返回的异步任务执行状态
    -   `Poll::Ready<T>` 对应 `CouroutineState::Complete<T>`，任务完成、返回 `T` 类型值
    -   `Poll::Pending` 对应 `CouroutineState::Yielded<()>`，任务未完成（无 `yield` 值）、阻塞等待调度

> - 17.1 Futures 和 Async 语法：<https://www.rust-book-cn.com/ch17-01-futures-and-syntax.html>
> - 17.5 深入探讨一步特性：<https://www.rust-book-cn.com/ch17-05-traits-for-async.html>
> - Rust中的 `async`、`await` 语法糖：展开原理深度解析：<https://cloud.tencent.com.cn/developer/article/2584247>
> - 25.5 协程-异步编程：<https://uxiew.github.io/rust_study/%E7%AC%AC%2025%20%E7%AB%A0%20%E5%8D%8F%E7%A8%8B/25.5%20%E5%BC%82%E6%AD%A5.html>

####    *Async Runtime* 异步运行时

-   *Async Runtime* 异步运行时：管理异步代码执行的环境
    -   异步运行时即 **在最外层调度、执行协程的事件循环**
        -   异步运行时在 `.await`（`yield`）处获取控制权
        -   异步运行时维护 *事件循环*（任务队列），并不断轮询
            -   即，调用 `Future::poll()` 直至 `Ready<T>`、返回 `T`
            -   事实上，`Pending` 异步任务不会自动进入队列
                -   需要任务自身负责 `Future::poll()` 中配置、通过 `Context.Waker` 唤醒
                -   即，仅在自身准备完成、可继续执行后才进入队列，避免无效轮询空转
        -   异步项目中至少设置一个运行时用于调度 `impl Future`
            -   *Rust* 未自带运行时，标准库外有很多不同异步运行时实现，如 *Tokio*
            -   不同运行时根据设计目标有不同侧重
    -   为此，异步框架核心是：异步实现的 *IO*、网络通讯之类底层任务（`Future` 对象）
        -   维护事件循环是简单的
        -   但，底层异步任务需要实现后台执行任务、唤醒注册等机制

> - 25.5 协程-异步编程：<https://uxiew.github.io/rust_study/%E7%AC%AC%2025%20%E7%AB%A0%20%E5%8D%8F%E7%A8%8B/25.5%20%E5%BC%82%E6%AD%A5.html>
> - 深入异步：<https://tokio.rust-lang.net.cn/tokio/tutorial/async>

####    `task::Context`、`task::Waker`

```rust
pub struct RawWaker {                           // 唤醒
    data: *const (),
    vtable: &'static RawWakerVTable,
}
pub struct Waker {
    waker: RawWaker,
}
pub struct RawWakerVTable {
    clone: unsafe fn(*const ()) -> RawWaker,
    wake: unsafe fn(*const ()),
    wake_by_ref: unsafe fn(*const ()),
    drop: unsafe fn(*const ()),
}
pub struct Context<'a> {
    waker: &'a Waker,
    local_waker: &'a LocalWaker,
    ext: AssertUnwindSafe<ExtData<'a>>,
    _maker: PhantomData<fn(&'a ()) -> &'a ()>,
    _maker2: PhantomData<*mut ()>,
}
```

> - 深入异步：<https://tokio.rust-lang.net.cn/tokio/tutorial/async>
> - *Std Core* 源码 - `core::task::wake`：<https://doc.rust-lang.org/src/core/task/wake.rs.html>
> - *Futures* 源码 - `futures_task::waker`：<https://docs.rs/futures-task/0.3.31/src/futures_task/waker.rs.html>
> - *Futures* 源码 - `futures_task::arc_wake::ArcWake`：<https://docs.rs/futures/latest/futures/task/trait.ArcWake.html>
> - *Futures* 和 *Tokio* 项目的前世今生：<https://rustcc.cn/article?id=8af74de6-1e3d-4596-94ca-c3da45509f58>

####    `async`、`await`

```rust
// *********************** `lower_expr_await` 将 `<expr>.await` 去糖化为如下
match ::std::future::IntoFuture::into_future(<expr>) {
    mut pinned => loop {                                        // 循环直至 `poll()` 返回 `Ready`
        match unsafe { ::std::future::Future::poll(
            <::std::pin::Pin>::new_unchecked(&mut pinned),
            ::std::future::get_context(task_context),
        ) } {
            ::std::task::Poll::Ready(result) => break result,   // 1. `Ready` 则 `break` 跳出、返回 result`
            ::std::task::Poll::Pending => {}                    // 2. `Pending` 则继续执行 `yield` 挂起、转移控制权给运行时
        }
        task_context = yield ();                                // 3. 并在运行时调用 `.resume()` 恢复执行时，接受上下文
    }
}

// *********************** `make_async_expr`
std::future::from_generator(static move? |_task_context| -> <ret_ty> {
    <body>
}
```

-   `async`、`await` 扩展发生在源码转换为 *High-Level Intermidiate Representation* 时
    -   `lower_expr_await`：将 `<impl Future>.await` **去糖化** 为循环 `poll` 检查任务状态
        -   `yield` 点在 `loop` 循环中
            -   则，从语义上允许异步运行时重复轮询、检查任务状态
            -   即，该 `yield` 点对应状态机状态可以保持不变（转移至自身）
            -   也即，异步任务未完成时将阻塞宿主执行
        -   `yield` 前已执行 `Future::poll` 检查依赖任务执行状态
            -   即，若任务已执行完成，`.await` 将直接返回依赖任务结果 `result`
                -   且，当前任务继续执行，而不会转移控制权
        -   运行时在 `.resume()` 恢复循环时，将上下文 `task::Context` （包装）传递给 `impl Future`
            -   `Context.waker` 唤醒器将被传给 `<impl Future>.poll` 用于通知异步运行时任务准备完成
    -   `make_async_expr`：将 `async {}`（`async fn`）转换为生成器、随后被封装为匿名 `impl Future`
        -   `async fn` 异步函数：将被编译为返回 `impl Future` （或异步代码块）的（同步）函数
            -   `Future::Output` 即为异步函数、异步代码块的返回类型
        -   `async {}` 异步块将被编译器包装为 `impl Coroutine` 的匿名协程状态机
            -   块内代码成为 `resume` 方法主体
            -   块内 `await` 点成为 `yield` 点（仅挂起，无输出、输入）
        -   匿名协程状态机被编译器进一步封装进 `impl Future` 的匿名适配器
            -   匿名适配器仅为 `impl Future`
    -   其他说明
        -   以下为 `.await` 去糖化结果，对任何异步运行时均如此
            -   对已完成任务，异步任务结果 `Ready(result)` 直接直接解包返回 `result` 
            -   异步任务获取运行时上下文 `Context`，传给 `Future::poll` 用于后续唤醒任务（加入事件循环）
        -   `Future` 通过 `await`、`async` 在编译器层面依赖 `Coroutine`
            -   得以分离需要保持长期稳定 `Future` 逻辑设计、与可能变化 `Couroutine` 具体实现
            -   `Future` 侧重异步场景下的设计逻辑、具体值
            -   `Coroutine` 侧重可挂起、恢复重入的具体实现
            -   `await`、`async` 是语法层面的连接、设计

> - **Lowering async in rust**：<https://wiki.cont.run/lowering-async-await-in-rust/>
> - *Rustc* 源码 - `make_async_expr`：<https://github.com/rust-lang/rust/blob/3ee016ae4d4c6ee4a34faa2eb7fdae2ffa7c9b46/compiler/rustc%5Fast%5Flowering/src/expr.rs#L518-L607>
> - *Rustc* 源码 - `lower_expr_await`：<https://github.com/rust-lang/rust/blob/3ee016ae4d4c6ee4a34faa2eb7fdae2ffa7c9b46/compiler/rustc%5Fast%5Flowering/src/expr.rs#L609-L800>
> - *Rustc HIR* 源码 - `CoroutineSource`、`CoroutineKind`：<https://gitcode.com/GitHub_Trending/ru/rust/blob/master/compiler/rustc_hir/src/hir.rs>
> - Rust中的 `async`、`await` 语法糖：展开原理深度解析：<https://cloud.tencent.com.cn/developer/article/2584247>
> - 25.5 协程-异步编程：<https://uxiew.github.io/rust_study/%E7%AC%AC%2025%20%E7%AB%A0%20%E5%8D%8F%E7%A8%8B/25.5%20%E5%BC%82%E6%AD%A5.html>
> - 25.6 协程-`Coroutine` 与 `Future`：<https://uxiew.github.io/rust_study/%E7%AC%AC%2025%20%E7%AB%A0%20%E5%8D%8F%E7%A8%8B/25.6%20Coroutine%20%E4%B8%8E%20Future.html>
> - Future Explained(0)：<https://hsqstephenzhang.github.io/2021/11/24/rust/futures/future-explained0/>
> - *Futures* 源码 - `futures_utils::join!`：<https://docs.rs/futures-util/0.3.31/src/futures_util/async_await/join_mod.rs.html>
> - *Futures* 源码 - `futures_macro::join_internal!` 源码：<https://docs.rs/futures-macro/latest/src/futures_macro/join.rs.html>

#####   `async` 逻辑

```rust
// ************************* `async fn` 异步函数将被编译为返回异步代码块
async fn async_fn(url: &str) -> Option<String> {        // 异步函数等价于如下同步函数
    Some(url.to_string())
}
fn async_fn(url: &url)
    -> impl Future<Output = Option<String>>             // 返回异步代码块执行结果，也即 `Future` 特征对象
{
    async move {                                        // 异步代码块
        Some(url.to_string())
    }
}

// ************************* `async {}` 代码块将被编译为类似匿名枚举状态机
enum OuterFuture{
    Initial,                                    // 初始状态
    Await1{                                     // 对应某个阻塞的中间状态
        inner_future: InnerFuture,              // 维护代码块内部状态
        ...
    },
    Await2{...},
    Terminated,                                 // 终止状态
}
impl Future for OuterFuture {
    Output = ();                                // `async` 异步代码块返回类型
    fn poll(
        mut self: Pin<&mut self>,
        cx: &mut Context<'_>
    ) -> Poll<Self::Output>
    {
        use OuterFuture::*;
        match *self {
            Initial => { }                      // 代码块开头直至下个 `await` 阻塞
            Await1 {
                match inner_future.poll() {     // 检查当前 `inner_future` 执行情况
                }
                ...                             // 执行直至下个 `await` 阻塞，并更新 `self` 切换状态
            }
            Await2 => { }
            Terminated => {
                panic!("")
            }
        }
    }
}
```

> - 深入异步：<https://tokio.rust-lang.net.cn/tokio/tutorial/async>

#####   `Coroutine` 封装为 `Future`

```rust
// ****************** 匿名协程状态机被封装
struct AsyncFnFuture<C: Coroutine> {
    coroutine: C,
}

impl<C> Future for AsyncFnFuture<C>
where
    C: Coroutine<(), Yield=(), Return=Self::Output>,
{
    type Output = C::Return;
    fn poll(
        self: Pin<&mut Self>,
        cs: &mut Context<'_>
    ) -> Poll<Self::Output> {
        let coroutine = unsafe {
            self.map_unchecked_mut(|s| &mut s.coroutine)    // 获取协程可变引用
        };
        // register_waker(cx.waker())                       // 协程恢复前注册 `waker`
        match coroutine.resume(()) {
            CoroutineState::Yielded(()) => {
                Poll::Pending
            }
            CoroutineState::Complete(result) => {
                Poll::Ready(result)
            }
        }
    }
}
```

> - Rust中的 `async`、`await` 语法糖：展开原理深度解析：<https://cloud.tencent.com.cn/developer/article/2584247>

### 模块系统

| 概念             | 描述                                     |
|------------------|------------------------------------------|
| *Packages* 包    | *Crate* 集合，方便构建、测试、分发       |
| *Crate* 代码单位 | 生成库文件、可执行文件的 **模块树**      |
| *Modules* 模块   | 组织函数，控制路径、作用域、可见性       |
| *Paths* 路径     | 命名（结构体、函数、模块）项的层次、位置 |

-   模块系统
    -   *Crate*：编译器单次处理、编译生成独立（库或可执行）文件最小代码单位
        -   *Crate Root*：编译器最开始处理的源文件
            -   *Crate* 实际上即存在依赖关系的模块树，模块可能存在不同文件中
            -   根模块所在的文件即 *Crate 根*
            -   当然，实际上是根据 *Crate 根* 确定模块树根，即源文件中顶层模块
        -   *Binary Crate*：可以编译成可执行文件的代码
            -   二进制 *Crate* 中必须包含 `main` 函数，作为执行入口
            -   包可包含任意数量 *二进制 Crate*
        -   *Lib Crate*：定义共享功能、编译为库的代码
            -   包最多包含一个 *库 Crate*
        -   *Cargo* 遵循如下约定
            -   `src/main.rs`：与包同名的 *二进制 Crate* 的 *Crate 根*
            -   `src/bin/<...>.rs`：额外的 *二进制 Crate* 的 *Crate 根*
            -   `src/lib.rs`：与包同名的 *库 Crate* 的 *Crate 根*
            -   *Cargo* 将 *Crate 根* 文件传递给 `rustc` 以构建库、可执行文件
    -   *Module* 模块、*Path* 路径
        -   `mod <PATH::MODULE>;`：声明构建 *Crate* 需编译的模块
            -   通过逐级 `mod` 声明模块形成模块树
                -   一般的，文件系统路径即对应模块路径
                -   另，`mod <MODULE> {}` 可在当前模块内定义子模块
            -   子模块模块成员（函数、结构体、子模块的子模块等）默认对父模块私有，需要 `pub` 以公开
                -   而，父模块成员对子模块是公开的
        -   `use <PATH>;`：将项名称引入当前作用域，在当前模块内创建命名项的快捷方式
            -   绝对路径：从 *Crate 根* 开始的完整路径
                -   外部 *Crate* 以其名称开头（需要在 `Cargo.toml` 中添加依赖）
                -   当前 *Crate* 以 `crate` 字面量开头
            -   相对路径：从当前模块开始，使用 `self`、`super` 或当前模块内标识符（子模块名）
            -   `pub use`：公开引入的项名称，允许其他模块通过当前模块一步直接访问公开项
                -   简化、调整当前模块的暴露结构
                -   也可避免当前模块子模块整体暴露

```rust
mod thread::threadpool;                 // 相对路径 `src/thread/threadpool.rs`
mod crate::thread::threadpool;          // 绝对路径 `crate` 表示当前 Crate
use std::io:Result as IoResult;         // 重命名
use std::io:::{self, Write};            // 嵌套引入 `std::io`、`std::io::Write`
```

> - 7 使用包、库和模块管理不断发展的项目：<https://www.rust-book-cn.com/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html>
> - 7.1 包和库：<https://www.rust-book-cn.com/ch07-01-packages-and-crates.html>
> - 7.2 定义模块以控制作用域和私有性：<https://www.rust-book-cn.com/ch07-02-defining-modules-to-control-scope-and-privacy.html>
> - 7.4 使用 `use` 关键字将路径引入作用域：<https://www.rust-book-cn.com/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html>

####    自动化测试

```rust
#[cfg(test)]                                // 注解：仅在 `cargo test` 时编译，仅单元测试需要
mod tests{
    use super::*;                           // 导入所属的外部模块

    #[test]                                 // 注解：测试函数
    #[should_panic]                         // 注解：应引起 `panic`
    #[ignore]                               // 注解：默认忽略
    fn it_works() {
        assert_eq!(1, 1);
        panic!("panic here.")
    }
}
```

-   `$ cargo test` 将在测试模式下编译代码、运行测试二进制文件
    -   可通过参数指定测试范围、线程数、测试显式内容，默认的
        -   并行运行所有未被 `#[ignore]` 测试
        -   若测试通过，*Rust* 将捕获标准输出、仅打印测试通过的指示
    -   单元测试：单独测试代码单元（单个文件内代码）
        -   通常与待测试代码在同一文件中，组织为 `mod tests` 子模块、使用 `#[cfg(test)]` 注解
        -   测试被组织为 `tests` 子模块使得可以测试（当前父模块）私有函数
    -   集成测试：独立于库整体、调用公共 *API* 测试库整体是否正常工作
        -   故，项目必须有库 `crate` 才能暴露 *API* 供测试
        -   测试代码模块为独立文件、位于在 `tests/` 目录（无需 `#[cfg(test)]` 注解）
            -   *Cargo* 将会自动查找 `tests/` 目录中测试函数
            -   测试函数同样需要 `#[test]` 注解

| 断言宏                 | 描述                     |
|------------------------|--------------------------|
| `assert!(bool, msg)`   | 断言失败将打印自定义信息 |
| `assert_eq!(LHS, RHS)` | 断言失败将打印表达式值   |
| `assert_ne!(LHS, RHS)` | 断言失败将打印表达式值   |

> - 11.1 如何编写测试：<https://www.rust-book-cn.com/ch11-01-writing-tests.html>
> - 11.2 控制测试的运行方式：<https://www.rust-book-cn.com/ch11-02-running-tests.html>
> - 11.3 测试组织：<https://www.rust-book-cn.com/ch11-03-test-organization.html>

### 简单实现

####    简单线程库实现

```rust
use std::{
    sync::{mpsc, Arc, Mutex},
    thread,
};

pub struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: Option<mpsc::Sender<Job>>,
}
type Job = Box<dyn FnOnce() + Send + 'static>;

// 简单的线程池实现
impl ThreadPool {
    pub fn new(size: usize) -> ThreadPool {
        assert!(size > 0);
        let mut workers = Vec::with_capacity(size);
        // 创建通道、并准备复制消费者
        // `mpsc::channel::<Job>` 无法推断泛型参数，需手动指明
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));

        for id in 0..size {
            // 复制消费者
            let receiver = receiver.clone();
            workers.push(thread::spawn(move || {
                loop{
                    match receiver.lock().unwrap().recv() {
                        Ok(job) => { job(); },
                        Err(_) => {
                            println!("Worker {id} shut down.");
                            // 若此处无 `break`，编译器能知道闭包永远不会返回
                            // 此时，`JobHandle<>` 中为任意类型均可编译通过
                            break;
                        },
                    }
                };
            }));
        }

        ThreadPool {
            workers,
            sender: Some(sender),
        }
    }

    pub fn execucte<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.as_ref().unwrap().send(job).unwrap();
    }
}

impl Drop for ThreadPool{
    fn drop(&mut self) {
        drop(self.sender.take());

        for worker in self.workers.drain(..) {
            worker.join().unwrap();
        }
    }
}
```

> - 16.1 使用线程同时运行代码：<https://www.rust-book-cn.com/ch16-01-threads.html>
> - 16.2 使用消息在线程间传输数据：<https://www.rust-book-cn.com/ch16-02-message-passing.html>
> - 21.2 将单线程服务器转换为多线程服务器：<https://www.rust-book-cn.com/ch21-02-multithreaded.html>

####    `impl future::Future` 实现

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

struct Delay {
    when: Instant,
}
impl Future for Delay {                             // 手动实现 `Future`
    type Output = &'static str;
    fn poll(
        self: Pin<&mut self>,
        cx: &mut Context<'_>
    ) -> Poll<&'static str>
    {
        if Instant::now() >= self.when {
            println!("hello world");
            Poll::Ready("done")
        } else {
            cx.waker().wake_by_ref();               // 唤醒调度器，要求进入待执行队列
            Poll::Pending
        }
    }
}

// **************************** 引入异步运行时调度、执行异步代码块
fn main() {
    let mut rt = tokio::runtime::Runtime::new().unwarp();
    rt.block_on(async {                             // 调度、执行异步代码块
        let when = Instant::new() + Duration::from_millis(10);
        let future = Delay { when };                // 初始化 `Delay: Future`
        let out = future.await;                     // 并阻塞 `await`
        assert_eq!(out, "done");
    })
}
```

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

// ************************** 上述 `main` 中 `async` 块将被近似编译如下状态机
enum MainFuture {                                   // 上述 `main` 中异步代码块对应状态机
    State0,                                         // 初始状态
    State1(Delay),                                  // `await` **阻塞导致、对应的中间状态**
    Terminated,                                     // 终止状态
}

impl Future for MainFuture {
    type Output = ();
    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>
    ) -> Poll<()>
    {
        use MainFuture::*;

        loop {
            match *self {
                State0 => {                                     // 初始状态应执行代码
                    let when = Instant::new()
                        + Duration::from millis(10);
                    let future = Delay { when };
                    *self = State1(future);                     // 更新状态
                }
                State1(ref mut my_future) => {
                    match Pin::new(my_future).poll(cx) {        // Pin 住 `future`，`poll` 检查状态
                        Poll::Ready(out) => {
                            assert_eq!(out, "done");
                            *self = Terminated;
                            return Poll::Ready(());
                        }
                        Poll::Pending => {
                            reutrn Poll::Pending;
                        }
                    }
                }
                Terminated => {
                    panic!("future polled after completion")
                }
            }
        }
    }
}
```

> - 深入异步：<https://tokio.rust-lang.net.cn/tokio/tutorial/async>

####    简单异步超时

```rust
extern crate tokio;
extern crate futures;

use std::{future::Future, time::Duration};
use futures::{future::Either, future::self};

async fn timeout<F: Future>(
    future_to_try: F,
    max_time: Duration,
) -> Result<F::Output, Duration> {
    match future::select(future_to_try, tokio::time::sleep(max_time)).await {
        Either::Left(output) => Ok(output),
        Either::Right(fut) => Err(max_time),
    }
}

fn main(){
    // 创建运行时执行异步块
    tokio::run(async {
        match timeout(
            tokio::time::sleep(Duration::from_millis(100)),
            Duration::from_millis(90)
        ).await {
            Ok(msg) => println!("Succeed with {msg}"),
            Err(duration) => println!("Failed after {}", duration.as_secs()),
        }
    })
}
```

> - 17.3 处理任意数量的 Futures：<https://www.rust-book-cn.com/ch17-03-more-futures.html>
