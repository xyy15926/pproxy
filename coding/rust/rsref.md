---
title: Rust Ref
categories:
  - Coding
  - Rust
tags:
  - Rust
date: 2025-12-22 10:38:52
updated: 2026-01-10 21:50:03
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
        -   遮蔽：同名变量可用 `let` 遮蔽、再次赋值，再次赋值类型可不同
        -   `let` 声明变量和初始化可以分开，可用于调整变量作用域
    -   `const` 关键字声明常量
        -   常量必须注解值类型
        -   常量可在任何作用域中声明，包括全局作用域
        -   常量在程序整个运行期间、在其声明的作用域内有效
        -   常量只能设置为常量表达式、字面量

> - 3.1 变量与可变性：<https://www.rust-book-cn.com/ch03-01-variables-and-mutability.html>

####    静态变量

```rust
static HALLO: &str = "Hello, world";                // 静态引用变量生命周期总是 `'static`
static mut COUNTER: u32 = 0;
unsafe {
    println!("{}", *(&raw const COUNTER));          // 通过裸指针访问可变静态变量
}
```

-   全局变量、静态变量：值在内存中有固定地址的变量
    -   静态变量内存地址固定，而常量在每次使用时可复制其数据
        -   静态引用变量只能存储具有 `'static` 声明周期的引用
        -   即，编译器总是可以推断此静态引用变量的生命周期，无需显式注解
    -   对不可变的静态变量的访问是安全的
    -   而，对可变的静态变量的访问、修改是不安全的，存在竞争风险
        -   编译器不允许创建对可变静态变量的引用，只能通过显式创建裸指针、并解引用访问（包括引用被隐式创建时）

> - 20.1 不安全的 Rust - 访问或修改可变的静态变量：<https://www.rust-book-cn.com/ch20-01-unsafe-rust.html>

### 数据类型

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
    -   数组 `[...]`：固定长度、单一类型
        -   数据类型 `[TYPE; LEN]`：成员类型、长度

```rust
let tup3: (i32, f64, u8) = (500, 6.4, 1);   // 显式指定元组类型
let (x, y, z) = tup3;                       // 模式匹配解构元组
let x = tup3.0;                             // 索引直接访问

let arr: [i33; 5] = [1, 2, 3, 4, 5];        // 显式指定数组类型
let fst = arr[0];                           // 索引直接访问
```

> - 3.2 数据类型：<https://www.rust-book-cn.com/ch03-02-data-types.html>

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

####    类型转换、强制转换

```rust
trait Trait {}
fn foo<X: Trait>(t: X) {}
impl<'a> Trait for &'a i32 {}
fn main() {
    let t: &mut i32 = &mut 0;
    foo(t);                                 // `t: &mut i32` 不会强制转换为 `&i32`
}
```

-   *Coercions* （类型）强制转换：在某些上下文中，类型可以被隐式的转换为其他类型
    -   强制转换通常是 **类型弱化**，主要是引用、生命周期的变化
    -   但，**强制转换应用以满足 *Trait Bound* **
-   *Cast* （类型）转换 `EXPR as TYPE`
    -   类型转换是强制转换超集，所有强制转换都可以通过转换显式完成
    -   （类型）转换不是 `unsafe` 的（通常不违反内存安全）
        -   转换很危险但不会执行失败，但可能出现 “难以理解” 情况
        -   转换往往围绕 *Raw Pointer* 裸指针、原始数据类型进行
    -   注意事项
        -   转换原始切片时不会自动调整长度：`*const [u16] as *const [u8]` 得到切片包含一半内存
        -   转换不可传递：`e as U1 as U2` 有效不保证 `e as U2` 有效


> - 4.1 强制转换：<https://doc.rust-lang.net.cn/nomicon/coercions.html>
> - 4.1 Coercions：<https://doc.rust-lang.org/nomicon/coercions.html>
> - 4.3 转换：<https://doc.rust-lang.net.cn/nomicon/casts.html>
> - 4.3 Cast：<https://doc.rust-lang.org/nomicon/casts.html>
> - 10.7 类型转换 - 可强制转换的类型：<https://doc.rust-lang.net.cn/reference/type-coercions.html#coercion-types>
> - 8.2.4 运算符表达式 - 类型转换表达式：<https://doc.rust-lang.net.cn/reference/expressions/operator-expr.html#type-cast-expressions>

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

> - 3.3 函数：<https://www.rust-book-cn.com/ch03-03-how-functions-work.html>
> - 3.5 控制流：<https://www.rust-book-cn.com/ch03-03-how-functions-work.html>

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
        -   *类单元结构体*：无字段结构体
            -   适合实例无需存储数据，仅需实现某 `trait` 的场合
    -   结构体类型、枚举类型通过 `impl` 块（可分多块）定义方法、关联函数（首个参数非 `self`）
        -   方法中 `self` 参数可以为多种类型（有限的）
        -   `.` 运算符（调用方法）会自动引用、解引用、强制转换直至调用者、方法签名类型匹配
            -   引用除 `&`、`&mut` 外，还 **包装进智能指针**
            -   解引用也包括智能指针解引用 `Deref`（甚至支持多层嵌套）
            -   强制转换更多指转换为 `trait` 对象、调用 `trait` 中方法

| `self` 类型        | 说明                 |
|--------------------|----------------------|
| `self: Self`       | 当前类型为 `self`    |
| `self: &Self`      | 可简写为 `&self`     |
| `self: &mut Self`  | 可简写为 `&mut self` |
| `self: Rc<Self>`   |                      |
| `self: Box<Self>`  |                      |
| `self: Arc<Self>`  |                      |
| `self: Pin<&Self>` |                      |

> - 2.7 方法 Method：<https://course.rs/basic/method.html>
> - 5.1 定义和实例化结构体：<https://www.rust-book-cn.com/ch05-01-defining-structs.html>
> - 4.2 点运算符：<https://doc.rust-lang.net.cn/nomicon/dot-operator.html>
> - How is it even possible? self type’s is not Self but Pin<&mut Self>：<https://users.rust-lang.org/t/how-is-it-even-possible-self-types-is-not-self-but-pin-mut-self/49683>
> - 6.15 The Rust Reference - Associated items - Method：<https://doc.rust-lang.org/stable/reference/items/associated-items.html#methods>

####    *Trait*

```rust
trait Summary{
    fn sum_user(&self) -> String;               // 方法签名
    fn summarize(&self) -> String {             // 带默认实现的方法
        format!("Read more from {}", self.sum_user())
    }
}
impl Summary for User{                          // 为结构体实现 Trait
    fn sum_user(&self) -> String {
        self.username.clone()
    }
}
impl Summary for Message {                      // 为枚举类型实现 Trait
    fn sum_user(&self) -> String {
        format!("{:?}", &self)
    }
}
```

-   `trait` 特性：将方法签名组合，以定义实现某些目的所需一组的行为
    -   **`trait` 方法总是公开的**
        -   调用 `trait` 中方法前，需要将单独将 `trait` 引入作用域（不仅需要引入 `struct`、`enum`）
    -   孤儿规则：只能在 `trait` 或类型是本地 *Crate* 的情况下可以实现 `trait`
        -   确保代码不被从外部破坏
        -   避免同一类型、`trait` 实现多次实现的冲突
    -   *Mark Trait* 标记特性：仅用于 *Trait Bound*，即用于编译器检查某类型是否满足要求
        -   标记特性没有类似方法的关联项，**实现本身即开发对类型的保证**
            -   正确的实现即类型具备标记特性应有的特征
            -   错误的实现可能导致未定义的行为
        -   标记特性一般成对、互补：默认派生、反默认 `!`
            -   `Send`、`!Send`
            -   `Sync`、`!Sync`
            -   `Unpin`、`!Unpin`

> - 10.2 Traits：定义共享行为：<https://www.rust-book-cn.com/ch10-02-traits.html>
> - 20.2 高级特性：<https://www.rust-book-cn.com/ch20-02-advanced-traits.html>

####    泛型、*Trait Bound*、*Super Trait*

```rust
use std::ops::Add;

fn first<T>(list: &[T]) -> &T {                 // 函数定义中泛型
    &list.0
}
enum Option<T> {                                // 枚举定义中泛型
    Some(T),
    None,
}
struct Point<T> {                               // 结构体定义中泛型
    x: T,
    y: T,
}
impl<T> Point<T> {                              // 方法定义中泛型
    fn x(&self) -> &T {
        &self.x
    }
}
```

-   泛型
    -   *Rust* 在编译时对泛型代码单态化，保证泛型参数没有额外运行时成本
        -   单态化：编译时使用具体类型将泛型代码填充为特定类型的代码
        -   即类似，手动为每个类型复制一套代码
    -   *Trait Bound*：限制泛型单态化时的类型，必须实现某些 `trait`
        -   函数返回类型可为泛型参数，但 **函数实际返回的类型必须一致**
            -   因为，泛型通过编译时单态化实现
            -   若确实需要返回不同类型，应用动态类型 `Box<dyn ...>`
    -   *Super Trait*：限制为某类型实现该 `trait` 前，该类型需实现其他 `trait`

```rust
use std::fmt::Display; 

fn out_sum<T: Summary + Display>(x: &T) {       // Trait Bound，`+` 分隔多个
    println!("{}", x.summarize());
}
fn out_sum(x: &(impl Summary + Display)) {      // 函数 Trait Bound 简写语法糖
    println!("{}", x.summarize());
}
fn out_sum<T>(x: &T) -> ()
where
    T: Summary + Display,                       // `where` 子句 Trait Bound
{
    println!("{}", x.summarize());
}
```

```rust
impl<T: Display> Point<T> {                     // Trait Bound：限制为特定类型实现方法
    fn display(&self) {}
}
impl<T: Display> Summary for T {}               // Trait Bound：限制为特定类型实现 `trait`
trait Summary: Display {}                       // Super Trait：实现 `Summary` 前需实现 `Display`
```

> - 10.1 泛型数据类型：<https://www.rust-book-cn.com/ch10-01-syntax.html>
> - 10.2 Traits：定义共享行为：<https://www.rust-book-cn.com/ch10-02-traits.html>
> - 20.2 高级特性：<https://www.rust-book-cn.com/ch20-02-advanced-traits.html>

####    `trait` 对象

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

-   `dyn` *Trait 对象*：代表编译时无法确定、但均实现某 `trait` 的运行时才能确定的类型
    -   *Trait 对象* 仅用于对通用行为进行抽象
    -   *Rust* 对 *Trait 对象* 进行动态分派
        -   编译器无法预知具体类型、需要调用的方法
        -   只能在运行时通过 *Trait 对象* 内部指针确定需调用的方法，由运行时成本

> - 18.2 使用允许不同类型值的 Trait 对象：<https://www.rust-book-cn.com/ch18-02-trait-objects.html>

####    `trait` 多态：关联类型、泛型

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

-   `trait` 相关的多态
    -   关联类型：在 `trait` 中定义的类型占位符，`trait` 中方法可在签名中使用
        -   实现 `trait` 时，需指定类型占位符为具体类型
        -   相较于带泛型参数的 `trait`
            -   带关联类型的 `trait` 只能为类型实现一次，无需注解类型
    -   运算符重载：*Rust* 中只能通过运算符相关 `trait` 重载有定义的运算符
    -   消除（来自多个 `trait`、`struct`）同名方法之间的歧异
        -   `INST.METHOD()`：优先调用直接在 `<INST>` 所属的具体类型上实现的方法
        -   `TRAIT::METHOD(&INST)` 完全限定语法：类似调用 `trait` 关联函数，显式指定调用 `trait` 中方法
            -   `<STRUCT as TRAIT>::REL_FN()` 关联函数的完全限定语法：显式指定调用 `trait` 关联函数的实现

> - 20.2 高级特性：<https://www.rust-book-cn.com/ch20-02-advanced-traits.html>

### 枚举、模式匹配

####    枚举类型

```rust
enum Message {
    Quit,                           // 枚举变体
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
    -   枚举变体是枚举类型的取值类型（而结构体字段是结构体的成员、组成部分）
        -   枚举变体可以关联不同类型、数量数据
        -   枚举变体在枚举名称（标识符）下命名空间化，且可以独立导入

> - 6.1 定义枚举：<https://www.rust-book-cn.com/ch06-01-defining-an-enum.html>

####    模式

| 模式                | 说明                                               |
|---------------------|----------------------------------------------------|
| 字面量              |                                                    |
| 命名变量            |                                                    |
| `竖杠`              | 分隔多个模式                                       |
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

> - 19 模式与匹配：<https://www.rust-book-cn.com/ch19-00-patterns.html>
> - 19.2 可反驳性：模式是否可能匹配失败：<https://www.rust-book-cn.com/ch19-02-refutability.html>
> - 19.3 模式语法：<https://www.rust-book-cn.com/ch19-03-pattern-syntax.html>

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
| `?` 宏                               | **函数直接返回** `None`、`Err` | 解包取值           |

> - 6.1 定义枚举：<https://www.rust-book-cn.com/ch06-01-defining-an-enum.html>
> - 9.2 使用 `Result` 处理可恢复的错误：<https://www.rust-book-cn.com/ch09-02-recoverable-errors-with-result.html>
> - Rust 错误处理：`Option` 和 `Result` 的使用总结：<https://zhuanlan.zhihu.com/p/668022700>

##  所有权、引用、切片、指针

### 所有权

-   所有权：指定 *Rust* 程序 **管理内存** 的一组规则
    -   所有权规则
        -   每个值在某个时点都有、且仅有唯一所有者（变量）
        -   所有者（变量）超出作用域时，值被丢弃
    -   对只包含栈上内存（大小、类型已知）的值，*Rust* 执行复制 `clone`，否则 **执行移动 `move`**
        -   `move` 移动：将原变量绑定值（的所有权）移动给（赋值给）新变量
            -   即，浅拷贝的同时，无效原变量
            -   移动可能发生在赋值、函数传参时
        -   `.clone()` 克隆（常见方法）：“深拷贝” 变量值，包括堆数据
            -   *Rust* 不会自动创建数据的 “深拷贝”，即任何自动复制都是廉价的
            -   `.clone()` 的显式调用将标识较大开销
        -   `Copy trait`：实现 `Copy` 特性的类型值将被简单复制而不是移动
            -   `Copy` 特性（可）被用于存储在栈上的数据
                -   任何简单标量类型（可以）实现 `Copy`
                -   只包含实现 `Copy` 类型的元组
            -   实现、或任意成员实现 `Drop` 特性不可使用 `Copy` 注解

> - 4.1 什么是所有权：<https://www.rust-book-cn.com/ch04-01-what-is-ownership.html>

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
    -   切片（类型）`&[S..E]`：引用集合中连续元素序列的引用
        -   字符串字面量是切片 `&str` 类型：指向二进制文件中特定位置的切片
        -   数组切片是 `&[i32]` 等类型
    -   **堆上值不可被移出所有权，只能创建引用**

> - 4.2 引用与借用：<https://www.rust-book-cn.kcom/ch04-02-references-and-borrowing.html>
> - 4.3 切片类型：<https://www.rust-book-cn.com/ch04-03-slices.html>

####    引用规则、检查

-   引用遵守以下 2 条规则（编译器 **通过生命周期检查引用是否遵守规则**）
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

#####   运行时检查

-   注意，编译器的引用检查 **局限在单个函数内部的 `&`、`&mut` 引用操作符**（显式借用、方法中 `self`）
    -   即，若单个函数内借用满足 “值有最多单一可变引用（所有者）”，则编译通过
        -   对指令，**编译器逐（调用栈）层内检查借用**，递归的确保整体满足要求
        -   对数据又有，结构体中成员可变性与所有者（引用）可变性一致
            -   即正常情况下，无法修改不可变所有者、引用的成员值
            -   除非，通过 `unsafe` 方式从不可变获取可变，实现不可变外层的内部可变性
    -   此时，需要智能指针 **自行实现运行时单一可变引用检查**
        -   `RefCell<T>.borrow_mut(&self)` 参数为自身的不可变引用（返回 `RefMut<T>`）
        -   故，同时创建、存在多个 `RefMut<T>` 可正常通过编译
        -   但，运行时 `RefCell<T>` 会检查、并 `panic`

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
                -   实参中引用的生命周期 `'a` 只需是函数签名中生命周期 `'b` 的子类型 `'a:'b` 即可，不必完全一致
                -   可视为，函数签名中 **生命周期形参 `'a` 被置为（泛型单态化）包含 `'a` 的各引用生命周期交集**
                -   即常见描述的，引用的实际生命周期（变异前）不小于函数签名中生命周期即可
            -   **当然引用的实际生命周期时变异后生命周期：参数、返回值中引用的生命周期并集**

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

-   生命周期标注省略规则
    -   函数签名：编译器在没有显式标注情况下根据以下 3 条件规则推断函数签名中所有引用的生命周期
        -   每个参数分配单独生命周期参数
        -   若只有 1 个生命周期参数，改生命周期被分给所有输出作为生命周期参数
        -   若有 `&self`、`&mut self` 作为参数（即方法），则 `self` 的生命周期参数被分配给所有输出作为生命周期参数
    -   `impl` 块
        -   `'_` *Anonymous Lifetime* 匿名生命周期：类型签名中包含生命周期，但是在 `impl` 块中未使用，可用 `'_` 替代
    -   `struct` 泛型生命周期约束
        -   `struct` 中泛型的生命周期约束可以被省略

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
        -   静态声明周期似乎是 “确定类型”，不是 “泛型”
    -   *Unbound Lifetime* 无界生命周期：凭空产生的引用的生命周期
        -   无界生命周期会根据上下文需要任意扩展，甚至比 `'static` “更泛用”
        -   无界生命周期来源
            -   对函数，任何不来源于输入参数的输出生命周期都是无界的
            -   最常见的来源是对解引用的裸指针取引用 `&*(s: *const)`

> - 4.1.2 `&'static` 和 `T: 'static`：<https://course.rs/advance/lifetime/static.html>
> - 4.1.1 深入生命周期：<https://course.rs/advance/lifetime/advance.html>
> - 3.7 无界生命周期：<https://doc.rust-lang.net.cn/nomicon/unbounded-lifetimes.html>
> - 3.7 Unbounded Lifetimes：<https://doc.rust-lang.org/nomicon/unbounded-lifetimes.html>

####    子类型和变异性

```rust
fn assign<T>(input: &mut T, val: T) {
    *input = val;
}

// ************************ 编译失败
fn main() {
    let mut hello: &'static str = "hello";          // `&mut hello`: `&mut &'static str`
    {
        let world = String::from("world");          // `&world`: `&'world String`，即 `T`
        assign(&mut hello, &world);                 // 1. `&mut T` 对 `T` 不变，所以不能降级为 `&mut &'world str`
                                                    //     即无法满足 `assign` 签名中两个 `T` 类型完全一致
    }                                               // 3. 即使此处无 `{}` 分隔作用域，实际无悬垂，也无法编译
    println!("{hello}");                            // 2. 否则，可能出现悬垂指针
}

// ************************ 编译通过
fn main() {
    let world = String::from("world");
    let refh = &mut (&world[..]);
    {
        let hello: &'static str = "hello";
        assign(refh, hello);                        // 4. `&T` 对 `T` 协变，故 `&'static str` 可降级为 `&'world str`
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
    -   *Rust* 中没有继承机制，子类型仅局限于生命周期的变异性
        -   对生命周期 `'b:'a`，**生命周期 `'b` 是生命周期 `'a` 的子类型**
    -   子类型可被 “降级” 为父类型，用于需要父类型的场合
        -   常见场合即赋值、函数调用
        -   因，类型 `&'a T` 对 `'a` 协变，若 `'b:'a` 则有 `&'b T: &'a T`
        -   故，即使函数签名中各引用参数的生命周期标注相同，引用实参可具有不同生命周期
-   *Variance* 变异性：描述 “组合泛型” `F<Sub>`、`F<Super>` 之间的父子类型关系
    -   即，描述类型 `F` 与、组成 `F` 的类型的父子类型关系的相关性
        -   *Covariance* 协变性：若 `F` 是协变的，则 `F<Sub>` 是 `F<Super>` 的子类型 `F<Sub>:F<Super>`
        -   *Contravariance* 逆变性：若 `F` 是逆变的，则 `F<Super>` 是 `F<Sub>` 的子类型 `F<Super>:F<Sub>`
        -   *Invariant* 不变：若 `F` 是不变的，则 `F<Sub>`、`F<Super>` 之间不存在父子关系
    -   内置类型的变异性
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
| `*mut T`        |      | 不变     |      |
| `[T]`、`[T; n]` |      | 协变     |      |
| `fn(T) -> U`    |      | 协变     | 逆变 |
| `Box<T>`        |      | 协变     |      |
| `Unsafacell<T>` |      | 不变     |      |
| `Vec<T>`        |      | 协变     |      |
| `Cell<T>`       |      | 不变     |      |
| `RefCell<T>`    |      | 不变     |      |


> - 3.8 子类型和协变性：<https://doc.rust-lang.net.cn/nomicon/subtyping.html>
> - 3.8 Subtyping and Variance：<https://doc.rust-lang.org/nomicon/subtyping.html>
> - 10.5 子类型和变异性：<https://doc.rust-lang.net.cn/reference/subtyping.html#variance>
> - 10.5 Subtyping and Variance://doc.rust-lang.org/reference/subtyping.html#variance>

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

### 智能指针、裸指针

| 智能指针                   | 说明                                                   | 线程安全版            |
|----------------------------|--------------------------------------------------------|-----------------------|
| `std::prelude::Box<T>`     | 堆内存分配、封装，对齐非同类数据                       |                       |
| `std::prelude::Rc<T>`      | 引用计数器，共享数据                                   | `std::sync::Arc<T>`   |
| `std::prelude::Weak<T>`    | 弱引用计数，由 `Rc<T>::downgrade()` 避免引用循环       |                       |
| `std::cell::Cell<T>`       | 内部 `Copy` 类型值可变性，作为结构体成员独立可变       |                       |
| `std::cell::RefCell<T>`    | 内部非 `Copy` 类型值可变性，配合 `Rc<T>` 共享可变数据  | `std::sync::Mutex<T>` |
| `std::cell::Ref<T>`        | 值的不可变 “引用”，`RefCell<T>.borrow()` 返回          |                       |
| `std::cell::RefMut<T>`     | 值的可变 “引用”，`RefCell<T>.borrow_mut()` 返回        |                       |
| `std::cell::UnsafeCell<T>` | 通过共享引用从不可变获取可变，`Cell`、`RefCell` 内核心 |                       |

-   智能指针：行为类似指针、拥有额外元数据和功能的数据结构
    -   与引用（通常意义上的简单指针）相比
        -   智能指针大部分情况下拥有指向的数据，而引用只是借用数据
        -   智能指针实现有 `Drop`、`Deref` 特性
            -   `Deref` 特性：自定义解引用运算符 `*`，使得智能指针实例可类似引用被使用
            -   `Drop` 特性：自定义智能指针实例超出作用域时行为
    -   某种意义上，`String`、`Vec` 也是智能指针
        -   `String` 类型拥有 `str`（包含指向堆上 `str` 的引用）

> - 15. 智能指针：<https://www.rust-book-cn.com/ch15-00-smart-pointers.html>
> - 15.1 使用 `Box<T>` 指向堆上的数据：<https://www.rust-book-cn.com/ch15-01-box.html>
> - 15.4 `Rc<T>`，引用计数智能指针：<https://www.rust-book-cn.com/ch15-01-box.html>
> - 15.5 `RefCell<T>` 和内部可变性模式：<https://www.rust-book-cn.com/ch15-05-interior-mutability.html>
> - 15.6 引用循环可能导致内存泄漏：<https://www.rust-book-cn.com/ch15-06-reference-cycles.html>
> - 16.3 共享状态并发：<https://www.rust-book-cn.com/ch16-03-shared-state.html>
> - 聊聊 Rust 的 `Cell` 和 `RefCell`：<https://zhuanlan.zhihu.com/p/659426524>
> - Rust 的源码阅读：`Cell/RefCell` 与内部可变性：<https://zhuanlan.zhihu.com/p/384388192>

####    解引用、强制解引用转换、`Deref`

```rust
use std::ops::Deref;

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x: T) -> MyBox<T> {
        MyBox(x)
    }
}

impl<T> Deref for MyBox<T>{
    type Target = T;

    fn deref(&self) -> &Self::Target{
        &self.0
    }
}
```

-   解引用 `*`
    -   `Deref Trait`：自定义解引用运算符 `*`，使得智能指针实例可类似引用被使用
        -   `Deref::deref` 方法应返回引用，避免所有权转移
        -   编译器将自动调用 `Deref::deref` 方法获取 `&` 引用
            -   编译器默认只能解引用 `&` 引用
    -   `*` 显式解引用：`*<impl Deref>` 将被自动转换为 `*(<impl Deref>.deref())`
    -   函数（方法）中（隐式、自动）强制解引用转换
        -   若实参类型与形参类型不匹配，*Rust* 将自动、隐式执行解引用强制转换
            -   通过链式调用 `.deref()` 方法将实参类型转换为形参类型
            -   当然，非方法场景需要手动解引用
                -   通过可变引用赋值 `*a = 1`
        -   *Rust* 在实参类型 `T: Deref` 特性实现、与形参类型 `U` 为以下情况时执行解引用强制转换
            -   `&T` 需转换为 `&U`：`T: Deref<Target=U>`
            -   `&mut T` 需转换为 `&mut U`：`T: DerefMut<Target=U>`
            -   `&mut T` 需转换为 `&U`：`T: Deref<Target=U>`
        -   也因此，*Rust* 中没有 `->` 操作符
            -   *Rust* 调用方法时会自动添加 `&`、`&mut`、`*` 执行自动引用、解引用

> - 15.2 使用 `Deref` 将智能指针当作常规引用处理：<https://www.rust-book-cn.com/ch15-02-deref.html>
> - 5.3 方法语法：<https://www.rust-book-cn.com/ch05-03-method-syntax.html>

####    *Destructor*、`Drop Trait`

```rust
use std::prelude::Drop;

struct CustomSmartPointer{
    data: String,
}

impl Drop for CustomSmartPointer{
    fn drop(&mut self) {
    }
}
```

-   *Destructor* 清理
    -   `Drop Trait`：自定义值离开作用域时需运行的代码
        -   编译器会在自动插入清理代码，避免资源泄露
        -   以下场景，*Rust* 自动调用 `drop` 函数、清理变量拥有的（堆）内存
            -   变量超出作用域
            -   可变变量被重新赋值，原值被丢弃
        -   若需强制提前清理值，应使用 `std::mem::drop` 函数
            -   手动调用 `Drop::drop()` 方法会导致重复释放资源，不被允许

> - 15.3 使用 `Drop` 特性在清理时运行代码：<https://www.rust-book-cn.com/ch15-03-drop.html>

####    裸指针

```rust
let mut num = 5;
let r1: *const i32 = &raw const num;                    // 创建不可变裸指针，无需在 `unsafe` 块中
let r2: *mut i32 = &raw num num;                        // 创建可变裸指针
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
    -   闭包可以从定义其的作用域中捕获值，编译器根据闭包主体确定捕获方式
        -   捕获不可变引用
        -   捕获可变引用：绑定至变量时，变量必须声明为 `mut` 才可被调用
        -   获取所有权：`move` 关键字指定（`Copy` 类值依然仅复制）
    -   闭包将被编译为实现以下一个或多个特性的值，由闭包捕获、处理值方式决定
        -   `FnOnce` 只调用一次：所有闭包至少实现此特性
        -   `FnMut` 可能改变捕获值、但不获取所有权
        -   `Fn` 不改变捕获值、且不获取所有权

> - 13.1 闭包：可以捕获环境的匿名函数：<https://www.rust-book-cn.com/ch13-01-closures.html>

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
            -   闭包、函数、`async` 代码块将被编译为 `impl Fuuture` 等对应类型实例
            -   其中项应被作为实例中成员，即包含 `!Send` 值，则整体 `!Send`
            -   `thread::spawn`、`tokio::spawn` 等函数涉及将闭包等传递给其他线程（执行），故要求参数整体 `Send`，即闭包中各项均 `Send`
    -   主要的 `!Send`、`!Sync` 例外
        -   裸指针 `*mut`、`*const` 既不 `Send`、也不 `Sync`
        -   `UnsafeCell<T>`、`Cell<T>`、`RefCell<T>` 不 `Sync`
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

### 异步并发

####    `Future trait`、`async`、`await`

```rust
use std::pid::Pin;
use std::task::{Context, Pool};

enum Poll<T> {
    Ready(T),
    Pending,
}

pub trait Future {                                      // `Future` 标准库实现
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll(Self::Output);
}

async fn async_fn(url: &str) -> Option<String> {        // 异步函数等价于如下同步函数
    Some(url.to_string())
}
fn async_fn(url: &url) -> impl Future<Output = Option<String>> {
    async move {                                        // 返回异步代码块，也即 `Future` 特征对象
        Some(url.to_string())
    }
}
```

-   异步并发：协程并发，手动调度任务
    -   `std::future::Future trait` 特征、*Async* 异步语法
        -   `Future trait`：标准库中异步操作的公共特性
            -   *Future*、*Task*、*Promise*：可能现在还没准备好，但未来某个时刻会准备好的值
            -   异步运行时的核心即对 `Future` 特征对象进行调度、执行
        -   `async` 关键字：指定代码块（函数）可以被中断、恢复
            -   `async` 异步代码块：将被编译为 `impl Future` 特征对象
            -   `async fn` 异步函数：将被编译为返回 `impl Future` 特征对象的（同步）函数
        -   `await` 关键字：在 `async` 块、函数中使用，阻塞当前 `async` 块等待 `impl Future`
            -   `await` 处将将控制权交给运行时，由允许时调度、控制任务执行
            -   `await` 之间的流程都是同步顺序执行
    -   *Async Runtime* 异步运行时：管理异步代码执行的环境
        -   `Future` 特征对象（异步代码块）涉及控制权移交、重入，可视为状态机
            -   `Future` 往往需要被 **传递** 给异步运行时，由运行时负责调度执行、状态维护、重入
            -   故，异步项目中至少设置一个运行时并执行 `Future`
                -   *Rust* 未自带运行时，标准库外有很多不同异步运行时实现，如 *Tokio*
                -   不同运行时根据设计目标有不同侧重

```rust
match fut.poll() {                              // `<impl Future>.await` 在异步运行时中将被类似处理
    Ready(ret) => ret,                          // 未来值已实现，则返回
    Pending => {
        ...                                     // 控制权交换给异步运行时，等待下次轮询
    }
}
```

> - 17 异步编程基础：Async、Await、Futures 和 Streams：<https://www.rust-book-cn.com/ch17-00-async-await.html>
> - 17.1 Futures 和 Async 语法：<https://www.rust-book-cn.com/ch17-01-futures-and-syntax.html>
> - 17.5 深入探讨一步特性：<https://www.rust-book-cn.com/ch17-05-traits-for-async.html>

####    `Unpin`、`Pin`

-   `Unpin trait`：标记特性，指示类型值可在内存中安全移动（即，类型值无自引用）
    -   `!Unpin trait`：标记特性，类型可能存在自引用，不可在内存中安全移动
    -   大部分类型实现 `Unpin`（所有原生类型均实现 `Unpin`），以下类型实现 `!Unpin`
        -   包含 `std::marker::PhantomPinned` 类型成员结构体类型
        -   `impl Future` 异步线程块对象
-   `std::pin::Pin<P<T>>`：固定（智能）**指针指向值** 内存位置、避免值移动导致自引用（内部引用）失效的包装器
    -   `Pin<P<T>>.get_mut()`、`Pin<P<T>>.deref_mut()` 方法中要求泛型 `T` 实现 `Unpin`
        -   由此，拒绝对 `!Unpin` 类型获取可变引用 `&mut T`，即 **通过拒绝创建可变引用防止移动值**
        -   若类型 `T` 实现 `Unpin`，则 `Pin<P<T>>` 等价于 `P<T>`，无特殊效果
    -   对 `T: impl !Unpin` 类型，可通过以下方式包装获取 `Unpin` 类型包装（固定类型 `T` 值）
        -   `Box::pin` 创建 `Pin<Box<T>>`：将值固定在堆上
        -   `pin_utils::pin_mut` 创建 `Pin<&mut T>`：将值固定在栈上

> - Rust 自引用结构、`Pin` 和 `Unpin`：<https://zhuanlan.zhihu.com/p/600784379>
> - 定海神针 `Pin`、`Unpin`：<https://course.rs/advance/async/pin-unpin.html>
> - Rust 的 Pin 机制：<https://www.cnblogs.com/RioTian/p/18135131>
> - 17.5 深入研究 `Async` 的特性：<https://www.rust-book-cn.com/ch17-05-traits-for-async.html>

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
