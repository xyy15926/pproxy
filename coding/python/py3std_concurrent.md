---
title: Python 并发
categories:
  - Python
tags:
  - Python
  - Async
  - Concurrent
date: 2026-05-09 09:30:53
updated: 2026-06-10 16:24:33
toc: true
mathjax: true
description: 
---

##  多线程

### `threading.Thread`

```python
class Thread:
    def __init__(
        self,
        group=None,                 # 为后续 `ThreadGroup` 扩展保留，应始终为 `None`
        target=None,                # `run` 调用、线程执行目标函数
        name=None,                  # 线程名
        args=(),                    # `target` 函数参数
        kwargs={}                   # `target` 函数参数
        *,
        daemon=None,                # 守护线程，缺省继承当前线程该配置
        context=None,               # 启动线程时的 `contextvars.Context`
    ):
        pass

    def run(self):                  # 执行函数体
        pass

    def start(self):                # 启动线程
        pass

    def join(self):                 # 等待线程完成
        pass
```

-   说明
    -   通过 `threading.Thread` 创建线程有两种主要方式
        -   初始化 `Thread` 实例时，指定 `target` 执行函数体、参数
            -   应当使用关键字参数调用、创建线程实例
        -   继承 `Thread`、重载 `run` 方法
            -   除  `__init__`、`run` 之外的方法不应被重载
    -   *Python* 中线程无法被强制杀死
        -   因为 *GIL* 的存在，强行中断线程可能导致锁未释放、内存泄露、数据损坏
        -   只能协作式实现线程终止
            -   进程隔离
            -   循环内按循环次数、时间检查，主动终止
    -   其他
        -   守护线程：主线程结束时不等待此守护线程结束
            -   死循环、等待主线程发送任务的工作线程应置位

| `Thread` 方法               | 描述                         | 备注                                         |
|-----------------------------|------------------------------|----------------------------------------------|
| `Thread.run()`              | 子类重载，线程实际执行函数体 | 可将结果存储为 `self` 属性、通过 `join` 返回 |
| `Thread.start()`            | 启动线程执行                 |                                              |
| `Thread.join(timeout=None)` | 等待线程执行完毕             | 可重载以返回值，获取线程执行结果             |
| `name`                      | 线程名                       |                                              |
| `ident`                     | 线程标识符                   | 线程未启动为 `None`、结束后也可获取          |
| `native_id`                 | 内核分配的线程 *TID*         | 线程未启动为 `None`                          |
| `is_alive()`                | 线程是否存活                 | `run()` 执行之间返回 `True`                  |
| `daemon`                    | 守护线程标志                 |                                              |

> - `threading.Thread` 线程对象：<https://docs.python.org/zh-cn/3.14/library/threading.html#thread-objects>

### `threading` 同步原语：锁、信号量

| `threading` 同步原语                  | 描述                 | 关键方法                             | 适用场合                 |
|---------------------------------------|----------------------|--------------------------------------|--------------------------|
| `threading.Lock()`                    | 互斥锁               | `acquire()`、`release()`             | 简单计数、值修改         |
| `threading.RLock()`                   | 可重入锁             | `acquire()`、`release()`             | 嵌套获取锁               |
| `threading.Semaphore(value=1)`        | 限制并发数           | `acquire()`、`release()`             | 资源并发限制             |
| `threading.BoundedSemaphore(value=1)` | 防溢出的 `Semaphore` | `acquire()`、`release()`             | 资源并发限制             |
| `threading.Event()`                   | 简单事件通知         | `wait()`、`set()`、`clear()`         | 线程启动信号             |
| `threading.Condition(lock=None)`      | 复杂事件通知         | `wait()`、`notify()`、`notify_all()` | *生产者-消费者* 模式     |
| `threading.Barrier()`                 | 进度同步             | `wait()`、`reset()`、`n_waiting`     | 多阶段、多依赖并行计算   |

-   多线程通信
    -   共享内存 + 同步原语：多线程共享内存空间，配合同步机制即可实现多线程通信

> - `threading` 线程：<https://docs.python.org/zh-cn/3.14/library/threading.html>

####    `threading.Lock` 实现

```c
typedef void *PyThread_type_lock;       // `pythread.h` 中对不同平台底层锁的屏蔽

typedef struct {                        // `_threadmodule.c` 中的锁对象封装
    PyObject_HEAD
    PyThread_type_lock lock_lock;       // 底层 C 锁
    PyObject *lock_lock;                // 拥有者线程 ID (用于 RLock)
    unsigned long lock_count;           // 重入计数 (用于 RLock)
    char locked;                        // 状态标志
    char in_weakreflist;                // 弱引用支持
} lockobject;
```

-   `threading.Lock` 基于各平台的内核锁实现（*CPython* 发行版中）
    -   实现层级
        -   `pythread.h` 线程抽象层的头文件，屏蔽不同平台的差异，提供统一的线程、锁 *API*
            -   *POSIX* 平台：`pthread_mutex_t` 作为底层锁，在 `thread_pthread.h` 中定义
            -   *Windows* 平台：`CRITICAL_SECTION`、`SRWLOCK` 作为底层锁，在 `thread_nt.h` 中定义
        -   `_threadmodule.c` 基于 `pythread.h` 创建、封装供 Python 层调用的线程、锁
        -   `threading.py` 调用 `_threaddmodule.c` 线程、锁并封装为 Python 对象
    -   `pthread_mutex_t` 等内核锁机制（数据对象）都是线程局部的
        -   故，`threading.Lock` 不可用于进程间同步

### `queue` 线程安全队列

| 核心类                           | 描述                 | 与 threading 的关系                          |
|----------------------------------|----------------------|----------------------------------------------|
| `queue.Queue(maxsize=0)`         | 先进先出线程安全队列 | `maxsize` 非正数时尺寸为无限大               |
| `queue.LifoQueue(maxsize=0)`     | 后进先出（栈）       |                                              |
| `queue.PriorityQueue(maxsize=0)` | 优先级队列           | 元素须全序可比，常为 `(priority, data)` 元组 |
| `queue.SimpleQueue()`            | 无界简单队列         | 性能更好，缺少任务跟踪等功能                 |

-   `queue` 模块包含各类线程安全队列
    -   `queue.SimpleQueue` 基于 C 的原子操作实现，性能更优
        -   永远非阻塞的入队
        -   不支持 `join`、`task_done` 等任务跟踪功能

####    `Queue` 队列方法

| 方法/属性/异常                              | 描述             | 说明                                             | `SimpleQueue`                 |
|---------------------------------------------|------------------|--------------------------------------------------|-------------------------------|
| `Queue.put(item, block=True, timeout=None)` | 入队             | `block=False` 时队列满直接抛 `Full`              | ❌ 不支持                     |
| `Queue.get(block=True, timeout=None)`       | 出队             | `block=False` 时队列空直接抛 `Empty`             | ✅ 支持                       |
| `Queue.put_nowait(item)`                    | 非阻塞入队       | 等价于 `put(item, block=False)`                  | ✅ 支持（同 `put`）           |
| `Queue.get_nowait()`                        | 非阻塞出队       | 等价于 `get(block=False)`                        | ✅ 支持                       |
| `Queue.empty()`                             | 判断队列是否为空 | 返回 `bool`，多线程下结果可能瞬间过时            | ✅ 支持                       |
| `Queue.full()`                              | 判断队列是否已满 | 返回 `bool`，多线程下结果可能瞬间过时            | ❌ 不支持（无界队列永远不满） |
| `Queue.qsize()`                             | 返回当前队列大小 | MacOS、部分 Unix 上可能未实现                    | ❌ 不支持                     |
| `Queue.maxsize`                             | 队列最大容量     | `0` 表示无界                                     | ❌ 不支持（无界）             |
| `Queue.task_done()`                         | 标记任务已完成   | 消费者调用，内部计数器 `-1`，须与 `join()` 配对  | ❌ 不支持                     |
| `Queue.join()`                              | 等待所有任务完成 | 生产者调用，阻塞直到所有入队项都被 `task_done()` | ❌ 不支持                     |
| `queue.Empty`                               | 队列空异常       | `get_nowait()`、`get(block=False)` 时抛出        | ✅ 支持                       |
| `queue.Full`                                | 队列满异常       | `put_nowait()`、`put(block=False)` 时抛出        | ❌ 不会抛出（无界）           |

-   队列对象应用说明
    -   多线程场合 `empty()`、`full()`、`qsize()` 结果不可靠
    -   *生产者-消费者* 模式下，使用 `put()`、`task_done()` 对内部计数增、减
        -   生产者 `put()`、并 `join()` 等待所有任务完成
        -   消费者 `get()`、任务处理完成后调用 `task_done()`

> - `queue` 同步队列类：<https://docs.python.org/zh-cn/3.14/library/queue.html>

### `contextvars` 上下文局部存储

-   `contextvars` 上下局部存储：数据在同一上下文（线程、**协程任务**）中独立、互不干扰
    -   `contextvars` 是 `threading.local()` 的升级，同时支持线程、协程任务

> - `contextvars` 上下文变量：<https://docs.python.org/zh-cn/3.14/library/contextvars.html>

####    `contextvars.ContextVar` 上下文变量

| `contextvars.ContextVar`、`Token`        | 说明                                 | 其他                                                    |
|------------------------------------------|--------------------------------------|---------------------------------------------------------|
| `contextvars.ContextVar(name,*,default)` | 创建上下文变量                       | `name` 用于调试，应在模块顶级定义                       |
| `ContextVar.name`                        | 变量名称                             |                                                         |
| `ContextVar.get([default])`              | 获取当前上下文中变量的值             | 优先级：传入 `default` > 创建 `default` > `LookupError` |
| `ContextVar.set(value)`                  | 设置当前上下文中变量的值             | 返回 `Token` 对象，可用于 `ContextVars.reset`           |
| `ContextVar.reset(token)`                | 利用 `Token` 重置到 `set` 之前的状态 |                                                         |
| `Token.var`                              | 指向创建它的 `ContextVar`            |                                                         |
| `Token.old_value`                        | `set` 之前的旧值                     | 未设置时为 `Token.MISSING`                              |
| `Token.MISSING`                          | 哨兵值                               | 表示变量此前未设置过值                                  |

-   说明
    -   `contextvars.ContextVar` 必须在模块顶层创建，否则可能导致内存泄露

####    `context.Context` 上下文

| `contextvars.Context`                    | 说明                         | 其他                                |
|------------------------------------------|------------------------------|-------------------------------------|
| `contextvars.Context()`                  | 创建空上下文                 | 通常不直接创建，用 `copy_context()` |
| `Context.run(callable, *args, **kwargs)` | 在指定上下文中运行可调用对象 | 用于临时切换上下文执行任务          |
| `Context.copy()`                         | 返回上下文的浅拷贝           |                                     |
| `contextvars.copy_context()`             | 返回当前上下文的拷贝         | 创建 `asyncio.Task` 时常自动调用    |

-   `contextvars` 通过栈管理当前上下文 `Context`
    -   `contextvars.Context` 实现了 `Mapping` 接口，可类似字典访问、遍历其中 `ContextVars` 上下文变量
    -   `contextvars.Context` 拷贝是浅拷贝
        -   若其中上下文变量是 `dict` 等可变对象，对副本修改会影响原上下文变量（即影响父环境）
    -   `asyncio.create_task`、`asyncio.run_couroutine_threadsafe` 等方法创建 `Task` 时，会缺省复制当前线程上下文

### `threading` 其他 *API*

| `threading` 函数                                   | 描述           | 备注                                   |
|----------------------------------------------------|----------------|----------------------------------------|
| `threading.Timer(interval,function[args,kwargs,])` | 延迟执行的线程 | `Timer.cancel()` 取消                  |
| `threading.local()`                                | 线程局部数据   | 用属性存储线程局部值（各线程独立副本） |
| `threading.current_thread()`                       | 当前线程对象   |                                        |
| `threading.enumerate()`                            | 存活线程列表   |                                        |
| `threading.active_count()`                         | 存活线程数量   |                                        |
| `threading.main_thread()`                          | 主线程对象     |                                        |
| `threading.get_ident()`                            | 当前线程 ID    |                                        |

-   说明
    -   `threading.local()` 仅确保线程间拥有独立副本，若需确保协程拥有独立副本，需使用 `contextvars`
    -   `threading.Timer` 时 `Thread` 子类，类似自定义线程工作
        -   经过 `interval` 指定的秒数后，线程开始执行
        -   额外方法 `Timer.cancel` 可用于取消未执行（等待状态）的线程

```python
def hello():
    print("hello, world")

t = Timer(30.0, hello)
t.start()                       # 30 秒后执行
```

> - `threading` 线程：<https://docs.python.org/zh-cn/3.14/library/threading.html>

##  `multiprocessing`

| 维度       | 多线程 `threading`                  | 多进程 `multiprocessing`         |
|------------|-------------------------------------|----------------------------------|
| 适用场景   | 同一进程内并发                      | 跨进程并行                       |
| 数据传递   | 共享内存（指针传递）                | `pickle` 序列化、反序列化        |
| 序列化开销 | 无（直接访问对象）                  | 有（数据需 `pickle` 编码）       |
| 共享内存   | 天然共享，需同步机制保护            | 不共享，需显式传递或使用共享内存 |
| GIL 影响   | 受 GIL 限制，CPU 密集型无法真正并行 | 不受 GIL 限制，可充分利用多核    |
| 上限       | 受 OS 线程数限制（通常数千）        | 受 OS 进程数限制（通常数百）     |
| 异常处理   | 主线程捕获较复杂                    | 子进程异常可能静默丢失           |
| 调试难度   | 较简单（共享内存易调试）            | 较复杂（进程隔离、日志分散）     |
| 适用任务   | I/O 密集型、需要共享状态            | CPU 密集型、计算密集型任务       |

### `mp.Process`

```python
class Process:
    def __init__(
        self,
        group=None,                 # 必须为 `None`
        target=None,                # 进程执行函数
        name=None,
        args=(),                    # `target` 函数参数
        kwargs={}
        *,
        daemon: bool = False,       # 守护线程，置位时：主线程结束时不等待此线程结束
    ):
        pass

    def run(self):                  # 执行函数体
        pass

    def start(self):                # 启动线程
        pass

    def join(self):                 # 等待线程完成
        pass
```

-   `multiprocssing.Process` 沿用了 `threading.Thread` 的 *API*
    -   通过 `threading.Thread` 创建线程有两种主要方式
        -   初始化 `Thread` 实例时，指定 `target` 执行函数体、`args` 参数
        -   继承 `Thread`、重载 `run` 方法
    -   *Python* 中线程无法被强制杀死
        -   因为 *GIL* 的存在，强行中断线程可能导致锁未释放、内存泄露、数据损坏
        -   只能协作式实现线程终止
            -   进程隔离
            -   循环内按循环次数、时间检查，主动终止

> - `multiprocessing.Process` 类：<https://docs.python.org/zh-cn/3.14/library/multiprocessing.html#the-process-class>
> - `multiprocessing.Process` 和异常：<https://docs.python.org/zh-cn/3.14/library/multiprocessing.html#process-and-exceptions>

### 进程创建模式

| 平台            | `fork`          | `spawn`         | `forkserver`    | 说明                                    |
|-----------------|-----------------|-----------------|-----------------|-----------------------------------------|
| *Windows*       | ❌ 不支持       | ✅ 支持（默认） | ❌ 不支持       | Windows 内核无 `fork` 系统调用          |
| *macOS*         | ⚪ 可用但不推荐 | ✅ 支持（默认） | ✅ 支持         | 曾默认 `fork`，后因安全问题改为 `spawn` |
| *Linux*、*Unix* | ✅ 支持         | ✅ 支持         | ✅ 支持（默认） | 三种方法都支持                          |

-   `mp.set_start_method()` 可指定全局进程创建方式
    -   `spawn` 模式：通过 `spawn()` 从头创建子进程
        -   子进程从头启动新 Python 解释器、导入模块
            -   `if __name__ == "__main__"` 内代码不会被执行
                -   即，其中放置只应在主进程内执行一次的代码，如 `set_start_method`、全局队列初始化等
            -   父进程将 `target` 函数、进程参数等通过 `pickle` 序列化传递至子进程
                -   全局对象在主模块导入时被创建，但可能与主进程中状态不一致
            -   子进程反序列化、重新构造对象，执行 `target(*args, **kwargs)`
        -   子进程只继承必要的运行时资源
            -   标准输入输出
            -   当前工作目录
    -   `fork` 模式：通过 `fork()` 复制父进程（的当前线程）创建子进程
        -   子进程直接复制父进程内存，在子进程中执行 `target` 函数
            -   无需初始化解释器
            -   进程创建完毕后，所有变量（全局变量、进程参数）已在子进程中存在，且状态与 `fork` 线程一致
                -   即，进程参数可不支持 `pickle` 序列化
        -   父进程中存在多线程、锁竞争（包括线程锁、进程锁）时，`fork` 将导致锁状态混乱、死锁
            -   当前线程持有进程锁：当前线程 `fork`、等待子进程结束，若子进程需获取锁，则永久阻塞
            -   其他线程持有线程锁：当前线程 `fork` 后永远无法等到线程锁释放、永久阻塞
    -   `forkserver` 模式：创建服务器进程，后续所有子进程由服务器进程 `fork`
        -   `fork` 的速度：无需重新导入模块
        -   `spawn` 的安全性：服务器进程干净、单线程，避免多线程情况下 `fork` 死锁

| 对比     | `fork`                               | `spawn`                            |
|----------|--------------------------------------|------------------------------------|
| 启动速度 | 快（复制内存，无需重新初始化）       | 慢（重新启动解释器，重新导入模块） |
| 内存开销 | 低（*copy-on-write*）                | 高（独立内存空间）                 |
| 线程安全 | 有风险（复制多线程状态可能导致死锁） | 安全（干净的新进程）               |
| 全局变量 | 继承父进程                           | 不继承                             |
| 适用场景 | 简单并行、数据共享                   | 复杂应用、多线程父进程             |

> - `multiprocessing` 上下文和启动方法：<https://docs.python.org/zh-cn/3.14/library/multiprocessing.html#contexts-and-start-methods>
> - 启动方法：<https://docs.python.org/zh-cn/3.14/library/multiprocessing.html#all-start-methods>

####    `mp.get_context`、`Process`

-   `mp.get_context()` 获取上下文对象，与 `multiprocessing` 模块有相同 *API*
    -   3 类进程创建方式对应 3 种不同的上下文对象类型
        -   `mp.context.SpawnContext(BaseContext)`
        -   `mp.context.ForkContext(BaseContext)`
        -   `mp.context.ForkServerContext(BaseContext)`
    -   不同上下文对象类型绑定不同 `mp.process.BaseProcess` 子类作为其 `Process` 属性
        -   故，不同上下文对象实例化的 `BaseProcess` 子类启动方式不同（本身即不同类型）
            -   `mp.context.ForkProcess(BaseProcess)`
            -   `mp.context.SpawnProcess(BaseProcess)`
            -   `mp.context.ForkServerProcess(BaseProcess)`
        -   `BaseProcess._Popen` 方法为不同 `BaseProcess` 子类间核心差异
            -   不同子类 `_Popen` 返回不同 `Popen` 对象，对应进程 3 种启动方式
            -   `Popen._launch` 方法即开始启动、执行进程
    -   对 `spawn`、`forkserver` 模式，在 `Popen._launch` 方法中会序列化 `Process` 对象、必要数据给子进程
        -   在序列化前、后调用 `context.set_spawning_popen` 设置线程局部变量 `spawning_popen`
        -   `context.assert_spawning` 即通过 `spawning_popen` 线程局部变量判断，当前进程是否处于创建子进程状态
        -   进而，判断序列化 `Lock`、`Queue` 等同步原语是否合法

> - `multiprocessing.context` 进程上下文、`multiprocessing` *API*：<https://github.com/python/cpython/blob/3.14/Lib/multiprocessing/context.py>
> - `multiprocessing.process` 进程类型：<https://github.com/python/cpython/blob/3.14/Lib/multiprocessing/process.py>
> - `multiprocessing.popen_forkserver` 进程创建、进程对象序列化：<https://github.com/python/cpython/blob/3.14/Lib/multiprocessing/popen_forkserver.py>

####    `multiprocessing` *API*

-   事实上，`mp` 中 *API* 本身就是 `mp.context.DefaultdContext(BaseContext)` 中方法
    -   `mp` *API* 通过 `mp.context._default_context` 模块级实例导出
    -   `DefualtContext` 有独特的方法（重载、新增）
        -   `DefaultContext.set_start_method` 可设置全局进程启动方式，即设置自身 `actual_context` 属性
        -   `DefaultContext.get_context` 默认不返回自身，而返回被设置 `actual_context` 属性
            -   `mp.get_context` 相较于 `mp.set_start_method`，可避免污染用户进程
    -   而 `mp.context.Process` 被定义为单独类型
        -   额外方法即 `_defualt_context.get_context().Process` 方法
        -   即，实际上等同于对应上下文类型绑定的 `BaseProcess` 子类
        -   此时，`Process` 类型不变，但是行为可根据 `_default_context` 内设置的 `actual_context` 改变
    -   类似的，`mp` 命名空间中的 `Lock`、`Queue` 各类同步原语实际上是 `BaseContext` 中同名方法
        -   调用方法时创建 `Lock` 等同步原语对象，并关联 `get_context()` 返回的上下文对象

> - `multiprocessing.context` 进程上下文、`multiprocessing` *API*：<https://github.com/python/cpython/blob/3.14/Lib/multiprocessing/context.py>

### 同步原语

| 线程模型                     | `fork` 进程模型                    | 服务进程                     | 说明                             |
|------------------------------|------------------------------------|------------------------------|----------------------------------|
| `queue.Queue`                | `multiprocessing.JoinableQueue`    | `Manager().JoinableQueue`    | 队列                             |
|                              | `multiprocessing.Queue`            | `Manager().Queue`            | 队列，不支持 `task_done`、`join` |
| `threading.Lock`             | `multiprocessing.Lock`             | `Manager().Lock`             | 互斥锁                           |
| `threading.RLock`            | `multiprocessing.RLock`            | `Manager().RLock`            | 可重入互斥锁                     |
| `threading.Semaphore`        | `multiprocessing.Semaphore`        | `Manager().Semaphore`        | （限制并发）信号量               |
| `threading.BoundedSemaphore` | `multiprocessing.BoundedSemaphore` | `Manager().BoundedSemaphore` | 防溢出信号量                     |
| `threading.Condition`        | `multiprocessing.Condition`        | `Manager().Condition`        | 条件变量                         |
| `threading.Event`            | `multiprocessing.Event`            | `Manager().Event`            | 事件通知                         |
| `threading.Barrier`          | `multiprocessing.Barrier`          | `Manager().Barrier`          | 栅栏同步                         |
|                              | `multiprocessing.Pipe`             |                              | （两进程间）双向管道             |
|                              | `multiprocessing.Value`            | `Manager().Value`            | 共享标量值                       |
|                              | `multiprocessing.Array`            | `Manager().Array`            | 共享数组                         |
| 天然共享                     |                                    | `Manager().list`             | 共享列表                         |
| 天然共享                     |                                    | `Manager().dict`             | 共享字典                         |
|                              |                                    | `Manager().Namespace`        | 共享命名空间（属性访问）         |
|                              |                                    | `Manager().Pool`             | 跨机器进程池                     |
| `threading.local()`          |                                    |                              | 私有数据                         |

-   同步原语说明
    -   不同进程间 `Lock`、`Queue` 位于不同进程内存空间
        -   则，不同进程 `Lock`、`Queue` 需共享同一底层内核资源以实现可跨进程通信（共享）
        -   而 `mp.Lock`、`mp.Queue` 等同步原语 `pickle` 序列化被限制
    -   不同模式下，`mp.Lock`、`mp.Queue` 的使用要求有差异
        -   `fork` 模式下，子进程直接复制父进程内存，不涉及序列化
            -   全局 `Lock`、`Queue` 对象共享同一底层内核资源，可直接在多进程间使用
            -   `Lock`、`Queue` 若作为进程启动参数也被直接复制，可作为参数传递给进程 `target`
        -   `spawn` 模式下，子进程启动、导入主模块，`Process` 进程对象、必要数据被序列化传给子进程
            -   全局 `Lock`、`Queue` 在模块导入时重新创建，与主进程中对应对象无关，无法在进程间同步
            -   `Lock`、`Queue` 作为进程启动参数时，在 `spawn` 进程时可被序列化、传给子进程
        -   `forkserver` 模式类似 `spawn`
    -   其他的一些同步机制
        -   `mmap` 内存映射文件
        -   `ctypes` 数据指针操作共享内存
        -   `socket` 网络服务器

####    进程上下文、同步原语

-   同一上下文创建的多进程对象应配合使用，不应跨上下文混用
    -   包括 `Lock`、`Queue` 的同步原语中均包含 `mp.synchronize.SemLock`
        -   `mp.synchronize.SemLock` 在关联不同上下文对象时，内部内核信号对象构造不同
    -   且 `mp.synchronnize.SemLock.__getstate__` 方法做两个检查，确保 `SemLock` 仅在 `spawn` 进程间序列化、传递
        -   `context.assert_spawning()`：要求 **当前进程正通过 `spawn`、`forkserver` 创建子进程**
            -   即，`SemLock` 无法、不应直接 `pickle` 序列化
        -   `self._is_fork_ctx`：要求 `SemLock` 自身关联 `SpawnContext`、`ForkServerContext`
            -   即，`ForkContext` 关联的同步原语不可 `pickle` 序列化
    -   且因此，`Lock`、`Queue` 等各类同步原语 **不可作为 `mp.Pool` 进程池分发任务方法参数**
        -   进程池分发任务时，`context.set_spawning_popen` 未被调用以设置线程局部变量 `spawning_popen`
            -   进程池内使用 `multiprocessing.SimpleQueue` 向池内进程分发任务（具体实现有差异）
            -   则，尝试序列化 `mp.Lock`、`mp.Queue` 时将 `raise RuntimeError`
        -   若需在进程池中使用 `mp.Lock`、`mp.Queue`，可作为 `mp.Pool` 中 `initializer` 函数实参的参数传递、并在其中被设置为全局变量
            -   `initializer` 函数实参将在进程池中每个进程创建时执行一次
            -   作为实参的 `mp.Lock`、`mp.Queue` 通过 `spawn` 进程时序列化、传递给进程
        -   此行为基于设计思路：同步原语应是进程粒度对象，而不是任务粒度对象
            -   即，同步原语不应随任务分发

```python
import multiprocessing as mp
import time

_lock = None                        # 全局变量，进程池进程继承得到

def init_worker(lock):
    global _lock
    _lock = lock                    # 初始化时设置全局锁

def worker(x):
    with _lock:
        print(f"Process {x} got the lock")
        ret = x * x
        time.sleep(2)
    print(f"Process {x} release the lock")
    return ret

if __name__ == '__main__':
    lock = mp.Lock()

    # `lock` 通过 `initargs` 传给 `initializer`
    # `initializer` 在进程创建时执行一次
    with mp.Pool(4, initializer=init_worker, initargs=(lock,)) as pool:
        results = pool.map(worker, range(4))
```

> - `multiprocessing.synchronize` 同步原语：<https://github.com/python/cpython/blob/3.14/Lib/multiprocessing/synchronize.py>

####    `mp.Manager`

-   `multiprocessing.Manager`：通过独立 *Server Process* 服务进程实现跨进程数据共享
    -   子进程通过获取、操作代理对象 *Proxy* 实现共享数据
        -   代理对象通过 *socket* 连接至服务进程
            -   服务进程单线程，高并发时存在瓶颈
        -   对代理对象的操作通过 **网络请求** 转发到服务进程执行
            -   可通过配置 `Manager` 监听端口实现跨机器数据共享
            -   `Manager().Queue` 相较于 `multiprocessing.Queue` 通信开销更大
        -   子进程、服务进程底层依然通过 *pickle* 序列化通信
            -   数据量较大时序列列开销大
    -   代理对象可序列化、在不同进程间传递
        -   相较于 `mp.Lock`、`mp.Queue` 使用限制更小
        -   尤其是在进程池 `multiprocessing.Pool`、`concurrent.futures.ProcessPoolExecutor` 场合
            -   池内进程任务需包含队列作为参数时，基本只能使用 `Manager().Queue`
            -   否则，只能通过 `initializer` 初始化设置通信队列

> - 管理器：<https://docs.python.org/zh-cn/3/library/multiprocessing.html#managers>

### 进程池

####    `mp.Pool`

```python
class Pool:
    def __init__(
        self,
        processes: int = None,              # 工作进程数，默认 `os.cpu_count()`
        initializer: callable = None,       # 每个工作进程启动时调用的初始化函数
        initargs: tuple = (),               # 初始化函数的参数
        maxtasksperchild: int = None,       # 每个工作进程最大任务数，达到后重启（防内存泄漏）
        context: Context = None             # 指定启动上下文（spawn/fork/forkserver）
    ):
        pass

    def close(self):
        pass
```

-   进程池 `Pool`（包括 `ProcessPoolExecutor`）内使用 `mp.SimpleQueue` 向池内进程分发任务（具体实现有差异）
    -   进程池同样关联特定进程上下文 `self._ctx`
        -   进程池通过 `self._ctx.Process()` 创建进程（确保与进程池关联上下文一致）

| 生命周期方法           | 功能                       | 说明                                                 |
|------------------------|----------------------------|------------------------------------------------------|
| `close()`              | 关闭进程池，不再接受新任务 | 调用后仍可等待已有任务完成                           |
| `join()`               | 等待所有工作进程执行完毕   | 必须在 `close()` 或 `terminate()` 之后调用           |
| `terminate()`          | 立即终止所有工作进程       | 未完成的任务会被丢弃                                 |
| `__enter__ / __exit__` | 上下文管理器支持           | 支持 `with Pool() as p:` 语法，自动 `close` + `join` |

> - `multiprocessing.pool` 进程池：<https://github.com/python/cpython/blob/3.14/Lib/multiprocessing/pool.py>

####    `mp.Pool` 任务分发

| 任务分发方法                                                         | 返回值                   | 功能                                    | 适用场景                     |
|----------------------------------------------------------------------|--------------------------|-----------------------------------------|------------------------------|
| `apply(func, args, kwds)`                                            | `Any`                    | 同步执行单个任务                        | 简单同步调用，不推荐用于并行 |
| `apply_async(func, args, kwds, callback, error_callback)`            | `AsyncResult[Any]`       | 异步执行单个任务                        | 需要非阻塞执行、自定义回调   |
| `map(func, iterable, chunksize)`                                     | `list[Any]`              | 同步映射，将 `iterable` 每项传给 `func` | 批量处理，自动分块           |
| `map_async(func, iterable, chunksize, callback, error_callback)`     | `AsyncResult[list[Any]]` | `map` 的异步版本                        | 大批量任务，不阻塞主进程     |
| `starmap(func, iterable, chunksize)`                                 | `list[Any]`              | 同步执行多参数 `map`                    | 函数需要多个位置参数         |
| `starmap_async(func, iterable, chunksize, callback, error_callback)` | `AsyncResult[list[Any]]` | `starmap` 的异步版本                    | 多参数 + 异步                |
| `imap(func, iterable, chunksize)`                                    | `Iterator[Any]`          | 惰性迭代器版`map`                       | 结果逐个产出，节省内存       |
| `imap_unordered(func, iterable, chunksize)`                          | `Iterator[Any]`          | 无序惰性 `map`                          | 不关心结果顺序，追求速度     |

-   进程池自身开启独立线程负责管理任务队列、与工作通信、通知 `AsyncResult` 任务结果
    -   进程池中工作进程执行无限循环，从 `SimpleQueue` 中接收任务、返回结果
        -   即，任务（闭包、参数）、返回值需可 `pickle` 序列化
        -   故，`mp.Lock`、`mp.Queue` 不可作为参数
    -   `Pool.map`、`Pool.startmap` 方法会将任务序列按 `chuncksize` 分批（而不是逐个）发送给工作进程，以降低进程间通信开销
        -   缺省会将任务序列分为 4 批，并据此确定 `chunksize`
        -   `Pool.imap` 迭代式方法同样支持分批，但默认 `chunksize` 为 1
            -   迭代式方法无需缓存任务结果，内存开销可能更小
    -   同步版本方法内即调用异步方法、并 `AsyncResult.get()` 阻塞获取结果

```python
# 进程池 worker 简化逻辑
def worker(inqueue, outqueue):
    while True:
        task = inqueue.get()
        if task is None:
            break
        result = task.func(*task.args)
        outqueue.put(result)
```

| `AsyncResult` 方法 | 功能                                              |
|--------------------|---------------------------------------------------|
| `get(timeout)`     | 获取结果，可设置超时（超时 `raise TimeoutError`） |
| `ready()`          | 任务是否已完成                                    |
| `successful()`     | 任务是否成功完成（需先 `ready()`）                |
| `wait(timeout)`    | 等待任务完成                                      |

### `concurrent.futures` 

| `concurrent.futures` 类型、函数          | 说明                                          |
|------------------------------------------|-----------------------------------------------|
| `futures.ThreadPoolExecutor`             | 线程池执行器                                  |
| `futures.ProcessPoolExecutor`            | 进程池执行器                                  |
| `futures.Future`                         | 任务执行结果句柄                              |
| `futures.as_completed(fs[,timeout])`     | 按完成顺序获取 `Future` 结果                  |
| `futures.wait(fs[,timeout,return_when])` | 按 `return_when` 指定时点返回结果、未完成任务 |

####    `futures.XXXExecutor` 

```python
class ThreadPoolExecutor:
    def __init__(
        self,
        max_workers: int = None,            # 最大线程数
        thread_name_prefix: str = "",       # 线程名前缀
        initializer: callable = None,       # 线程初始化函数，初始化线程上下文
        initargs: tuple = (),               # 线程初始化函数参数
        **ctxkwargs,                        # 看源码，不应传参
    ):
        pass

    def submit(self, fn, *args, **kwargs):
        pass

    def map(
        self,
        fn: callable,
        *iterable: Any,                     # 位置参数迭代器，分别迭代各位置参数
        timeout: int = None,                # 超时时间
        chunksize: int = 1,                 # 序列化批大小，仅对进程池有意义
        buffersize: int = None,             # 最大未处理、迭代任务数
    ):
        pass


class ProcessPoolExecutor:
    def __init__(
        self,
        max_workers: int = None,            # 最大线程数
        mp_context: Context = None,         # 进程上下文
        initializer: callable = None,       # 线程初始化函数，初始化线程上下文
        initargs: tuple = (),               # 线程初始化函数参数
        *,
        max_tasks_per_child: int = None,    # 每个进程最多处理任务数
    ):
        pass

def worker(a, b):
    print(a, b)

with ProcessPoolExecutor(4) as ppe:
    ppe.map(worke, range(4), range(4))      # `map` 与 `starmap` 参数传递逻辑不同

with Pool(4) as p:
    p.starmap(worke, [range(4), range(4)])
```

-   `ThreadPoolExecutor`、`ProcessPoolExecutor` 基本共享 *API*
    -   核心方法
        -   `submit(fn, *args, **kwargs)`：提交任务
        -   `map(fn, *iterables)`：多个参数执行任务
            -   `ProcessPoolExecutor.map` 与 `Pool.starmap` 都可向任务传递多个位置参数
                -   但 `ProcessPoolExecutor.map` 中参数是 `*iterable`，各位置参数独立迭代
                -   而 `Pool.starmap` 参数是 `iterable`，参数列表需封装为列表

> - `concurrent.futures.thread` 线程池：<https://github.com/python/cpython/blob/3.14/Lib/concurrent/futures/thread.py>
> - `concurrent.futures.process` 进程池：<https://github.com/python/cpython/blob/3.14/Lib/concurrent/futures/process.py>

####    进程池对比

| 对比维度         | `multiprocessing.pool.Pool`                                    | `concurrent.futures.ProcessPoolExecutor`                |
|------------------|----------------------------------------------------------------|---------------------------------------------------------|
| 引入版本         | Python 2.6+                                                    | Python 3.2+                                             |
| API 风格         | 传统/底层                                                      | 现代/高层（基于 `Future`）                              |
| 任务提交方式     | `apply()`、`map()`、`starmap()`、`imap()` 及 `_async` 异步版本 | `submit()`, `map()`                                     |
| 异步任务提交     | `apply_async()` 返回 `AsyncResult`                             | `submit()` 返回 `Future`                                |
| 同步 `map`       | `map()` 返回完整列表                                           | 无                                                      |
| 惰性迭代 `map`   | `imap()`、`imap_unordered()` 惰性生成                          | `map()` 本身惰性返回                                    |
| 异步批量任务提交 | `map_async()`、`starmap_async()` 返回 `AsyncResult`            | `submit()` 多次 + `as_completed()`、`wait()`            |
| 多参数传递       | `starmap()`、`starmap_async()`                                 | 无 `starmap()`，需手动解包                              |
| 结果获取顺序     | `map()` 按提交顺序；`imap_unordered()` 按完成顺序              | `map()` 按提交顺序；`as_completed()` 按完成顺序         |
| 取消任务         | 不支持                                                         | `future.cancel()`（仅未开始执行的任务）                 |
| 强制终止         | `terminate()`、`close()` + `join()`                            | 不支持强制终止                                          |
| 异常处理         | 异常在 `get()` 时重新抛出，无法直接访问异常对象                | `future.exception()` 可直接获取异常对象                 |
| 回调函数         | `AsyncResult` 支持 `callback`、`error_callback`                | `Future.add_done_callback()`                            |
| 任务超时         | `AsyncResult.get(timeout=...)`                                 | `future.result(timeout=...)`                            |
| 最大处理任务数   | `maxtasksperchild` 参数支持（定期重启 worker 防内存泄漏）      | Python 3.11 开始支持 `max_tasks_per_child` 参数         |
| 进程初始化       | `initializer` + `initargs`                                     | Python 3.7 开始支持 `initializer` + `initargs`          |
| 资源管理         | 需显式 `close()` + `join()` 或 `terminate()`                   | `with` 语句自动 `shutdown()`                            |
| 线程池等价类     | `multiprocessing.pool.ThreadPool`（未公开接口）                | `concurrent.futures.ThreadPoolExecutor`（完全统一接口） |
| 性能             | 略高（任务分发开销稍低）                                       | 略低（`Future` 抽象层有额外开销）                       |
| 适用场景         | 长期运行、需要精细控制、内存泄漏风险                           | 快速开发、统一线程/进程接口、动态任务流                 |

-   关于进程池、线程池的说明
    -   线程池、进程池涉及任务队列管理，*worker* 调度、结果收集等复杂逻辑
        -   `concurrent.futures`：更现代、统一的进程池、线程池接口
        -   `multiprocessing.pool`：更底层，功能更丰富、控制更精细
            -   兼容旧版本 `Pool` *API* 实现
    -   `threading`、`multiprocessing`（自身） 模块被设计为仅提供 `Process`、`Thread`、`Lock` 等底层原语
        -   在高层抽象引入前，`Pool` 直接在 `multiprocessing` 主模块中实现
            -   后 `Pool` 移至 `multiprocessing.pool` 子模块中实现，并在主模块中重导出
        -   早期并无线程池设计：线程创建成本较低，池化收益不如进程，且受多线程受 *GIL* 限制应用范围有限
            -   后续设计有 `multiprocessing.pool.ThreadindPool` 类（但不稳定）兼容 `Pool` *API*

####    `futures.Future`

| `futures.Future` 方法            | 描述                                | 阻塞 |
|----------------------------------|-------------------------------------|------|
| `Future.result(timeout=None)`    | 获取执行结果，未完成则阻塞          | 可选 |
| `Future.exception(timeout=None)` | 获取异常（如有），无异常返回 `None` | 可选 |
| `Future.done()`                  | 是否已完成（正常/异常/取消）        | 否   |
| `Future.running()`               | 是否正在运行                        | 否   |
| `Future.cancel()`                | 尝试取消未开始的任务                | 否   |
| `Future.cancelled()`             | 是否已成功取消                      | 否   |
| `Future.add_done_callback(fn)`   | 注册完成回调                        | 否   |

##  `asyncio`

### 事件循环

| 创建、配置事件循环函数         | 描述                         | 备注               |
|--------------------------------|------------------------------|--------------------|
| `asyncio.run(coro)`            | 自动创建、运行并关闭事件循环 | 推荐入口           |
| `asyncio.get_event_loop()`     | 获取当前线程的事件循环       | 事件循环可以未运行 |
| `asyncio.get_running_loop()`   | 获取当前正在运行的事件循环   | 协程内使用         |
| `asyncio.new_event_loop()`     | 创建新的事件循环             |                    |
| `asyncio.set_event_loop(loop)` | 设置默认事件循环             |                    |

-   事件循环：负责调度协程、管理 *I/O* 事件、处理定时器和回调
    -   事件循环是单线程调度器，核心即注册事件、回调
        -   *I/O* 事件：通过 *epoll/kqueue* 等系统调用实现高效的 *I/O* 多路复用
        -   计时事件：定时器时间片处理超时、定时任务
    -   `asyncio.get_event_loop()` 可以在协程外使用，获取 `asyncio.set_event_loop()` 设置的事件循环
        -   注意，`asyncio.run()` 会清除 `asyncio.set_event_loop()` 的设定（刻意设计）
            -   另，在某些 *IPython* 版本中 *Cell* 执行依赖 `asyncio.run()`
            -   即，在 *IPython* 交互式环境中 `asyncio.get_event_loop` 可能无法正常获取事件循环
        -   而，`asyncio.get_running_loop()` 只应在协程内使用
            -   否则，不可能获取到当前线程内的在运行事件循环

####    事件循环启停、配置

| 事件循环启停                      | 描述                         | 备注                                           |
|-----------------------------------|------------------------------|------------------------------------------------|
| `loop.run_until_complete(future)` | 运行直到 `future` 完成       |                                                |
| `loop.run_forever()`              | 持续运行直到 `stop()` 被调用 | 常在后台线程执行                               |
| `loop.stop()`                     | 停止事件循环                 | 需在事件循环所在线程被调用才生效               |
| `loop.close()`                    | 关闭事件循环                 | 在运行事件循环无法关闭                         |
| `loop.set_task_factory(factory)`  | 设置事件循环工厂函数         | `asyncio.create_task` 自定义 `Task` 创建的钩子 |

####    事件循环回调

| 回调                                         | 返回值                | 描述                     | 备注                               |
|----------------------------------------------|-----------------------|--------------------------|------------------------------------|
| `loop.call_soon(callback, *args)`            | `asyncio.Handle`      | 尽快执行回调函数（同步） | 非线程安全，仅可在事件循环线程调用 |
| `loop.call_later(delay, callback, *args)`    | `asyncio.TimerHandle` | 延迟后回调               | 非线程安全                         |
| `loop.call_at(when, callback, *args)`        | `asyncio.TimerHandle` | 在指定时间执行回调       | 非线程安全                         |
| `loop.call_soon_threadsafe(callback, *args)` | `asyncio.Handle`      | 尽快执行回调函数（同步） | 线程安全，可向任意事件循环线程提交 |

-   `asyncio.Handle`：已调度但未执行的回调句柄
    -   `asyncio.TimerHandle`：`Handle` 子类，额外关联定时信息
    -   `asyncio.Handle` 非线程安全
        -   `call_soon_threadsafe` 返回句柄不应直接跨线程调用
        -   若需取消任务，需封装为函数、并再通过 `call_soon_threadsafe` 在事件循环中执行

| `asyncio.Handle` 方法 | 返回值  | 描述               | 备注                 |
|-----------------------|---------|--------------------|----------------------|
| `Handle.cancel()`     |         | 取消回调           | 回调已执行则无效     |
| `Handle.cancelled()`  | `bool`  | 回调是否已取消     |                      |
| `Handle._callback`    |         | 原回调函数         |                      |
| `Handle._args`        |         | 原回调函数参数     |                      |
| `TimerHandle.when()`  | `float` | 计划执行的绝对时间 | 相对于 `loop.time()` |

### 可等待对象

-   可等待对象：实现 `__await__` 方法的对象，可用在 `await` 表达式中
    -   *Coroutine* 协程：`async def` 函数返回、待后续在事件循环中执行的对象
    -   `asyncio.Future` 异步未来值：不涉及任务执行，只关注异步结果的占位符
        -   常用于桥接、包装外部异步任务（第三方异步系统、线程池结果等）的结果
        -   即，外部异步任务执行完毕之后通过 `Future.set_result()`、`set_exception()` 设置结果、异常
    -   `asyncio.Task` 协程任务：封装协程、隔离上下文的 `Future`，在事件循环中驱动协程执行
        -   是 `async.Future` 的子类，在强调异步结果外，驱动异步任务（协程）执行
        -   即，`Task` 在事件循环被唤醒执行协程、完成后设置结果
-   其他间接可等待对象
    -   异步生成器：`async def` 内 `yield`
        -   需通过 `async for` 迭代
    -   异步上下文管理器：实现 `__aenter__`、`__aexit__` 的对象
        -   需通过 `async with` 进入

> - Python 异步: 异步上下文管理器（17）：<https://zhuanlan.zhihu.com/p/613324037>

####    `Task`、`Future` 创建

| `Task` 创建、执行                      | 返回值           | 描述                                  | 备注                                |
|----------------------------------------|------------------|---------------------------------------|-------------------------------------|
| `loop.create_future()`                 | `asyncio.Future` | 创建绑定到当前循环的 `Future` 对象    |                                     |
| `loop.create_task(coro)`               | `asyncio.Task`   | 将协程封装为 `Task`、加入当前事件循环 | 复制主线程 `contextvars`            |
| `asyncio.create_task(coro, name=None)` | `asyncio.Task`   | 将协程封装为 `Task`、加入默认事件循环 |                                     |
| `asyncio.ensure_future(obj)`           | `asyncio.Task`   | 将可等待对象转换为 `Task`、`Future`   | `Task`、`Future` 原样返回；不再推荐 |

-   `asyncio.Task` 是赋予协程 “I/O 并行” 能力核心 *API*
    -   相较于协程，`Task` 拥有独立上下文、独立绑定事件循环
        -   而协程共享所属 `Task` 上下文，被事件循环类似生成器、非并发地调度
        -   即，协程内对 `contextvars` 的修改影响范围为所属的整个 `Task`
        -   某种意义上，若无 `Task` 创建，单纯的 `await coroutine` 与函数嵌套调用类似
    -   `asycio.Task` 可直接实例化
        -   直接实例化 `Task` 虽然会执行很多检查、设置，但不推荐
            -   验证入参为 *Couroutine*
            -   若未指定上下文，调用 `contextvars.copy_context()` 创建当前线程上下文（写时复制）副本
            -   调用 `context.run(coro)`、确保协程复制隔离上下文中运行
            -   若未指定事件循环，获取当前在 *RUNNING* 事件循环并绑定
        -   更推荐 `asyncio.create_task` 创建 `Task`
            -   支持 `task_factory` 机制，兼容 `uvloop` 等第三方异步库
            -   避免直接使用不稳定 `Task` *API*

> - *Asyncio Task*：<https://docs.python.org/3/library/asyncio-task.html>

####    `Task`、`Future` 结果

| 执行、收集结果                                                     | `await` 返回值               | 描述                             | 备注                   |
|--------------------------------------------------------------------|------------------------------|----------------------------------|------------------------|
| `async asyncio.gather(*aws, return_exception=False)`               | `list[Any]`                  | 等待多个任务结果                 | 全部完成后返回结果列表 |
| `async asyncio.wait(aws, timeout=None, return_when=ALL_COMPLETED)` | `(set[Future], set[Future])` | 返回已完成、未完成两个任务集合   |                        |
| `async asyncio.wait_for(aw, timeout)`                              | `Any`                        | 带超时的等待                     | 超时后任务被取消       |
| `async asyncio.shield(aw)`                                         | `Any`                        | 保护内部任务免受外部协程取消影响 |                        |
| `asyncio.as_completed(*aws, timeout=None)`                         | `Iterator[await Any]`        | 按完成顺序迭代返回 `Future`      | 迭代结果需 `await`     |

####    `Future`、`Task` 方法

| 方法                                                  | 返回值              | 描述                               | 备注                                                                    |
|-------------------------------------------------------|---------------------|------------------------------------|-------------------------------------------------------------------------|
| `Future.done()`                                       | `bool`              | 是否已完成（有结果、异常或被取消） | 完成态即 done，不区分成功/失败/取消                                     |
| `Future.cancelled()`                                  | `bool`              | 是否已被取消                       | 仅当通过 `.cancel()` 取消且协程/回调正确处理了 `CancelledError`         |
| `Future.get_loop()`                                   | `AbstractEventLoop` | 返回该 Future 绑定的事件循环       | `create_future()` 创建的会正确绑定；裸 `Future()` 可能绑定默认循环      |
| `Future.result()`                                     | `Any`               | 获取结果值                         | 未完成抛 `InvalidStateError`；有异常抛该异常；已取消抛 `CancelledError` |
| `Future.exception()`                                  | `Exception`、`None` | 获取异常对象                       | 无异常返回 `None`；未完成抛 `InvalidStateError`                         |
| `Future.set_result(value)`                            | `None`              | 设置结果并标记完成                 | 通常由内部/框架调用；已完成态调用抛 `InvalidStateError`                 |
| `Future.set_exception(exc)`                           | `None`              | 设置异常并标记完成                 | `exc` 必须是 `Exception` 实例；已完成态调用抛 `InvalidStateError`       |
| `Future.add_done_callback(callback, *, context=None)` | `None`              | 注册完成回调                       | `callback(fut)` 签名；已完成的 Future 会立即触发回调                    |
| `Future.remove_done_callback(callback)`               | `int`               | 移除指定回调，返回移除数量         | 回调已触发则无法移除                                                    |
| `Future.cancel(msg=None)`                             | `bool`              | 请求取消                           | 返回 `True` 表示取消成功；已执行则返回 `False`                          |
| `Task.get_name()`                                     | `str`               | 获取任务名称                       | 默认形如 `Task-1`；`create_task(name=...)` 可指定                       |
| `Task.set_name(value)`                                | `None`              | 设置任务名称                       | 用于调试和日志识别                                                      |
| `Task.get_coro()`                                     | `coroutine`         | 获取包装的协程对象                 | Python 3.8+；协程已执行完毕可能返回 `None`                              |
| `Task.get_stack(*, limit=None)`                       | `list[FrameType]`   | 获取协程当前栈帧列表               | 用于调试；`limit` 限制栈深度                                            |
| `Task.print_stack(*, limit=None, file=None)`          | `None`              | 打印协程栈跟踪                     | 类似 `traceback.print_stack()`                                          |
| `Task.uncancel()`                                     | `None`              | 撤销取消请求                       | Python 3.11+；用于嵌套取消场景，减少取消计数                            |
| `Task.cancelling()`                                   | `int`               | 返回当前取消计数                   | Python 3.11+；嵌套 `cancel()` 调用时递                                  |

-   说明
    -   `Task.cancel()` 取消协程会 `raise asyncio.CancelledError`，而普通 `Future` 不会

| `Task` 属性         | 描述                           | 备注 |
|---------------------|--------------------------------|------|
| `Task._coro`        | 原始协程对象（生成器）         |      |
| `Task._loop`        | 关联的事件循环                 |      |
| `Task._context`     | 独立的 `ContextVars` 环境      |      |
| `Task._callbacks`   | 完成时的回调列表               |      |
| `Task._result`      | 执行结果、或异常               |      |
| `Task._state`       | `PENDING`、`CANCELLED`、`DONE` |      |
| `Task._fut_waiter`  | 当前正在等待的子 Future        |      |
| `Task._must_cancel` | 是否需要取消标志               |      |
| `Task._name`        | 任务名称                       |      |

####    异步任务组

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(task(1))
        tg.create_task(task(2))
```

-   `asyncio.TaskGroup` 任务组：分组管理的任务的上下文管理器
    -   退出上下文时，组内未完成任务自动取消

###    同步、异步桥接

| 同步桥接                                            | 返回值           | 描述                                                   | 备注                             |
|-----------------------------------------------------|------------------|--------------------------------------------------------|----------------------------------|
| `async asyncio.wrap_future(future,*,loop=None)`     | `asyncio.Future` | 封装 `concurrent.futures.Future`（进程池、线程池返回） | 最底层 *API*                     |
| `async loop.run_in_executor(executor, func, *args)` | `asyncio.Future` | 封装 `wrap_future`，提交 `func` 至进程池、线程池并封装 | 进程池可置 `None` 使用默认线程池 |
| `async asyncio.to_thread(func, *args,**kwargs)`     | `asyncio.Future` | 封装 `run_in_executor`、更简洁的线程桥接               | 会复制主线程 `contextvars`       |
| `asyncio.run_coroutine_threadsafe(coro, loop)`      | `futures.Future` | 向其他线程事件循环提交协程                             | 会复制主线程 `contextvars`       |

-   关于同步、异步桥接的说明
    -   `asyncio.wrap_future`、`loop.loop_run_executor`、`asyncio.to_thread` 可视为递进封装
    -   `asyncio.run_coroutine_threadsafe`、`asyncio.call_soon_threadsafe` 均可以用于向其他线程的事件循环提交任务
        -   `call_soon_threadsafe` 向事件循环提交任务（同步函数）
            -   事件循环将在下次迭代时执行任务
            -   返回 `asyncio.Handle` 非线程安全，不应直接跨线程使用，可认为不
        -   `run_coroutine_threadsafe` 在事件循环中执行协程，并返回跨线程句柄 `concurrent.futures.Future` 供注册回调、阻塞等待
            -   `run_coroutine_threadsafe` 内部即通过 `call_soon_threadsafe` 投递内部包装函数
                -   在其他线程的事件循环内 `create_task`
                -   创建跨线程句柄 `concurrent.futures.Future` 并返回
            -   `futures.Future` 存在额外开销，若无需获取任务结果，可使用 `call_soon_threadsafe`

```python
# `asyncio.to_thread` 内封装 `run_in_executor`
async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    funccall = ctx.run(func, *args, **kwargs)
    return await loop.run_in_executor(None, funccall)
```

```python
# `asyncio.run_coroutine_threadsafe` 内封装 `call_soon_threadsafe`
def run_coroutine_threadsafe(coro, loop):
    future = concurrent.futures.Future()
    def wrapper():
        try:
            task = loop.create_task(coro)
            task.add_done_callback(
                lambda t: _set_future_result(future, t)
            )
        except Exception as e:
            future.set_exception(e)
    loop.call_soon_threadsafe(wrapper)
    return future

def _set_future_result(cf: concurrent.futures.Future, task: asyncio.Task):
    try:
        result = task.result()
        cf.set_result(result)
    except asyncio.CancelledError:
        cf.cancel()
    except Exception as e:
        cf.set_exception(e)
```

> - Add Support For Context Parameter In `run_coroutine_threadsafe`：<https://discuss.python.org/t/add-support-for-context-parameter-in-run-coroutine-threadsafe/24711>
