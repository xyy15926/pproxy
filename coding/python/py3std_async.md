---
title: Python 并发
categories:
  - Python
tags:
  - Python
  - Async
  - Concurrent
date: 2026-05-09 09:30:53
updated: 2026-05-13 08:47:26
toc: true
mathjax: true
description: 
---

##  `threading`

### `threading.Thread`

```python
class Thread:
    def __init__(
        self,
        target,                     # 线程执行函数体
        args,                       # `target` 线程函数体参数
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

-   说明
    -   通过 `threading.Thread` 创建线程有两种主要方式
        -   初始化 `Thread` 实例时，指定 `target` 执行函数体、`args` 参数
        -   继承 `Thread`、重载 `run` 方法
    -   *Python* 中线程无法被强制杀死
        -   因为 *GIL* 的存在，强行中断线程可能导致锁未释放、内存泄露、数据损坏
        -   只能协作式实现线程终止
            -   进程隔离
            -   循环内按循环次数、时间检查，主动终止

| `Thread` 方法    | 描述                         | 备注                                         |
|------------------|------------------------------|----------------------------------------------|
| `Thread.run()`   | 子类重载，线程实际执行函数体 | 可将结果存储为 `self` 属性、通过 `join` 返回 |
| `Thread.start()` | 启动线程执行                 |                                              |
| `Thread.join()`  | 等待线程执行完毕             | 可重载以返回值，获取线程执行结果             |

### 锁、信号量

| `threading` 锁、同步    | 描述         | 备注                                                  |
|-------------------------|--------------|-------------------------------------------------------|
| `threading.Lock()`      | 互斥锁       | 临界区互斥                                            |
| `threading.RLock()`     | 可重入互斥锁 | 临界区互斥、同线程可重入                              |
| `threading.Event()`     | 事件通知     | `event.ready()` 发送信号、`event.wait()` 阻塞等待信号 |
| `threading.Condition()` |
| `threading.Semaphore()` |

### 其他

| `threading` 函数        | 描述        | 备注 |
|-------------------------|-------------|------|
| `threading.get_ident()` | 当前线程 ID |      |

##  `concurrent.futures`

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
    -   `asycio.Task` 可直接实例化，但不推荐
        -   验证入参为 *Couroutine*
        -   若未指定上下文，调用 `contextvars.copy_context()` 创建当前线程上下文（写时复制）副本
        -   调用 `context.run(coro)`、确保协程复制隔离上下文中运行
        -   若未指定事件循环，获取当前在 *RUNNING* 事件循环并绑定
    -   更推荐 `asyncio.create_task` 创建 `Task`
        -   支持 `task_factory` 机制，兼容 `uvloop` 等第三方异步库
        -   避免直接使用不稳定 `Task` *API*

> - *Asyncio Task*：<https://docs.python.org/3/library/asyncio-task.html>

####    `Task`、`Future` 结果

| 执行、收集结果                                                     | `awati` 返回值               | 描述                             | 备注                   |
|--------------------------------------------------------------------|------------------------------|----------------------------------|------------------------|
| `async asyncio.gather(*aws, return_exception=False)`               | `list[Any]`                  | 等待多个任务结果                 | 全部完成后返回结果列表 |
| `async asyncio.wait(aws, timeout=None, return_when=ALL_COMPLETED)` | `(set[Future], set[Future])` | 返回已完成、未完成两个任务集合   |                        |
| `async asyncio.wait_for(aw, timeout)`                              | `Any`                        | 带超时的等待                     | 超时后任务被取消       |
| `async asyncio.shield(aw)`                                         | `Any`                        | 保护内部任务免受外部协程取消影响 |                        |
| `async asyncio.as_completed(*aws, timeout=None)`                   | `Iterator[await Any]`        | 按完成顺序迭代返回 `Future`      | 迭代结果需 `await`     |

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

```python
async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    funccall = ctx.run(func, *args, **kwargs)
    return await loop.run_in_executor(None, funccall)
```

> - Add Support For Context Parameter In `run_coroutine_threadsafe`：<https://discuss.python.org/t/add-support-for-context-parameter-in-run-coroutine-threadsafe/24711>
