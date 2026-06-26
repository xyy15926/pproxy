---
title: Tokio
categories:
  - Rust
tags:
  - Rust
  - Tokio
  - Async
date: 2026-03-25 18:20:08
updated: 2026-03-27 21:58:43
toc: true
mathjax: true
description: Tokio 异步运行时
---

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

    pub fn execute<F>(&self, f: F)
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
impl Future for Delay {               // 手动实现 `Future`
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
            cx.waker().wake_by_ref(); // 唤醒调度器，要求进入待执行队列
            Poll::Pending
        }
    }
}

// **************************** 引入异步运行时调度、执行异步代码块
fn main() {
    let mut rt = tokio::runtime::Runtime::new().unwarp();
    rt.block_on(async {               // 调度、执行异步代码块
        let when = Instant::new() + Duration::from_millis(10);
        let future = Delay { when };  // 初始化 `Delay: Future`
        let out = future.await;       // 并阻塞 `await`
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
enum MainFuture {                                        // 上述 `main` 中异步代码块对应状态机
    State0,                                              // 初始状态
    State1(Delay),                                       // `await` **阻塞导致、对应的中间状态**
    Terminated,                                          // 终止状态
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
                State0 => {                              // 初始状态应执行代码
                    let when = Instant::new()
                        + Duration::from millis(10);
                    let future = Delay { when };
                    *self = State1(future);              // 更新状态
                }
                State1(ref mut my_future) => {
                    match Pin::new(my_future).poll(cx) { // Pin 住 `future`，`poll` 检查状态
                        Poll::Ready(out) => {
                            assert_eq!(out, "done");
                            *self = Terminated;
                            return Poll::Ready(());
                        }
                        Poll::Pending => {
                            return Poll::Pending;
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

####    `futures` Crate

-   或 `futures::join!`、`futures::try_join!` 等封装 `.await` 的类似宏
