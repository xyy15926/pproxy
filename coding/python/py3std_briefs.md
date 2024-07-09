---
title: Python3 标准库
categories:
  - 
tags:
  - 
date: 2023-10-18 18:16:32
updated: 2023-10-22 16:06:11
toc: true
mathjax: true
description: 
---

##  文本处理

| 模块          | 描述                      |
|---------------|---------------------------|
| `string`      | 字符串操作                |
| `re`          | 正则表达式                |
| `difflib`     | 增量计算                  |
| `textwrap`    | 文本包装、填充            |
| `unicodedata` | Unicode 数据库            |
| `stringprep`  | 网络字符串处理            |
| `readline`    | *GNU* `readline` 接口     |
| `rlcompleter` | *GNU* `readline` 补全函数 |

### `string`

| 常量              | 描述       |
|-------------------|------------|
| `ascii_letters`   | 大小写字母 |
| `ascii_lowercase` |            |
| `ascii_uppercase` |            |
| `digits`          |            |
| `hexdigits`       |            |
| `octdigits`       |            |
| `punctuation`     | 标点符号   |
| `prinatable`      | 可打印     |
| `whitespace`      | 空白符     |

| 类          | 描述         | 说明 |
|-------------|--------------|------|
| `Formatter` | 格式化字符串 |      |

##  二进制数据

| 模块     | 描述                  |
|----------|-----------------------|
| `struct` | 打包、拆包 C `struct` |
| `codecs` | 编解码注册、查找      |

### `struct`

| 类                         | 描述       | 说明 |
|----------------------------|------------|------|
| `struct.Struct(self, fmt)` | 结构体对象 |      |

| 函数                                          | 描述             | 返回值   | 说明 |
|-----------------------------------------------|------------------|----------|------|
| `struct.pack(fmt,v1,v2,...)`                  | 封装数据         | `bytes`  |      |
| `struct.packint(fmt,buffer,offset,v1,v2,...)` | 封装数据至buffer |          |      |
| `struct.unpack(fmt,buffer)`                   | 拆包数据         | `tuples` |      |
| `struct.unpack_from(fmt,buffer[,offset])`     | 拆包数据         | `tuples` |      |
| `struct.iter_unpack(fmt,buffer)`              | 迭代拆包         |          |      |
| `struct.calsize(fmt)`                         | 空间占用         | `int`    | Byte |

-   说明
    -   `struct.Struct` 类型具备 `struct` 模块中函数的同名方法
    -   用途
        -   封装、解压数据
        -   `reinterpret_cast` 类型转换

> - *Python Library Struct*：<https://docs.python.org/3/library/struct.html>

####    格式化字符串

| 标识符 | 字节序          | 空间占用 | 字段对齐方式 |
|--------|-----------------|----------|--------------|
| `@`    | 原生字节序      | 原生大小 | 原生对齐     |
| `=`    | 原生字节序      | 标准     | 无           |
| `<`    | *little-endian* | 标准     | 无           |
| `>`    | *big-endian*    | 标准     | 无           |
| `!`    | *network*       | 标准     | 无           |

-   说明
    -   原生字节序：平台原生字节序，常见平台均为 *little-endian*
    -   原生大小、对齐：C 编译器 `sizeof` 决定的大小、对齐方式
        -   字段起始位置须用 `0` 补齐至其长度整数倍

####    类型

| Format | C Type                | Python             | Bytes |
|--------|-----------------------|--------------------|-------|
| `x`    | pad byte              | no value           | 1     |
| `?`    | `_Bool`               | `bool`             | 1     |
| `h`    | `short`               | `integer`          | 2     |
| `H`    | `unsigned short`      | `integer`          | 2     |
| `i`    | `int`                 | `integer`          | 4     |
| `I`    | `unsigned int`        | `integer`          | 4     |
| `l`    | `long`                | `integer`          | 4     |
| `L`    | `unsigned long`       | `long`             | 4     |
| `q`    | `long long`           | `long`             | 8     |
| `Q`    | `unsigned long long`  | `long`             | 8     |
| `f`    | `float`               | `float`            | 4     |
| `d`    | `double`              | `float`            | 4     |
| `c`    | `char`                | `str` of length1   | 1     |
| `b`    | `signed char`         | `bytes` of length1 | 1     |
| `B`    | `unsigned char`       | `bytes` of length1 | 1     |
| `s`    | `char[]`              | `str`              | 1     |
| `p`    | pascal string，带长度 | `str`              | NA    |
| `n`    | `ssize_t`             | `integer`          | NA    |
| `N`    | `size_t`              | `integer`          | NA    |
| `P`    | `void *`              | 足够容纳指针的整形 | NA    |

-   说明
    -   在类型符前添加数字可以指定类型重复次数
    -   字符、字符串类型实参须以字节串形式给出
        -   字符：必须以 1B 串形式给出，重复须对应多参数
        -   字符串：可以是任意长度字节串，重复对应单个字符串
            -   长于格式指定长度被截断
            -   短于格式指定长度用`\0x00`补齐
    -   Python 按以上长度封装各类型，但 C 各类型长度取决于平台
        -   64bits 平台 C 类同标准长度

### `codecs`

| 类                                                           | 描述             | 说明 |
|--------------------------------------------------------------|------------------|------|
| `codecs.CodecInfo(encode,decodep[,streamreader...])`         | 编解码器信息包装 |      |
| `codecs.EncodedFile(file,data_encoding[,file_encoding,...])` |                  |      |
| `codecs.IncrementalEncoder([errors])`                               |                  |      |
| `codecs.IncrementalDecoder([errors])`                               |                  |      |
| `codecs.StreamWriter(stream[,errors])`                              |                  |      |
| `codecs.StreamReader(stream[,errors])`                              |                  |      |
| `codecs.StreamReaderWriter(stream,Reader,Writer[,errors])`          |                  |      |
| `codecs.StreamRecoder(stream,encode,decode,...)`                    |                  |      |

| 函数                                         | 描述                 | 返回值                   | 说明 |
|----------------------------------------------|----------------------|--------------------------|------|
| `codecs.encode(obj[,encoding,errors])`       | 编码                 | 输入、返回取依赖编解码器 |      |
| `codecs.decode(obj[,encoding,erros])`        | 解码                 | 同上                     |      |
| `codecs.lookup(encoding)`                    | 编解码器描述         | `CodecInfo`              |      |
| `codecs.getencoder(encoding)`                | 编码函数、类         | `function`               |      |
| `codecs.gedecoder(encoding)`                 | 解码函数、类         | `function`               |      |
| `codecs.getincrementalencoder(encoding)`     | 增量编码函数、类     | `function`               |      |
| `codecs.getincrementaldecoder(encoding)`     | 增量解码函数、类     | `function`               |      |
| `codecs.getreader(encoding)`                 | 读流                 | `ReadStreaming`          |      |
| `codecs.getwriter(encoding)`                 | 写留                 | `WriterStreaming`        |      |
| `codecs.register(search_function)`           | 注册编解码器查找函数 | `CodecInfo`              |      |
| `codecs.unregister(search_function)`         | 取消注册             |                          |      |
| `codecs.open(filename[,mode,encoding,...])`  | 打开文件             | `StreamingReaderWriter`  |      |
| `codecs.iterencode(iterator,encoding[,...])` | 增量迭代编码         | `Iterator`               |      |
| `codecs.iterdecode(iterator,decoding[,...])` | 增量迭代解码         | `Iterator`               |      |

-   说明
    -   标准编码
        -   均为字符串、字节串编解码，通过 C、映射表实现
        -   可由 `str.encode`、`bytes.decode` 直接提供支持
    -   Python 专属编码
        -   字符串、字节串编码：类似 *Unicode* 编码
        -   二进制转换：类字节对象、字节串之间转换
        -   字符串内转换
    -   `codecs.open` 可视为 Py2、Py3 的兼容措施，Py3 中推荐使用内置 `open`

| 常量           | 描述           | 说明                |
|----------------|----------------|---------------------|
| `BOM`          | *BOM* 字节序   | `BOM_UTF16` 别名    |
| `BOM_BE`       |                | `BOM_UTF16_BE` 别名 |
| `BOM_LE`       |                | `BOM_UTF16_LE` 别名 |
| `BOM_UTF8`     | *Unicode* 标志 |                     |
| `BOM_UTF16`    |                | 依赖平台            |
| `BOM_UTF16_BE` |                |                     |
| `BOM_UTF16_LE` |                |                     |
| `BOM_UTF32`    |                | 依赖平台            |
| `BOM_UTF32_BE` |                |                     |
| `BOM_UTF32_LE` |                |                     |

> - Python Library Codecs：<https://docs.python.org/3/library/codecs.html>
> - 编码类型：<https://docs.python.org/3/library/codecs.html#standard-encodings>

##  运行时服务

| 模块     | 描述       |
|----------|------------|
| `sys` | Python 解释器组件|

### `sys`

##  数据类型

| 模块              | 描述             |
|-------------------|------------------|
| `datetime`        | 日期、时间类型   |
| `zoneinfo`        | *IANA* 时区      |
| `calender`        | 通用日历相关函数 |
| `collections`     | 容器             |
| `collections.abc` | 日期抽象基类     |
| `heapq`           | 堆队列           |
| `bisect`          | 二分算法         |
| `array`           | 高效数值数组     |
| `weakref`         | 弱引用           |
| `types`           | 类型创造         |
| `copy`            | 浅、深拷贝       |
| `pprint`          | 打印美化         |
| `reprlib`         | `repr` 替代实现  |
| `enum`            | 枚举             |
| `graphlib`        | 图               |

##  数值、数学

| 模块        | 描述         |
|-------------|--------------|
| `numbers`   | 数值抽象类   |
| `math`      | 数学函数     |
| `cmath`     | 复数数学函数 |
| `decimal`   | 定点数     |
| `fractions` | 实数         |
|`random`|随机数|
|`statistics`|统计函数|

##  函数编程

| 模块        | 描述         |
|-------------|--------------|
| `itertools` | 迭代器、循环 |
| `functools` | 高阶函数     |
| `operator`  | 标准符号函数 |

##	文件、目录

| 模块        | 描述                 |
|-------------|----------------------|
| `pathlib`   | 面向对象文件系统路径 |
| `os.path`   | 通用文件路径操作     |
| `fileinput` | 多输入流迭代行       |
| `stat`      | 解释`os.stat()` 结果 |
|`filecmp`|文件、目录比较|
|`tempfile`|临时文件、目录|
|`glob`|Unix 风格文件路径扩展|
|`fnmatch`|Unix 风格文件名匹配|
|`linecache`|随机获取文本行|
|`shutil`|高层级文件操作|

##	数据持久化

| 模块      | 描述                   |
|-----------|------------------------|
| `pickle`  | Python 对象序列化      |
| `copyreg` | 注册 `pickle` 支持函数 |

##	数据压缩、归档

| 模块      | 描述             |
|-----------|------------------|
| `zlib`    | 兼容 *gzip* 压缩 |
| `gzip`    | *gzip* 压缩      |
| `bz2`     | *bzip2* 压缩     |
| `lzma`    | *LZMA* 压缩      |
| `zipfile` | *ZIP* 压缩       |
| `tarfile` | *tar* 归档       |

##	文件格式

| 模块           | 描述     |
|----------------|----------|
| `csv`          | *CSV*    |
| `configparser` | 配置文件 |
| `tomllib`      | *TOML*   |
| `netrc`        | `netrc`  |
| `plistlib`     | `.plist` |

##	加密

| 模块      | 描述               |
|-----------|--------------------|
| `hashlib` | 安全哈希、信息摘要 |
| `hmac`    | 基于密钥的消息认证 |
| `secrets` | 密钥随机数         |

##	通用 OS 服务

| 模块               | 描述                         |
|--------------------|------------------------------|
| `os`               | OS 接口                      |
| `io`               | 流处理                       |
| `time`             | 时间获取、转换               |
| `argparse`         | 命令行参数、选项、子命令解析 |
| `getopt`           | C 风格命令选项解析           |
| `logging`          | 日志                         |
| `logging.config`   | 日志配置                     |
| `logging.handlers` | 日期处理器                   |
| `getpass`          | 密码输入                     |
| `curses`           | 屏幕绘制、键盘处理           |
| `curses.textpad`   | *curses* 程序文本输入控件    |
| `curses.ascii`     | *ASCII* 字符工具             |
| `curses.panel`     | *curses* 面板扩展            |
| `platform`         | 平台标识                     |
| `errno`            | 标准 *errno* 系统符号        |
| `ctypes`           | 封装 C 库                    |

##	并发执行

| 模块                            | 描述           |
|---------------------------------|----------------|
| `threading`                     | 线程并行       |
| `multiprocessing`               | 进程并行       |
| `multiprocessing.shared_memory` | 跨进程共享内存 |
| `concurrent.futures`            | 启动并行任务   |
| `subprocess`                    | 子进程管理     |
| `sched`                         | 事件调度       |
| `queue`                         | 同步队列       |
| `contextvars`                   | 上下文变量     |
| `_thread`                       | 底层多线程 API |

##	网络、进程通信

| 模块        | 描述                     |
|-------------|--------------------------|
| `asyncio`   | 异步 I/O                 |
| `socket`    | 底层网络接口             |
| `ssl`       | Socket 的 *TLS/SSL* 包装 |
| `select`    | 等待 I/O 完成            |
| `selectors` | 高级 I/O 复用库          |
| `signal`    | 异步事件处理             |
| `mmap`      | 内存映射文件             |

##	网络数据处理

| 模块        | 描述                   |
|-------------|------------------------|
| `email`     | 电邮、MIME 处理        |
| `json`      | JSON 编解码            |
| `mailbox`   | 邮箱格式支持           |
| `mimetypes` | MIME、文件名映射       |
| `base64`    | *Base64* 系编码        |
| `binascii`  | 二进制、ASCII 转换     |
| `quopri`    | 编解码 *MIME* 转码数据 |

##	标记文本处理

| 模块                    | 描述                           |
|-------------------------|--------------------------------|
| `html`                  | *HTML* 支持                    |
| `html.parser`           | *HTML*、*XHTML* 解析           |
| `html.entities`         | *HTML* 实体                    |
| `xml.etree.ElementTree` | `ElementTree XMl` API          |
| `xml.dom`               | 文档对象                       |
| `xml.dom.minidom`       | 最小化 *DOM*                   |
| `xml.dom.pulldom`       | 构建部分 *DOM* 树              |
| `xml.sax`               | *SAX2* 解析器支持              |
| `xml.sax.handler`       | *SAX* 处理句柄基类             |
| `xml.sax.saxutils`      | *SAX* 工具集                   |
| `xml.sax.xmlreader`     | *XML* 解析器接口               |
| `xml.parsers.expat`     | 使用 `Expat` 得快速 *XML* 解析 |

##	互联网

| 模块                 | 描述                        |
|----------------------|-----------------------------|
| `webbrowser`         | 浏览器控制                  |
| `wsgiref`            | *WSGI* 工具参考             |
| `urllib`             | *URL* 处理                  |
| `urllib.request`     | 打开 *URL*                  |
| `urllib.response`    | *Response* 类               |
| `urllib.parser`      | 解析 *URL*                  |
| `urllib.error`       | `urllib.request` 引发的异常 |
| `urllib.robotparser` | `robots.txt` 语法分析       |
| `http`               | *HTTP*                      |
| `http.client`        | *HTTP* 协议客户端           |
| `ftplib`             | *FTP* 协议客户端            |
| `poplib`             | *POP3* 协议客户端           |
| `imaplib`            | *IMAP4* 协议客户端          |
| `smtplib`            | *SMTP* 协议客户端           |
| `uuid`               | *UUID* 对象                 |
| `socketserver`       | 服务器框架                  |
| `http.server`        | *HTTP* 服务器               |
| `http.cookies`       | *HTTP Cookie*               |
| `http.cookiejar`     | *HTTP* 客户端 *Cookie* 处理 |
| `xmlrpc`             | *XMLRPC*                    |
| `xmlrpc.client`      | *XMLRPC* 客户端             |
| `xmlrpc.server`      | *XMLRPC* 服务器             |
| `ipaddress`          | *IPv4/IPv6* 操作            |

##	多媒体服务

| 模块       | 描述           |
|------------|----------------|
| `wave`     | *WAV* 格式读写 |
| `colorsys` | 颜色系统转换   |

##	国际化

| 模块      | 描述             |
|-----------|------------------|
| `gettext` | 多语种国际化服务 |
| `locale`  | 国际化服务       |

##	程序框架

| 模块     | 描述               |
|----------|--------------------|
| `turtle` | 海龟绘图           |
| `cmd`    | 面向行的命令解释器 |
| `shlex`  | 词法分析           |

##	Tk GUI

| 模块                   | 描述          |
|------------------------|---------------|
| `tkinter`              | *Tcl/Tk* 接口 |
| `tkinter.colorchooser` | 颜色选择框    |
| `tkinter.font`         | 字体封装      |
| `tkinter.messagebox`   | 消息提示      |
| `tkinter.scrolledtext` | 滚动文字      |
| `tkinter.dnd`          | 拖放操作      |
| `tkinter.ttk`          | *Tk* 风格控件 |
| `tkinter.tix`          | *Tk* 扩展包   |
| `IDLE`                 |               |

##	开发工具

| 模块                            | 描述                 |
|---------------------------------|----------------------|
| `typing`                        | 类型提示             |
| `pydoc`                         | 文档生成器、在线帮助 |
| `doctest`                       | 文档交互性代码测试   |
| `unitest`                       | 单元测试框架         |
| `unitest.mock`                  | 模拟对象             |
| `2to3`                          | Python 代码转写      |
| `test`                          | 回归测试             |
| `test.support`                  | 测试套件             |
| `test.support.socket_helper`    | *Socket* 测试        |
| `test.support.script_helper`    | 执行测试             |
| `test.support.bytecode_helper`  | 字节码生成测试       |
| `test.support.threading_helper` | 线程测试             |
| `test.support.os_helper`        | 操作系统测试         |
| `test.support.import_helper`    | 导入测试             |
| `test.support.warnings_helper`  | 警告测试             |

##	调试、分析

| 模块           | 描述             |
|----------------|------------------|
| `bdb`          | 调试器框架       |
| `faulthandler` | 转储跟踪信息     |
| `pdb`          | 调试器           |
| `timeit`       | 代码片段执行时间 |
| `trace`        | 跟踪语句执行     |
| `tracemalloc`  | 跟踪内存分配     |

##	打包分发软件

| 模块        | 描述                      |
|-------------|---------------------------|
| `ensurepip` | 引导 `pip` 安装器         |
| `venv`      | 虚拟环境                  |
| `zipapp`    | 管理可执行 *Zip* 打包文件 |

##	运行时服务

| 模块             | 描述               |
|------------------|--------------------|
| `sys`            | 系统相关形参、函数 |
| `sys.monitoring` | 执行事件监控       |
| `sysconfig`      | 配置访问支持       |
| `builtins`       | 内建对象           |
| `__main__`       | 最高层代码执行环境 |
| `warnings`       | 警告信息控制       |
| `dataclasses`    | 数据类             |
| `contextlib`     | 上下文处理工具     |
| `abc`            | 抽象基类           |
| `atexit`         | 退出处理器         |
| `traceback`      | 堆栈跟踪信息       |
| `__future__`     | `Future` 语句定义  |
| `gc`             | 垃圾回收器         |
| `inspect`        | 检查对象           |
| `site`           | 指定域配置钩子     |

##	自定义解释器

| 模块     | 描述       |
|----------|------------|
| `code`   | 解释器基类 |
| `codeop` | 编译代码   |

##	导入模块

| 模块                      | 描述                   |
|---------------------------|------------------------|
| `zipimport`               | 从 *Zip* 导入模块      |
| `pkgutil`                 | 包扩展工具             |
| `modulefinder`            | 查找脚本使用的模块     |
| `runpy`                   | 查找并执行模块         |
| `importlib`               | `import` 实现          |
| `importlib.resources`     | 包资源读取、打开和访问 |
| `importlib.resources.abc` | 资源抽象基类           |
| `importlib.metadata`      | 包元数据               |
| `sys.path`                | 模块搜索路径初始化     |

##	语言服务

| 模块          | 描述                |
|---------------|---------------------|
| `ast`         | 抽象语法树          |
| `symtable`    | 编译器符号表        |
| `token`       | 解释树同用常量      |
| `keyword`     | 关键字              |
| `tokenize`    | 标记解析器          |
| `tabnanny`    | 模糊缩进检测        |
| `pyclbr`      | 浏览器支持          |
| `py_compile`  | 编译源文件          |
| `compileall`  | 字节编译            |
| `dis`         | 字节码反汇编器      |
| `pickletools` | `pickle` 开发工具集 |

