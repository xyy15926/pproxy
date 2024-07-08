---
title: Windows Linux Subsystem
categories:
  - Tool
  - Windows
tags:
  - Tool
  - Windows
  - Linux
  - Terminal
  - File System
date: 2019-03-21 17:27:37
updated: 2024-07-08 16:14:55
toc: true
mathjax: true
comments: true
description: Windows Linux Subsystem
---

##  *Windows Subsystem Linux*

-   *WSL*：兼容层，类似反过来的 *Wine* ，但更底层
    -   Linux、Windows 程序不兼容，是因为二者内核提供的接口不同
        -   WSL 提供 Linux 内核接口，并转换为 *NT* 内核接口
    -   以 `ls/dir` 命令为例
        -   内核调用逻辑不同
            -   在 Linux 下调用 `getdents` 内核调用
            -   在 Windows 下调用 `NtQueryDirectoryFile` 内核调用
        -   WSL 兼容逻辑
            -   在 WSL 中执行 `ls` 仍然调用 `getdents`
            -   WSL 收到请求，将系统调用转换为 *NT* 内核接口 `NTQueryDirectoryFile`
            -   *NT* 内核收到 WSL 请求，返回执行结果
            -   WSL 将结果包装后返回
    -   相较于真正 Linux 系统不足
        -   *Docker* 等涉及未实现的内核特性软件如法使用
        -   *Raw socket* 相关相关操作容易出错
        -   *I/O* 性能相对孱弱

### 与 *Cygwin* 对比

![wsl_architecture](imgs/wsl_architecture.png)

-   *Cygwin*：提供了完整的 *POSIX* 系统调用 API（以运行库`Cygwin*.dll`形式提供）
    -   工作在 *User Mode*
    -   Cygwin 将 *POSIX* 系统调用转换为 Win32 API（因为其架设在 Win32 子系统上）
        -   很多内核操作（如：`fork`）受限于 Win32 实现
    -   Linux 应用程序必须链接到 `Cynwin*.dll`，需要修改源码重新编译
        -   编译出的可执行文件为 *Win32 PE* 格式封装，只能在 Windows 下执行
        -   应用程序不直接请求内核，而是调用 Cygwin 运行库

-   WSL
    -   Linux 应用程序进程被包裹在 *Pico Process* 中，其发出的所有系统调用会被直接送往 *Kernel Mode* 中的 `lxcore.sys`、`lxss.sys`
    -   WSL 将 *POSIX* 系统调用转换为更底层的 *NT API* 调用（WSL 和 Win32 平行，直接架设在 *NT* 内核上）
    -   可以直接执行 *ELF* 格式封装的 *Linux* 可执行程序

##  *WSL* 基本使用

-   功能启用
    -   `控制面板 ->`
    -   `程序和功能 ->`
    -   `启用或关闭 Windows 功能 ->`
    -   `适用于 Linux 的 Windows 子系统`

-   进入 WSL
    -   图形界面中点击图标，以默认参数启动
    -   `wsl.exe`：打开默认发行版中默认 Shell
    -   `distroname.exe`：打开指定发行版中默认 Shell
    -   `bash.exe`：打开默认发行版中 Bash Shell
    > - 这些应用程序默认在 `Path` 中，可以直接执行

-   版本管理
    -   `wslconfig.exe`：可以用于管理多个子系统的发行版

### WSL、Windows 互操作

-   文件系统
    -   Windows 所有盘符挂载在 WSL 中 `/mnt` 目录下
    -   WSL 中所有数据存放在 `%HOME%/AppData/Local/Packages/{linux发行包名}/LocalState/rootfs` 中
        -   不要在 Windows 下直接修改，避免造成权限错误
-   端口、环境变量
    -   WSL 与 Windows 共享端口
    -   WSL 继承 Windows 的部分环境变量，如：`PATH`
-   命令
    -   在cmd中直接调用WSL命令
        ```shell
        PS> wsl [-e] ls -al
            // wsl带参数执行
        ```
    -   在WSL中调用Windows命令行程序（在`$PATH`中）
        ```shell
        $ which ipconfig.exe
        $ ipconfig.exe
        ```
    -   在WSL中启动Windows应用（在`$PATH`中）
        ```shell
        $ notepad.exe
        ```
    -   通过pipes通信
        ```shell
        $ cat foo.txt | clip.exe
        PS> ipconfig | wsl grep IPv4
        ```

### 文件权限问题

-   为 WSL，Windows 实现了两种文件系统用于支持不同使用场景
    -   *VolFs*：着力于在 Windows 文件系统上提供完整的 Linux 文件系统特性
        -   通过各种手段实现了对 *Inodes*、*Directory Entries*、*File Objects*、*File Descriptors*、*Special File Types* 的支持
        -   为支持 *Inodes*，*VolFS* 会把文件权限等信息保存在 *NTFS Extended Attributes* 中
            -   在 Windows 中新建的文件缺少此扩展参数，有些编辑器也会在保存文件是去掉这些附加参数
            -   在 Windows 中修改 WSL 中文件，将导致 *VolFs* 无法正确获得文件 metadata
        -   WSL中 `/` 就是VolFs文件系统
    -   *DrvFs*：着力提供于Windows系统的互操作性
        -   从 Windows的文件权限（即 `文件->属性->安全选项卡` 中的权限）推断出文件对应 Linux 权限
        -   所有 Windows 盘符挂在在 WSL 中 `/mnt` 是都使用 *DrvFs* 文件系统
        -   由于 *DrvFs* 文件权限继承机制很微妙，结果就是所有文件权限都是 `0777`
            -   所以 `ls` 结果都是绿色的
            -   早期 *DrvFs* 不支持 metadata，在 *Build 17063* 之后支持文件写入metadata，但是需要重新挂载磁盘
        -   可以通过设置 *DrvFs* metadata 设置默认文件权限
            ```shell
            $ sudo umount /mnt/e
            $ sudo mount -t drvfs E: /mnt/e -o metadata
            # 此时虽然支持文件权限修改，但默认权限仍然是*0777*
            $ sudo mount -t drvfs E: /mnt/e -o metadata,uid=1000,gid=1000,umask=22,fmask=111
            # 此时磁盘中默认文件权限为*0644*
            # 一般通过 `/etc/wsl.conf` 配置 *DrvFs* 自动挂载属性，而不直接命令行手动挂载
            ```

> - <https://blogs.msdn.microsoft.com/wsl/2016/06/15/wsl-file-system-support/>
> - <https://blogs.msdn.microsoft.com/commandline/2018/01/12/chmod-chown-wsl-improvements/>

### *AutoMatically Configuring WSL*

```cnf
 # `/etc/wsl.conf`
[automount]
 # 是否自动挂载
enabled = true
 # 是否处理`/etc/fstab`文件
mountFsTab = true
 # 挂载路径
root = /mnt/
 # DrvFs 挂载选项，若需要针对不同drive配置，建议使用`/etc/fstab`
options = "metadata,umask=023,dmask=022,fmask=001"
[network]
generateHosts = true
generateResolvConf = true
[interop]
 # 是否允许WSL载入windows进程
enabled = true
appendWindowsPath = true
```

-   说明
    -   如果需要给不同盘符设置不同挂载参数，需要再修改 `/etc/fstab`
        ```cnf
        E: /mnt/e drvfs rw,relatime,uid=1000,gid=1000,metadata,umask=22,fmask=111 0 0
        ```

> - <https://blogs.msdn.microsoft.com/commandline/2018/02/07/automatically-configuring-wsl/>
> - <https://devblogs.microsoft.com/commandline/automatically-configuring-wsl/>

### 其他

#### Terminal 推荐

-   WSL 默认终端是 *CMD*，功能较差
    -   [wsl-terminal](https://github.com/goreliu/wsl-terminal)：专为 WSL 开发的终端模拟器，基于 *mintty*、*wslbridge*，稳定易用
        -   受限于 *wslbridge*的 原因，WSL-Terminal 必须在 *NTFS* 文件系统中使用
        -   *mintty* 本身依赖 *CMD*，包括字体等在内配置受限于 *CMD*
    -   [ConEmu](https://conemu.github.io)：老牌终端模拟器，功能强大
    -   [Hyper](https://hyper.is)：基于 *Electron* 的跨平台终端模拟器

-   WSL-Terminal中包含一些快捷工具
    -   `tools` 目录中包含一些脚本，可以通过 `wscripts.exe` 执行修改注册列表，添加一些功能
        -   添加 WSL 中 *Vim*、*Emacs* 等到右键菜单
        -   添加 *在 WSL 中打开文件夹* 到右键菜单
    -   `run-wsl-file.exe` 可以用于在快捷执行 WSL 脚本，只需要将其选择为文件打开方式
    -   `vim.exe` 可以用 WSL 中 *Vim* 打开任何文件
        -   一般是配合 `tools/` 中脚本在右键注册后使用
    -   配置文件
        -   配置文件`etc/wsl-terminal.conf`
        -   主题文件`etc/themes/`
        -   *mintty*配置文件`etc/mintty`

> - <https://zhuanlan.zhihu.com/p/22033219>
> - <https://www.zhihu.com/question/36344262/answer/67191917>、
> - <https://www.zhihu.com/question/38752831>

#### 其他参考

-   子系统可以替换为其他非官方支持发行版，如 [archlinux](https://wiki.archlinux.org/index.php/Install_on_WSL_(简体中文))
-   WSL 可以可以通过 *X Server* 执行 GUI 应用程序 <https://news.ycombinator.com/item?id=13603451>
-   WSL 官博
    -   <https://blogs.msdn.microsoft.com/wsl/>
    -   <https://blogs.msdn.microsoft.com/commandline/tag/wsl/>

