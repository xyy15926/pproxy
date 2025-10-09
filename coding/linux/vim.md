---
title: *VIM* 功能
categories:
  - Linux
tags:
  - Linux
  - Tool
  - Vi
  - Diff
date: 2025-09-25 15:21:59
updated: 2025-10-09 09:34:59
toc: true
mathjax: true
description: 
---

#   *Vim* 启动、配置

##  *Vim* 启动

| *VIM* 编辑不同类型文件     | 说明                                 |
|----------------------------|--------------------------------------|
| `vim [options]`            | 启动 *VIM* 并开启一个空白缓冲区      |
| `vim [options] {file} ..`  | 启动并编辑一个或多个文件             |
| `vim [options] -`          | 从标准输入读入文件                   |
| `vim [options] -t {tag}`   | 编辑与标签 `{tag}` 关联的文件        |
| `vim [options] -q [fname]` | 以快速修复模式开始编辑并显示首个错误 |

-   *VIM* 启动
    -   *VIM* 开启编辑时，以下 5 种参数仅必选其一以 **确定待编辑文件类型**
        -   `<FILE>`：编辑已有文件，读入缓冲区，首个文件作为当前文件
        -   `-`：效果取决于是否使用 *Ex* 模式
            -   `$ vim -`、`$ ex -v -`：普通模式，从标准输入读取文本
            -   `$ ex -`、`$ vim -e -`、`$ evim -`、`$ vim -E`：*Ex* 模式，安静模式开始编辑
        -   `-t <TAG>`：从标签文件中查找 `TAG`，以相关文件作为当前文件，并执行相关命令
            -   通常用于 *C* 程序，查找、定位 `TAG`
        -   `-q [ERROR_FILE]`：快速修复模式，读入错误文件并显示第一个错误
            -   `ERROR_FILE` 缺省为 `error_file` 选项值
        -   空白：编辑新的空白、无名称缓冲区

> - *VIM starting*：<https://yianwillis.github.io/vimcdoc/doc/starting.html>
> - *VIM* 学习笔记：*Ex Mode*：<https://zhuanlan.zhihu.com/p/78778165>

###   启动模式、常见别名

| Alias        | 等价 *Vim* 启动选项 | 含义                          |
|--------------|---------------------|-------------------------------|
| `$ ex`       | `$ vim -e`          | *Ex* 模式                     |
| `$ exim`     | `$ vim -E`          | 增强的 *Ex* 模式              |
| `$ view`     | `$ vim -R`          | 只读                          |
| `$ gvim`     | `$ vim -g`          | *GUI* 启动                    |
| `$ gex`      | `$ vim -eg`         | *Ex* 模式启动 *GUI*           |
| `$ gview`    | `$ vim -Rg`         | *GUI* 启动只读                |
| `$ rvim`     | `$ vim -Z`          | 受限模式                      |
| `$ rview`    | `$ vim -RZ`         | 受限模式 `view`               |
| `$ rgvim`    | `$ vim -gZ`         | 受限模式 `gview`              |
| `$ rgview`   | `$ vim -RgZ`        | 受限模式 `gview`              |
| `$ evim`     | `$ vim -y`          | 简易 *Vim*，置位 `insertmode` |
| `$ eview`    | `$ vim -yR`         | 只读 `evim`                   |
| `$ vimdiff`  | `$ vim -d`          | 比较模式                      |
| `$ gvimdiff` | `$ vim -gd`         | *GUI* 启动比较模式            |

-   *VIM* 默认以 *visual* 模式启动，特殊模式
    -   *EVim* 简易模式：设置了 *点击-输入* 风格的 *Vim*
        -   便于新手使用键鼠操作
        -   任何按键均将进入 `insertmode`，即总是可以直接编辑
            -   `CTRL-L` 可以退出此模式
            -   或 `CTRL-O` 进入暂时 *Ex* 模式，再 `:set noinsertmode`
    -   *Ex* 模式：以 *Ex* 命令进行编辑操作
        -   仅可使用 *Ex* 命令（即 *visual* 模式下 `:` 引导的命令）进行编辑
        -   缓冲区渲染结果不再随输入更新
        -   可在 *visual* 模式下以 `Q`、`gQ` 命令进入
            -   以 `vi(sual)` 命令退出

###    启动选项

| *VIM* 启动选项      | 说明                                          | *Alias*     |                 |
|---------------------|-----------------------------------------------|-------------|-----------------|
| `-g`                | 启动 `GUI`（同时允许其他选项）                | `$ gvim`    |                 |
| `+[num]`            | 置光标于第 `[num]` 行（缺省 末行）            |             |                 |
| `+{command}`        | 载入文件后执行命令 `{command}`                |             |                 |
| `+/{pat} {file} ..` | 置光标于首次出现 `{pat}` 处                   |             |                 |
| `-v`                | *Vi*  模式，以普通模式启动 `ex`               |             | `vi`            |
| `-e`                | *Ex*  模式，以 *Ex* 模式启动 *VIM*            | `$ ex`      | `ex`            |
| `-E`                | 增强的 *Ex*  模式                             | `$ exim`    |                 |
| `-R`                | 只读模式，隐含 `-n`                           | `$ view`    | `Read-only`     |
| `-m`                | 禁止修改（复位 '`write`' 选项）               |             | `modifications` |
| `-d`                | 比较模式  `diff`                              | `$ vimdiff` | `diff`          |
| `-b`                | 二进制模式                                    |             | `binary`        |
| `-y`                | 简易模式模式（置位 `insertmode`）             | `$ evim`    |                 |
| `-l`                | *lisp*  模式                                  |             | `lisp`          |
| `-A`                | 阿拉伯语模式（置位 `arabic`）                 |             | `Arabic`        |
| `-F`                | 波斯语模式（置位 `fkmap` 和 '`rightleft`'）   |             | `Farsi`         |
| `-H`                | 希伯来语模式（置位 `hkmap` 和 '`rightleft`'） |             | `Hebrew`        |
| `-V`                | 详细，给出更多信息                            |             | `Verbose`       |
| `-C`                | 兼容，置位 `compatible` 选项                  |             | `Compatible`    |
| `-N`                | 不兼容，复位 `compatible` 选项                |             | `Nocompatible`  |
| `-S {file} ..`      | 从会话文件恢复                                |             | `session`       |
| `-r`                | 列出交换文件                                  |             |                 |
| `-r {file} ..`      | 恢复中断的编辑（从交换文件恢复）              |             | `recover`       |
| `-n`                | 不创建交换文件                                |             |                 |
| `-o [num]`          | 打开 `[num]` 个窗口（缺省 每个文件一个窗口）  |             | `open`          |
| `-f`                | *GUI* 时作为前台进程，不调用 `fork`           |             | `foreground`    |
| `-s {scriptin}`     | 先从文件 `{scriptin}` 读入命令                |             | `script`        |
| `-c {cmd}`          | 载入首个文件后执行命令 `{cmd}`                |             | `cmd`           |
| `-cmd {cmd}`        | 载入所以文件前执行命令 `{cmd}`                |             | `cmd`           |
| `-w {scriptout}`    | 把键入的字符写进文件 `{scriptout}`（添加）    |             | `write`         |
| `-W {scriptout}`    | 把键入的字符写进文件 `{scriptout}`（覆盖）    |             | `Write`         |
| `-T {terminal}`     | 设置终端名                                    |             | `Terminal`      |
| `-u {vimrc}`        | 从文件 `{vimrc}` 而非其它文件读入初始化命令   |             | `user`          |
| `-U {gvimrc}`       | 同上，但用于启动 `GUI` 时                     |             | `User`          |
| `-i {viminfo}`      | 从文件 `{viminfo}` 而不是其它文件读入信息     |             | `info`          |
| `--`                | 结束选项，其余的参数都将是文件名              |             |                 |
| `--help`            | 显示参数列表并退出                            |             |                 |
| `--version`         | 显示版本信息并退出                            |             |                 |
| `-`                 | 从标准输入读入文件                            |             |                 |

### *Vim* 初始化

####    初始化流程

-   *Vim* 初始化顺序（非 *GUI* 版本）
    -   设置 `shell`、`term` 选项
        -   `shell`：优先使用环境变量 `$SHELL` 设置（*Win32* 上使用 `COMSPEC`）
        -   `term`：优先使用环境变量 `$TERM` 设置
    -   处理参数：检查命令行上给出的选项、文件名
        -   为所有文件创建缓冲区（还未载入文件）
        -   `-V`：可显示、记录执行情况，方便调试
    -   从初始化文件、环境变量执行 *Ex* 命令
        -   载入 `$VIMRUNTIME/evim.vim`：仅 *Vim* 以 `evim`、`eview` 模式启动
        -   载入系统 `vimrc` 文件：`$VIM/vimrc`
            -   总是按 `compatible` 模式载入（选项还未设置）
        -   载入用户 `vimrc` 文件、*vim* 环境变量：仅首个被找到的文件、环境被执行
            -   `$VIMINIT` 环境变量
                -   多个命令用 `|`、`<NL>` 分隔
            -   用户 *vimrc* 文件：`$ vim --version` 查看详细优先级
                -   *Unix*：`$HOME/.vimrc`、`$HOME/.vim/vimrc`
            -   `$EXINIT` 环境变量
            -   用户 `exrc` 文件
            -   默认 `vimrc` 文件：`$VIMRUNTIME/defaults.vim`
        -   搜索当前目录 `vimrc` 文件：仅在 `exrc` 选项被设置时
            -   `.vimrc`、`_vimrc`
            -   `.exrc`、`_exrc`
        -   说明事项
            -   可用 `-u` 指定初始化文件，此时以上 3 步初始化均被跳过，另外
                -   `-u NORC`：跳过此初始化
                -   `-u NONE`：跳过插件载入
            -   `vimrc` 文件：包含初始化命令，每行作为 *Ex* 命令执行
            -   `vimrc` 是 *Vim* 专用名称，`exrc` 被 *Vi* 使用
    -   载入插件脚本：相当于执行命令 `:runtime! plugin/**/*.vim`
        -   按以下顺序载入插件脚本
            -   插件脚本：`runtimepath` 选项值中除 `after` 结尾目录下 `plugin/**/*.vim` 文件
                -   按 `runtimepath` 中顺序依次搜索，但跳过以 `after` 结尾的目录
                -   递归搜索、载入全部 `.vim` 脚本
            -   插件包：`packpath` 选项值中各目录下 `pack/*/start/*/plugin/**/*.vim` 文件
                -   事实上，在上步载入插件脚本前会先查找、记录插件包中插件：`packpath` 各目录下 `pack/*/start/*`
                -   插件脚本执行完毕后，`pack/*/start/*`、`pack/*/start/*/after` 添加进 `runtimepath`
            -   `after` 插件：新 `runtimepath` 中 `after` 结尾目录下 `plugin/**/*.vim` 文件
                -   以上 2 步后，插件包中 `after` 被加入 `runtimepath`，同一逻辑中被载入
                -   事实上，软件包 `after` 位置更靠前，较先执行
        -   以下情况将不载入插件脚本
            -   `loadplugins` 选项复位
                -   命令行 `-c 'set noloadplugins'` 不生效，此时命令行命令未执行
                -   命令行 `--cmd 'set noloadplugins'` 生效
            -   `--noplugins` 命令行参数
            -   `--clean` 命令行参数
            -   `-u NONE` 命令行参数
            -   *Vim* 无 `+eval` 特性
    -   启动前命令行参数、选项设置处理
        -   设置选项 `shellpipe`、`shellredir`
            -   根据 `shell` 选项值设置，除非已设置
        -   命令行参数 `-n`：设置 `updatecount` 选项为 0
        -   命令行参数 `-b`：置位 `binary` 选项
        -   *GUI* 初始化
        -   `viminfo` 选项非空：读入 *viminfo* 文件
        -   命令行参数 `-q`：读入快速修复文件，失败则 *Vim* 退出
    -   打开窗口
        -   命令行标志 `-o`：打开所有窗口
        -   命令行标志 `-p`：打开所有标签页
        -   切换屏幕，启动
        -   命令行标志 `-q`：跳到首个错误
        -   载入所有窗口的缓冲区，不触发 `BufAdd` 自动命令
    -   执行启动命令
        -   命令行标志 `-t`：跳转至标签处
        -   命令行 `-c` 、`+cmd`：执行给出命令
        -   `insertmode` 选项置位：进入插入模式
        -   复位启动标志位，`has("vim_starting")` 返回 0
        -   `v:vim_did_enter` 变量设为 1
        -   执行 `VimEnter` 自动命令

####    流程说明

-   *VIM* 启动说明
    -   `-s` 参数将跳过 *载入插件脚本* 及之前的 4 步初始化动作
        -   但 `-u` 指定初始化文件仍生效
    -   *Vi* 兼容性问题
        -   *Vi* 启动时， `compatible` 选项默认置位，初始化时使用该设置
            -   `vimrc` 总是以 `compatible` 模式读入，选项复位在其后发生
        -   以下情况下存在会复位 `compatible` 选项
            -   找到用户 `vimrc` 文件、`$VIMINIT` 环境变量、当前目录下 `vimrc` 文件、`gvimrc` 文件
            -   `-N`、`--clean` 命令行参数
            -   载入 `defaults.vim` 脚本
    -   默认用户 `vimrc`：`$VIMRUNTIME/defaults.vim`
        -   置位 `skip_defaults_vim` 变量可避免执行
        -   若确定需要执行，最好 `unlet! skip_defaults_vim` 后再 `source`

## *Vim Runtime*

### *Vim Runtime* 目录结构

-   `$VIMRUNTIME` 目录结构：仅包含涉及初始化关键目录、文件
    -   `vimrc`：系统初始化文件，其中
        -   `:syntax on`
        -   `:filetype on`
    -   `syntax/`：语法高亮目录
        -   `syntax/syntax.vim`：其中
            -   `:runtime syntax/synload.vim`
        -   `syntax/synload.vim`：其中
            -   `:runtime syntax/<FILETYPE>.vim`
            -   `:runtime syntax/syncolor.vim`
        -   `syntax/syncolor.vim`：
    -   `filetype.vim`、`ftoff.vim`：文件类型探测，其中
        -   `:runtime ftdetect/*.vim`
        -   `:runtime scripts.vim`
    -   `indent.vim`、`indoff.vim`：文件类型缩进，其中
        -   `:runtime indent/<FILETYPE>.vim`
    -   `ftplugin.vim`、`ftplugof.vim`：文件类型插件，其中
        -   `:runtime ftplugin/<FILETYPE>.vim`
    -   `autoload/`：自动载入脚本
        -   `:call <FILENAME>#<FUNC>` 自动载入 `autoload/FILENAME`
    -   `colors/`：颜色配置脚本
        -   `:colorscheme <FILENAME>` 载入 `colors/<FILENAME>` 中配色
    -   `plugin/`：插件脚本目录，初始化时自动载入
    -   `default.vim`：缺省用户初始化文件

-   初始 `runtimepath` 中 *runtime* 目录结构
    -   `vimrc`：用户初始化温婉
    -   `plugin/`：插件脚本目录，初始化时自动载入
    -   `autoload/`：自动载入脚本
    -   `colors/`：颜色配置脚本
    -   `pack/`：插件包目录
    -   被命令触发
        -   `filetype.vim`、`ftoff.vim`
        -   `ftplugin.vim`、`ftplugof.vim`
        -   `indent.vim`、`indoff.vim`
    -   `autocmd` 关联文件类型触发
        -   `syntax/*.vim`
        -   `ftplugin/*.vim`
        -   `indent/*.vim`

> - *Runtime* 目录结构以 *Vim82* 为基准

###    *Runtime* 相关命令

-   相关命令：仅涉及脚本触发方式部分
    -   `syntax [on|off|enable|manual]`：语法设置
        -   `on`/`enable`：打开语法，载入 `$VIMRUNTIME/syntax/syntax.vim`
        -   `off`：关闭语法，载入 `$VIMRUNTIME/syntax/nosyntax.vim`
        -   `manual`：手动语法，载入 `$VIMRUNTIME/syntax/manual.vim`
        -   `clear`：清除当前缓冲区语法配置
    -   `filetype [indent] [plugin] [on|off]`：文件类型探测、缩进、插件设置
        -   `indent`：文件类型缩进
            -   开启时：载入 `runtimepath/indent.vim`
            -   关闭时：载入 `runtimepath/indentoff.vim`
        -   `plugin`：文件类型插件
            -   开启时：载入 `runtimepath/ftplugin.vim`
            -   关闭时：载入 `runtimepath/ftplugof.vim`
        -   `on`：开启，可认为总是省略开启探测
            -   即包含 `on` 时总是会开启文件探测
            -   开启文件探测：载入 `runtimepath/filetype.vim`
        -   `off`：关闭
            -   仅在 `filetype off` 才关闭文件探测
            -   关闭文件探测：载入 `runtimepath/ftoff.vim`

### *Runtime* 目录

####    `after/` 延迟处理

-   （以）`after/`（结尾）：延迟处理脚本目录
    -   目录结构应类似普通 *runtimepath* 目录
    -   需位于 `runtimepath` 才生效
        -   *Vim* 启动时，延迟处理其中插件 `plugin`
        -   *Vim* 启动后，和其他目录角色、功能一致
    -   但其中不能包含 `pack/`，会崩溃

####    `pack/` 插件包

-   `pack/`：存放插件包目录，应该位于 `packpath` 中目录下
    -   插件包 *bundle*/*package*：包含一个、多个插件的目录
        -   统一管理，多个插件间可包含依赖关系
        -   避免和其他插件的文件混杂
        -   方便更新
    -   插件包 `pack/<PACKAGE_1>` **内** 目录结构
        -   `start/`：启动时在 `runtimepath/plugin` 加载后自动加载的插件
            -   `<PLUGIN_1>/`
            -   `<PLUGIN_2>/`
        -   `opt/`：启动后 `:packadd <PLUGIN_NAME>` 手动加载的插件
            -   `<PLUGIN_EXTRA_1>/`
            -   `<PLUGIN_EXTRA_2>/`

-   `pack/<PACKAGE_1>/start/<PLUGIN_1>`：插件目录
    -   目录结构应类似普通 *runtimepath* 目录
    -   *Vim* 启动时
        -   类似 `runtimepath` 其他目录，其中 `plugin/**/*.vim` 被载入
        -   目录自身、其中 `after/` 目录被添加进 `runtimepath`
    -   *Vim* 启动后，和 `runtimepath` 中其他目录角色、功能一致

-   `pack/<PACKAGE_1>/opt/<PLUGIN_1>` 类似
    -   但不在 *Vim* 启动时载入，需手动 `:packadd` 载入插件
    -   插件目录、`/after` 加入 `runtimepath` 首、尾
    -   事实上，插件中 `ftdetect/*.vim` 会在 `plugin/**/*.vim` 后被载入，而其中 `filetype.vim` 不被载入

### `$VIM`、`$VIMRUNTIME` 环境变量

-   `$VIM`、`$VIMRUNTIME` 环境变量
    -   `$VIM` 用于定位 *Vim* 使用的用户文件，按照如下顺序取值
        -   `$VIM` 环境变量
        -   `helpfile` 选项值（若其中不包含其他环境变量）确定的目录
        -   *Unix*：编译时定义的安装目录
        -   *Win32*：可执行文件的目录名确定的目录
    -   `$VIMRUNTIME` 用于定位支持文件，如：帮助文档、语法高亮文件，按如下顺序取值
        -   `$VIMRUNTIME` 环境变量
        -   `$VIM/vim{VERSION}`
        -   `$VIM/runtime`
        -   `$VIM`
        -   `helpfile` 选项值确定的目录
    -   说明
        -   可通过 `:let $VIM=`、`:let $VIMRUNTIME=` 修改二者取值

##  *VIM* 持久化

### 视图

| 命令                 | 说明                   |
|----------------------|------------------------|
| `:mkvie[w]! [FILE]`  | 将窗口配置写入视图文件 |
| `:lo[adview] [FILE]` | 从文件中恢复视图       |

-   视图：应用于一个窗口的设置的集合
    -   可保存窗口配置并在之后恢复，具体内容由 `viewoptions` 指定
        -   窗口的配置选项、映射
        -   窗口使用参数列表
        -   窗口编辑的文件
        -   光标位置

| `set` 选项   | 说明                           |
|--------------|--------------------------------|
| `viewdir`    | 视图文件目录，缺省 `$VIM/view` |
| `viewoption` | 视图存储的内容（列表）         |

| `viewoptions` 取值 | 说明                     |
|--------------------|--------------------------|
| `options`          | 全局映射、缩写、全局选项 |
| `localoptions`     | 局部映射、缩写、选项     |
| `folds`            | 折叠                     |
| `curdir`           | 当前目录                 |


> - *VIM* 视图和会话：<https://yianwillis.github.io/vimcdoc/doc/starting.html#:views-session>

### 会话

| `sessionoptions` 取值 | 说明                                               |
|-----------------------|----------------------------------------------------|
| `options`             | 恢复全局映射、选项（局部选项的全局值）             |
| `globals`             | 恢复大写字母开始、至少包含一个小写字母的全局变量   |
| `curdir`              | 恢复当前目录                                       |
| `sesdir`              | 设置当前目录为会话文件所在位置                     |
| `winpos`              | 恢复 *GUI* 窗口位置                                |
| `resize`              | 恢复屏幕大小                                       |
| `buffers`             | 恢复所有缓冲区，包括隐藏、未载入，否则仅打开缓冲区 |
| `help`                | 恢复帮助窗口                                       |
| `blank`               | 恢复编辑无名缓冲区的窗口                           |
| `winsize`             | 若无任何窗口舍弃，则恢复窗口大小                   |
| `tabpages`            | 包含所有标签页                                     |

| 命令                   | 说明                                         |
|------------------------|----------------------------------------------|
| `:mks[ession]! [FILE]` | 将当前会话写入会话文件（缺省 `Session.vim`） |

-   会话：所有窗口的视图、全局设置
    -   可以会话并在之后恢复，保存内容由 `sessionoptions` 选项决定
        -   建立多个会话以在不同项目间快速切换
    -   恢复会话
        -   已启动 *VIM*，执行 `:source <SESSION_FILE>` 即可恢复
        -   直接 `$ vim -S <SESSION_FILE>` 恢复会话
        -   与会话文件同名，但以 `x.vim` 结尾文件同样被执行
            -   可用于给指定会话的附加设置、动作
        -   恢复会话后，`v:this_session` 变量将保存会话的完整文件名

| 自动化监听事件    | 说明                 |
|-------------------|----------------------|
| `SessionLoadPost` | 会话文件载入、执行后 |

> - 须编译时启用 `+mksession` 特性
> - *VIM* 视图和会话：<https://yianwillis.github.io/vimcdoc/doc/starting.html#:views-session>

### *viminfo*

-   *viminfo* 文件：记住所有视图、会话都使用的信息，允许继续上次退出的编辑
    -   保存内容
        -   命令行历史
        -   搜索字符串历史
        -   输入行历史
        -   非空寄存器历史
        -   多个文件的位置标记
        -   文件标记：指向文件位置
        -   最近搜索、替换模式
        -   缓冲区列表
        -   全局变量
    -   *viminfo* 文件不依赖工作内容
        -   通常只有一个 *viminfo* 文件
    -   *viminfo* 文件默认被设置为不能被其他用户读取，避免泄露可能包含文本、命令
        -   每次 `vim` 替换其时会保留被用户主动更改的权限
        -   *Vim* 不会覆盖当前实际用户不能写入的 *viminfo* 文件，避免 `$ su root` 生成无法读取的文件
        -   *viminfo* 文件不能是符号链接

> - 须编译时启用 `+viminfo` 特性
> - *VIM* 视图和会话：<https://yianwillis.github.io/vimcdoc/doc/starting.html#:viminfo>
> - *VIM option viminfo*：<https://yianwillis.github.io/vimcdoc/doc/options.html#'viminfo'>

####    相关选项、命令

| `set` 选项    | 说明                         |
|---------------|------------------------------|
| `viminfofile` | *viminfo* 文件名             |
| `viminfo`     | *viminfo* 保存内容、数量限制 |

| `viminfo` 取值项 | 说明                                         |
|------------------|----------------------------------------------|
| `!`              | 仅保存大写字母开头全局变量                   |
| `"`              | 设置寄存器保存最大行数                       |
| `%`              | 保存缓冲区列表                               |
| `'`              | 设置记住位置标记的文件数目                   |
| `/`              | 设置记住搜索历史数目                         |
| `:`              | 设置记住命令行历史数目                       |
| `@`              | 设置记住输入行历史数目                       |
| `c`              | 将 *viminfo* 中文本转换为当前 `encoding`     |
| `f`              | 不存在、非零则保存文件位置标记               |
| `h`              | 关闭 `hlsearch`                              |
| `n`              | *viminfo* 文件名                             |
| `r`              | 不保存文件名前缀匹配的文件的位置标记，可多次 |
| `s`              | 保存寄存器内容 `KB` 限制                     |

-   *viminfo* 文件处理逻辑
    -   若启动时 `viminfo` 选项非空，*viminfo* 文件内容被读入
        -   其中信息在适当地方被应用
            -   `v:oldfiles` 变量被填充值
                -   有位置标记的文件名
            -   启动时不读入位置标记
        -   *viminfo* 文件可以手动修改
        -   若读入 *viminfo* 文件时检查到错误
            -   之后不会覆盖该文件
            -   超过 10 个错误则停止
    -   若退出时 `viminfo` 选项非空，相关信息保存在 *viminfo* 文件中
        -   保存内容、数量限制由 `viminfo` 选项定义
            -   位置标记：可为每个文件存储位置标记
                -   位置标记只在退出 *Vim* 时保存
                -   若须 `:bdel` 清除缓冲区，可以手动使用 `:wv` 保存位置标记
                    -   `[`、`]` 不被保存
                    -   `"` 、`A-Z` 被保存
                    -   `0` 会被设置为当前光标，并依次覆盖之后数字位置标记
        -   多数选项：保存当前会话中被改变的值，未变动的值则从原 *viminfo* 文件中填充
        -   部分选项：使用时间戳保留最近改动版本，总是保留最新项目
            -   命令行历史
            -   搜索字符串历史
            -   输入行历史
            -   非空寄存器内容
            -   跳转表
            -   文件标记

| 命令                     | 说明                                              |
|--------------------------|---------------------------------------------------|
| `:rv[iminfo] [file]`     | 从文件 `[file]` 读入 *viminfo* 信息               |
| `:rv[iminfo]! [file]`    | 同上，但覆盖已存在的信息                          |
| `:wv[iminfo] [file]`     | 向文件 `[file]` 中加入 *viminfo* 信息             |
| `:wv[iminfo]! [file]`    | 向文件 `[file]` 中写入 *viminfo* 信息             |
| `:ol[files]`             | 列出 *viminfo* 文件中有存储位置标记的文件列表     |
| `:bro[wse] ol[files][!]` | 类似 `:ol` 列出文件名，之后可输入编号编辑指定文件 |

### 帮助

-   `:help [THEME]`：获得特定主题的帮助，缺省显示总览帮助窗口
    -   `THEME` 帮助主题可以是命令、功能、命令行参数、选项
        -   包含控制字符命令：控制字符前缀 `:help CTRL-A`
        -   特殊按键：尖括号 `:help i_<Up>`
        -   不同模式下命令：模式前缀 `:help i_CTRL-A`
            -   普通模式：无前缀
            -   可视模式：`v_`
            -   插入模式：`i_`
            -   命令行编辑、参数：`c_`
            -   *Ex* 命令：`:`
            -   用于调试的命令：`>`
            -   正则表达式：`/`
        -   命令行参数：横杠 `:help -t`
        -   选项：引号括起 `:help 'number'`
    -   `:helpgrep [THEME]`：在所有帮助页面中搜索（包括已安装插件）

### 交换文件

-   交换文件处理
    -   确认需要恢复：直接恢复 `R`
    -   确认丢弃：直接删除 `D`
    -   没有明确目标：只读打开 `O`
    -   比较交换文件、现在文件差别
        -   恢复打开 `R`
        -   另存为其他文件 `:saveas filname.bak`
        -   与当前文件对比 `:diffsplit filename`
    -   一般不会直接 `E`（*Edit Anyway*）
#TODO

#   *VIM* 操作功能

```python
import re


def generate_markdown_table(lines: str):
    cts = []
    for ll in lines.split("\n"):
        al, bl, cl, dl = [0] * 4
        if len(ll.strip()) == 0:
            continue
        mt = re.match(r" *(\S+) +(N?) +([^\u4e00-\u9fa5]+) +([\u4e00-\u9fa5].+)", ll)
        if mt is not None:
            a, b, c, d = mt.groups()
            c = c.strip()
            d = d.replace(" (", "（").replace(") ", "）").replace(":", "").replace("\"", "")
            d = re.sub(r"([^\u4e00-\u9fa5（）： 、，]+)", r"`\1`", d)
            al = max(len(a), al)
            bl = max(len(b), bl)
            cl = max(len(c), cl)
            dl = max(len(d), dl)
            cts.append((a, b, c, d))
    sts = []
    for a, b, c, d in cts:
        st = f"| {a.rjust(al)} | {b.rjust(bl)} | {c.rjust(cl)} | {d.rjust(dl)} |"
        sts.append(st)

    return "\n".join(sts)
```

##  动作

-   统一说明
    -   *VIM* 分为普通命令、*Ex* 命令两种
        -   普通命令：各模式下直接按键生效命令
        -   *Ex* 命令：`:` 引导、在命令行输入、回车后执行的命令
    -   `N` 普通命令重复执行次数

> - *VIM quikref 快速参考*：<https://yianwillis.github.io/vimcdoc/doc/quickref.html>

### 动作：左右


| `N` | 命令      | 说明                                                        |
|-----|-----------|-------------------------------------------------------------|
| `N` | `h`       | 左（亦 `CTRL-H`、`<BS>` 或 `<Left>` 键)                     |
| `N` | `l`       | 右（亦 `<Space>` 或 `<Right>` 键)                           |
|     | `0`       | 至本行首个字符（亦 `<Home>` 键)                             |
|     | `^`       | 至本行首个非空白字符                                        |
| `N` | `$`       | 至本行（加上 `N` - `1` 个后续行）末个字符（亦 `<End>` 键)   |
|     | `g0`      | 至屏幕行首个字符（行回绕时不同于 "`0`")                     |
|     | `g^`      | 至屏幕行首个非空白字符（行回绕时不同于 "^")                 |
| `N` | `g$`      | 至屏幕行末个字符（行回绕时不同于 "$")                       |
|     | `gm`      | 至屏幕行中点                                                |
|     | `gM`      | 至本行中点                                                  |
| `N` | `|`       | 至第 `N` 列（缺省 `1)`                                      |
| `N` | `f{char}` | 至右边第 `N` 次出现 `{char}` 之处（`find`）                 |
| `N` | `F{char}` | 至左边第 `N` 次出现 `{char}` 之处（`Find`）                 |
| `N` | `t{char}` | 至右边第 `N` 次出现 `{char}` 之前（`till`）                 |
| `N` | `T{char}` | 至左边第 `N` 次出现 `{char}` 之前（`Till`）                 |
| `N` | `;`       | 重复前次 "`f`"、"`F`"、"`t`" 或 "`T`" 命令 `N` 次           |
| `N` | `,`       | 以相反方向重复前次 "`f`"、"`F`"、"`t`" 或 "`T`" 命令 `N` 次 |

### 动作：上下

| `N` | 命令          | 说明                                                     |
|-----|---------------|----------------------------------------------------------|
| `N` | `k`           | 上移 `N` 行（亦 `CTRL-P` 和 `<Up>`）                     |
| `N` | `j`           | 下移 `N` 行（亦 `CTRL-J`、`CTRL-N`、`<NL>` 和 `<Down>`） |
| `N` | `-`           | 上移 `N` 行，至首个非空白字符处                          |
| `N` | `+`           | 下移 `N` 行，至首个非空白字符处（亦 `CTRL-M` 和 `<CR>`） |
| `N` | `_`           | 下移 `N-1` 行，至首个非空白字符处                        |
| `N` | `G`           | 至第 `N` 行（缺省 末行） 首个非空白字符处                |
| `N` | `gg`          | 至第 `N` 行（缺省 首行） 首个非空白字符处                |
| `N` | `%`           | 至全文件行数百分之 `N` 处；必须给出 `N`，否则是 `%` 命令 |
| `N` | `gk`          | 上移 `N` 屏幕行（回绕行时不同于 "`k`"）                  |
| `N` | `gj`          | 下移 `N` 屏幕行（回绕行时不同于 "`j`"）                  |
|     | `CTRL-<HOME>` | 光标移至文件首                                           |
|     | `CTRL-<END>`  | 光标移至文件末                                           |

### 动作：文本对象

| `N` | 命令 | 说明                                                      |
|-----|------|-----------------------------------------------------------|
| `N` | `w`  | 向前（正向，下同） `N` 个单词（`word`）                   |
| `N` | `W`  | 向前 `N` 个空白隔开的字串 `WORD`（`WORD`）                |
| `N` | `e`  | 向前至第 `N` 个单词词尾（`end`）                          |
| `N` | `E`  | 向前至第 `N` 个空白隔开的字串`WORD` 的词尾（`End`）       |
| `N` | `b`  | 向后（反向，下同） `N` 个单词（`backward`）               |
| `N` | `B`  | 向后至第 `N` 个空白隔开的字串 `WORD` 的词尾（`Backward`） |
| `N` | `ge` | 向后至第 `N` 个单词词尾                                   |
| `N` | `gE` | 向后至第 `N` 个空白隔开的字串`WORD`的词尾                 |
| `N` | `)`  | 向前 `N` 个句子                                           |
| `N` | `(`  | 向后 `N` 个句子                                           |
| `N` | `}`  | 向前 `N` 个段落                                           |
| `N` | `{`  | 向后 `N` 个段落                                           |
| `N` | `]]` | 向前 `N` 个小节，置于小节的开始                           |
| `N` | `[[` | 向后 `N` 个小节，置于小节的开始                           |
| `N` | `][` | 向前 `N` 个小节，置于小节的末尾                           |
| `N` | `[]` | 向后 `N` 个小节，置于小节的末尾                           |
| `N` | `[(` | 向后至第 `N` 个未闭合的 '('                               |
| `N` | `[{` | 向后至第 `N` 个未闭合的 '{'                               |
| `N` | `[m` | 向后至第 `N` 个方法（`method`） 的开始（用于 `Java`）     |
| `N` | `[M` | 向后至第 `N` 个方法的结束（`Method`） （用于 `Java`）     |
| `N` | `])` | 向前至第 `N` 个未闭合的 '）'                              |
| `N` | `]}` | 向前至第 `N` 个未闭合的 '}'                               |
| `N` | `]m` | 向前至第 `N` 个方法（`method`） 的开始（用于 `Java`）     |
| `N` | `]M` | 向前至第 `N` 个方法的结束（`Method`） （用于 `Java`）     |
| `N` | `[#` | 向后至第 `N` 个未闭合的 "#`if`" 或 "#`else`"              |
| `N` | `]#` | 向前至第 `N` 个未闭合的 "#`else`" 或 "#`endif`"           |
| `N` | `[*` | 向后至第 `N` 个注释的开始 "/*"                            |
| `N` | `]*` | 向前至第 `N` 个注释的结束 "*/"                            |

### 动作：模式查找


| `N` | 命令    | 说明                                           |
|-----|---------|------------------------------------------------|
| `N` | `/<CR>` | 向前重复前次查找                               |
| `N` | `?<CR>` | 向后重复前次查找                               |
| `N` | `n`     | 重复前次查找                                   |
| `N` | `N`     | 相反方向重复前次查找                           |
| `N` | `*`     | 向前查找光标下的标识符                         |
| `N` | `#`     | 向后查找光标下的标识符                         |
| `N` | `g*`    | 同 "*"，但也查找部分匹配                       |
| `N` | `g#`    | 同 "#"，但也查找部分匹配                       |
|     | `gd`    | 至光标下标识符的局部声明（`goto declaration`） |
|     | `gD`    | 至光标下标识符的全局声明（`goto Declaration`） |

####  *Pattern*

| 模式            | 选项      | 表达式开关 | 元字符处理                                         |
|-----------------|-----------|------------|----------------------------------------------------|
| *Magic*（默认） | `magic`   | `\m`       | 除 `$`、`.`、`*`、`^` 外其他元字符需添加反斜杠转义 |
| *NoMagic*       | `nomagic` | `\M`       | 除 `$`、`^` 外其他元字符需添加反斜杠转义           |
| *Very Magic*    |           | `\v`       | 所有元字符均无需反斜杠 `\` 转义                    |
| *Very NoMagic*  |           | `\V`       | 所有元字符均需反斜杠 `\` 转义                      |

-   *VIM* 有 4 种不同解析正则表达式的模式
    -   不同模式下对正则表达式元字符解析逻辑不同
        -   将正则表达式中全部元字符全部引入可能导致查找功能过于复杂
        -   简单搜索 `(){}[]` 之类符号时操作繁琐，影响编辑器查找功能定位
    -   正则表达式解析模式可在配置文件中配置、表达式前添加引导符切换
        -   *Magic*、*NoMagic* 模式可在配置文件中配置默认逻辑
        -   *Very Magic*、*Very NoMagic* 模式可在表达式前添加引导开关 `\v`、`\V` 触发

> - 正则表达式：<https://vim80.readthedocs.io/zh/latest/basic/regular.expression.html>
> - 神级编辑器 *VIM* 使用-正则替换篇：<https://zhuanlan.zhihu.com/p/346058975>
> - *Pattern*：<https://yianwillis.github.io/vimcdoc/doc/pattern.html>

### 动作：位置标记

| 命令                  | 说明                                                  |
|-----------------------|-------------------------------------------------------|
| `m{a-zA-Z}`           | 用标记 `{a-zA-Z}` 记录当前位置                        |
| <code>`{a-z}</code>   | 至当前文件中的标记 `{a-z}`                            |
| <code>`{A-Z}</code>   | 至任何文件中的标记 `{A-Z}`                            |
| <code>`{0-9}</code>   | 至 *VIM* 前次退出的位置                               |
| <code>``</code>       | 至前次跳转之前的位置                                  |
| <code>`"</code>       | 至前次编辑此文件的位置                                |
| <code>`[</code>       | 至前次被操作或放置的文本的开始                        |
| <code>`]</code>       | 至前次被操作或放置的文本的结尾                        |
| <code>`<</code>       | 至（前次） 可视区域的开始                             |
| <code>`></code>       | 至（前次） 可视区域的结尾                             |
| <code>`.</code>       | 至当前文件最后被改动的位置                            |
| `'{a-zA-Z0-9[]'"<>.}` | 同 <code>`</code>，但同时移动至该行的首个非空白字符上 |

### 动作：其他

| `N` | 命令                   | 说明                                        |          |
|-----|------------------------|---------------------------------------------|----------|
|     | `%`                    | 找到本行中后一个括号、方括号、注释或        |          |
| `N` | `H`                    | 至窗口的第 `N` 行的首个非空白字符处         | `Home`   |
|     | `M`                    | 至窗口的中间行的首个非空白字符处            | `Middle` |
| `N` | `L`                    | 至窗口的从下方算第 `N` 行的首个非空白字符处 | `Last`   |
| `N` | `go`                   | 至缓冲区的第 `N` 个字节处                   |          |
|     | `:[range]go[to] [off]` | 至缓冲区的第 `[off]` 个字节处               |          |

### 屏幕滚动


| `N` | 命令              | 说明                                   |             |
|-----|-------------------|----------------------------------------|-------------|
| `N` | `CTRL-E`          | 窗口下滚 `N` 行（缺省 `1`）            | `Extra`     |
| `i` | `i_CTRL-X CTRL-E` | 窗口下滚 `N` 行（缺省 `1`）            |             |
| `N` | `CTRL-D`          | 窗口下滚 `N` 行（缺省 `1/2` 窗口）     | `Downwards` |
| `N` | `CTRL-F`          | 窗口下滚 `N` 页                        | `Forwards`  |
| `N` | `CTRL-Y`          | 窗口上滚 `N` 行（缺省 `1`）            |             |
| `i` | `i_CTRL-X CTRL-Y` | 窗口上滚 `N` 行（缺省 `1`）            |             |
| `N` | `CTRL-U`          | 窗口上滚 `N` 行（缺省 `1/2` 窗口）     | `Upwards`   |
| `N` | `CTRL-B`          | 窗口上滚 `N` 页                        | `Backwards` |
|     | `z<CR>`、`zt`     | 重画，当前行置于窗口顶端               |             |
|     | `z.`、`zz`        | 重画，当前行置于窗口正中               |             |
|     | `z-`、`zb`        | 重画，当前行置于窗口底端               |             |
| `N` | `zh`              | 屏幕右滚 `N` 个字符（`wrap` 选项复位） |             |
| `N` | `zl`              | 屏幕左滚 `N` 个字符（`wrap` 选项复位） |             |
| `N` | `zH`              | 屏幕右滚半个屏宽（`wrap` 选项复位）    |             |
| `N` | `zL`              | 屏幕左滚半个屏宽（`wrap` 选项复位）    |             |

##  插入

### 插入：插入文本

| `N` | 命令                  | 说明                                            |
|-----|-----------------------|-------------------------------------------------|
| `N` | `a`                   | 在光标后添加文本（`N` 次）                      |
| `N` | `A`                   | 在行末添加文本（`N` 次）                        |
| `N` | `i`                   | 在光标前插入文本（`N` 次）（亦 `<Insert>`）     |
| `N` | `I`                   | 在当前行首个非空白字符前插入文本（`N` 次）      |
| `N` | `gI`                  | 在第一栏中插入文本（`N` 次）                    |
|     | `gi`                  | 最近一次插入模式停止处继续插入文本（`N` 次）    |
| `N` | `o`                   | 在当前行下方打开新行，添加文本（`N` 次）        |
| `N` | `O`                   | 在当前行上方打开新行，添加文本（`N` 次）        |
|     | `:star[tinsert][!]`   | 开始插入模式 `i`，当使用 `[!]` 时添加文本       |
|     | `:startr[eplace][!]`  | 开始替换模式 `R`，当使用 `[!]` 时从行末开始     |
|     | `:startg[replace][!]` | 启动虚拟替换 `gR`                               |
|     | `:[RANGE]a[ppend][!]` | 指定行下方添加行，持续输入直至输入仅包含 `.` 行 |
|     | `:[RANGE]i[nsert][!]` | 指定行上方添加行，持续输入直至输入仅包含 `.` 行 |
|     | `:stopi[nsert]`       | 尽快停止插入模式，类似 `<Esc>`                  |

-   替换模式说明
    -   替换模式 `R`：输入的每个字符会删除行内字符，直至在末尾附加输入的字符
        -   若输入 `<NL>`，则插入换行符，不删除任何字符
        -   `<Tab>` 是单个字符，但占据多个位置，可能影响窗口展示
        -   用 `<BS>`、`CTRL-w`、`CTRL-u` 删除字符，实际上是删除修改，被替换的字符将复原
    -   虚拟替换模式 `gR`：类似替换模式，但按屏幕位置替换，保持窗口中字符不移动
        -   `<Tab>` 可能会替换多个字符，在 `<Tab>` 上替换可能等同于插入
        -   `<NL>` 会替换光标至行尾
        -   用 `<BS>`、`CTRL-w`、`CTRL-u` 删除字符，实际上是删除修改，被替换的字符将复原
        -   适合用于编辑 `<Tab>` 分隔表格列的场合
-   *Ex* 命令插入文本说明
    -   `:a`、`:i` 等后不跟文本，回车后继续输入待插入文本
        -   持续输入直至输入仅包含 `.` 行
        -   通用参数
            -   `RANGE`：插入文本范围
            -   `!`：切换 `autoindent` 选项

| *VIM* 选项      | 说明                                                 |
|-----------------|------------------------------------------------------|
| `backspace`     | 指定退格功能（可删除项目 `indent`、`eol`、`start`）  |
| `[no]linebreak` | 文本自动断行（回绕不断行）                           |
| `textwidth`     | 断行长度                                             |
| `formatoptions` | 限制断行时机                                         |
| `formatexpr`    | 断行处理函数                                         |
| `wrapmargin`    | 根据屏幕宽度动态断行，即等价于 `columns - textwidth` |
| `fileformat`    | 当前缓冲区换行符风格（可在读取、写入时设置不同风格） |
| `fileformats`   | 打开文件时尝试的换行符风格                           |

-   回车、换行说明
    -   `fileformat` 可指定（三种平台）文本格式的 *EOL* 标识
        -   `<NL><CR>`：*MS-DOS* `dos`
        -   `<NL>`：*Unix* `unix`
        -   `<CR>`：*Macintosh* （*OSX* 前系统）`mac`
    -   *Vim* 对 `<NL>`、`<CR>` 处理
        -   文本格式下用于换行的控制字符被转换为 `<Nul>`（*ASCII* 码值 0）插入缓冲区
            -   `ff=mac` 无法插入 `<CR>`/`^M`
            -   `ff=unix`、`ff=dos`  无法插入 `<NL>`/`^J`
        -   `unix`、`dos` 与 `mac` 格式互转时，文本结构不发生变化
            -   可认为存在 `<EOL>` 标记，在缓冲区切换格式时，在 `<NL><CR>`、`<NL>`、`<CR>` 之间转换
            -   多余的、显示出的 `<CR>` 与 `<NL>` 互相转化
            -   可利用此特性在文本中插入 `<NL>`、`<CR>`

> - *VIM insert*：<https://yianwillis.github.io/vimcdoc/doc/insert.html>
> - *VIM insert*：<https://vimdoc.sourceforge.net/htmldoc/insert.html>
> - `<NL>` *New Line* 即 `<LF>` *Line Feed*，即 *ASCII* 码值 14 的字符

####    `Visaul Block` 模式插入

| `N`  | 命令   | 说明                             |
|------|--------|----------------------------------|
| `vb` | `vb_I` | 在所有选中的行之前插入相同的文本 |
| `vb` | `vb_A` | 在所有选中的行之后添加相同的文本 |

### `Insert` 模式插入

#### `Insert` 模式：移动

| `N` | 命令                                    | 说明                                |
|-----|-----------------------------------------|-------------------------------------|
| `i` | `i_<Esc>`、`CTRL-[`                     | 结束插入模式，回到普通模式          |
| `i` | `i_CTRL-C`                              | 同 `<Esc>`，但不检查缩写            |
| `i` | `i_CTRL-O {command}`                    | 执行命令 `{command}` 并回到插入模式 |
| `i` | `i_<Up>`、`<Down>`、`<Left>`、`<Right>` | 上、下、左、右移动光标              |
| `i` | `i_shift-left/right`                    | 左、右移动一个单词                  |
| `i` | `i_shift-up/down`                       | 后、前移动一个满屏                  |
| `i` | `i_<End>`                               | 光标至本行的末字符之后              |
| `i` | `i_<Home>`                              | 光标至本行的首字符之上              |

####    `Insert` 模式：特殊输入

| `N` | 命令                                 | 说明                                   |
|-----|--------------------------------------|----------------------------------------|
| `i` | `i_CTRL-V {char}..`                  | 按本义插入字符，或插入十进制数的字节值 |
| `i` | `i_<NL>`、`<CR>`、`CTRL-M`、`CTRL-J` | 开始新行                               |
| `i` | `i_CTRL-E`                           | 插入光标下方的字符                     |
| `i` | `i_CTRL-Y`                           | 插入光标上方的字符                     |
| `i` | `i_CTRL-A`                           | 插入前次插入的文本                     |
| `i` | `i_CTRL-@`                           | 插入前次插入的文本并结束插入模式       |
| `i` | `i_CTRL-R {register}`                | 插入寄存器的内容                       |
| `i` | `i_CTRL-R CTRL-R {register}`         | 插入寄存器的内容，保持本义             |
| `i` | `i_CTRL-R CTRL-0 {register}`         | 插入寄存器的内容，无自动缩进           |
| `i` | `i_CTRL-R CTRL-P {register}`         | 插入寄存器的内容，修复缩进             |
| `i` | `i_<BS>`、`CTRL-H`                   | 删除光标前的一个字符                   |
| `i` | `i_<Del>`                            | 删除光标下的一个字符                   |
| `i` | `i_CTRL-W`                           | 删除光标前的一个单词                   |
| `i` | `i_CTRL-U`                           | 删除当前行中所有的输入字符             |
| `i` | `i_CTRL-T`                           | 在当前行首插入一个 `shiftwidth` 的缩进 |
| `i` | `i_CTRL-I`                           | 在当前位置插入一个 `shiftwidth` 的缩进 |
| `i` | `i_CTRL-D`                           | 从当前行首删除一个 `shiftwidth` 的缩进 |
| `i` | `i_0 CTRL-D`                         | 删除当前行的所有缩进                   |
| `i` | `i_^ CTRL-D`                         | 删除当前行的所有缩进，恢复下一行的缩进 |
| `i` | `i_CTRL-X ...`                       | 以各种方式补全光标前的单词             |
| `i` | `i_CTRL-N`                           | 从 `complete` 指定源正向补全单词       |
| `i` | `i_CTRL-P`                           | 从 `complete` 指定源反向补全单词       |

> - 注意：`i_CTRL-N`、`i_CTRL-X CTRL-N` 补全逻辑不同，前者是 `complete` 补全，后者是缓冲区内关键字补全
> - 保持本义：指控制字符将直接插入其 *ASCII* 字符本身，而不会被 *VIM* 解读、并控制输入内容

####    `Insert` 模式：`CTRL-X` 补全

| `N` | 不同补全            | 说明                                                            |
|-----|---------------------|-----------------------------------------------------------------|
| `i` | `i_CTRL-X CTRL-L`   | 反向搜索补全整行，忽略缩进                                      |
| `i` | `i_CTRL-X CTRL-N`   | 缓冲区内正向搜索补全单词                                        |
| `i` | `i_CTRL-X CTRL-P`   | 缓冲区内反向搜索补全单词                                        |
| `i` | `i_CTRL-X CTRL-K`   | 根据 `dictionary` 选项补全单词                                  |
| `i` | `i_CTRL-X CTRL-T`   | 根据 `thesaurus` 选项（同义词）补全单词                         |
| `i` | `i_CTRL-X CTRL-I`   | 在当前文件、头文件中补全关键字                                  |
| `i` | `i_CTRL-X CTRL-]`   | 在当前文件、头文件中补全标签                                    |
| `i` | `i_CTRL-X CTRL-F`   | 补全文件名                                                      |
| `i` | `i_CTRL-X CTRL-D`   | 补全定义、宏                                                    |
| `i` | `i_CTRL-X CTRL-V`   | 补全 *Vim* 命令，包括 *Ex* 命令的参数                           |
| `i` | `i_CTRL-X CTRL-Q`   | 同 `CTRL-x CTRL-v`                                              |
| `i` | `i_CTRL-X CTRL-U`   | 通过 `completefunc` 选项自定义函数补全                          |
| `i` | `i_CTRL-X CTRL-O`   | 通过 `ominifunc` 选项自定义函数补全，通常用于特定文件类型的补全 |
| `i` | `i_CTRL-X [CTRL-]S` | 单词拼写补全                                                    |
| `i` | `i_CTRL-X CTRL-R`   | 寄存器内匹配                                                    |

-   `CTRL-X` 可以进入 *插入补全子模式*：可给出命令补全单词、滚动窗口菜单
    -   `CTRL-X` 按下后，*VIM* 命令行回提示支持的插入补全子模式
    -   菜单可能处于 3 个状态：一般处于 1、3 状态，即完整匹配、增删后重新匹配
        -   完整匹配插入：匹配项完整插入，如在 `CTRL-N`、`CTRL-P` 切换匹配项后
        -   仅匹配不插入：此时不插入仅高亮，仅在 `<Up>`、`<Down>` 移动光标后
        -   部分匹配插入：部分匹配文本，且输入字符、或退格，此时匹配项列表根据光标前内容调整，如在匹配初始、完全匹配后增删后
    -   全部 3 个状态下，可以使用按键
        -   `CTRL-Y`：是，接受当前匹配项并停止补全
        -   `CTRL-E`：结束补全，回到匹配前原有内容
        -   `<PageUp>`、`<PageDown>`：反向、正向若干项后选择匹配项，不插入
        -   `<Up>`、`<Down>`：选择前、后个匹配项，同 `CTRL-p`、`CTRL-n`，不插入
        -   `<Space>`、`<Tab>`：停止补全，不改变匹配，插入键入字符
        -   `<Enter>`：状态 1、3 时插入现有文本、换行符；状态 2 时插入选择项
        -   `CTRL-N`：下个候选项
        -   `CTRL-P`：上个候选项
        -   `CTRL-R`：插入寄存器内容（主要是为允许通过 `=` 寄存器调用函数决定下个操作）|
    -   状态 1 下
        -   `<BS>`、`CTRL-H`：删除字符，重新查找匹配项，会减少匹配项数目
        -   其他非特殊字符：停止补全不改变匹配，插入输入字符
    -   状态 2、3 下
        -   `<BS>`、`CTRL-H`：删除字符，重新查找匹配项，会增加匹配项数目
        -   `<CTRL-L>`：从当前匹配项中增加字符，减少匹配项数量
        -   任何可显示的空白字符：插入字符，减少匹配项数量

| *VIM* 选项     | 说明                               |
|----------------|------------------------------------|
| `infercase`    | 调整匹配的大小写                   |
| `complete`     | 决定匹配搜索的缓冲区               |
| `include`      | 指定如何找到含有头文件名字的行     |
| `define`       | 包含定义的行                       |
| `path`         | 指定搜索头文件的位置               |
| `isfname`      | 文件名可包含的字符                 |
| `completefunc` | `CTRL-x CTRL-u` 用户补全自定义函数 |
| `ominifunc`    | `CTRL-x CTRL-o` 全能补全函数       |
| `compeletopt`  | 补全选项                           |
| `pumheight`    | 菜单最大高度，缺省整个有效空间     |
| `pumwidth`     | 菜单最小宽度，缺省 15 字符         |

> - *VIM insert ins-completion*：<https://yianwillis.github.io/vimcdoc/doc/insert.html#ins-completion>

#### 插入：二合字母

| 命令                                      | 说明                               |
|-------------------------------------------|------------------------------------|
| `:dig[raphs]`                             | 显示当前二合字母列表               |
| `:dig[raphs] {char1}{char2} {number} ...` | 加入一或多个二合字母到二合字母列表 |
| `CTRL-K {char1} {char2}`                  | 键入二合字母                       |
| `{char1} <BS> {char2}`                    | 置位 `digraph` 时，键入二合字母    |

### 插入：特殊插入

| 命令            | 说明                                         |
|-----------------|----------------------------------------------|
| `:r [file]`     | 将文件 `[file]` 的内容插入到光标之下         |
| `:r! {command}` | 将命令 `{command}`  的标准输出插入到光标之下 |

-   插入文件命令
    -   `:[RANGE]r[ead] [++<OPT>] [!<CMD>] [NAME]`：光标下插入内容
        -   `NAME`：文件包含内容
        -   `CMD`：执行 `CMD` 的标准输出
        -   `RANGE`：添加新行位置，缺省为当前行
        -   `OPT`：指定 `fileformat`、`fileencoding`、`binary`、坏字符处理方案
            -   `++fileformat`/`++ff`
            -   `++encoding`/`++enc`
            -   `++binary`/`++bin`、`++nobinary`/`++nobin`
            -   `++bad=<X>|keep|drop`：用 `X` 替换、维持、删除

| 相关选项     | 说明                                |
|--------------|-------------------------------------|
| `shellredir` | Shell 重定向选项，缺省为 `>%s &2>1` |

##  改动

### 改动：删除文本

| `N` | 命令            | 说明                                        |
|-----|-----------------|---------------------------------------------|
| `N` | `x`             | 删除光标之下及之后的 `N` 个字符             |
| `N` | `<Del>`         | 删除光标之下及之后的 `N` 个字符             |
| `N` | `X`             | 删除光标之前的 `N` 个字符                   |
| `N` | `d{motion}`     | 删除动作 `{motion}` 跨越的文本              |
|     | `{visual}d`     | 删除高亮的文本                              |
| `N` | `dd`            | 删除 `N` 行                                 |
| `N` | `D`             | 删除至行尾（及 `N-1` 后续行）               |
| `N` | `J`             | 连接 `N-1` 行（删除 `<EOL>`）               |
|     | `{visual}J`     | 连接高亮的行                                |
| `N` | `gJ`            | 同 `J`，但不插入空格                        |
|     | `{visual}gJ`    | 同 `{visual}J`，但不插入空格                |
|     | `:[range]d [x]` | 删除范围 `[range]` 覆盖的行，存入寄存器 `x` |

### 改动：复制与移动

| `N` | 命令         | 说明                                                |
|-----|--------------|-----------------------------------------------------|
|     | `"{char}`    | 在接下来的删除、抽出或放置命令中使用寄存器 `{char}` |
|     | `:reg`       | 显示所有寄存器的内容                                |
|     | `:reg {arg}` | 显示寄存器 `{arg}` 的内容                           |
| `N` | `y{motion}`  | 抽出动作 `{motion}` 跨越的文本至寄存器              |
|     | `{visual}y`  | 抽出高亮的文本至寄存器                              |
| `N` | `yy`         | 抽出 `N` 行至寄存器                                 |
| `N` | `Y`          | 抽出 `N` 行至寄存器                                 |
| `N` | `p`          | 在光标位置后放置寄存器的内容（`N` 次）              |
| `N` | `P`          | 在光标位置前放置寄存器的内容（`N` 次）              |
| `N` | `]p`         | 同 `p`，但调整当前行的缩进                          |
| `N` | `[p`         | 同 `P`，但调整当前行的缩进                          |
| `N` | `gp`         | 同 `p`，但将光标留在新文本之后                      |
| `N` | `gP`         | 同 `P`，但将光标留在新文本之后                      |

### 改动：修改文本

| `N` | 命令                       | 说明                                                 |
|-----|----------------------------|------------------------------------------------------|
| `N` | `r{char}`                  | 以 `{char}` 替换 `N` 个字符                          |
| `N` | `gr{char}`                 | 替换 `N` 个字符，但不影响布局                        |
| `N` | `R`                        | 进入替换模式（重复键入的文本 `N` 次）                |
| `N` | `gR`                       | 进入虚拟替换模式 同替换模式，但不影响布局            |
| `N` | `c{motion}`                | 修改动作 `{motion}` 跨越的文本                       |
|     | `{visual}c`                | 修改高亮的文本                                       |
| `N` | `cc`                       | 修改 `N` 行                                          |
| `N` | `S`                        | 修改 `N` 行                                          |
| `N` | `C`                        | 修改至行尾（及 `N-1` 后续行）                        |
| `N` | `s`                        | 修改 `N` 个字符                                      |
|     | `{visual}c`                | 在可视列块模式下 用键入的文本修改选中文本的每一行    |
|     | `{visual}C`                | 在可视列块模式下 用键入的文本修改选中各行直至行末    |
| `N` | `~`                        | 反转 `N` 个字符的大小写并前进光标                    |
|     | `{visual}~`                | 反转高亮文本的大小写                                 |
|     | `{visual}u`                | 改高亮的文本为小写                                   |
|     | `{visual}U`                | 改高亮的文本为大写                                   |
|     | `g~{motion}`               | 反转动作 `{motion}` 跨越的文本的大小写               |
|     | `gu{motion}`               | 改动作 `{motion}` 跨越的文本为小写                   |
|     | `gU{motion}`               | 改动作 `{motion}` 跨越的文本为大写                   |
|     | `{visual}g?`               | 用 `rot13` 编码高亮的文本                            |
|     | `g?{motion}`               | 用 `rot13` 编码动作 `{motion}` 跨越的文本            |
| `N` | `CTRL-A`                   | 将光标之上或之后的数值增加 `N`                       |
| `N` | `CTRL-X`                   | 将光标之上或之后的数值减少 `N`                       |
| `N` | `<{motion}`                | 左移动作 `{motion}` 跨越的多行一个 `shiftwidth`      |
| `N` | `<<`                       | 左移 `N` 行一个 `shiftwidth`                         |
| `N` | `>{motion}`                | 右移动作 `{motion}` 跨越的多行一个 `shiftwidth`      |
| `N` | `>>`                       | 右移 `N` 行一个 `shiftwidth`                         |
| `N` | `gq{motion}`               | 排版动作 `{motion}` 跨越的多行到 '`textwidth`' 宽    |
|     | `:[range]ce[nter] [width]` | 将范围 `[range]` 中的多行居中对齐                    |
|     | `:[range]le[ft] [indent]`  | 将范围 `[range]` 中的多行靠左对齐（使用 `[indent]`） |
|     | `:[range]ri[ght] [width]`  | 将范围 `[range]` 中的多行靠右对齐                    |

-   说明
    -   修改：删除文本并进入插入模式

### 改动：复杂改动、过滤、替换

| `N` | 命令                                             | 说明                                                        |
|-----|--------------------------------------------------|-------------------------------------------------------------|
| `N` | `!{motion}{command}<CR>`                         | 用命令 `{command}` 过滤动作跨越的多行                       |
| `N` | `!!{command}<CR>`                                | 用命令 `{command}` 过滤 `N` 行                              |
|     | `{visual}!{command}<CR>`                         | 用命令 `{command}` 过滤高亮的行                             |
|     | `:[range]! {command}<CR>`                        | 用命令 `{command}` 过滤范围 `[range]` 覆盖的多行            |
| `N` | `={motion}`                                      | 用 `equalprg` 过滤动作跨越的多行                            |
| `N` | `==`                                             | 用 `equalprg` 过滤 `N` 行                                   |
|     | `{visual}=`                                      | 用 `equalprg` 过滤高亮的多行                                |
|     | `:[range]s[ubstitute]/{pattern}/{string}/[g][c]` | 用 `{string}` 替代范围 `[range]` 覆盖的多行中的 `{pattern}` |
|     | `:[range]s[ubstitute] [g][c]`                    | 以新的范围和选项重复前次的 `s` 命令                         |
|     | `&`                                              | 不带选项在当前行上重复前次的 `s` 命令                       |
|     | `:[range]ret[ab][!] [tabstop]`                   | 置 `tabstop` 为新值并依据此值调整空白字符                 |

-   说明
    -   `:substitute` 选项说明  
        -   `g`：替换各行中全部匹配，否则替换各行首个匹配
        -   `c`：替换前确认
    -   `:[RANGE]ret[ab][!] [NEW_TABSTOP]`：将空白序列替换为 `NEW_TABSTOP` 确定的空白序列
        -   根据 `expandtab` 选项决定替换为 `<Tab>`、空格
        -   `NEW_TABSTOP`：缺省、0 则使用 `tabstop` 选项值
        -   已有 `<Tab>` 用 当前 `tabstop` 选项值确定
        -   `!`：把包含正常空格字符串替换为 `<Tab>`

| *VIM* 选项    | 说明                                                                    |
|---------------|-------------------------------------------------------------------------|
| `tabstop`     | 制表符真实占位（影响换算、展示）                                        |
| `softtabstop` | （非 0 时）`<Tab>` 插入、`<BS>` 删除的位置数量                          |
| `expandtab`   | 空格填充制表符位置                                                      |
| `smarttab`    | `<Tab>` 在行首插入 `shiftwidth` 个位置，在其他地方插入 `tabstop` 个位置 |
| `shiftwidth`  | 缩进位置数量，影响 `>>` 等命令                                          |

##  对象操作

### 可视对象

| `N` | 命令         | 说明                                 |
|-----|--------------|--------------------------------------|
|     | `v`          | 以字符方式开始高亮                   |
|     | `V`          | 以行方式开始高亮                     |
|     | `CTRL-V`     | 以列块方式开始高亮                   |
| `v` | `v_o`        | 交换高亮区域的开始处和光标位置       |
|     | `gv`         | 在前次可视区域上开始高亮             |
| `v` | `v_v`        | 以字符方式开始高亮或停止高亮         |
| `v` | `v_V`        | 以行方式开始高亮或停止高亮           |
| `v` | `v_CTRL-V`   | 以列块方式开始高亮或停止高亮         |
| `v` | `v_g CTRL-g` | 显示可视区域单词、字符、行、字节计数 |

-   说明
    -   光标移动：移动光标或使用操作符来作用于高亮的文本

### 文本对象

| `N` | 命令            | 说明                                        |              |
|-----|-----------------|---------------------------------------------|--------------|
| `N` | `aw`            | 选择 “一个单词”                             | `a word`     |
| `N` | `iw`            | 选择 “内含单词”                             | `inner word` |
| `N` | `aW`            | 选择 “一个字串”                             | `WORD`       |
| `N` | `iW`            | 选择 “内含字串”                             |              |
| `N` | `as`            | 选择 “一个句子”                             | `sentence`   |
| `N` | `is`            | 选择 “内含句子”                             |              |
| `N` | `ap`            | 选择 “一个段落”                             | `paragraph`  |
| `N` | `ip`            | 选择 “内含段落”                             |              |
| `N` | `ab`            | 选择 “一个块”（从 "[(" 至 "])"）            | `block`      |
| `N` | `ib`            | 选择 “内含块”（从 "[(" 到 "]）"）           |              |
| `N` | `aB`            | 选择 “一个大块”（从 "[{" 到 "]}"）          | `Block`      |
| `N` | `iB`            | 选择 “内含大块”（从 "[{" 到 "]}"）          |              |
| `N` | `a>`            | 选择 “一个 <> 块”                           |              |
| `N` | `i>`            | 选择 “内含 <> 块”                           |              |
| `N` | `at`            | 选择 “一个标签块”（从 `<aaa>` 到 `</aaa>`） | `tag`        |
| `N` | `it`            | 选择 “内含标签块”（从 `<aaa>` 到 `</aaa>`） |              |
| `N` | `a'`            | 选择 “一个单引号字符串”                     |              |
| `N` | `i'`            | 选择 “内含单引号字符串”                     |              |
| `N` | `a"`            | 选择 “一个双引号字符串"                     |              |
| `N` | `i"`            | 选择 “内含双引号字符串"                     |              |
| `N` | <code>a`</code> | 选择 “一个反引号字符串”                     |              |
| `N` | <code>i`</code> | 选择 “内含反引号字符串”                     |              |

### 重复命令

| `N` | 命令                                | 说明                                                                         |
|-----|-------------------------------------|------------------------------------------------------------------------------|
| `N` | `.`                                 | 重复最近一次改动（但计数改为 `N`）                                           |
|     | `q{a-z}`                            | 记录键入的字符，存入寄存器 `{a-z}`                                           |
|     | `q{A-Z}`                            | 记录键入的字符，附加至寄存器 `{a-z}`                                         |
|     | `q`                                 | 终止记录                                                                     |
| `N` | `@{a-z}`                            | 执行寄存器 `{a-z}` 的内容（`N` 次）                                          |
| `N` | `@@`                                | 重复前次的 `@{a-z}` 操作（`N` 次）                                           |
|     | `:@{a-z}`                           | 将寄存器 `{a-z}` 的内容当作 `Ex` 命令来执行                                  |
|     | `:@@`                               | 重复前次的 `@{a-z}` 操作                                                     |
|     | `:[range]g[lobal]/{pattern}/[cmd]`  | 对 `[range]` 内所有匹配 `{pattern}` 的行执行 *Ex* 命令 `[cmd]`（缺省 `p`）   |
|     | `:[range]g[lobal]!/{pattern}/[cmd]` | 对 `[range]` 内所有不匹配 `{pattern}` 的行执行 *Ex* 命令 `[cmd]`（缺省 `p`） |
|     | `:so[urce] {file}`                  | 从文件 `{file}` 读入 *Ex* 命令                                               |
|     | `:so[urce]! {file}`                 | 从文件 `{file}` 读入 *VIM* 命令                                              |
|     | `:sl[eep] [sec]`                    | 在 `[sec]` 秒钟内不做任何事                                                  |
| `N` | `gs`                                | 睡 `N` 秒（`goto sleep`）                                                    |

##  其他操作

### 撤销/重做

| `N` | 命令             | 说明                                              |
|-----|------------------|---------------------------------------------------|
| `N` | `u`              | 撤销最近的 `N` 此改动                             |
| `N` | `CTRL-R`         | 重做最近的 `N` 个被撤销的改动                     |
|     | `U`              | 恢复最近被改动的行                                |

### 外部命令

| 命令             | 说明                                              |
|------------------|---------------------------------------------------|
| `:sh[ell]`       | 开启 Shell                                        |
| `:!{command}`    | 通过 Shell 执行命令 `{command}`                   |
| `K`              | 用 '`keywordprg`' 程序（缺省 "`man`"） 查光标下的 |

### 其他命令

#TODO

| 命令                       | 说明                                           |
|----------------------------|------------------------------------------------|
| `CTRL-L`                   | 清除并重画屏幕                                 |
| `CTRL-G`                   | 显示当前文件名（包括路径） 和光标位置          |
| `ga`                       | 以十进制、十六进制和八进制显示光标所在字符的   |
| `g8`                       | 用于 *utf-8* 编码 以十六进制显示光标所在字符的 |
| `g CTRL-G`                 | 显示光标所在的列、行、以及字符位置             |
| `CTRL-C`                   | 在查找中 中断查找                              |
| `CTRL-Break`               | MS-Windows 中断查找                            |
| `<Del>`                    | 输入计数时 删除最近输入字符                    |
| `:ve[rsion]`               | 显示版本信息                                   |
| `:mode N`                  | 置屏幕模式为 `N`（已废弃）                     |
| `:norm[al][!] {commands}`  | 执行普通命令                                   |
| `Q`                        | 切换至 `Ex` 模式                               |
| `:redir >{file}`           | 重定向消息至文件 `{file}`                      |
| `:silent[!] {command}`     | 安静地执行 `{command}` 命令                    |
| `:confirm {command}`       | 退出、写入等有未保存的改动或文件只读时         |
| `:browse {command}`        | 使用文件选择对话框打开/读入/写入文件           |

####    对话框

| *Ex* 命令、普通命令 | 说明                                        |
|---------------------|---------------------------------------------|
| `:conf[irm] {CMD}`  | 执行 `CMD`，若存在待确认事项，显示对话框    |
| `:bro[wse] {CMD}`   | 为（支持浏览的） `CMD` 的参数显示选择对话框 |

#TODO
-   对话框命令
    -   `:conf[irm] {CMD}`：执行 `CMD`，若存在待确认事项，显示对话框
        -   用于 `:q`、`:qa`、`:w` 及其他会以类似方式失败的命令，如：`:only`、`:buffer`、`bdelete`
    -   `:bro[wse] {CMD}`：为（支持浏览的） `CMD` 的参数显示选择对话框
        -   文件浏览：用于 `:e`、`:w`、`:wall`、`:mkexrc`、`:mkvimrc`、`:split`、`:cgetfile` 等命令
        -   可用 `g:browsefilter`、`b:browsefilter` 变量过滤选项：`<TAG>\t<PTN>;<PTN>\n`
            -   `TAG`：*File of Type* 组合框中文字
            -   `PTN`：过滤文件名的模式，多个模式用 `;` 分隔
            -   多个选项直接可直接合并
        -   `:browse set`：类似 `:options`

> - `:browse` 浏览文件需要 `+browse` 特征

#   *VIM* 配置类功能

##  配置

###    配置文件

| 命令                        | 说明                                                                       |
|-----------------------------|----------------------------------------------------------------------------|
| `:mk[exrc][!] [file]`       | 将当前的键盘映射、缩写及设置写入文件 `[file]`（缺省 `.exrc`，`!` 覆盖文件）|
| `:mkv[imrc][!] [file]`      | 同 `mkexrc`，但缺省为 `.vimrc`                                             |
| `:mks[ession][!] [file]`    | 同 `mkvimrc`，但同时存储当前文件、窗口等信息，使得用户将来可以继续当前对话 |

-   说明
    -   `:mk`、`:mkv` 会将 `:set`、`:map`、`:abbr` 命令配置写入文件
        -   部分和终端、文件有关的配置不被保存
        -   只有保存全局映射，局部于缓冲区的映射被忽略

### 键盘映射

| Mode         | Norm | Ins | Cmd | Vis | Sel | Opr | Term | Lang |
|--------------|------|-----|-----|-----|-----|-----|------|------|
| `[nore]map`  | yes  | -   | -   | yes | yes | yes | -    | -    |
| `n[nore]map` | yes  | -   | -   | -   | -   | -   | -    | -    |
| `[nore]map!` | -    | yes | yes | -   | -   | -   | -    | -    |
| `i[nore]map` | -    | yes | -   | -   | -   | -   | -    | -    |
| `c[nore]map` | -    | -   | yes | -   | -   | -   | -    | -    |
| `v[nore]map` | -    | -   | -   | yes | yes | -   | -    | -    |
| `x[nore]map` | -    | -   | -   | yes | -   | -   | -    | -    |
| `s[nore]map` | -    | -   | -   | -   | yes | -   | -    | -    |
| `o[nore]map` | -    | -   | -   | -   | -   | yes | -    | -    |
| `t[nore]map` | -    | -   | -   | -   | -   | -   | yes  | -    |
| `l[nore]map` | -    | yes | yes | -   | -   | -   | -    | yes  |

-   键盘映射：`<MODE>map <MARK> {LHS} {RHS}` 为不同模式配置键盘映射
    -   `<MODE>` 映射的工作模式可区分 6 种
        -   *Normal* 模式：输入命令时
        -   *Visual* 模式：可视区域高亮并输入命令时
        -   *Select* 模式：类似可视模式，但键入的字符对选择区替换
        -   *Operator-pending* 模式：操作符等待中
        -   *Insert* 模式：包括替换模式
        -   *Command-line* 模式：输入 `:`、`/` 命令时
    -   说明
        -   `<MODE>noremap[!]`：非递归映射，即不会在其他映射中再次被展开
        -   `<MODE>unmap[!]`：取消映射
        -   `<MODE>mapclear[!]`：清除 `MODE` 下所有键位映射
        -   映射后不能跟注释，Vim 会认为整行都是命令

> - <https://yianwillis.github.io/vimcdoc/doc/map.html>

#### 特殊参数

-   可通过配置 `<MARK>` 特殊参数调整映射命令生效逻辑
    -   特殊参数
        -   `<buffer>`：映射将局限于当前缓冲区
            -   优先级比全局映射高
            -   清除映射时同样需要添加参数
            -   可使用 `<leader>` 替代 `<localleader>` 可工作，但是不推荐
        -   `<nowait>`：存在较短映射时，失效以其作为前缀的较长映射
        -   `<silent>`：映射不在命令行上回显
        -   `<special>`：特殊键可以使用`<>`记法
        -   `<script>`：映射只使用通过以`<SID>`开头来定义的脚本局部映射来重映射优右值中的字符
        -   `<unique>`：若存在相同命令、缩写则定义失败
            -   定义局部映射时，同样会检查全局映射
        -   `<expr>`：映射的右值将被作为表达式被计算
    -   特殊参数说明
        -   特殊参数的尖括号 `<>` 是本身具有的，必须紧跟命令后面
        -   有些特殊参数在取消映射时同样需注明

#### *leaders*、*localleader*

-   *leader*、*localleader*：作为“前缀”的不常用的按键，后接其他字符作为整体映射
    -   用途
        -   避免覆盖太多按键原始功能 
        -   约定俗成的规范，容易理解
        -   方便更改 `<leader>`、`<localleader>` 作为前缀设置
            -   `<leader>`：对全局映射而设置的映射的前缀
            -   `<localleader>`：只对某类（个）文件而设置的映射的前缀
        -   `<leader>` 和 `<localleader>` 除了设置不同以外，没有太大区别，应用场合时约定规范，不是强制性的
    -   `<leader>`、`<localleader>` 设置
        ```vimscripts
        :let mapleader = "-"
        :nnoremap <leader>d dd
        :let maplocalleader = "\\"
        :nnoremap <buffer> <localleader>c I#<esc>
        ```
        -   *Vim* 会对 `mapleader`、`maplocalleader` 进行特殊的处理，不是简单的声明

####    `omap`

| 按键  | 操作          | 移动         |
|-------|---------------|--------------|
| `dw`  | 删除 *delete* | 到下一个单词 |
| `ci(` | 修改 *change* | 在括号内     |
| `yt,` | 复制          | 到逗号前     |

#TODO
-   `omap` 操作符操作映射：配合范围、移动命令处理指定范围的内容
    -   应用方法：`operator （操作命令） + operator-pending （移动、范围选择）`
    -   预定义的 *operator-pending* 映射如 `w`、`aw`、`i(`、`t,`
    -   自定义的 *operator-pending* 映射则需要
        -   选取一定范围：可同时指定开头、结尾（一般通过进入 *visual* 模式下选择范围）
            ```vimscripts
            " 下个括号内内容
            onoremap in( :<c-u>normal! f(vi(<cr>
            " 当前括号内容
            onoremap il( :<c-u>normal! f)vi(<cr>`
            " 选取使用 `===` 标记 markdown 标题
            onoremap ih :<c-u>execute "normal! ?^==\\+$\r:nohlsearch\rkvg_"<cr>
            onoremap ah :<c-u>execute "normal! ?^==\\+$\r:nohlsearch\rg_vk0"<cr>
            ```
        -   指定光标位置：光标当前位置为开头、指定位置为结尾
            ```vimscripts
            " 移动至 `return` 前一行
            onoremap b /return<cr>
            ```

### 缩写

| 命令                        | 说明                           |
|-----------------------------|--------------------------------|
| `:ab[breviate] {lhs} {rhs}` | 为 `{rhs}` 加入缩写 `{lhs}`    |
| `:ab[breviate] {lhs}`       | 显示以 `{lhs}` 开始的缩写      |
| `:ab[breviate]`             | 显示所有缩写                   |
| `:una[bbreviate] {lhs}`     | 删除 `{lhs}` 对应的缩写        |
| `:norea[bbrev] [lhs] [rhs]` | 同 `ab`，但不对 `[rhs]` 重映射 |
| `:iab/:iunab/:inoreab`      | 同 `ab`，但仅适用于插入模式    |
| `:cab/:cunab/:cnoreab`      | 同 `ab`，但仅适用于命令行模式  |
| `:abc[lear]`                | 清除所有缩写                   |
| `:cabc[lear]`               | 清除所有命令行模式缩写         |
| `:iabc[lear]`               | 清除所有插入模式缩写           |

-   缩写命令说明
    -   `iabvrev`：紧跟缩写输入非关键字后，缩写会替换为相应的完整字符串
        -   相较于映射
            -   `iabbrev` 用于 *insert*、*replace*、*command-line* 模式
            -   `iabbrev` 会注意缩写前后的字符，只在需要的时候替换
        -   `iabbrev` 同样支持特殊参数
            -   `<buffer>`：仅限本地缓冲区

### 选项设置


| 命令                       | 说明                                |
|----------------------------|-------------------------------------|
| `:se[t]`                   | 显示所有改动过的选项                |
| `:se[t] all`               | 显示所有非 `termcap` 选项           |
| `:se[t] termcap`           | 显示所有 `termcap` 选项             |
| `:se[t] {option}`          | 置位布尔选项（开启）                |
| `:se[t] no{option}`        | 复位布尔选项（关闭）                |
| `:se[t] inv{option}`       | 反转布尔选项的值                    |
| `:se[t] {option}={value}`  | 设置字符串/数值选项的值为 `{value}` |
| `:se[t] {option}+={value}` | 将 `{value}` 加到字符串、数值选项中 |
| `:se[t] {option}-={value}` | 从 `{value}` 字符串、数值选项里减去 |
| `:se[t] {option}?`         | 显示 `{option}` 的值                |
| `:se[t] {option}&`         | 重置 `{option}` 为其缺省值          |
| `:setl[ocal]`              | 同 `set`，但对局部选项设定其局部值  |
| `:setg[lobal]`             | 同 `set`，但对局部选项设定其全局值  |
| `:fix[del]`                | 根据 `t_kb` 的值来设置 `t_kD`       |
| `:opt[ions]`               | 打开一个新窗口，用来参看并设置选项  |

##  自动化

### 自动命令

| 命令                       | 说明                                                           |
|----------------------------|----------------------------------------------------------------|
| `:au`                      | 列出所有自动命令                                               |
| `:au {event}`              | 列出针对事件 `{event}` 的所有自动命令                          |
| `:au {event} {pat}`        | 列出针对事件 `{event}` 并匹配 `{pat}` 的所有自动命令           |
| `:au {event} {pat} {cmd}`  | 加入针对事件 `{event}` 及匹配 `{pat}` 的新自动命令             |
| `:au!`                     | 清除所有自动命令                                               |
| `:au! {event}`             | 清除所有针对事件 `{event}` 的自动命令                          |
| `:au! * {pat}`             | 清除所有匹配 `{pat}` 的自动命令                                |
| `:au! {event} {pat}`       | 清除所有针对事件 `{event}` 及匹配 `{pat}` 的自动命令           |
| `:au! {event} {pat} {cmd}` | 清除所有针对事件 `{event}` 及匹配 `{pat}` 的自动命令并输入新的 |

#TODO
-   自动命令实现方式
    -   *viminfo-file*：启动时读入寄存器、标记、历史记录，退出时存储这些信息
    -   *modeline*：置于文件的前面或后面数行，为文件配置 `:set` 选项
    -   *autocommand*：特定事件发生时自动执行命令

> - *VIM autocmd*：<https://yianwillis.github.io/vimcdoc/doc/autocmd.html>
> - *VIM option modeline*：<https://yianwillis.github.io/vimcdoc/doc/options.html#modeline>

### `autocmd` 事件监听自动命令

| 事件监听                 | 描述         |
|--------------------------|--------------|
| `bufnewfile <FILENAME>`  | 新建文件     |
| `bufwritepre <FILENAME>` | 写入前       |
| `bufread <FILENAME>`     | 读文件       |
| `filetype <FILETYPE>`    | 设置文件类型 |

-   `autocmd` 注意事项
    -   同时监听多个事件，使用 `,` 分隔，中间不能有空格
        -   一般同时监听 `bufnewfile`、`bufread`，这样打开文件时无论文件是否存在都会执行命令
    -   所有事件后面都需要注明适用场景，可用`*`表示全部场景，中间也不能有空格
    -   `autocmd` 是定义命令，不是执行命令
        -   每次执行都会定义命令，而*vim* 不会忽略重复定义
        -   如：`:autocmd bufwrite * :sleep 200m`，每次执行时都会重复定义命令

```vimscriptss
" 缓冲区事件
autocmd bufnewfile * :write
autocmd bufnewfile *.txt :write
autocmd bufwritepre *.html :normal gg=g
autocdm bufnewfile,bufread *.html setlocal nowrap
" *filetype* 事件（*vim* 设置缓冲区 *filetype* 时触发）
autocmd filetype javascript nnoremap <buffer> <localleader>c i//<esc>
autocmd filetype python nnoremap <buffer> <localleader>c i#<esc>
autocmd filetype javascript :iabbrev <buffer> iff if ()<left>
autocmd filetype python :iabbrev <buffer> iff if:<left>
```

### `augroup` 自动命令组

-   自动命令组：用事件监听自动命令编组，方便统一管理、清除
    -   类似 `autocmd`，*vim* 不会忽略重复定义，需可以通过 `:autocmd!` 清除一个组

```vimscripts
augroup cmdgroup
    autocmd bufwrite * :echom "foo"
    autocmd bufwrite * :echom "bar"
augroup end
augroup cmdgroup
   autocmd!                             " 清除同名自动命令组
   autocmd bufwrite * :echom "foo"
   autocmd bufwrite * :echom "bar"
augroup end
```

##  界面展示

### 语法高亮

| 命令                                        | 说明                                 |
|---------------------------------------------|--------------------------------------|
| `:syntax on`                                | 开始使用语法高亮                     |
| `:syntax off`                               | 停止使用语法高亮                     |
| `:syntax keyword {group-name} {keyword} ..` | 添加语法关键字项目                   |
| `:syntax match {group-name} {pattern} ...`  | 加入语法匹配项目                     |
| `:syntax region {group-name} {pattern} ...` | 添加语法区域项目                     |
| `:syntax sync [comment | lines {N} | ...]`  | 设置语法高亮的同步方式               |
| `:syntax [list]`                            | 列出当前语法项目                     |
| `:syntax clear`                             | 清除所有语法信息                     |
| `:highlight clear`                          | 清除所有高亮信息                     |
| `:highlight {group-name} {key}={arg} ..`    | 为语法组 `{group-name}` 设置高亮     |
| `:filetype on`                              | 开启文件类型检测，不启用语法高亮     |
| `:filetype plugin indent on`                | 开启文件类型检测，包括自动缩进及设置 |

### *GUI* 命令

| 命令                         | 说明                          |
|------------------------------|-------------------------------|
| `:gui`                       | *UNIX* 启动 *GUI*             |
| `:gui {fname} ..`            | 同上，并编辑 `{fname} ..`     |
| `:menu`                      | 列出所有菜单                  |
| `:menu {mpath}`              | 列出 `{mpath}` 下的所有菜单   |
| `:menu {mpath} {rhs}`        | 把 `{rhs}` 加入菜单 `{mpath}` |
| `:menu {pri} {mpath} {rhs}`  | 同上，并带有优先权 `{pri}`    |
| `:menu ToolBar.{name} {rhs}` | 把 `{rhs}` 加入工具栏         |
| `:tmenu {mpath} {text}`      | 为菜单 `{mpath}` 加入工具提示 |
| `:unmenu {mpath}`            | 删除菜单 `{mpath}`            |

### 折叠

| 命令           | 说明                                    |          |
|----------------|-----------------------------------------|----------|
| `zf{motion}`   | 操作符 手动定义一个折叠                 | `fold`   |
| `:{range}fold` | 将范围 `{range}` 包括的行定义为一个折叠 |          |
| `zd`           | 删除光标下的一个折叠                    | `delete` |
| `zD`           | 删除光标下的所有折叠                    | `Delete` |
| `zo`           | 打开光标下的折叠                        | `open`   |
| `zO`           | 打开光标下的所有折叠                    | `Open`   |
| `zc`           | 关闭光标下的一个折叠                    | `close`  |
| `zC`           | 关闭光标下的所有折叠                    | `Close`  |
| `zm`           | 折起更多 减少 `foldlevel`               | `more`   |
| `zM`           | 关闭所有折叠 置 `foldlevel` 为 `0`      |          |
| `zr`           | 减少折叠 增加 `foldlevel`               | `reduce` |
| `zR`           | 打开所有折叠 置 `foldlevel` 为最大      |          |
| `zn`           | 不折叠 复位 `foldenable`                | `none`   |
| `zN`           | 正常折叠 置位 `foldenable`              | `Normal` |
| `zi`           | 反转 `foldenable`                       | `invert` |

####    相关选项

| `set` 选项   | 说明     |
|--------------|----------|
| `foldmethod` | 折叠方式 |
| `foldlevel`  | 折叠层级 |
| `foldmarker` | 折叠标记 |

| `foldmethd` 取值 | 说明                             |
|------------------|----------------------------------|
| `manumal`        | 手动折叠                         |
| `indent`         | 按缩进折叠                       |
| `<expr>`         | 按 `<expr>` 表达式折叠           |
| `syntax`         | 按语法区域折叠                   |
| `foldmaker`      | 按 `foldmarker` 选项指定标记折叠 |

#   文件、缓冲区、窗口

##  文件编辑

### 编辑文件

| 命令                     | 说明                                               |
|--------------------------|----------------------------------------------------|
| `:e[dit][!] {file} {#N}` | 编辑 `{file}`、编号 `{N}` 缓冲区（缺省轮换缓冲区） |
| `:vi[sual][!]`           | *Ex* 模式退出 *Ex* 模式，否则同 `:e`               |
| `:vie[w][!]`             | *Ex* 模式退出 *Ex* 模式，否则同只读 `:e`           |
| `:e[dit][!]`             | 重新载入当前文件                                   |
| `:ene[w][!]`             | 编辑无名新缓冲区                                   |
| `:fin[d][!] {file}`      | 在 `path` 当中查找文件 `{file}` 并编辑之           |
| `gf`、`]f`、`gF`         | 编辑光标下的文件名对应的文件（`goto file`）        |
| `v_gf`、`]f`、`gF`       | 编辑选中内容作为文件名对应的文件（`goto file`）    |
| `:f[ile]`                | 显示当前文件名及光标位置                           |
| `:f[ile] {name}`         | 置当前文件名为 `{name}`                            |
| `:files`                 | 显示所有的轮换文件名                               |

-   `:e`、`:find` 等载入文件编辑通用参数：`:e [++OPT] [+CMD] [FILE] [#N]`
    -   `FILE`：编辑文件名，缺省重新载入当前文件
        -   *Wildcards* 方式搜索、匹配文件：其中通配符被扩展，具体支持取决于平台
        -   反引号后紧跟 `=` 将被视为 *Vim* 表达式被计算
        -   *Unix* 平台上，可用反引号括起 Shell 命令，将其输出作为结果
    -   `#N`：编号 `N` 缓冲区
        -   缺省轮换缓冲区
        -   缓冲区编号必须存在，可为隐藏缓冲区
    -   `++OPT`：指定 `fileformat`、`fileencoding`、`binary`、坏字符处理方案
        -   `++fileformat`/`++ff`
        -   `++encoding`/`++enc`
        -   `++binary`/`++bin`、`++nobinary`/`++nobin`
        -   `++bad=<X>|keep|drop`：用 `X` 替换、维持、删除
    -   `+CMD`：在新打开文件中定位光标、执行命令
        -   `+`：从最后一行开始
        -   `+{NUM}`：从第 `NUM` 行开始
        -   `+/{PTN}`：从首个匹配 `PTN` 开始
        -   `+{CMD}`：打开文件后执行 *Ex* 命令

| *VIM* 相关选项 | 说明                                       |
|----------------|--------------------------------------------|
| `isfname`      | 决定组成文件名的字符                       |
| `suffixesadd`  | 查找文件所需附加的后缀                     |

####    编辑二进制文件

#TODO
-   *Vim* 用 `-b` 表示以二进制模式进行文件读写
    -   相关的文件编辑选项被设置
        -   `binary`
        -   `textwidth=0`
        -   `nomodeline`
        -   `noexpandtab`

####    加密编辑

| *Ex* 命令、普通命令 | 说明                                     |
|---------------------|------------------------------------------|
| `:X`                | 加密，提示输入加密密钥（未输入即不加密） |

-   *Vim* 支持文件加密独写
    -   加密文件头部有魔术数字，*Vim* 据此确认加密文件
        -   可将如下配置写入 *magic* 文件（`/etc/magic`、`/usr/share/misc/magic`）使得加密文件可被 `file` 命令识别
            ```cnf
            0   string  VimCrypt~   Vim encrypted file
            >9  string  01          - "zip" cryptmethod
            >9  string  02          - "blowfish" cryptmethod
            >9  string  03          - "blowfish2" cryptmethod
            ```
    -   但常规的选项设置存在问题
        -   交换文件、撤销文件此时被分块加密
        -   内存中文本、`:!filter`、`:w {CMD}` 过滤文本未加密
        -   *viminfo* 文本未加密

| *VIM* 相关选项     | 说明                           |
|--------------------|--------------------------------|
| `key`              | 存储密钥                       |
| `cryptmethod`/`cm` | 加密方法（须在写入文件前设置） |

-   相关选项
    -   `key`：存储密钥
        -   写入时若该选项非空，则用其值作为密钥加密，否则不加密
        -   读取时若该选项非空，使用其值解密，否提示输入密钥
    -   `cryptmethod`/`cm`：加密方法（须在写入文件前设置）
        -   `zip`：弱加密
        -   `blowfish`：有漏洞
        -   `blowfish2`：中强度

### 工作目录

| *Ex* 命令、普通命令            | 说明                                   |
|--------------------------------|----------------------------------------|
| `:cd [path]`、`:chd[dir]`      | 切换当前目录到 `[path]`                |
| `:tcd[!] <PATH>`、`:tchd[dir]` | 类似 `:cd`，仅为当前标签页设置当前目录 |
| `:lcd[!] <PATH>`、`:lchd[dir]` | 类似 `:cd`，仅为当前窗口设置当前目录   |
| `:pwd`                         | 显示当前目录名                         |

-   `cd` 类命令说明
    -   `PATH`：目标目录（基本同 `$ cd` 命令）
        -   缺省 *Unix* 上改变当前目录到主目录，非 *Unix* 显示当前目录名
        -   `-`：切换到上个当前目录
        -   若为相对路径，则在 `cdpath` 列出的目录中搜索
    -   对当前目录的修改不改变当前已经开文件，但是可能会改变参数列表中文件

### 文件搜索、*Wildcards*

| *VIM* 相关选项 | 说明                                       |
|----------------|--------------------------------------------|
| `cdpath`       | `:cd` 命令查找相对路径的路径               |
| `path`         | `:find`、`gf` 等查找命令查找相对路径的路径 |

-   文件搜索：`path`、`cdpath`、`tags` 选项值设置，以及 `finddir()`、`findfile()` 搜索文件的逻辑
    -   向下搜索：可使用 `*`、`**` 或其他操作系统支持的通配符
        -   `*`：匹配 0 个或更多字符
        -   `**[N]`：匹配 `N` 层目录，缺省 30 层
    -   向上搜索：给定起、止目录，沿目录树向上搜索
        -   起始目录、多个终止目录 `;` 分隔
    -   混合向上、向下搜索
        -   若 `set path=**;/path`，当前目录为 `/path/to/current`，则须在 `path` 中搜索时搜索
            -   `/path/to/current/**`
            -   `/path/to/**`
            -   `/path/**`

-   *Wildcards* 匹配：其余 *Ex* 命令均使用此方式匹配文件（无需预设搜索范围）
    -   具体支持取决于平台，但以下通用
        -   `?`：一个字符
        -   `*`：任何东西，包括空
        -   `**`：任何东西，包括空，递归进入目录
        -   `[abc]`：`a`、`b` 或 `c`

### *VIM* 参数列表处理

| 命令                                 | 说明                                         |
|--------------------------------------|----------------------------------------------|
| `:ar[gs]`                            | 显示参数列表，当前文件显示在 `[]` 中作为标识 |
| `:arge[dit] {file}`                  | 将 `{file}` 加入参数列表                     |
| `:arga[dit] {file}`                  | 将 `{file}` 加入参数列表                     |
| `:argd[elete] {file}`                | 将 `{file}` 加入参数列表                     |
| `:all`、`sall`                       | 为参数列表中的每个文件打开一个窗口           |
| `:wn[ext][!]`                        | 写入当前文件并编辑后一个文件                 |
| `:wn[ext][!] {file}`                 | 写到 `{file}` 并编辑后一个文件               |
| `:wN[ext][!] [file]`、`:wp[revious]` | 写入当前文件并编辑前一个文件                 |
| `:argg[lobal] {arglist}`             | 定义全局参数列表                             |
| `:argl[ocal] {arglist}`              | 定义当前窗口局部参数列表                     |
| `:[range]argdo[!] {cmd}`             | 对参数列表应用 `{cmd}`                       |

| 在当前窗口编辑           | 在新建窗口编辑             | 说明                   |
|--------------------------|----------------------------|------------------------|
| `:argu[ment] N`          | `:sar[gument] N`           | 编辑第 `N` 个文件      |
| `:n[ext]`                | `:sn[ext]`                 | 编辑后一个文件         |
| `:n[ext] {arglist}`      | `:sn[ext] {arglist}`       | 定义新的文件列表并编辑 |
| `:N[ext]`、`:prev[ious]` | `:sN[ext]`、`:sprev[ious]` | 编辑前一个文件         |
| `:fir[st]`、`:rew[ind]`  | `:sfir[st]`、`:srew[ind]`  | 编辑首个文件           |
| `:la[st]`                | `:sla[st]`                 | 编辑末个文件           |

-   参数列表：启动 *Vim* 时给出的多个文件名
    -   不同于 `:buffers` 的缓冲区列表，参数列表中文件名在缓冲区列表中存在，反之不然
    -   所有窗口缺省使用相同全局参数列表，但可以通过 `:arglocal` 创建局部

-   参数列表命令中通用参数说明：`:[COUNT][RANGE]argu[!] [++OPT] [+CMD] [ARGLIST] [FILE] [#N] [ARGLIST]`
    -   `!`：忽略缓冲区更改
    -   `PTN`：删除匹配 `PTN` 的文件
        -   `%`：当前项
    -   `COUNT`：第 `COUNT` 个参数列表中元素
    -   `RANGE`：参数列表中 `RANGE` 范围，缺省全部 `1,$`
        -   `$`：最后项
        -   `.`、空白：删除当前
        -   `%`：全部项
    -   `FILE`：编辑文件名，缺省重新载入当前文件
        -   其中通配符被扩展，具体支持取决于平台
        -   反引号后紧跟 `=` 将被视为 *Vim* 表达式被计算
        -   *Unix* 平台上，可用反引号括起 Shell 命令，将其输出作为结果
    -   `#N`：参数列表编号，缺省为轮换缓冲区
    -   `ARGLIST`：指定新的参数列表
    -   `++OPT`：指定 `fileformat`、`fileencoding`、`binary`、坏字符处理方案
        -   `++fileformat`/`++ff`
        -   `++encoding`/`++enc`
        -   `++binary`/`++bin`、`++nobinary`/`++nobin`
        -   `++bad=<X>|keep|drop`：用 `X` 替换、维持、删除
    -   `+CMD`：在新打开文件中定位光标、执行命令
        -   光标定位
            -   `+`：从最后一行开始
            -   `+{NUM}`：从第 `NUM` 行开始
            -   `+/{PTN}`：从首个匹配 `PTN` 开始
            -   `+{CMD}`：打开文件后执行 *Ex* 命令
        -   执行命令
            -   可用 `|` 分隔多个命令

> - *VIM editing arugment-list*：<https://yianwillis.github.io/vimcdoc/doc/editing.html#argument-list>

###  *Diff*

| 命令                     | 说明                                                     |
|--------------------------|----------------------------------------------------------|
| `:diffs[plit] [FILE]`    | 上下分割窗口，并设置比较状态                             |
| `:difft[his]`            | 将当前窗口设置为比较状态                                 |
| `:diffo[ff][!]`          | 关闭当前窗口比较状态（`!` 关闭所有窗口比较状态）         |
| `:diffu[pdate][!]`       | 更新当前比较窗口，重新生成比较信息（如：修改当前文件后） |
| `:[range]diffg[et] [#N]` | 从其他比较窗口中更新差异内容                             |
| `:diffpu[t] [#N]`        | 向其他比较窗口中输出差异内容                             |
| `]c`、`[c`               | 比较窗口中移至下个不同的为止                             |
| `do`、`dp`               | 向、从其他窗口更新差异                                   |

-   窗口比较
    -   `$ vimdiff`、`$ vimdiff -d` 可直接在命令行开启比较模式
    -   比较基于缓冲区内容、局限于当前 *Tab* 内 `diff` 置位缓冲区
        -   比较状态窗口自动参与当前 *Tab* 内比较，其左侧多出空白列
    -   每个被比较的文件中，以下选项被设置（编辑其他值时，选项被重设为全局值）
        -   `diff`
        -   `scrollbind`
        -   `cursorbind`
        -   `scrollopt`：包含 "hor"
        -   `wrap`：关闭，`diffopt` 包含 "followrap" 时保持不变
        -   `foldmethod`："diff"
        -   `foldcolumn`：来自 `diffopt` 的值，缺省为 2

| *VIM* 配置选项 | 说明                             |
|----------------|----------------------------------|
| `diff`         | 加入窗口至比较窗口组             |
| `diffopt`      | 比较选项                         |
| `diffexpr`     | 计算文件不同版本差异文件的表达式 |

### 写入、退出

| 命令                           | 说明                                           |
|--------------------------------|------------------------------------------------|
| `:[range]w[rite][!]`           | 写入当前文件                                   |
| `:[range]w[rite] {file}`       | 写入至文件 `{file}`，除非其已经存在            |
| `:[range]w[rite]! {file}`      | 写入至文件 `{file}`，覆盖已存在的文件          |
| `:[range]w[rite][!] >>`        | 添加至当前文件                                 |
| `:[range]w[rite][!] >> {file}` | 添加至文件 `{file}`                            |
| `:[range]w[rite] !{cmd}`       | 执行命令 `{cmd}`，以 `[range]` 限定的行作      |
| `:[range]up[date][!]`          | 如果当前文件被改动则写入                       |
| `:wa[ll][!]`                   | 写入所有被改动的缓冲区                         |
| `:q[uit]`                      | 退出当前缓冲区，无非 *Help* 缓冲区则退出 *VIM* |
| `:q[uit]!`                     | 强制退出当前缓冲区，放弃所有的改动             |
| `:qa[ll]`                      | 退出 *VIM*，除非作了改动                       |
| `:qa[ll]!`                     | 退出 *VIM*，放弃所有改动                       |
| `:cq`                          | 退出，不写入文件并返回错误代码                 |
| `:wq[!]`                       | 写入当前文件并退出                             |
| `:wq[!] {file}`                | 写入文件 `{file}` 并退出                       |
| `:x[it][!] [file]`             | 同 `wq` 但是仅当有改动时写入                   |
| `ZZ`                           | 同 `x`                                         |
| `ZQ`                           | 同 `q!`                                        |
| `:xa[ll][!]`                   | 或 `wqall[`!] 写入所有改动的缓冲区并退出       |

-   文件写入参数选项：`:[RANGE]w[!] [++OPT] [>>] [FILE]`
    -   `!`：强制写入，即使置位 `readonly`、文件（不）已存在、无法创建备份文件
        -   可能会破坏文件、权限位
    -   `RANGE`：指定缓冲区范围写入，缺省整个缓冲区
    -   `>>`：将缓冲区内容附加到文件后
    -   `FILE`：写入目标文件，缺省缓冲区名称
        -   `FILE` 被给出时，与当前缓冲区名将互为轮换文件
    -   `OPT`：指定 `fileformat`、`fileencoding`、`binary`
        -   `++fileformat`/`++ff`
        -   `++encoding`/`++enc`
        -   `++binary`/`++bin`、`++nobinary`/`++nobin`

####    挂起

| 命令            | 说明                      |
|-----------------|---------------------------|
| `:st[op][!]`    | 挂起 *VIM* 或开始新 Shell |
| `:sus[pend][!]` | 同 `:stop`                |
| `CTRL-Z`        | 同 `:stop`                |

-   说明
    -   `:st[op][!]`、`:sus[pend][!]`：挂起 *VIM*
        -   无 `!`、且置位 `autowrite` 时，每个修改过、由文件名的缓冲区被写回
        -   `!`、或 `autowrite` 未置位，则修改过的缓冲区不被写回
    -   禁止处理输入
        -   `<C-s>`：阻止 *Vim* 处理输入（中间输入均被记录，恢复后被处理）
        -   `<C-q>`：恢复 *Vim* 处理输入
#TODO

####    文件写入相关选项

#TODO
| *VIM* 相关选项 | 说明                                 |
|----------------|--------------------------------------|
| `write`        | 允许写入文件                         |
| `backup`       | 备份原文件                           |
| `writebackup`  | 写入时备份，写入完成后删除备份文件   |
| `backupskip`   | 备份时忽略匹配的文件名               |
| `backupdir`    | 存放备份路径，缺省为写入文件相同目录 |
| `backupcopy`   | 决定复制、改名实现备份               |

#### 修改时间

| 命令                          | 说明                          |
|-------------------------------|-------------------------------|
| `:[N]checkt[ime] [FILE] [#N]` | 检查文件是否在 *Vim* 外被修改 |

| *VIM* 相关选项 | 说明                          |
|----------------|-------------------------------|
| `autoread`     | 自动载入在 *Vim* 外修改的文件 |

-   *Vim* 会检查 *Vim* 之外的文件修改，避免文件的不同版本
    -   记住文件开始开始编辑时的修改时间、模式、大小
    -   执行 Shell 命令（`:!<CMD>`、`:suspend`、`:read!`、`K`）后
        -   *Vim* 比较缓冲区的修改时间、模式、大小
        -   并对修改的文件执行 `FileChangedShell` 自动命令、或显示警告
    -   若文件在缓冲区中未经编辑，*Vim* 会自动读取、比较

##  窗口、缓冲区

-   缓冲区、窗口、标签页
    -   缓冲区：内存中的文件文本
    -   窗口：缓冲区的视窗
    -   标签页：窗口的集合

> - *VIM windows*：<https://yianwillis.github.io/vimcdoc/doc/windows.html>

### 多窗口命令

| 命令                   | 说明                              |            |
|------------------------|-----------------------------------|------------|
| `CTRL-W s`             | 或  `split`    将窗口分割成两部分 | `split`    |
| `:split {file}`        | 分隔窗口并在其中一个编辑 `{file}` |            |
| `:vsplit {file}`       | 同上，但垂直分割                  |            |
| `:vertical {cmd}`      | 使命令 `{cmd}` 垂直分割           |            |
| `:sf[ind] {file}`      | 分割窗口，从 `{path}` 中找到文件  |            |
| `:terminal {cmd}`      | 打开新终端窗口                    |            |
| `CTRL-W ]`             | 分割窗口并跳转到光标下的标签      |            |
| `CTRL-W f`             | 分割窗口并编辑光标下的文件名      | `file`     |
| `CTRL-W ^`             | 分割窗口并编辑轮换文件            |            |
| `CTRL-W n`、`:new`     | 创建新空白窗口                    | `new`      |
| `CTRL-W q`、`:q[uit]`  | 退出编辑并关闭窗口                | `quit`     |
| `CTRL-W c`、`:clo[se]` | 隐藏当前缓冲区并关闭窗口          | `close`    |
| `CTRL-W o`、`:on[ly]`  | 使当前窗口成为唯一窗口            | `only`     |
| `CTRL-W j`             | 跳转到下方窗口                    |            |
| `CTRL-W k`             | 跳转到上方窗口                    |            |
| `CTRL-W CTRL-W`        | 移动光标至下方窗口（折转）        | `Wrap`     |
| `CTRL-W W`             | 移动光标至上方窗口（折转）        | `wrap`     |
| `CTRL-W t`             | 跳转到顶端窗口                    | `top`      |
| `CTRL-W b`             | 跳转到底端窗口                    | `bottom`   |
| `CTRL-W p`             | 跳转到上一次激活的窗口            | `previous` |
| `CTRL-W r`             | 向下旋转窗口                      | `rotate`   |
| `CTRL-W R`             | 向上旋转窗口                      | `Rotate`   |
| `CTRL-W x`             | 将当前窗口与后一个窗口对调        | `eXchange` |
| `CTRL-W =`             | 使所有窗口等高等宽                |            |
| `CTRL-W -`             | 减少当前窗口高度                  |            |
| `CTRL-W +`             | 增加当前窗口高度                  |            |
| `CTRL-W _`             | 设置当前窗口高度（缺省 很高）     |            |
| `CTRL-W <`             | 减少当前窗口宽度                  |            |
| `CTRL-W >`             | 增加当前窗口宽度                  |            |
| `CTRL-W`               | 设置当前窗口宽度（缺省 尽可能宽） |            |

### 缓冲区列表

| 命令                        | 说明                                                                       |
|-----------------------------|----------------------------------------------------------------------------|
| `:buffers`、`:files`、`:ls` | 列出所有已知缓冲区和文件名                                                 |
| `:ball`、`:sball`           | 编辑所有参数、缓冲区                                                       |
| `:unhide`、`:sunhide`       | 编辑所有载入的缓冲区                                                       |
| `:badd {fname}`             | 加入文件名 `{fname}` 到缓冲区列表                                          |
| `:bunload[!] [N]`           | 从内存中卸载缓冲区 `[N]`                                                   |
| `:bdelete[!] [N]`           | 从内存中卸载缓冲区 `[N]` 并从列表中删除                                    |
| `[count]CTRL-g`             | 显示当前缓冲区文件名、光标位置、文件状态（`[count]>1` 同时给出缓冲区编号） |
| `g CTRL-g`                  | 显示当前光标位置，分别按：列、和、单词、字符、字节计数                     |
| `:keepalt {CMD}`            | 执行 `CMD`，在此期间保持轮换文件名不变                                     |

-   缓冲区列表：记录所有缓冲区名称，可用 `:ls` 打印
    -   *Vim* 编辑文件过程
        -   将文件读取至缓冲区
        -   用编辑器命令修改缓冲区
        -   将缓冲区内容写回文件：保存缓冲区前，文件内容不改变
    -   编辑、写回时，对应 **文件名** 被加入缓冲区列表，作为缓冲区名称
        -   保存缓冲区时将写入缓冲区名称指向的文件中
        -   即，缓冲区名称修改、删除后保存文件将不会写入原文件
        -   每个窗口维护独立的文件名、轮换文件名
    -   各缓冲区有唯一、不变的递增编号
    -   缓冲区标识符（`%`、`#` 等标记可用于切换缓冲区时替代编号）
        -   `%` 当前在编辑缓冲区
        -   `#` 轮换的缓冲区（上次编辑缓冲区）
        -   `+` 已修改未保存缓冲区

####    缓冲区相关选项

| *VIM* 相关选项 | 说明                                             |
|----------------|--------------------------------------------------|
| `backup`       | 文件被覆盖前备份文件                             |
| `backupext`    | 备份文件名后缀（备份文件名为原文件名 + 此后缀）  |
| `shortname`    | 替换多余点好为 `_`                               |
| `backupdir`    | 备份文件存储路径                                 |
| `shortmess`    |                                                  |
| `autowriteall` | 自动写回改动                                     |
| `hidden`       | 允许对缓冲区修改且不写回后将其切换至后台（隐藏） |

####    缓冲区相关内建函数

| *VIM* 内建函数         | 说明                                     |
|------------------------|------------------------------------------|
| `bufnr()`              | 根据缓冲区名称获取编号（缺省当前缓冲区） |
| `bufname()`            | 根据缓冲区编号获取名称（缺省当前缓冲区） |
| `win_getid(win, tab)`  | 获取标签下窗口的唯一编号（缺省当前窗口） |
| `win_id2tabwin(winid)` | 获取窗口所在标签序号、自身序号           |
| `win_goto(winid)`      | 跳转至指定窗口                           |

####    缓冲区标识

| 标识符 | 说明                                     |
|--------|------------------------------------------|
| `a`    | 激活缓冲区                               |
| `h`    | 隐藏缓冲区（已载入内存但无窗口展示）     |
| ` `    | 非激活缓冲区                             |
| `u`    | 列表外缓冲区（需 `!` 显示）              |
| `%`    | 当前窗口缓冲区                           |
| `#`    | 轮换缓冲区                               |
| `-`    | 不可更改缓冲区（选项 `modifiable` 复位） |
| `=`    | 只读缓冲区                               |
| `R`    | 作业运行中的终端缓冲区                   |
| `F`    | 作业已完成的终端缓冲区                   |
| `?`    | 无作业的终端缓冲区                       |
| `+`    | 已修改缓冲区                             |
| `x`    | 有读错误缓冲区                           |

####    缓冲区类型

| 缓冲区类型       | 说明                                          |
|------------------|-----------------------------------------------|
| `""`（空字符串） | 普通缓冲区                                    |
| `nofile`         | 未关联文件缓冲区，无法写入磁盘                |
| `nowrite`        | 未写入磁盘缓冲区                              |
| `acwrite`        | 总随 `BufWriteCmd` 自动命令自动写入磁盘缓冲区 |
| `quickfix`       | *QuickFix* 缓冲区                             |
| `help`           | 帮助缓冲区                                    |
| `terminal`       | 中断缓冲区                                    |
| `prompt`         | 提示符缓冲区，仅应用于插件                    |
| `popup`          | 弹窗缓冲区                                    |

-   缓冲区有不同类型，存储在 `buftype` 选项中
    -   不建议直接修改 `buftype` 选项值

### 缓冲区切换

| 命令        | 说明                                                |
|-------------|-----------------------------------------------------|
| `[N]CTRL-^` | 切换至编号 `N` 缓冲区（同 `:e #N`），缺省轮换缓冲区 |

| 当前窗口内            | *Split* 窗口           | 说明                      |
|-----------------------|------------------------|---------------------------|
| `:[N]buffer [N]`      | `:[N]sbuffer [N]`      | 至编号 `N` 参数/缓冲区    |
| `:[N]bn[ext] [N]`     | `:[N]sbn[ext] [N]`     | 往后第 `N` 个参数/缓冲区  |
| `:[N]bNext [N]`       | `:[N]sbNext [N]`       | 往前第 `N` 个参数/缓冲区  |
| `:[N]bp[revious] [N]` | `:[N]sbp[revious] [N]` | 往前第 `N` 个参数/缓冲区  |
| `:bfirst`             | `:sbfirst`             | 至首个参数/缓冲区         |
| `:blast`              | `:sblast`              | 至末个参数/缓冲区         |
| `:[N]bmod [N]`        | `:[N]sbmod [N]`        | 至第 `N` 个改动过的缓冲区 |

## *QuickFix*

| 命令             | 说明                                              |
|------------------|---------------------------------------------------|
| `:cc [nr]`       | 显示第 `[nr]` 个错误（缺省首个、当前错误）        |
| `:cn`            | 显示后一个错误                                    |
| `:cp`            | 显示前一个错误                                    |
| `:cl`            | 列出所有错误                                      |
| `:cf`            | 从文件 '`errorfile`' 读入错误                     |
| `:cgetb`         | 类似于 `cbuffer` 但不跳转到首个错误               |
| `:cg`            | 类似于 `cfile` 但不跳转到首个错误                 |
| `:cgete`         | 类似于 `cexpr` 但不跳转到首个错误                 |
| `:caddf`         | 从错误文件加入错误到当前快速修复列表              |
| `:cad`           | 从表达式计算结果加入错误到当前快速修复列表        |
| `:cb`            | 从缓冲区文本读入错误                              |
| `:cex`           | 从表达式计算结果读入错误                          |
| `:cq`            | 退出不保存并返回错误代码（至编译器）              |
| `:make [args]`   | 启动 `make`，读入错误，并跳转到首个错误           |
| `:gr[ep] [args]` | 执行 '`grepprg`' 程序以找出匹配并跳转到首个匹配   |

-   快速修复命令：加快 *编辑-编译-编辑* 循环
    -   可通过 `$ vim -q <FILENAME>` 读取保存在文件中的出错信息
    -   （全局）快速修复列表
        -   唯一标识：在 *Vim* 会话中保持不变
        -   列表号：在快速修复栈中加入超过 10 个列表时可能改变
        -   由 `:vimgrep`、`:grep`、`:helpgrep`、`:make` 等命令产生
    -   位置列表：窗口局部的快速修复列表
        -   与窗口相关联，每个窗口有独立位置列表，与快速修复列表独立
        -   窗口分割时新窗口得到位置列表的备份
        -   由 `:lvimgrep`、`:lgrep`、`:lhelpgrep`、`:lmake` 等命令产生

> - *VIM quickfix*：<https://yianwillis.github.io/vimcdoc/doc/quickfix.html>

### *Quickfix* 窗口

| 全局快速修复列表命令  | 说明                       |
|-----------------------|----------------------------|
| `:cope[n] [HEIGHT]`   | 打开窗口显示当前列表       |
| `:ccl[ose]`           | 关闭窗口                   |
| `:cw[indow] [HEIGHT]` | 存在可识别错误时，打开窗口 |
| `:cbo[ttom]`          | 光标置于窗口末行           |

| 窗口局部快速修复列表命令 | 说明                       |
|--------------------------|----------------------------|
| `:lope[n] [HEIGHT]`      | 打开窗口显示当前列表       |
| `:lcl[ose]`              | 关闭窗口                   |
| `:lw[indow] [HEIGHT]`    | 存在可识别错误时，打开窗口 |
| `:lbo[ttom]`             | 光标置于窗口末行           |

### *Quickfix* 跳转

| 全局快速修复列表命令                 | 说明                                                               |
|--------------------------------------|--------------------------------------------------------------------|
| `:[NR]cc[!] [NR]`                    | 跳转编号 `NR` 错误                                                 |
| `:[COUNT]cn[ext][!]`                 | 跳转列表中后 `COUNT` 个错误                                        |
| `:[COUNT]cN[ext][!]`、`:cp[revious]` | 跳转列表中后、前 `COUNT` 个错误                                    |
| `:[COUNT]cabo[ve]`                   | 跳转缓冲区当前行前 `COUNT` 个错误                                  |
| `:[COUNT]cbel[ow]`                   | 跳转缓冲区当前行后 `COUNT` 个错误                                  |
| `:[COUNT]cbe[fore]`                  | 跳转缓冲区光标前 `COUNT` 个错误                                    |
| `:[COUNT]caf[ter]`                   | 跳转缓冲区光标后 `COUNT` 个错误                                    |
| `:[COUNT]cnf[ile][!]`                | 若包含文件名，跳转后 `COUNT` 个文件最后错误，否则后 `COUNT` 个错误 |
| `:[COUNT]cNf[ile][!]`、`:cpf[ile]`   | 若包含文件名，跳转前 `COUNT` 个文件最后错误，否则前 `COUNT` 个错误 |
| `:cr[ewind][!] [NR]`、`:cfir[st]`    | 跳转错误 `NR`，缺省首个                                            |
| `:cla[st][!] [NR]`                   | 跳转错误 `NR`，缺省末尾                                            |

| 窗口局部快速修复列表命令             | 说明                                                               |
|--------------------------------------|--------------------------------------------------------------------|
| `:[NR]ll[!] [NR]`                    | 跳转编号 `NR` 错误                                                 |
| `:[COUNT]ln[ext][!]`                 | 显示列表中后 `COUNT` 个错误                                        |
| `:[COUNT]lN[ext][!]`、`:lp[revious]` | 显示列表中后、前 `COUNT` 个错误                                    |
| `:[COUNT]labo[ve]`                   | 当前缓冲区当前行前 `COUNT` 个错误                                  |
| `:[COUNT]lbel[ow]`                   | 当前缓冲区当前行后 `COUNT` 个错误                                  |
| `:[COUNT]lbe[fore]`                  | 当前缓冲区光标前 `COUNT` 个错误                                    |
| `:[COUNT]laf[ter]`                   | 当前缓冲区光标后 `COUNT` 个错误                                    |
| `:[COUNT]lnf[ile][!]`                | 若包含文件名，显示后 `COUNT` 个文件最后错误，否则后 `COUNT` 个错误 |
| `:[COUNT]lNf[ile][!]`、`:lpf[ile]`   | 若包含文件名，显示前 `COUNT` 个文件最后错误，否则前 `COUNT` 个错误 |
| `:lr[ewind][!] [NR]`、`:lfir[st]`    | 显示错误 `NR`，缺省首个                                            |
| `:lla[st][!] [NR]`                   | 显示错误 `NR`，缺省末尾                                            |

### *Quickfix* 建立

| 全局快速修复列表命令                  | 说明                                            |
|---------------------------------------|-------------------------------------------------|
| `:cf[file][!] [ERRORFILE]`            | 读入错误文件，跳转到首个错误                    |
| `:cg[etfile][!] [ERRORFILE]`          | 读入错误文件，不跳转到首个错误                  |
| `:caddf[ile][!] [ERRORFILE]`          | 读取错误文件，将错误文件里的错误加入列表中      |
| `:cb[uffer][!] [BUFNR]`               | 从缓冲区 `BUFNR` 读入错误列表，跳转到首个错误   |
| `:cgetb[uffer][!] [BUFNR]`            | 从缓冲区 `BUFNR` 读入错误列表，不跳转到首个错误 |
| `:cad[dbuffer] [BUFNR]`               | 从缓冲区 `BUFNR` 读取错误列表加入列表           |
| `:cex[pr][!] <EXPR>`                  | 用 `EXPR` 计算结果建立列表，跳转到首个错误      |
| `:cgete[xpr] <EXPR>`                  | 用 `EXPR` 计算结果建立列表，不跳转到首个错误    |
| `:cad[dexpr] <EXPR>`                  | 将 `EXPR` 计算结果读入列表                      |
| `:cl[ist][!] [<FROM> [TO]] [+<COUNT]` | 显示有效错误                                    |
| `:[N]cq[uit][!] [N]`                  | 以错误码 `N` 退出                               |

| 窗口局部快速修复列表命令              | 说明                                            |
|---------------------------------------|-------------------------------------------------|
| `:lf[file][!] [ERRORFILE]`            | 读入错误文件，跳转到首个错误                    |
| `:lg[etfile][!] [ERRORFILE]`          | 读入错误文件，不跳转到首个错误                  |
| `:laddf[ile][!] [ERRORFILE]`          | 读取错误文件，将错误文件里的错误加入列表中      |
| `:lb[uffer][!] [BUFNR]`               | 从缓冲区 `BUFNR` 读入错误列表，跳转到首个错误   |
| `:lgetb[uffer][!] [BUFNR]`            | 从缓冲区 `BUFNR` 读入错误列表，不跳转到首个错误 |
| `:lad[dbuffer] [BUFNR]`               | 从缓冲区 `BUFNR` 读取错误列表加入列表           |
| `:lex[pr][!] <EXPR>`                  | 用 `EXPR` 计算结果建立列表，跳转到首个错误      |
| `:lgete[xpr] <EXPR>`                  | 用 `EXPR` 计算结果建立列表，不跳转到首个错误    |
| `:lad[dexpr] <EXPR>`                  | 将 `EXPR` 计算结果读入列表                      |
| `:ll[ist][!] [<FROM> [TO]] [+<COUNT]` | 显示有效错误                                    |
| `:[N]lq[uit][!] [N]`                  | 以错误码 `N` 退出                               |

-   说明
    -   通用选项
        -   `!`：强制跳转缓冲区，即使可能丢失当前缓冲区的修改
            -   或，所有错误
        -   `ERRORFILE`：缺省为 `errorfile` 选项值
        -   `RANGE`：列表范围
        -   `CMD`：*Ex* 命令，可用 `|` 连接多个命令
        -   `FROM-TO`：指定错误范围
        -   `+COUNT`：当前和之后 `COUNT` 个错误行

| *VIM* 选项     | 说明                                   |
|----------------|----------------------------------------|
| `errorformat`  | 错误信息格式（缺省值可能无法正常工作） |
| `errorfile`    | 错误文件缺省值                         |
| `makeencoding` | 错误文件码格式                         |

### *Quickfix* 处理

| 全局快速修复列表命令    | 说明                             |
|-------------------------|----------------------------------|
| `:[RANGE]cdo[!] <CMD>`  | 在列表的每个有效项目上应用 `CMD` |
| `:[RANGE]cfdo[!] <CMD>` | 在列表的每个文件上应用 `CMD`     |
| `:col[der] [COUNT]`     | 切换至后 `COUNT` 个列表          |
| `:cnew[er] [COUNT]`     | 切换至前 `COUNT` 个列表          |
| `:[COUNT]chi[story]`    | 给出修复列表列表                 |

| 窗口局部快速修复列表命令 | 说明                             |
|--------------------------|----------------------------------|
| `:[RANGE]cdo[!] <CMD>`   | 在列表的每个有效项目上应用 `CMD` |
| `:[RANGE]cfdo[!] <CMD>`  | 在列表的每个文件上应用 `CMD`     |
| `:lol[der] [COUNT]`      | 切换至后 `COUNT` 个列表          |
| `:lnew[er] [COUNT]`      | 切换至前 `COUNT` 个列表          |
| `:[COUNT]lhi[story]`     | 给出修复列表列表                 |

-   说明
    -   通用选项
        -   `!`：强制跳转缓冲区，即使可能丢失当前缓冲区的修改
        -   `RANGE`：列表范围
        -   `CMD`：*Ex* 命令，可用 `|` 连接多个命令

##  *Tags*

| `N` | 命令                     | 说明                                           |
|-----|--------------------------|------------------------------------------------|
|     | `:ta[g][!] {tag}`        | 跳转到标签 `{tag}`（可谓正则表达式）           |
|     | `:[count]ta[g][!]`       | 跳转到标签列表里第 `[count]` 个较新的标签      |
|     | `CTRL-]`                 | 跳转到光标下的标签，除非文件被改动             |
|     | `:ts[elect][!] [tag]`    | 列出匹配的标签并选择其中一个跳转               |
|     | `:tj[ump][!] [tag]`      | 跳转到标签 `[tag]`，当有多个匹配时从列表中选择 |
|     | `:lt[ag][!] [tag]`       | 跳转到标签 `[tag]` 并把匹配的标签加到位置列表  |
|     | `:tags`                  | 显示标签列表                                   |
| `N` | `CTRL-T`                 | 跳转到标签列表中第 `N` 个较早的标签            |
|     | `:[count]po[p][!]`       | 跳转到标签列表中第 `[count]` 个较早的标签      |
|     | `:[count]tn[ext][!]`     | 跳转到向后第 `[count]` 个匹配的标签            |
|     | `:[count]tp[revious][!]` | 跳转到向前第 `[count]` 个匹配的标签            |
|     | `:[count]tr[ewind][!]`   | 跳转到第 `[count]` 个匹配的标签                |
|     | `:tl[ast][!]`            | 跳转到前次匹配的标签                           |
|     | `:pt[ag] {tag}`          | 打开预览窗口来显示 `{tag}` 标签                |
|     | `CTRL-W }`               | 同 `CTRL-]` 但在预览窗口显示标签               |
|     | `:pts[elect]`            | 同 `:tselect` 但在预览窗口显示标签             |
|     | `:ptj[ump]`              | 同 `:tjump` 但在预览窗口显示标签               |
|     | `:pc[lose]`              | 关闭标签预览窗口                               |
|     | `CTRL-W z`               | 关闭标签预览窗口                               |

-   *Tags*：出现在 *tags* 文件中、用于跳转的标识符
    -   *tags* 文件由 `ctags` 类似的程序生成

| *VIM* 选项     | 说明           |
|----------------|----------------|
| `tagcase`      | 标签大小写匹配 |
| `[no]tagstack` | 标签压入栈中   |

> - *VIM tagsrch*：<https://yianwillis.github.io/vimcdoc/doc/tagsrch.html>

### *Tags* 匹配、跳转

| `N` | 命令                                           | 说明                                               |
|-----|------------------------------------------------|----------------------------------------------------|
|     | `:ta[g][!] {tag}`                              | 跳转到标签 `{tag}`（可谓正则表达式）               |
| `N` | `CTRL-]`、`g<LeftMouse>`、`CTRL-<LeftMouse>`   | 跳转到光标下的标签，除非文件被改动                 |
| `v` | `v_CTRL-]`                                     | 跳转到选中文本的标签，除非文件被改动               |
|     | `:ts[elect][!] [tag]`                          | 列出匹配的标签，等待选择其中一个跳转               |
|     | `:sts[elect][!] [tag]`                         | 列出匹配的标签，等待选择其中一个分割窗口           |
|     | `:tj[ump][!] [tag]`                            | 跳转到标签 `[tag]`，当有多个匹配时从列表中选择     |
|     | `g CTRL-]`                                     | 类似 `CTRL-]`，但使用 `:tjump`                     |
| `v` | `v_g CTRL-]`                                   | 类似 `v_CTRL-]`，但使用 `:tjump`                   |
|     | `:stj[ump][!] [tag]`                           | 分屏跳转到标签 `[tag]`，当有多个匹配时从列表中选择 |
|     | `:lt[ag][!] [tag]`                             | 跳转到标签 `[tag]` 并把匹配的标签加到位置列表      |
|     | `:tags`                                        | 显示标签列表                                       |
|     | `:[count]tn[ext][!]`                           | 跳转到向后第 `[count]` 个匹配的标签                |
|     | `:[count]tp[revious][!]`、`:tN[ext]`           | 跳转到向前第 `[count]` 个匹配的标签                |
|     | `:[count]tr[ewind][!]`、`:tf[irst]`            | 跳转到第 `[count]` 个匹配的标签，缺省首个          |
|     | `:tl[ast][!]`                                  | 跳转到前次匹配的标签                               |

### *Tag Stack* 

| `N` | 命令                                           | 说明                                    |
|-----|------------------------------------------------|-----------------------------------------|
|     | `:[count]ta[g][!]`                             | 跳转到标签栈中第 `[count]` 个较新的标签 |
| `N` | `CTRL-T`、`g<RightMouse>`、`CTRL-<RightMouse>` | 跳转到标签栈中第 `[count]` 个较早的标签 |
|     | `:[count]po[p][!]`                             | 跳转到标签栈中第 `[count]` 个较早的标签 |

-   标签栈：记录跳转过的标签历史
    -   最多容纳 20 项，较早项目被前移直至移除
    -   `>`：标识当前激活项目
    -   `T0` 列：标签匹配输目

### *Tag* 预览

| 命令                                  | 说明                                           |
|---------------------------------------|------------------------------------------------|
| `:pts[elect][!] [tag]`                | 列出匹配的标签，等待选择其中一个预览           |
| `:ptj[ump][!] [tag]`                  | 预览到标签 `[tag]`，当有多个匹配时从列表中选择 |
| `:[count]ptn[ext][!]`                 | 预览到向后第 `[count]` 个匹配的标签            |
| `:[count]ptp[revious][!]`、`:tN[ext]` | 预览到向前第 `[count]` 个匹配的标签            |
| `:[count]ptr[ewind][!]`、`:tf[irst]`  | 预览到第 `[count]` 个匹配的标签，缺省首个      |
| `:ptl[ast][!]`                        | 预览到前次匹配的标签                           |
| `CTRL-W }`                            | 同 `CTRL-]` 但在预览窗口显示标签               |
| `CTRL-W z`                            | 关闭标签预览窗口                               |
| `:pc[lose]`                           | 关闭标签预览窗口                               |

> - 标签跳转命令添加前缀 `p`，在预览窗口中查看标签


#   *VimScript* 编程

##  变量

-   `let`：创建变量、变量赋值都需要用到`let`关键字

> - *Vimscript* 变量：<https://www.cnblogs.com/jzy996492849/p/7194276.html>

### 变量作用域

| 作用域前缀 | 描述                           |
|------------|--------------------------------|
| `g:`       | 全局变量（非函数内变量默认）   |
| `b:`       | 缓冲区级                       |
| `w:`       | 窗口级                         |
| `t:`       | 标签级                         |
| `s:`       | *VIM* 脚本文件级               |
| `l:`       | 函数内部变量（函数内变量默认） |
| `a:`       | 函数参数                       |
| `v:`       | *VIM* 专用全局                 |
| `&`        | 访问属性                       |
| `@`        | 访问寄存器                     |

-   *VIM* 变量可以显式指定作用域
    -   变量作用域：`<PREFIX>:` 前缀引导变量名表示变量作用域
        -   `g:`：定义于函数外的变量默认
        -   `l:`：定义于函数内的变量默认
    -   `&`：可通过 `&<OPTION>` 符号访问属性值，类似普通变量赋值、取值、打印、参与运算
        -   选项取值
            -   *Bool* 选项输出 0、1
            -   键值选项取数值、字符串、列表等
        -   类似变量作用域，选项取值也通过 `g:`、`b:` 等区分作用域
    -   `@`：可通过 `@<REGISGER>` 访问寄存器值，类似普通变量赋值、取值、打印、参与运算

```vimscript
echo &wrap
echo &g:textwidth
let &textwidth = 80                 " 等价于 `set textwidth=80`
let &textwidth = &textwidth + 10    " `set` 命令无法实现
let @a = "hello"                    " 寄存器赋值
```

### *VIM* 变量类型

-   *VimScript* 提供 9 种基本数据类型：可通过 `type()` 函数查看
    -   `Number` 32 位带符号整形（`+num64` 标志即表示 64 位符号整数）
        -   `:echo 0xef`：16 进制
        -   `:echo 017`：8 进制（鉴于以下，不建议使用）
        -   `:echo 019`：10 进制（8 进制中 9 不可能出现，此例子 *VIM* 自动识别为 10 进制）
        -   说明
            -   整形之间除法同 *C/CPP*
    -   `Float` 浮点数（需 `+float` 标志才支持）
        -   `:echo 5.1e-3`：科学计数法）
        -   `:echo 5.0e3`：科学计数法中一定要有小数点
        -   说明
            -   类型转换：`Number` 和 `Float` 运算时会强制转换为 `Float`
    -   `String` 字符串
    -   `List` 有序列表
    -   `Dictionary` 字典（哈希表）
    -   `Funcref` 函数引用
    -   特殊值
        -   `v:false`：`0`、`v:false` 表示假值，其余均为真值
        -   `v:true`
        -   `v:none`
        -   `v:null`
    -   `Job` 任务
    -   `Channel` 通道

### 整数、浮点、字符串

-   整数、浮点、字符串
    -   *VimScript* 中字符串、整数之间回自动进行变量类型转换
        -   算术运算、布尔判断：`+`、`if` 这些 “运算” 中，*VIM* 会强制转换变量类型
            -   数字开头的字符串会根据字面值转为相应 `Number`（整形）
                -   转换时会考虑进制（`"0100"` 转换为 `64`）
                -   符合浮点数模式的字符串会舍弃小数点后内容，直接转换为 `Number`
            -   而非数字开头则转换为 `0`
        -   字符串连接 `.`
            -   `.` 连接时 *VIM* 可以自动将 `Number` 转换为字符串然后连接
            -   但，对于 `Float`，*VIM* 不会自动转换
    -   字符串字面量：`""`、`''` 扩起
        -   字符串可以像列表一样切割、索引
            -   但是不可以使用负数索引，却可以使用负数切片
        -   `""` 内支持转义 `\`，`''` 内保持字面值
            -   连续两个单引号 `''` 表示转义自身
        -   字符串可按码值顺序比较
            -   `==`、`==？`、`==#` 对大小写是否敏感结果
                -   `==`：对字符串比较是否大小写敏感取决于 `set [no]ignorecase` 配置项
                -   `==?`：对字符串比较大小写永远不敏感
                -   `==#`：对字符串比较大小写永远敏感
            -   `<`、`>`：类似 `==`，也有 3 种

| 内建字符串函数           | 描述                                   |
|--------------------------|----------------------------------------|
| `strlen(str)`            |                                        |
| `len(str)`               | 对字符串变量同 `strlen`                |
| `split(str, token=" ")`  |                                        |
| `tolower(str)`           |                                        |
| `toupper(str)`           |                                        |
| `==`                     | `set [no]ignorcase` 决定是否大小写敏感 |
| `==?`                    | 大小写不敏感                           |
| `==#`                    | 大小写敏感                             |
| `>`、`>?`、`>#`          |                                        |
| `<`、`<?`、`<#`          |                                        |

###    列表

| 列表内建函数                    | 描述                                          |
|---------------------------------|-----------------------------------------------|
| `empty(list)`                   | 空判断                                        |
| `min(list)`                     | 数值最小值（即使包含字符串）                  |
| `max(list)`                     | 数值最大值（即使包含字符串）                  |
| `count(list)`                   | 计数                                          |
| `index(list, item)`             | 返回元素索引，不存在返回 `-1`                 |
| `split(str, token=" ")`         | 拆分字符串为列表                              |
| `join(list, token=" ")`         | 将列表中元素转换为字符串后，使用 `token` 连接 |
| `insert(list, item, 0)`         | 插入元素                                      |
| `add(list, item)`               | 添加新素                                      |
| `extend(list, other)`           | 扩展列表                                      |
| `len(list)`                     | 列表长度                                      |
| `get(list, index, default_val)` | 获取列表元素，越界则返回默认值                |
| `reverse(list)`                 | 反转列表                                      |
| `sort(list)`                    | 按字符序排序（即使全为数值）                  |
| `uniq(list)`                    | 剔除重复值                                    |
| `copy(list)`                    | 浅拷贝（内部引用仅拷贝）                  |
| `deepcopy(list)`                | 深拷贝                                    |
| `+`                             | 连接两个列表                                  |
| `==`                            | 逐元素比较                                    |
| `is`                            | 是否为同一引用                                |

-   *VIM* 列表
    -   定义、声明语法：`let list = [1, "a", "b", ]`
        -   允许结尾的 `,`
        -   支持多种类型元素混杂
        -   修改列表中元素同样使用 `let`
        -   可通过 `unlet` 剔除列表中指定元素（列表长度改变）
    -   *VIM* 列表特点
        -   有序、异质
        -   索引从 `0` 开始，支持负数索引，使用下标得到对应元素
        -   支持切片（引用）
            -   左右均为闭区间
            -   可以负数区间切片
            -   可以忽略起始、结尾索引，表示从 0 开始、至末尾截止
            -   切片越界是安全的
            -   切片可被赋值
        -   列表赋值时只是传递引用（需 `copy()` 拷贝）
            -   支持模式匹配赋值

```vimscript
let list = [1, 2, "a", [4, 5, 6]]
let list[0] = 99                        " 修改元素
let list[:1] = [3, 4]                   " 切片赋值
let slice = list[:100]                  " 切片越界安全
let ref = list                          " 应用传递
let copyed = copy(list)                 " 拷贝副本
let [a, b; rest] = list                 " 模式匹配赋值
let listlsit = [list]
for [a, b; rest] in listlist            " 支持模式匹配 `for`
    echo rest
endfor
unlet list[0]                           " 剔除列表元素
```

###    字典

| 字典内建函数                            | 描述                                        |
|-----------------------------------------|---------------------------------------------|
| `get(dict, index, default_val)`         | 获取字典元素                                |
| `has_key(dict, index)`                  | 检查字典中是否有给定键，返回 `1` 真、`0` 假 |
| `items(dict)`                           | 返回字典键值对列表，不保证有序              |
| `keys(dict)`                            | 返回字典所有键                              |
| `values(dict)`                          | 返回字典所有值                              |
| `remove(dict, index)`                   | 移除字典元素                                |
| `unlet dict.index`、`unlet dict[index]` | 移除字典元素                                |
| `copy(list)`                            | 浅拷贝（内部引用仅拷贝）                    |
| `deepcopy(list)`                        | 深拷贝                                      |

-   *VIM* 字典
    -   定义、声明语法：`let dict = {"a": 2, 2: 3,}`
        -   允许结尾的 `,`
        -   使用 `[]`、`.` 语法注册、查找
        -   注册、修改新元素语法同普通赋值 `let dict.100 = 100`
        -   可通过 `unlet` 剔除字典中指定元素
            -   移除不存在的元素时报错
    -   字典特性
        -   值是异质的
        -   键都是字符串
            -   非字符串会被自动、强制转换为字符串注册、查找
        -   非字符串键均被自动转换为字符串

```vimscript
let dict = {1: "a", 2: "b"}
dict."1" = 100                          " 键均转换为字符串
unlet dict.1                            " 剔除列表元素
for [k, v] in items(dict)               " 同时迭代键值对必须 `items`
    echo k v
endfor
```

##  流程、函数

| 流程控制命令                 | 描述                      |
|------------------------------|---------------------------|
| `!`                          | 否                        |
| `||`                         | 或                        |
| `&&`                         | 与                        |
| `if`、`elseif`、`else`、`fi` | 逻辑控制                  |
| `finish`                     | 终止整个 *VimScript* 执行 |
| `for ... in ...`、`endfor`   | 迭代循环                  |
| `while`、`endwhile`          | `while` 循环              |
| `function`、`endfunction`    | 定义、列出函数            |
| `delfunction`                | 删除函数                  |

-   *VimScript* 支持基本逻辑控制、函数等
    -   流程控制命令
        -   `end<KW>` 配合结束逻辑控制语句
        -   没有 `;`、`:` 等分隔符、标识符

### 预定义函数

| 预定义函数                       | 描述                                |
|----------------------------------|-------------------------------------|
| `getline(lineno)`                | 获取指定行号内容，`.` 表示当前行    |
| `setline(lineno, content)`       | 设置指定行号行内容，`.` 表示当前行  |
| `range(start, end, step)`        | 生成序列，首尾均包含                |
| `expand(option)`                 | 根据参数返回当前文件相关信息        |
| `fnamemodify(file_name, option)` | 返回当前文件夹下文件信息            |
| `globpath(dir, ptn)`             | 返回匹配的文件名字符串，`<cr>` 分隔 |

> - *VIM builtin*：<https://yianwillis.github.io/vimcdoc/doc/builtin.html>

### 自定义函数

```vimscript
function[!] [g:]FuncName!(arg1, arg2, ...) <KEYWORD>
    let argn = 1
    while argn <= a:0
        echo a:{argn}               " 获取不可变参数 TODO
        let argn += 1
    end while
endfunction
```

-   自定义函数
    -   函数签名
        -   名称必须以大写字母（或 `s:`）开头
            -   没有作用域限制的 *VimScript* 函数必须以大写字母开头（有作用域限制最好也是大写字母开头）
            -   *Vim* 中函数可以赋值给变量，同样的此时变量需要大写字母开头
        -   也可以作为集合变量的元素、参数传递
        -   `!`：强制覆盖同名函数，否则同名函数报错
        -   `<KEYWARD>` 函数特殊关键字
            -   `range`：定义范围函数，针对范围内每行重复执行操作
            -   `abort`：函数遇到执行错误时退出
    -   函数参数
        -   可定义具名参数、不定长参数（最多 20 个）
        -   参数可以带默认值（可选参数）
            -   调用时可使用 `v:none` 触发用默认值调用
            -   默认值参数必须在必须必选参数（无默认值参数之前）定义
            -   默认值可引用前序参数值（需 `a:`）
        -   函数参数均通过 `a:` 作用域获取
            -   `a:arg1`：具名参数 `arg1`
            -   `a:firstline`、`a:lastline`：起始、终止行号
            -   `a:0`：不定长参数数量
            -   `a:000`：包含不定长参数的列表
            -   `a:1`、`a:000[0]`：不定长参数首个参数值
        -   参数变量不可重新赋值
    -   函数体
        -   函数内变量为局部变量，即默认带 `l:` 作用域
        -   全局变量需要显式带 `g:` 作用域
    -   函数返回值
        -   无显式 `return` 返回值时，默认返回 `0`
    -   函数调用
        -   函数通过 `<RANGE>call Func()` 调用（`call` 现已可选）
            -   未给出范围 `<RANGE>`：函数执行一次
            -   给出函数 `<RANGE>`、函数不接受范围（函数签名有 `range` 标志）：函数对每行单独、分别执行一次
                -   每次执行位置为范围内不同行
                -   但，`a:firstline`、`a:lastline` 起止行号始终为 `<RANGE>` 指定的范围
            -   给出范围 `<RANGE>`、函数接受范围（函数签名有 `range` 标志）：函数范围整体执行一次
        -   在表达式中调用 `let a = Func()`、`echo Func()`

> - 函数：<https://vimscript.haikebang.com/chapters/23.html>
> - *VIM* 学习笔记：脚本-自定义函数：<https://zhuanlan.zhihu.com/p/27353286>
> - *VIM userfunc*：<https://yianwillis.github.io/vimcdoc/doc/userfunc.html>

####    自动载入函数

```vimscript
autocmd FuncUndefined FuncPtn* source /path/to/script.vim       # 定义自动命令事件
call g:filename#FuncName                                        # VIM 自动载入 `autoload` 目录下脚本
```
-   自动载入函数：大量函数需要延迟定义时，可能需要在需使用函数才（执行）定义
    -   定义 `FuncUndefined` 函数未定义自动事件
    -   在 `autoload` 目录中脚本定义函数，*VIM* 将函数调用时自动载入
        -   此方法同样适合载入未定义变量

#  *Ex* 命令

##  *Ex* 命令行

### *Ex* 命令行编辑

| 命令                 | 说明                                             |
|----------------------|--------------------------------------------------|
| `Q`                  | 切换至 `Ex` 模式                                 |
| `<Esc>`              | 放弃命令行                                       |
| `CTRL-V {char}`      | 按本义插入 `{char}`                              |
| `CTRL-V {number}`    | 输入十进制数表示的字符（可达 `3` 个数位）        |
| `CTRL-R {register}`  | 插入指定寄存器的内容                             |
| `<Left>/<Right>`     | 光标左移/右移                                    |
| `<S-Left>/<S-Right>` | 光标左移/右移一个单词                            |
| `CTRL-B/CTRL-E`      | 光标移动至命令行行首/行尾                        |
| `<BS>`               | 删除光标前的字符                                 |
| `<Del>`              | 删除光标下的字符                                 |
| `CTRL-W`             | 删除光标前的单词                                 |
| `CTRL-U`             | 删除所有字符                                     |
| `<Up>/<Down>`        | 搜索以当前命令开始的较早/较晚的命令行            |
| `<S-Up>/<S-Down>`    | 从命令行历史中重放较早/较晚的命令                |
| `CTRL-G`             | `incsearch` 激活时、查找类命令时，跳转至后一匹配 |
| `CTRL-T`             | `incsearch` 激活时、查找类命令时，跳转至前一匹配 |
| `:his[tory]`         | 显示命令行历史                                   |
| `CTRL-D`             | 列出可能匹配的 *Ex* 命令                         |
| `CTRL-A`             | 插入可能匹配的 *Ex* 命令                         |
| `CTRL-L`             | 插入所有匹配最长共同部分                         |
| `CTRL-N`             | 在 `wildchar` 之后存在多个匹配时至后一匹配       |
| `CTRL-P`             | 在 `wildchar` 之后存在多个匹配时至前一匹配       |


> - *VIM options* `insearch`：<https://yianwillis.github.io/vimcdoc/doc/options.html#'incsearch'>

### *Ex* 命令应用范围

| 命令            | 说明                                           |
|-----------------|------------------------------------------------|
| `,`             | 分隔两个行号                                   |
| `;`             | 同上，但在解释第二个行号之前先移动光标至第一个 |
| `{number}`      | 绝对行号                                       |
| `.`             | 当前行                                         |
| `$`             | 文件的末行                                     |
| `%`             | 同 `1,$`（整个文件）                           |
| `*`             | 同 `'<,'>`（可视区域）                         |
| `'t`            | 位置标记 `t` 的位置                            |
| `/{pattern}[/]` | 后一个匹配 `{pattern}` 的行                    |
| `?{pattern}[?]` | 前一个匹配 `{pattern}` 的行                    |
| `+[num]`        | 在前面的行号上加 `[num]`（缺省 `1`）           |
| `-[num]`        | 从前面的行号里减 `[num]`（缺省 `1`）           |

-   *Ex* 命令应用范围：执行 *Ex* 命令时可以选择命令应用的行范围

### *Ex* 特殊字符

| 特殊字符   | 说明                                                 |
|------------|------------------------------------------------------|
| `|`        | 分隔两个命令（不适用于 `global` 及 `!`）             |
| `"`        | 开始注释                                             |
| `%`        | 当前文件名（仅当期望文件名时）                       |
| `#[num]`   | 第 `[num]` 个轮换文件名（仅当期望文件名时）          |
| `<abuf>`   | 缓冲区号，用于自动命令（仅当期望文件名时）           |
| `<afile>`  | 文件名，用于自动命令（仅当期望文件名时）             |
| `<amatch>` | 匹配模式的内容，用于自动命令（仅当期望文件名时）     |
| `<cword>`  | 光标下的单词（仅当期望文件名时）                     |
| `<cWORD>`  | 光标下的字串（仅当期望文件名时）（见 `WORD` ）       |
| `<cfile>`  | 光标下的文件名（仅当期望文件名时）                   |
| `<sfile>`  | `:source` 的文件里该文件的文件名（仅当期望文件名时） |

#TODO
-   说明
    -   在 `%`、`#`、`<cfile>`、`<sfile>` 或 `<afile>` 之后可添加处理后缀获取部分文件名
        | 文件名处理后缀     | 说明                        |
        |--------------------|-----------------------------|
        | `:p`               | 完整路径（`path`）          |
        | `:h`               | 头部（除去文件名 `head`）   |
        | `:t`               | 尾部（仅使用文件名 `tail`） |
        | `:r`               | 根部（除去扩展名 `root`）   |
        | `:e`               | 扩展名（`extension`）       |
        | `:s/{pat}/{repl}/` | 以 `{repl}` 替换 `{pat}`    |

##  执行命令

###  `execute`、`normal`

```vimscript
execute "echom 'hello, world'"<cr>
execute "normal! gg/foo\<cr>dd"
execute "normal! mqA;\<esc>`q"
```

-   `:execute`：把字符串当作 *Ex* 命令执行（命令行输入）
    -   注意事项
        -   `:execute` 命令用于配置文件时，不能忽略结尾的 `<cr>` 换行，以 “执行命令”
    -   代码中应避免使用 `eval` 之类构造可执行字符串
        -   但是 *VimScripts* 代码大部分只接受用户输入，安全不是严重问题
        -   使用 `:execute` 命令能够极大程度上简化命令
-   `:normal`：接受一串 **按键值**，并当作是 *Normal* 模式接受按键
    -   `:normal` 后的按键会执行映射，`:normal!` 忽略所有映射
    -   `:normal` 无法识别形如 `<cr>` 非打印字符
        -   `:normal /foo<cr>`这样的命令并不会搜索，因为 “没有按回车”
        -   需将 `:execute` 和 `:normal` 命令结合使用，让 `:normal` 接受 “按下非打印字符”（`<cr>`、`<esc>`等）

> - *Execute* 命令：<https://vimscript.haikebang.com/chapters/28.html>
> - *Normal* 命令：<https://vimscript.haikebang.com/chapters/29.html>

##  *Ex Misc*

| *Ex* 命令、普通命令 | 说明                                                             |
|---------------------|------------------------------------------------------------------|
| `q:`                | *Ex* 命令历史窗口                                                |
| `|`                 | 管道符，用于行内分隔多个命令 `:echom "bar" | echom "foo"`        |
| `:!{SHELL COMMAND}` | 执行 Shell 命令并暂时跳出 *VIM*                                  |
| `:shell`            | 暂时进入到 Shell 环境中（`$ exit` 即可回到 *Vim*）               |
| `<C-z>`             | 暂时后台挂起 VIM 回到 Shell 环境中（`$fg` 可回到之前挂起的进程） |
| `K`                 | 对当前单词调用 `keywordprg` 设置的外部程序，默认 `man`           |
| `:echo`             | 打印信息，但是信息不会保存                                       |
| `:echom`            | 打印、并保存信息在 `:messages` 可展示的消息队列中                |
| `:messages`         | 查看 `:echom` 保存的信息                                         |
| `g<`                | 查看上个命令、命令输出                                           |

-   说明
    -   `:echom` 打印的字符串中 `\` 转义字符将以 `<HEX>` *ASCII* 码值形式打印
    -   `:echo errmsg`：打印最近一条报错信息

> - *VIM* 学习笔记：信息（message）：<https://zhuanlan.zhihu.com/p/161370897>
