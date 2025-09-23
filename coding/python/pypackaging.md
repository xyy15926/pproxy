---
title: Python Package and Setup
categories:
  - Python
tags:
  - Python
  - Setuptools
  - Pip
date: 2025-01-19 20:13:19
updated: 2025-09-23 20:38:05
toc: true
mathjax: true
description: 
---

## Python 项目开发

###	Python相关环境变量

-   Python相关环境变量
    -   `PYTHONHOME`：标准 Python 库位置
    -	`PYTHONPATH`：模块文件查找路径，格式同 `PATH`
    -   `PYTHONSTARTUP`：该文件中的 Python 命令会在交互模式的首个提示符显示之前被执行

> - 命令行与环境：<https://docs.python.org/zh-cn/3.13/using/cmdline.html#envvar-PYTHONSTARTUP>

### Python 交互式运行

-   自动补全
    ```python
    import readline
    import rlcompleter                          # 为自动补全`rlcompleter`不能省略
    import atexit
    readline.parse_and_bind("tab:complete")     # 绑定`<tab>`为自动补全
    try:
        # 读取上次存储输入历史
        readline.read_history("/path/to/python_history") 
    except:
        pass
    # 将函数注册为退出 Python 环境时执行，将历史输入存储在文件中
    atexit.register(readline.write_history_file, "/path/to/python_history")
    del readline, rlcompleter
    ```
    -   Python3.4 之后交互模式下 `rlcompleter` 会自动激活，否则需要手动执行上述脚本（或配置在 `PYTHONSTARTUP` 中）以生效

> - readline - GUN readline 接口：<https://docs.python.org/zh-cn/3.13/library/readline.html#module-readline>
> - rlcompleter - 用于 GNU readline 的补全函数：<https://docs.python.org/zh-cn/3.13/library/rlcompleter.html>
> - 交互模式：<https://docs.python.org/zh-cn/3.13/tutorial/appendix.html#tut-interac>
> - site - 本地环境钩子：<https://docs.python.org/zh-cn/3.13/library/site.html#module-site>

###  包管理、环境依赖

| 包、应用                  | 说明                                                              |
|---------------------------|-------------------------------------------------------------------|
| `venv`                    | Py3.3 引入标准库，部分发行版中被独立为 `python3-venv`             |
| `pyvenv`                  | Py3.3 引入的脚本，Py3.8 被移除                                    |
| `virtualenv`              | 创建、隔离 Python 环境的 Python 包                                |
| `virtualenvwrapper`       | `virtualenv` 扩展，方便管理、切换虚拟环境                         |
| `pyenv`                   | 隔离不同版本 Python 的 Bash 扩展                                  |
| `pyenv-virtualenv`        | `pyenv` 插件，方便同时使用 `pyenv`、`virtualenv`                  |
| `pyenv-virtualenvwrapper` | 类上                                                              |
| `pip`                     | Python 包管理器，包括安装、打包                                   |
| `conda`                   | 系统级包管理器                                                    |
| `pipenv`                  | 整合 `Pipfile`、`pip`、`virtualenv`，方便开发应用程序（而不是库） |
| `poetry`                  | 整合 `pip`、`virutalenv`、`setuptools`，管理项目全周期            |
| `pixi`                    | 系统级包管理器，基于 *Conda* 生态同时支持 *PyPI* 生态             |
| `hatch`                   | 涵盖项目的依赖管理、环境管理、测试、打包、发布全流程              |
| `uv`                      | Python 包管理、环境管理                                           |

-   说明
    -   `virtualenv` 是最初的 Python 虚拟环境管理工具
        -   后续 `pipenv`、`poetry` 等工具均依赖此包实现环境隔离
        -   通过为不同环境创建目录、修改 `PATH` 环境变量实现环境隔离
    -   `venv` 是 Python3.3 开始引入的标准库，是 `virtualenv` 子集
        -   具备功能较 `virtualenv` 弱
            -   较慢、扩展性弱、无法自主升级
            -   不支持创建低版本 Python 虚拟环境
        -   部分 Python 发行版中 `venv` 被独立为 `python3-venv`
        -   `python -m venv` 即等价于 `pyvenv`
    -   `pyenv` 是 Bash 扩展，用于管理 **系统中不同版本 Python 工具链**
        -   即，`pyenv` 无需 Python 环境，不支持 Windows 平台
        -   `pyenv` 介入 Bash 中 `python` 命令执行链路，将分发 `python` 调用至指定的 Python 工具链
        -   `pyenv-virtualenv` 在 Python3.3 后将尝试 `python -m venv` 而不是 `virtualenv`
    -   `pip` 依赖 *PyPI* 实现 Python 包管理，包括安装、**构建**（前端）
    -   `conda` 是系统级包管理器
        -   `conda` 提供单独的执行环境，可以以独立、用户空间的方式安装软件包
        -   即，`conda` 可在无需 root 权限方式安装软件链，可视为 *Docker* 的轻量级替代
    -   `pixi` 是基于 *Conda* 生态、对标 *Conda* 的系统级包管理器
        -   相较于 *Conda* 侧重项目管理，即为项目配置环境、而不是激活全局环境
            -   默认环境在项目目录目录 `.pixi` 文件夹中（也可更改为集中放置）
            -   无默认全局激活环境（但可配置全局暴露环境）
        -   整合、统一管理 *Conda* 生态、*PyPI* 生态

> - *The difference between venv, pyenv...*：<https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe>
> - *Conda v.s. Pip*：<https://www.reddit.com/r/Python/comments/w564g0/can_anyone_explain_the_differences_of_conda_vs_pip/>
> - *The relationship between virtualenv and pyenv*：<https://stackoverflow.com/questions/29950300/what-is-the-relationship-between-virtualenv-and-pyenv>

###  打包 *Packaging*

| 包、应用     | 说明                                                 |
|--------------|------------------------------------------------------|
| `distutils`  | 标准库，后续所有打包工具的依赖，通过 `setup.py` 配置 |
| `setuptools` | `distutils` 增强版，最常用的打包工具，构建后端       |
| `distribute` | `setuptools` 分支，现已合并回 `setuptools`           |
| `distutils2` | 试图加入标准库失败，已废弃                           |
| `packaging`  | 打包核心模块                                         |
| `build`      | 兼容 *PEP 517* 的构建前端                            |
| `twine`      | 上传包至 *PyPI*                                      |
| `hatch`      | 涵盖项目的依赖管理、环境管理、测试、打包、发布全流程 |

-   *Python* 打包：需要根据项目受众、运行环境选择相应的打包技术
    -   打包 *Python* 库和工具：为开发人员在开发环境中使用的库、工具、基本程序
        -   依据对打包内容、格式分类，打包、分发方式
            -   模块打包、分发：仅适用于仅依赖标准库、兼容环境间少量脚本打包、分发
            -   源码打包、分发：适合包含多个文件纯 Python 包的打包 *sdist*、分发
            -   二进制打包、分发：适合集成其他语言的的包 *wheels*
    -   打包 *Python* 应用程序：完整的、在生产环境中使用的应用程序
        -   根据对目标环境的依赖性（同时也决定打包内容），打包方式
            -   依赖框架
            -   依赖预安装 Python 环境
            -   依赖单独的软件分发生态
            -   自带 Python 可执行文件打包
            -   自带用户空间（操作系统虚拟化、容器化）
            -   自带内核（经典虚拟化）
            -   自带硬件
-   标准构建工具涉及内容、目标
    -   *Source Tree* 源码树
        -   标准构建应支持直接从源码安装（类似 `$ pip install -e <SOME-DIR>`）
        -   *Configuration File* 配置文件（`pip`、`build` 等构建前端所需）
            -   `pyproject.toml`：*PEP 518* 中定义的标准规范
            -   `setup.py`：基于代码的、可直接执行（构建后端）、传统配置方式
    -   *Distribution* 分发内容
        -   *Source Distribution*、*sdists* 源码静态快照
        -   *Wheel*
    -   工具链（角色）：同一工具可能具备多种功能、承担不同角色（如 `pip` 可作为构建前端、集成前端）
        -   *Build Frontend* 构建前端：从源码树、快照构建 *wheels* 的工具
        -   *Build Backend* 构建后端：实际执行构建的工具
        -   *Integration Frontend* 集成前端：环境依赖管理工具，集成包定位、构建、安装功能

> - Python 软件发布生态系统目前由 *Python Packaging Authority* 管理
> - *PEP 517 - A build-system independent format for source trees*：<https://peps.python.org/pep-0517/#terminology-and-goals>
> - *Python 打包指南*：<https://packaging.pythonlang.cn/en/latest/guides/>
> - *Python Packaging Guide*：<https://packaging.python.org/en/latest/guides/>
> - *Conda & PyPI*：<https://pixi.sh/latest/concepts/conda_pypi/>

####    `pyproject.toml`

```toml
# Project infomation.
[project]
name = "pyproj"
version = 0.0.1
authors = [
    { name="author", email="author@example.com" },
]
description = "Description"
readme = "Readme"
requires-python = ">= 3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
]
license = "MIT"
license-files = ["LICEN[CS]E*"],
dependencies = [
    "Request",
]
[project.optional-dependencies]
cli = [
    "rich",
]

[project.urls]
Homepage = "https://example.com"
Issues = "https://example.com/issues"

# Specify build system.
[build-system]
requires = ["setuptools >= 60.0"]
build-backend = "setuptools.build_meta"

# Specifications for tools such as linter, build-backend and etc.
[tool.<MOD>]
```

-   `pyproject.toml`：现代 Python 项目配置文件（代码检查工具等也使用此配置文件）
    -   `[project]` 表：项目基本元信息
        -   构建工具将根据此部分项目元信息设置 Python 包信息
    -   `[build-system]` 表：指定构建后端
    -   `[tool]` 表：项目中涉及的、特定工具相关的配置
        -   包括构建后端配置、代码检查工具配置等
        -   子表名一般即工具包名，子表具体配置项取决于工具自身

> - *Writing your pyproject.toml*：<https://packaging.python.org/en/latest/guides/writing-pyproject-toml/>
> - Python 打包发布：`pyproject.toml` 现代配置指南：<https://zhuanlan.zhihu.com/p/1893271684603159649>

####    `MANIFEST.in`

```conf
# Include files matching pattern.
include data/*.txt
# Exclude files matching pattern.
exclude data/*.rst
# Include the whole directory and keep the file-tree structure.
recursive-include templates *
```

-   `MANIFEST.in` 文件：指示打包、发布 Python 应用程序时应包含的文件、目录
    -   使用 *Setuptools* 工具构建、打包 Python 项目时，工具默认会自动检查、打包所需文件目录
        -   大部分情况下，无需配置 `MANIFEST.in` 文件
        -   仅在需要自定义打包范围，添加、排除特定文件、目录时需要配置 `MANIFEST.in`

> - *Controlling files in the distribution*：<https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html>
> - Python 项目是否需要 `MANIFEST.in` 文件：<https://geek-docs.com/python/python-ask-answer/511_python_do_python_projects_need_a_manifestin_and_what_should_be_in_it.html>

## *Pip*

-   *Pip*：Python 包管理工具，从 *PyPI* 中获取 Python 包
    -   Pip 配置文件：`~/.config/pip/pip.conf`
    -   Pip 依赖管理：通过纯文本文件（一般命名为`requirements.txt`）来记录、管理 Python 项目依赖
        -	`$ pip freeze`：按照 `package_name=version` 的格式输出已安装包
        -   `$ pip install -r`：可按照指定文件（默认`requirements.txt`）安装依赖
    -   Pip 环境信息：无中心化包管理注册文件，而是通过收集 `PYTHONPATH` 路径下 `<XXX>.egg-info` 、`<XXX>.dist-info` 目录中信息获取已安装包信息
        -   `<XXX>.egg-info` 目录：由 `setuptools` 打包时在包开发目录生成
            -   `SOURCES.txt` 作为包文件记录，指导 `pip uninstall` 删除包
            -   `requires.txt` 包依赖
            -   `top_level.txt` 包包含的顶级包名、模块名
        -   `<XXX>.dist-info` 目录：由 `pip` 安装时在安装目录生成
            -   `RECORD` 安装记录：包含当前包所有已安装内容
            -   `METADATA` 记录包基本信息，`pip show` 内容
            -   `REQUESTED` 依赖此包的包
        -   `<XXX>.egg-link` 文件: `pip install -e` 以可编辑模式安装时生成
            -   实际即，包含指向开发目录的文件地址，方便包开发
        -   说明
            -  `LOCATION` 根据 `PYTHONPATH` 中查到的包位置确认，未在 `METADATA` 中
                -   而，`pip` 依赖 `LOCATION` 与 `RECORD` 删除包
                -   故，开发环境路径若在 `PYTHONPYTHON` 中，可能导致 `pip` 无法正常删除包

##  *Setuptools*

-   *Setuptools*：Python 最常用打包与分发工具
    -   *Setuptools* 最开始即是 Python 项目构建后端
        -   现在已扩展功能包括构建、安装、元信息管理等
        -   常被用于构建陈旧的项目，历史包袱重
        -   为此，为确保功能稳定，开发者不应依赖具体实现细节
    -   工具链
        -   构建前端 `build`：
        -   包上传管理 `twine`：
> - *Setuptools User Guide*：<https://setuptools.pypa.io/en/latest/userguide/index.html>
> - *Supported Interface*：<https://setuptools.pypa.io/en/latest/userguide/interfaces.html>

### `setuptools.setup`

-   `setuptools.setup` 函数：打包配置函数
    -   `setup` 函数参数即打包配置：`pyproject.toml` 中 `[project]`、`[tool.setuptools]` 表即来源其参数
        -   `name`：项目名
        -   `packages`：待打包包（子包需要分别独立指定）
        -   `package_dir`：包所在目录（仅在包不位于根目录、或包结构与目录结构不对应时需配置）
    -   传统配置 `setup.py` 文件核心即此函数
        -   带参数执行文件（即调用 `setup` 函数）即可完成打包工作（不再推荐）

> - *Why you shouldn't invoke setup.py directly*：<https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html>

### *Setuptools* 功能、配置

```toml
# Dependency management.
[project]
...
dependencies = [
    "<SOME_DEP>",
    "<SOME_DEP> >= 0.1",
    ...
]

# Entry points.
[project.scritps]
cli-name = "<PKG>.<MOD>:some_func"

# Package discovery.
[tool.setuptools.packages.find]
where = ["."]
include = ["*"]
exclude = []
namespaces = true
```

-   *Setuptools* 功能
    -   Python 版本管理、依赖版本管理、包安装
        -   安装项目时，未安装依赖将通过 *PyPI* 获取、构建、安装
        -   可分别配置必须依赖、可选依赖
        -   通过 `pip install -e <PKG>` 以可编辑方式安装项目
            -   `-e` 选项将在安装包目录创建链接指向项目
            -   项目的更改被整个环境事实感知
    -   项目打包与分发
        -   包、命名空间包发现
            -   对简单项目目录结构，*Setuptools* 能自动发现所有包、命名空间包
                -   `src-layout`：`<PROJECT-ROOT>/src/pkg/.../`
                -   `flat-layout`：`<PROJECT-ROOT>/pkg/.../`
            -   对复杂项目目录结构，可通过配置打包范围
                -   `setuptools.packages` 列表：直接指定所有需打包包
                    -   包结构需要与项目目录结构相匹配
                    -   当，包结构需要与项目目录结构不匹配时可用 `package-dir` 表为各包指定路径
                -   `setuptools.packages.find` 表：配置包发现逻辑
                    -   包发现根路径
                    -   指定待打包的包包名、路径的模式
        -   资源（数据）文件
            -   `setuptools.include-package-data` 置位：打包包内 `MANIFEST.in` 文件指定、或版本控制系统管理的资源文件
                -   由版本控制系统控制范围时，需配置 `setuptools-scm` 等插件
            -   `setuptools.package-data` 表：打包包内名称符合模式的资源
                -   可为各包分别指定资源文件名模式（`*` 表示所有包）
            -   `setuptools.exclude-package-data` 表：排除包内名称符合模式的资源
    -   配置 *Entry Points* 执行入口
        -   项目安装时将为执行入口创建可执行脚本
            -   `[project.scripts]` 表单：定义执行入口
    -   构建 *C/C++* 扩展模块

> - *Setuptools Quickstart*：<https://setuptools.pypa.io/en/latest/userguide/quickstart.html>
> - 构建与发布：<https://pyloong.github.io/pythonic-project-guidelines/guidelines/project_management/distribution/>
> - Python 打包分发工具 `setuptool`：<https://zhuanlan.zhihu.com/p/460233022>
> - Python `setup.py`：<https://zhuanlan.zhihu.com/p/276461821>

##  *Conda*

-   *Conda*：包、环境管理工具

> - *Conda* 入门：<https://docs.conda.org.cn/projects/conda/en/stable/user-guide/getting-started.html>

### *Conda* 配置

-   *Conda* 配置：`~/.condarc`
    -   包源配置
        -   `channels`：实际搜索包源通道，列表
            -   默认仅包含 `defaults`
                -   即，默认包源为下述 `default_channels` 字段中全体
                -   若其被修改，且不包含 `defaults` 项，则 `default_channels` 默认包源中包源不被搜索
            -   非 *URL* 格式通道被解释为 *Anaconda.org* 用户、组织名
                -   即，其之前被添加 `channel_alias` 指定的 *URL* 路径组成完整路径
        -   `default_channels`：默认包源通道，列表
            -   默认指向 `https://repo.anaconda.org/` 库中的多个包源通道
                -   硬编码，在 25.3.0 版本后将被移除
                -   可视为包含多项形如 `https://repo.anaconda.org/XXX` 的列表
            -   可自定义、覆盖
        -   `channel_alias`：通道别名，即 **添加在社区包源通道（非 *URL* 通道）的前缀**
            -   社区包源通道
                -   `conda-forge`
                -   `bioconda`
            -   默认为 `https://conda.anaconda.org/`
            -   用于简化 *Anaconda* 社区包源配置
                -   `channels` 字段中非 *URL* 通道
                -   命令行 `-c`、`--channel` 指定的非 *URL* 通道
        -   `custom_channels`：指定 **特定社区包源通道地址**，字典
            -   其中社区包源通道将被直接指定为配置地址，优先级高于 `channel_alias`
            -   其余未指定的社区（非 *URL* 格式）包源通道依然添加 `channel_alias` 字段作为前缀
        -   包源说明配置
            -   `channels`：优先级最高，真正包源通道配置
            -   `default_channels`：优先级最低，默认包源通道，需要 `defaults` 被加入 `channels` 中生效
            -   `channels_alias`：社区包源通道前缀（镜像站点）
            -   `custom_channels`：特定社区包源镜像地址

> - *Conda Settings*：<https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/settings.html>
> - *Conda Mirror Channels*：<https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/mirroring.html>
> - *Conda* 设置：<https://docs.conda.org.cn/projects/conda/en/stable/user-guide/configuration/settings.html>

##  *Pixi*

-   *Pixi*：*Pixi* 基于 *Conda* 生态系统，且同时支持 *PyPI* 包管理（依赖 `uv` 项目）
    -   功能支持
        -   工作台（项目）环境管理
            -   多环境管理
            -   跨平台环境支持
        -   全局环境管理 `$ pixi global`
        -   任务配置、执行
    -   `pixi.toml`：*Pixi* 默认配置文件
        -   *Pixi* 也支持 `pyproject.toml` 模式的配置文件

> - *Basic Usage of Pixi*：<https://pixi.sh/latest/getting_started/>
> - *Windows* 中使用 *Pixi* 替代 *Conda*：<https://zhuanlan.zhihu.com/p/1891081313118839047>
> - *Pixi Global*：<https://pixi.sh/latest/reference/cli/pixi/global/>

### *Pixi* 配置

-   *Pixi* 配置：`~/.pixi/config.toml` 全局配置、`<PROJ_ROOT>/.pixi/config.toml` 项目配置
    -   *Conda* 包源
        -   `default-channels`：默认包源，列表
            -   默认为 `[ conda-forge ]` 
            -   仅用于初始化工作目录时，项目包源由项目内配置确定
        -   `mirrors`：替换包源 *URL* 地址中匹配前缀部分为镜像站地址，值为列表的字典
            -   *Pixi* 优先匹配最长前缀并替换
            -   镜像站地址应为列表，按顺序尝试作为替换目标
                -   首个镜像地址被用于获取 *repodata*
            -   此配置同样影响 *PyPI* 包源
    -   *PyPI* 包源
        -   `index-url`：默认 *PyPI* 包索引地址
            -   `$ pixi init` 时，将被加入项目的 *manifest* 文件中
            -   全局配置对项目配置不生效
        -   `extra-index-url`：额外的 *PyPI* 包索引地址，列表
            -   `$ pixi init` 时，将被加入项目的 *manifest* 文件中
            -   全局配置对项目配置不生效
        -   `allow-insecure-host`：允许的不安全地址（非 `https`），列表
            -   全局配置对项目配置不生效

> - *The Configuration of Pixi Itself*：<https://pixi.sh/dev/reference/pixi_configuration/>
> - *Pixi Manifest*：<https://pixi.sh/latest/reference/pixi_manifest/>

### 工作台（项目）管理

| 工作台（项目）命令 | 说明                 |
|--------------------|----------------------|
| `pixi init`        | 在当前工作目录初始化 |
| `pixi add`         | 添加依赖             |
| `pixi remove`      | 移除依赖             |
| `pixi update`      | 更新配置文件中依赖   |
| `pixi upgrade`     | 更新依赖项           |
| `pixi lock`        | 创建 lockfile        |
| `pixi info`        | 工作台信息           |
| `pixi shell`       | 启动激活环境 Shell   |
| `pixi list`        | 列出依赖             |
| `pixi tree`        | 列出依赖树           |
| `pixi clean`       | 移除环境             |

####    多特性、环境管理

| `feature.<NAME>` 环境描述字段 | 描述                                   |
|-------------------------------|----------------------------------------|
| `dependecies`                 | 依赖                                   |
| `pypi-dependencies`           | *PyPI* 依赖                            |
| `system-requirements`         | 系统需求                               |
| `activation`                  | 环境激活配置                           |
| `platforms`                   | 平台要求                               |
| `channels`                    | 包源通道，可添加 `priority` 项避免覆盖 |
| `target`                      | 目标平台子项                           |
| `tasks`                       | 任务                                   |


-   *Pixi* 通过配置 `feature`、`environment` 管理多环境
    -   每个 `feature` 可以包含完整的环境描述
        -   默认环境即为 `default`，无前缀环境描述字段可认为是省略 `feature.default` 的缩写
    -   环境 `environment` 可由多个 `feature` 组成
        -   未置位 `no-default-feature` 时，所以环境默认包含 `default` 环境描述
        -   可配置 `solve-group` 以保证多个环境共同解析，确保多个环境依赖一致

> - *Pixi Multi Environment*：<https://pixi.sh/latest/workspace/multi_environment/>
> - 一分钟带你上手 *Pixi* 多环境：<https://zhuanlan.zhihu.com/p/1943035961463243343>

### *Pixi Global*

| `pixi global` 子命令 | 说明                                       |
|----------------------|--------------------------------------------|
| `install`            | 全局安装包至其自身环境                     |
| `uninstall`          | 从全局空间移除环境                         |
| `add`                | 向环境中添加包                             |
| `sync`               | 依全局配置文件 `pixi-global.toml` 同步环境 |
| `edit`               | 编辑全局配置文件                           |
| `update`             | 更新全局环境                               |
| `list`               | 列出全局环境                               |

-   *Pixi Global*：安装、管理全局工具（非全局工具无意义）
    -   *Pixi Global* 将隔离安装全局工具在各自环境
        -   只暴露各环境中必要调用点，可通过 `--with`、`add` 避免暴露
        -   注意，多个环境是 **同时都在生效**
    -   *Pixi Global* 支持从源文件 `--path`、`--git` 安装
    -   配置文件为 `~/.pixi/manifest/pixi-global.toml`
        -   其中可配置 `channels`、`dependencies`、`exposed`

> - *Pixi Global Tools*：<https://pixi.sh/latest/global_tools/introduction/>
> - *Pixi Global Manifes*：<https://pixi.sh/latest/global_tools/manifest/>
