---
title: Python Package and Setup
categories:
  - Python
tags:
  - Python
  - Setuptools
  - Pip
date: 2025-01-19 20:13:19
updated: 2025-02-08 16:33:08
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
    -   `pip` 依赖 *PyPi* 实现 Python 包管理，包括安装、**构建**（前端）
    -   `conda` 是系统级包管理器
        -   `conda` 提供单独的执行环境，可以以独立、用户空间的方式安装软件包
        -   即，`conda` 可在无需 root 权限方式安装软件链，可视为 *Docker* 的轻量级替代

> - *The difference between venv, pyenv...*：<https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe>
> - *Conda v.s. Pip*：<https://www.reddit.com/r/Python/comments/w564g0/can_anyone_explain_the_differences_of_conda_vs_pip/>
> - *The relationship between virtualenv and pyenv*：<https://stackoverflow.com/questions/29950300/what-is-the-relationship-between-virtualenv-and-pyenv>

###  打包 *Packaging*

| 包、应用     | 说明                                                 |
|--------------|------------------------------------------------------|
| `distutils`  | 标准库，后续所有打包工具的依赖，通过 `setup.py` 配置 |
| `setuptools` | `distutils` 增强版，最常用的打包工具                 |
| `distribute` | `setuptools` 分支，现已合并回 `setuptools`           |
| `distutils2` | 试图加入标准库失败，已废弃                           |
| `packaging`  | 打包核心模块                                         |
| `build`      | 兼容 *PEP 517* 的构建前端                            |
| `twine`      | 上传包至 *PyPI*                                      |

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

####    `pyproject.toml`

```toml
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

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool]
# Build backend specific.
```

-   `pyproject.toml`：打包工具所需的配置文件（代码检查工具等也使用此配置文件）
    -   `[project]` 表：包含构建工具所需的项目基本元信息
    -   `[build-system]` 表：指定构建后端
    -   `[tool]` 表：包含与工具相关的子表，具体配置依赖构建工具

> - *Writing your pyproject.toml*：<https://packaging.python.org/en/latest/guides/writing-pyproject-toml/>

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
    -   功能
        -   Python 库的打包与分发：资源文件、数据文件、命名空间包
        -   依赖包安装与版本管理：必须依赖、可选依赖
        -   指定 Python 版本
        -   可执行脚本生成：*Entry Points*
        -   *C/C++* 扩展
    -   `setuptools.setup` 函数：打包配置函数
        -   `setup` 函数参数即打包配置：类似 `pyproject.toml` 中 `[project]`、`[tool.setuptools]` 表
            -   `name`：项目名
            -   `packages`：待打包包（子包需要分别独立指定）
            -   `package_dir`：包所在目录（仅在包不位于根目录、或包结构与目录结构不对应时需配置）
        -   传统配置 `setup.py` 文件核心即此函数
            -   带参数执行文件（即调用 `setup` 函数）即可完成打包工作（不再推荐）
-   `setuptools` 功能
    -   （命名空间）包自动发现：`setuptools` 支持两种常见的项目层次下的包、命名空间包自动发现
        -   `src-layout`：`<PROJECT-ROOT>/src/pkg/.../`
        -   `flat-layout`：`<PROJECT-ROOT>/pkg/.../`

> - *Setuptools User Guide*：<https://setuptools.pypa.io/en/latest/userguide/index.html>
> - 构建与发布：<https://pyloong.github.io/pythonic-project-guidelines/guidelines/project_management/distribution/>
> - *Why you shouldn't invoke setup.py directly*：<https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html>
> - Python 打包分发工具 `setuptool`：<https://zhuanlan.zhihu.com/p/460233022>
> - Python `setup.py`：<https://zhuanlan.zhihu.com/p/276461821>
