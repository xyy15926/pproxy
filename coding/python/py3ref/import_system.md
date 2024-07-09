---
title: Python包、模块
categories:
  - Python
  - Py3Ref
tags:
  - Python
  - Py3Ref
  - Module
  - Package
date: 2019-06-09 16:54:21
updated: 2023-08-02 17:25:22
toc: true
mathjax: true
comments: true
description: Python 包、模块及导入机制
---

##  模块、包

-   `module` 模块对象
    -   Python 只有一种模块对象类型，所有模块（C、Python 包）都属于该类型
    -   所有模块都有其完整限定名称，来自于可发起导入机制的语句
        -   模块（包）的完整限定名称被用于搜索各阶段
        -   子包名与其父包名以 `.` 分隔，同 Python 标准属性访问语法
            -   限定名称为带点号路径时，尝试依次导入路径中各父模块
            -   导入中任意模块失败均 `raise ModuleNotFoundError`
    -   模块首次被导入时
        -   Python 搜索模块
            -   找到就创建 `module` 对象并初始化
            -   若指定名称模块未找到，则 `raise ModuleNotFoundError`
        -   发起调用导入机制时，Python 有多种策略搜索指定名称模块

### *Packages*

-   包：为帮助组织模块、并提供名称层次结构而引入
    -   可将包视为文件系统中目录、模块视为目录中文件
        -   但包、模块不是必须来自文件系统
    -   类似文件系统，包通过层次结构进行组织：包内包括模块、子包
    -   所有包都是模块，但并非所有模块都是包
        -   包是一种特殊的模块
        -   特别的，任何具有 `__path__` 属性的模块都被当作包

-   *Regular Packages* 正规包：通常以包含 `__init__.py` 文件的目录形式出现
    -   `__init__.py` 文件可以包含和其他模块中包含 Python 模块相似的代码
    -   正规包被导入时
        -   `__init__.py` 文件会隐式被执行，其中定义对象被绑定到该包命名空间中名称
        -   Python 会为模块添加额外属性

-   *Namespace Packages* 命名空间包：将分散的、多个同名包（目录）组织为一个包
    -   其不一定直接对应到文件系统对象，可能是无实体表示的虚拟模块
    -   可由多个部分（同名目录）构成，每个部分提供不同子模块、子包
        -   包各部分可以物理不相邻
        -   可能处于文件系统不同位置
        -   可能处于 Zip 文件、网络上，或在导入期间其他可搜索位置
    -   其 `__path__` 属性不是普通列表，而是定制的可迭代类型
        -   若父包、或最高层级包 `sys.path` 路径发生改变，对象会在包内的下次导入尝试时，自动执行新的对包部分的搜索
    -   命名空间包中没有 `__init__.py` 文件
        -   也即，命名空间包本身除提供命名空间外无法造成其他影响
        -   事实上，无 `__init__.py` 目录即可视为命名空间包

```conf
// 若 `dir1`、`dir2` 均在导入路径中，则 `foo` 即可导入为命名空间包
|-- dir1/
|   |-- foo/
|       |-- mod1.py
|
|-- dir2/
|   |-- foo/
|       |-- mod2.py
```

> - 利用命名空间包导入目录分散的代码：<https://python3-cookbook.readthedocs.io/zh_CN/latest/c10/p05_separate_directories_import_by_namespace.html>

###  导入机制

-   触发导入机制 3 种方式
    -   `import` 语句：包含命名模块搜索（加载）、绑定包至局部作用域两个操作
        -   搜索（加载）操作：调用内置 `__import__` 函数
            -   内建 `__import__` 函数可被覆盖，此时 `import` 关键字由其他路径强制实现类似 `__import__`
        -   绑定操作：绑定 `__import__` 返回结果、返回结果属性
    -   `__import__(name[,globals,locals,fromlist,level])` 函数：包含模块搜索、模块创建两个操作
        -   包含一些副作用
            -   导入父模块
            -   更新 `sys.modules` 等缓存
        -   参数说明
            -   `globals`：导入上下文
            -   `locals`：未使用
            -   `fromlist`：`from M import ...` 中导入列表
                -   其中元素可不必存在
                -   但，仅 `fromlist` 为子模块名称时有价值，此时会导入子模块并 **绑定为父模块属性**
                -   但，`name` 为 `P.C` 格式导入子模块时，`fromlist` 为空将导入、返回父模块 `P`，非空才导入、返回 `P.C`
            -   `level`：`0` 绝对导入、正值相对导入向上搜索层级
    -   `importlib.import_module()` 等其他导入机制
        -   可能绕过 `__import__` 函数，使用自身解决方案

> - Python Import System：<https://zhuanlan.zhihu.com/p/348559778>
> - Python Import System 流程原理：<https://zhuanlan.zhihu.com/p/356081029>
> - Python Reference 导入系统：<https://docs.python.org/zh-cn/3/reference/import.html>
> - Python Library `importlib` 提供与导入系统交互的 *API*：<https://docs.python.org/zh-cn/3/library/importlib.html>
> - Python `__import__ fromlist`：<https://www.imooc.com/article/289030>

#### *Import Protocol*

-   导入协议
    -   *Finder*：查找器，确定能否使用所知策略找到指定名称模块
        -   不真正加载模块，仅返回模块规格说明供后续导入机制使用
    -   *Loader*：加载器，加载找到的指定模块
    -   *Importer*：导入器，同时实现两种查找器、加载器的对象
        -   在确定能加载所需模块时会返回自身
        -   导入器的 `find_spec()` 返回的模块规格说明中加载器 `loader` 即为自身 `self`

##  模块搜索

-   搜索流程
    -   检查 `sys.modules` 模块缓存
        -   若存在需要导入模块，则导入完成
        -   若模块限定名称对应值为 `None` 则 `raise ModuleNotFoundError`
    -   按顺序调用 `sys.meta_path` 中元路径查找器 `find_spec` 方法获取模块规格说明
        -   若 `sys.meta_path` 处理到列表末尾仍未返回说明对象，则 `raise ModuleNotFoundError`
        -   导入过程中引发的任何异常直接向上传播，并放弃导入过程
        -   对非最高层级模块的导入请求可能会多次遍历元路径
    -   加载器将利用查找器返回的模块规格说明加载模块

### `sys.modules` 模块缓存

-   `sys.modules` 映射：缓存之前导入的所有模块
    -   其中每个键值对就是限定名称、模块对象
    -   缓存包括中间路径，即导入子模块会注册父模块条目
    -   映射可写、可删除其中键值对
    -   说明
        -   删除键值对不影响已导入模块内容
            -   但下次导入时需重新搜索、加载、注册，且与上次保留内容引用不同
            -   `importlib.reload` 将重用相同模块对象，仅通过重新运行模块代码重新初始化模块内容

### 元路径查找器

-   `find_spec(fullname[,path,target])` 方法：元路径查找器需实现此方法
    -   返回模块规格说明
    -   参数说明
        -   `fullname`：被导入模块的完整限定名称
        -   `path`：供模块搜索使用的路径条目
            -   对最高层级模块应为 `None`，缺省使用 `sys.path`
            -   对子模块、子包应为父包 `__path__` 属性值，若相应 `__path__` 属性无法访问将 `raise ModuleNotFoundError`
            -   部分元路径查找器仅支持顶级导入，`path` 参数不为 `None` 时总返回 `None`
        -   `target`：将被作为稍后加载目标的现有模块对象
            -   导入系统仅在重加载期间传入目标模块

> - Python3.4 前 `find_module` 方法被用于查找模块

####    `sys.meta_path`

-   `sys.meta_path` 列表：存储元路径查找器
    -   其中元路径查找器按顺序被调用以确认是否可用于处理指定命名模块，缺省包含
        -   `_frozen_importlib.BuiltinImporter`：定位、导入内置模块
        -   `_frozen_importlib.FrozenImporter`：定位、导入冻结模块
        -   `_frozen_importlib.PathFinder`：定位、导入来自 *Import Path* 中模块
    -   说明
        -   元路径查找器需实现 `find_spec(fullname[,path,target])` 方法
        -   缺省，内置模块、冻结模块导入器优先级较高（靠前），所以解释器首先搜索内置模块
        -   单个导入请求可能会多次遍历元路径查找器
        -   元路径查找器可使用任何策略确定其是否能处理（模块）限定名称
            -   若可处理模块名称则返回模块规格说明，否则返回 `None`

### *Path Based Finder*

-   `PathFinder` 即为 *Path Based Finder* 基于路径的（元）查找器
    -   在导入路径中查找命名模块，并返回模块规格说明
        -   `PathFinder` 不直接在导入路径中搜索模块
        -   而是，根据导入路径条目类型，将各路径条目委托、关联至给路径条目查找器
    -   模块查找流程
        -   按顺序遍历导入路径中路径条目
        -   对各路径条目查找 `sys.path_impporter_cache` 缓存的路径条目查找器
        -   若未在缓存中找到，则迭代调用 `sys.path_hooks` 中钩子函数并缓存
        -   `raise ImportError` 表示无法处理当前路径条目，将被忽略并继续迭代路径条目
    -   有如下路径条目查找器（对应下述两个钩子函数）
        -   `FileFinder`：可处理 `.py`、`.pyc`、`.so` 文件类型的模块
        -   `zipimport.zipimporter`：*Zip* 封装的上述文件类型

####    `sys.path_hooks`、`sys.path_importer_cache`

-   `PathFinder` 元路径查找器中
    -   `sys.path_hooks`：存储根据路径条目确定路径条目查找器的钩子
        -   `PathFinder` 按顺序遍历其中钩子，确定合适的路径查找器
            -   导入路径作为入参
            -   返回值即绑定特定路径的路径条目查找器，或 `raise ImportError` 被忽略
        -   缺省包括
            -   `zipimport.zipimporter`
            -   `FileFinder.path_hook.<locals>.path_hook_for_FileFinder`
    -   `sys.path_importer_cache`：缓存导入路径条目、路径查找器映射
        -   减少查找路径条目对应路径条目查找器的开销
        -   可从中移除缓存条目，以强制基于路径查找器执行路径条目搜索

####    *Import Path*

-   *Import Path* 导入路径：文件系统路径、Zip 文件等路径条目组成的位置列表
    -   其中元素不局限于文件系统位置，可扩展为字符串指定的任意可定位资源
        -   *URL* 指定资源
        -   数据库查询
    -   位置条目来源通常为 `sys.path`
        -   对次级包可能来自上级包的 `__path__` 属性
    -   其中每个路径条目指定一个用于搜索模块的位置
        -   *Path Based Finder* 将在其中查找导入目标

-   `sys.path`：模块、包搜索位置的字符串列表
    -   初始化自 `PYTHONPATH` 环境变量、特定安装和实现的默认设置、执行脚本目录（或当前目录）
    -   只能出现字符串、字节串，其他数据类型被忽略
        -   字节串条目使用的编码由导入路径钩子、路径查找器确定
    -   其中条目可以指定文件系统中目录、*Zip* 文件、可用于搜索模块的潜在位置
    -   当前工作目录在 `sys.path` 为空字符串表示，与其他路径条目处理方式不同
        -   若当前工作目录不存在，则 `sys.path_importer_cache` 中不存放任何值
        -   每次模块搜索时，当前工作目录总被重新确认路径条目查找器
        -   `sys.path_importer_cache` 中键、 `importlib.machinery.PathFinder.find_spec()` 返回路径为实际路径而非空字符串

###    *Path Entry Finder*

-   路径条目查找器：负责路径条目指定位置的模块实际搜索
    -   路径条目查找器需实现 *Path Entry Finder Protocol*，即 `find_spec` 方法
    -   说明
        -   基于路径的查找器是元路径查找器，作用于导入过程的开始、遍历 `sys.meta_path` 时启动
        -   路径条目查找器某种意义上是基于路径查找器的实现细节

####    路径条目查找协议

-   `find_spec(fullname[,target])` 方法即路径条目查找器协议
    -   参数说明
        -   `fullname`：要导入模块的完整限定名称
        -   `target`：目标模块
    -   返回值：完全填充好的模块规格说明
        -   模块规格说明总是包含加载器集合

> - Python3.4 后，`find_spec` 方法替代了 `find_loader`、`find_module` 方法，后二者仅为保持兼容性会在 `find_spec` 未定义时被调用

##  模块加载

-   加载流程
    -   创建模块对象
        -   `loader.create_module` 方法（若有）：接受模块规格说明，在加载期间创建模块对象
        -   `types.ModuleType` 自行创建模块对象
    -   设置模块导入相关属性
    -   在 `sys.modules` 中注册模块
        -   在加载器执行代码前注册，避免模块代码导入自身导致无限递归、多次加载
        -   若模块为命名空间包，直接注册空模块对象
        -   加载失败模块将从 `sys.modules` 中移除，作为附带影响被成功加载的模块仍然保留
            -   重新加载模块会保留加载失败模块最近成功版本
    -   模块规格说明中加载器负责模块执行，填充模块命名空间

```python
module = None
if spec.loader is not None and hasattr(spec.loader, 'create_module'):
    # 模块说明中包含加载器，使用加载器创建模块
    module = spec.loader.create_module(spec)
if module is None:
    # 否则创建空模块
    module = types.ModuleType(spec.name)
 # 设置模块导入相关属性
_init_module_attrs(spec, module)

if spec.loader is None:
    # 模块说明中不包含加载器
    raise ImportError
 # 检查模块是否为为命名空间包
if spec.origin is None and spec.submodule_search_locations is not None:
    sys.modules[spec.name] = module
elif not hasattr(spec.loader, "exec_module"):
    # 向下兼容现有`load_module`
    module = spec.loader.load_module(spec.name)
else:
    sys.modules[spec.name] = module
    try:
        # 模块执行
        spec.loader.exec_module(module)
    except BaseException:
        try:
            # 加载模块失败则从`sys.modules`中移除
            del sys.modules[spec.name]
        except KeyError:
            pass
        raise
return sys.modules[spec.name]
```

### 模块规格说明

|模块规格说明属性|模块属性|描述|
|-----|-----|-----|
|`name`|`__name__`|完整限定名称|
|`loader`|`__loader__`|加载器|
|`orgin`|`__file__`|加载源|
|`submodule_search_locations`|`__path__`|子模块搜索路径|
|`loader_state`| |加载模块所需的额外数据|
|`cached`|`__cached__`|模块缓存源|
|`parent`|`__package__`|父模块名|
|`has_location`| |`origin` 是否为可加载地址|

-   `_frozen_importlib.ModuleSpec` 模块规格说明：用于封装导入相关信息
    -   用于在导入系统各组件之间传递状态
        -   其属性由查找器负责填充
        -   各属性与模块属性存在对应关系（事实上即模块属性来源，但二者不会同步更新）
    -   方便导入机制执行加载的样板操作

> - `ModuleSpec`：<https://docs.python.org/zh-cn/3/library/importlib.html#importlib.machinery.ModuleSpec>

### *Loader*

-   模块加载器负责模块执行
    -   加载器应满足
        -   若模块是 Python 模块（非内置、非动态加载），加载器应该在模块全局命名空间 `module.__dict__` 中执行模块代码
        -   若加载器无法执行指定模块，则应 `raise ImportError`
    -   `create_module`：可选方法，在加载期间创建模块对象
        -   接受模块规格说明作参数，返回供加载期间使用的新模块对象
        -   不在模块对象上设置属性
        -   若返回 `None`、未实现，则导入机制将调用 `types.ModuleType` 自行创建模块对象
    -   `exec_module`：可选方法，执行模块对象
        -   `loader.exec_module` 不定返回传入（加载过程中创建）模块
            -   其返回值将被忽略，`importlib` 避免直接使用返回值
            -   而是通过在 `sys.modules` 中查找模块名称获取模块对象，可能会间接导致被导入模块可能在 `sys.modules` 中替换其自身

> - Python3.4 前，由 `load_module` 方法负责模块加载
> - Python3.6 后，加载器中定义 `exce_module` 同时未定义 `create_module` 将 `raise ImportError`
> - `importlib.abc` 中包含用于导入的抽象基类 `import.abc.Finder`、`import.abc.Loader` 等，但外部

#### 缓存字节码失效

-   Python 从 `.pyc` 文件加载缓存字节码前会检查其是否最新
    -   缺省通过比较文件元数据确定缓存有效性
        -   在 `.pyc` 缓存文件中保存源文件修改时间戳、大小
    -   支持基于哈希的缓冲文件
        -   即在 `.pyc` 文件中保存源文件哈希值
        -   检查型：求源文件哈希值再和缓存文件中哈希值比较
        -   非检查型：只要缓存文件存在就直接认为缓存文件有效

> - 可通过 `--check-hash-based-pycs` 命名行选项配置基于哈希的缓存文件检查行为

### 子模块

-   任意机制加载子模块时，父模块命名空间中会添加对子模块对象的绑定
    -   任意机制包括
        -   `importlib` 模块
        -   `import`、`import...from...` 语句
        -   `__import__` 函数
    -   即，若 `sys.modules` 中 `P`、`P.C` 同时存在，则 `sys.modules['P.C']` 必为 `sys.modules['P'].C`
        -   即使，仅在父模块中相对导入子模块：从未显式绑定属性
        -   即使，仅导入子模块内容、父模块：从未导入子模块

```python
 # spam/__init__.py
from .foo import Foo

 # other scripts
import spam
import sys
spam.foo is sys['spam.foo']
```

### 模块属性

-   导入机制在加载期间（加载器执行模块前）会根据模块规格说明填充模块下述属性
    -   `__name__`：模块完整限定名称，唯一标识模块
    -   `__loader__`：导入系统加载模块时使用的加载器对象
        -   主要用于内省
        -   也可用于额外加载器专用功能
    -   `__package__`：取代 `__name__` 用于主模块计算显式相对导入
        -   模块为包：应设置为 `__name__`
        -   模块非包：最高层级模块应设为空字符串，否则为父包名
        -   预期同 `__spec__.parent` 值相同，未定义时，以 `__spec__.parent` 作为回退项
    -   `__spec__`：导入模块时要使用的模块规格说明
        -   解释器启动时的初始化模块亦需设置此属性（仅 `__main__` 模块有时置 `None`）
    -   `__path__`：父模块 `__path__` 被作为搜索子模块的位置列表
        -   具有 `__path__` 属性模块即为包
            -   包模块必须设置 `__path__` 属性，非包模块不应设置
            -   可在包的 `__init__.py` 中设置、更改
        -   迭代时应返回字符串（若无用处可置空）
            -   在导入机制内部功能同 `sys.path`，需满足作用于 `sys.path` 的规则
            -   迭代结果被用于调用 `sys.path_hooks` 用于搜索模块
    -   `__repr__`：模块字符串代表
        -   缺省使用 `__spec__` 属性中信息生成
    -   `__file__`：模块对应的被加载文件的路径名字符串
        -   为可选项，可在其无语法意义时不设置
        -   对从共享库动态加载的扩展模块，应为共享库文件路径名
    -   `__cached__`：模块编译版本代码（字节码文件）路径
        -   不要求编译文件已经存在，可以表示**应该存放**编译文件的位置
        -   不要求 `__file__` 已经设置
            -   有时加载器可以从缓存加载模块但是无法从文件加载
            -   加载静态链接至解释器内部的C模块

> - *PEP 420* 前，在 `__init__.py` 中设置 `__path__` 属性是实现命名空间包的典型方式，之后导入机制会自动为命名空间包正确设置的 `__path__` 属性

##  其他导入事项

### 相对导入

-   相对导入使用模块 `__name__`（`__package__`）属性确定模块在包层次中位置
    -   若 `__name__` 中不包含包信息（`__package__=None`），相对引用认为该模块为顶级模块（而不考虑实际文件系统中位置）

###    脚本执行

-   Python 脚本执行
    -   `$ python <PYSCRIPT>` 直接执行脚本时
        -   `__name__`被设置为 `__main__`、`__package__` 设置为 `None`，此时导入器无法解释相对导入中 `.`，相对导入报错
    -   `python -m <PYSCRIPT>` 则会按模块逻辑设置 `__name__`、`__package__`，相对导入可以正常执行

### `__main__` 模块

-   `__main__` 模块在解释器启动时直接初始化
    -   类似`sys`、`builtins`，但是不被归类为内置模块
        -   其初始化的方式取决于启动解释器的旗标（命令行参数）
    -   `__main__.__spec__` 设置依赖 `__main__` 初始化的方式
        -   Python 附加 `-m` 选项启动时
            -   `__spec__` 被设置为相应模块、包规格说明
            -   `__main__` 模块作为执行某个目录、Zip 文件、其他 `sys.path` 条目的一部分加载时，`__spec__` 也被填充
        -   其余情况 `__spec__` 被设置为 `None`
            -   交互型提示
            -   `-c` 选项
            -   从 *stdin* 运行
            -   从源码、字节码文件运行
    -   说明
        -   `__main__.__spec__` 大部分场合被置 `None`，是因为此时填充 `__main__` 的代码与作为可导入模块时不直接对应
        -   `__main__`（即使 `__main__.__spec__` 被设置）与对应的可导入模块被视为不同模块

> - `-m` 执行模块时 `sys.path` 首个值为空字符串，而直接执行脚本时首个值为脚本所在目录

###  导入钩子

-   导入机制中设计有两个钩子（位点）以提供扩展性
    -   *Meta Hooks* 元（路径）钩子：向 `sys.meta_path` 添加查找器
        -   在检查 `sys.modules` 缓存后即被调用
        -   可用于重载 `sys.path` 过程、冻结模块甚至内置模块
    -   *Import Hooks* 导入（路径）钩子：向 `sys.path_hooks` 添加可调用对象
        -   `_frozen_importlib_external.PathFinder` 机制中调用
        -   在处理 `sys.path`、`package.__path__` 中对应路径项目时被调用

#### 导入路径修改

-   `sys.path` 可通过以下方式修改
    -   运行时修改 `sys.path` 列表：`sys.path.insert(0, /path/to/modules)`
        -   仅对当前进程生效
    -   配置 `PYTHONPATH` 环境变量：`export PYTHONPATH=/path/to/modules`
        -   会改变所有 Python 应用的搜索路径
    -   配置导入路径中 `.pth` 文件
        -   在遍历已知库文件目录过程中，遇到 `.pth` 文件会将其中路径加入 `sys.path` 中
        -   文件内容：`/path/to/python/site-packages`

```conf
 # extras.pth
/path/to/fold/contains/module
```

