---
title: SQLAlchemy 常用基础
categories:
  - Python
  - DataBase
tags:
  - Python
  - DataBase
  - Briefs
  - SQL
date: 2023-03-10 14:35:03
updated: 2023-12-31 22:24:18
toc: true
mathjax: true
description: 
---

##  SQLAlchemy 概述

-   *SQLAlchemy* 由两组 *API* 组成
    -   *Core*：基础架构，管理数据连接、查询、结果交互、*SQL* 编程
        -   记录 *SQL* 引擎、*DBAPI* 集成、事务继承、模式描述服务
        -   提供以模式为中心的使用模式
    -   *ORM*：基于 *CORE* ，提供对象关系映射能力
        -   允许 *类-表* 映射、绘画
        -   扩展 *Core* 级别的 *SQL* 表达式语言，允许通过对象组合调用 *SQL* 查询

##  *Core* 引擎、连接

| `Engine` 方法                                     | 描述                     |
|---------------------------------------------------|--------------------------|
| `Engine.connect([close_with_result])`             | 创建 *边走边做* 风格连接 |
| `Engine.begin([close_with_result])`               | 创建 *开始一次* 风格连接 |
| `Engine.execute(statement,*multiparams,**params)` | 绑定参数、执行语句       |

-   *Engine* 引擎：连接到特定数据库的中心源
    -   是数据库、其 *DBAPI*（驱动）、连接池、数据库方言的封装
        -   连接池 *Pool* 即根据数据库地址建立连接
        -   *SQLAlchemy* 通过对应驱动支持常用数据库后端方言 *DBAPI*
    -   一般的，特定数据库实例仅需创建一个全局对象
        -   单个引擎可管理多个进程连接，并可并发调用
    -   引擎、连接是 *Core* 风格 *API* 的核心
        -   提供访问 `Connection` 执行 SQL 语句，直接用于与数据库交互
        -   *ORM* 风格中则被传递给 `Session` 对象进一步封装

| 方言（数据库） | 驱动                 |
|----------------|----------------------|
| `postgrqsql`   | `psycopg2`、`pg8000` |
| `mysql`        | `mysqldb`、`pymysql` |
| `oracle`       | `cx_oracle`          |
| `mssql`        | `pyodbc`、`pymssql`  |
| `sqlite`       |                      |


> - 创建 *Engine*：<https://www.osgeo.cn/sqlalchemy/tutorial/engine.html>
> - *Establishing Connectivity*：<https://docs.sqlalchemy.org/en/14/tutorial/engine.html>

### *Engine* 配置

| `Engine` 相关 *API*                                     | 描述                  |
|---------------------------------------------------------|-----------------------|
| `sa.create_engine(url, **kwargs)`                       | 创建引擎              |
| `sa.create_mock_engine(url,executor,**kw)`              | 创建回显 *DDL* 假引擎 |
| `sa.engine_from_config(configuration[,prefi],**kwargs)` | 根据配置创建引擎      |
| `sa.make_url(name_or_url)`                              | 生成 URL 实例         |
| `URL`                                                   | 连接数据库 URL 组件   |

-   说明
    -   部分参数说明
        -   `url`：数据地址，类似普通地址
            -   其中特殊字符需要 `urllib.parse.quote_plus` 编码才可正常解析
        -   `echo`：启用 *SQl* 记录，写入标准输出
        -   `future`：*SQLAlchemy 2.0* 风格标志

```python
from sqlalchemy import create_engine
engine = create_engine(
    url="{DIALECT}[+{DRIVER}]://USER:PWD@HOST/DBNAME[?KEY=VAL...]",
    echo=True, future=True)
```

### *Transaction and DBAPI*

```python
with engine.connect() as conn:
    result = conn.execute(text("select * from tbl"))
    conn.commit()
```

| `Connection` 方法      | 描述     |
|------------------------|----------|
| `Connection.commit()`  | 提交事务 |
| `Connection.execute()` | 执行语句 |

-   `sqlalchemy.engine.base.Connection` 连接：表示针对数据库的开放资源
    -   可通过 `with` 语句将对象的使用范围限制在特定上下文中
        -   *边走边做* 风格
            -   `with` 执行块中包含多个事务，手动调用 `Connect.commmit()` 方法提交事务
            -   事务不会自动提交，上下文环境释放时会自动 `ROLLBACK` 回滚事务
        -   *开始一次* 风格
            -   `with` 执行块整体时视为单独事务
            -   上下文环境释放签自动 `COMMIT` 提交事务

-   `sqlalchemy.engine.cursor.Result` 结果：表示语句执行结果的可迭代对象
    -   说明
        -   `CursorResult.inserted_primary_key` 内部实现透明，根据数据库方言自动选择
            -   `CursorResult.lastrowid`
            -   数据库函数
            -   `INSERT...RETURNING` 语法

| `Result` 方法                       | 描述                                               |
|-------------------------------------|----------------------------------------------------|
| `CursorResult.all()`                | 获取所有行结果                                     |
| `CursorResult.fetchall()`           | 执行语句                                           |
| `CursorResult.mappings()`           | 生成 `MappingResult` 对象，可迭代得到 `RowMapping` |
| `CursorResult.inserted_primary_key` | 插入数据主键值                                     |
| `CursorResult.lastrowid`            |                                                    |

-   `sqlalchemy.engine.row.Row` 行：语句执行结果行
    -   行为类似 `namedtuples`，支持多种访问方式
        -   元组赋值
        -   整数索引
        -   属性（*SQL* 语句中字段名）

##  *Core* 语句、表达式

-   *SQLAlchemy* 表达式语言提供了使用 Python 构造表示关系型数据库结构、表达式的系统
    -   特点
        -   尽可能接近底层数据库构造
        -   同时提供对各种数据库后端的实现差异的少量抽象
        -   即，可编写后端无关 *SQL* 表达式，但是不强制使用
    -   `sa.sql` 模块内容
        -   `sql.expression` 中定义表达式，其中构造器、修饰器均在已在顶级模块、`sa.sql` 中导入
        -   `sql.functions` 中定义 *SQL* 函数，仅 `func` 默认导入顶级模块

> - *SQLALchemy 1.4 Core* 语句和表达式：<https://www.osgeo.cn/sqlalchemy/core/expression_api.html>

### *Column Elements* 列元素和表达式

> - *SQLALchemy 1.4 Core* 列元素和表达式：<https://www.osgeo.cn/sqlalchemy/core/sqlelement.html>

####    列元素基础构造函数

| 列元素基础构造器         | *SQL* 关键字                   | 描述     | 对应 *Python* 算符 |
|--------------------------|--------------------------------|----------|--------------------|
| `and_(*clauses)`         | `AND`                          | 与       | `&`                |
| `or_(*clauses)`          | `OR`                           | 或       | `|`                |
| `not_(clause)`           | `NOT`                          | 非       | `!`                |
| `null()`                 | `NULL`                         | 空       | `is None`          |
| `distinct(expr)`         | `DISTINCT`                     | 唯一值   |                    |
| `case(*whens, **kw)`     | `CASE`                         | 匹配     |                    |
| `cast(expression,type_)` | `CAST`                         | 类型转换 |                    |
| `false()`                | `false`、`0 = 1`（不支持场合） | 否       |                    |
| `true()`                 | `true`、`1 = 1`（不支持场合）  | 是       |                    |

| 函数                                                             | 描述                                |
|------------------------------------------------------------------|-------------------------------------|
| `extract(field,expr,**kwargs)`                                   | `EXTRACT`                           |
| `func.<ATTR>`                                                    | 基于属性名称 `ATTR` 生成 *SQL* 函数 |
| `column(text[,type_,is_literal,_selectable])`                    | 列对象，生成 `ColumnClause` 对象    |
| `bindparam(key[,value,type_,unique,...])`                        | 生成绑定表达式                      |
| `custom_op(opstring[,precedence,...])`                           | 自定义运算符                        |
| `lambda_stmt(lmb[,enable_tracking,track_closure_variables,...])` |                                     |
| `literal(value[,type])`                                          |                                     |
| `literal_column(text[,type_])`                                   |                                     |
| `outparam(key[,type_])`                                          |                                     |
| `quoted_name`                                                    |                                     |
| `text(text)`                                                     | *SQL* 文本串                        |
| `tuple_(*clauses,**kw)`                                          |                                     |
| `type_coerce(expression,type_)`                                  |                                     |

-   说明
    -   `and_`、`or_` 对应 `&`、`|` 操作符
    -   绑定表达式为 *SQL* 表示式中形如 `:` 占位符
        -   表达式执行前绑定具体实参
        -   除 `bindparam` 显式创建外，传递给 *SQL* 表达式的所有 *Python* 文本值都以此方式设置

####    列元素修饰构造函数

| 列元素修饰构造器                                    | 描述 |
|-----------------------------------------------------|------|
| `all_(expr)`                                        |      |
| `any_(expr)`                                        |      |
| `asc(column)`                                       |      |
| `between(expr,lower_bound,upper_bound[,symmetric])` |      |
| `collate(expression,collation)`                     |      |
| `desc(column)`                                      |      |
| `funcfilter(func,*criterion)`                       |      |
| `label(name,element[,type_])`                       |      |
| `nulls_first(column)`                               |      |
| `nulls_last(column)`                                |      |
| `over(element[,partiion_by,order_by,range_,...])`   |      |
| `within_group(element,*order_by)`                   |      |

####    列元素类

| 列元素类                                      | 含义                                       | 基类                               |
|-----------------------------------------------|--------------------------------------------|------------------------------------|
| `Operators`                                   | 比较、逻辑操作符基类                       |                                    |
| `ColumnOperators`                             | 用于 `ColumnElement` 的操作符              | `Operators`                        |
| `ClauseElement`                               | *SQL* 表达式元素基类                       |                                    |
| `ColumnClause`                                | 字符串创建的列表达式，模拟 `Column` 类     |                                    |
| `ClauseList`                                  | 操作符分割的多个子句                       | `ClauseElement`                    |
| `ColumnElement`                               | *SQL* 中用于列、文                         | `ColumnOperators`、`ClauseElement` |
| `ColumnCollection`                            | `ColumnElement` 实例集合                   |                                    |
| `BinaryExpression(left,right,operator[,...])` | `LEFT <OP> RIGHT` 表达式                   | `ColumnElement`                    |
| `Case`                                        | `CASE` 表达式                              | `ColumnElement`                    |
| `Cast(expression,type_`)                      | `CAST` 表达式                              | `ColumnElement`                    |
| `Extract`                                     | `EXTRACT` 子句、`extract(field FROM expr)` | `ColumnElement`                    |
| `FunctionFilter(func,*criterion)`             | 函数 `FILTER` 子句                         | `ColumnElement`                    |
| `Label`                                       | 列标签 `AS`                                | `ColumnElement`                    |
| `Over`                                        | `Over` 子句                                | `ColumnElement`                    |
| `NULL`                                        | `NULL` 关键字                              | `ColumnElement`                    |
| `False_`                                      | `FALSE` 关键字                             | `ColumnElement`                    |
| `True`                                        | `TRUE` 关键字                              | `ColumnElement`                    |
| `Tuple`                                       | *SQL* 元组                                 | `ColumnElement`                    |
| `TextClause`                                  | *SQL* 文本字面值                           | `ColumnElement`                    |
| `TypeCoerce`                                  | *Python* 侧类型强转包装器                  | `ColumnElement`                    |
| `UnaryExpression`                             | 一元操作符表达式                           | `ColumnElement`                    |
| `WithinGroup(element,*order_by)`              | 组内聚合子句                               | `ColumnElement`                    |
| `WrapsColumnExpression`                       | 具名 `ColumnElement` 包装器                | `ColumnElement`                    |
| `BindParameter(key[,value,...])`              | 绑定表达式                                 | `ColumnElement`                    |

-   说明
    -   列元素类一般由列元素基础构造器、列元素修饰构造器生成，不直接实例化

### *Operators* 运算符 `Operators`

-   操作符（方法）定义在 `Operators`、`ColumnOperators` 基类中
    -   `ColumnOperators` 继承自 `Operators` ，其衍生类包括
        -   `Column`
        -   `ColumnElement`
        -   `InstructmentedAttribute`

> - *SQLAlchemy 2.0 Operator Reference*：<https://docs.sqlalchemy.org/en/20/core/operators.html>
> - *SQLAlchemy 1.4 Operator Reference*：<https://docs.sqlalchemy.org/en/14/core/operators.html>
> - *SQLAlchemy 1.4* 操作符参考：<https://www.osgeo.cn/sqlalchemy/core/operators.html>

####    逻辑运算符

| `Operators` 连接操作符 | 函数版本    | *SQL* 关键字 | 描述   |
|------------------------|-------------|--------------|--------|
| `__and__()`            | `sa.and_()` | `AND`        | `&` 与 |
| `__or__()`             | `sa.or_()`  | `OR`         | `|` 或 |
| `__invert__()`         | `sa.not_()` | `NOT`        | `~` 非 |

-   逻辑运算符（方法）定义在 `Operators` 类中
    -   在 `Select.where`、`Update.where`、`Delete.where` 子句中 `AND` 在以下场合自动应用
        -   `.where` 方法重复调用
        -   `.where` 方法中传入多个表达式

####    比较运算符

| `ColumnOperators` 比较运算符 | *SQL* 关键字           | 描述                                 |
|------------------------------|------------------------|--------------------------------------|
| `__eq__()`                   | `=`                    |                                      |
| `__ne__()`                   | `!=`、`<>`             |                                      |
| `__gt__()`                   | `>`                    |                                      |
| `__lt__()`                   | `<`                    |                                      |
| `__ge__()`                   | `>=`                   |                                      |
| `__le__()`                   | `<=`                   |                                      |
| `between()`                  | `BETWEEN...AND...`     | 区间比较                             |
| `in()`                       | `IN`                   | 支持列表、空列表、元组各元素独立比较 |
| `not_in()`                   | `NOT IN`               | 支持列表、空列表、元组各元素独立比较 |
| `is_()`                      | `IS`                   | 主要用于 `is_(None)`，即 `IS NULL`   |
| `is_not()`                   | `IS NOT`               |                                      |
| `is_distinct_from()`         | `IS DISTINCT FROM`     |                                      |
| `isnot_distinct_from()`      | `IS NOT DISTINCT FROM` |                                      |

-   说明
    -   *SQLAlchemy* 通过渲染临时、第二步再渲染为绑定参数列表的 *SQL* 字符串以实现 `ColumnOperators.in`
        -   参数可为不定长列表：`IN` 表达式绑定参数数量执行时才确定
        -   参数可为空列表：`IN` 表达式渲染为返回空行字查询
        -   参数可为元组
        -   参数可为子查询
    -   上述魔法方法即对应 Python 算符实现，其中
        -   `__eq__(None)`（`== None`）会触发 `is_(None)`

####    算数运算符

| `ColumnOperators` 算术操作符 | *SQL* 关键字        | 描述 |
|------------------------------|---------------------|------|
| `__add__()`、`__radd__()`    | 数值 `+`、字符 `||` | `+`  |
| `__sub__()`、`__rsub__()`    | `-`                 | `-`  |
| `__mul__()`、`__rmul__()`    | `*`                 | `*`  |
| `__div__()`、`__rdiv__()`    | `/`                 | `/`  |
| `__mod__()`、`__rmod__()`    | `%`                 | `%`  |

####    字符串比较、操作

| `ColomnOperators` 字符串 | *SQL* 关键字                    | 描述             |
|--------------------------|---------------------------------|------------------|
| `like()`                 | `LIKE`                          |                  |
| `ilike()`                | `lower(_) LIKE lower(_)`        | 大小写不敏感匹配 |
| `notlike()`              | `NOT LIKE`                      |                  |
| `notilike()`             | `lower(_) NOT LIKE lower(_)`    |                  |
| `startswith()`           | `LIKE _ || '%'`                 |                  |
| `endswith()`             | `LIKE '%' || _`                 |                  |
| `contains()`             | `LIKE '%' || _ || '%'`          |                  |
| `match()`                | `MATCH`                         |                  |
| `regexp_match()`         | `~ %(_)s`、`REGEXP`             | 正则匹配         |
| `concat()`               | `||`                            | 字符类型 `+` 同  |
| `regex_replace()`        | `REGEXP_REPALCE(_,%(_)s,%(_)s)` | 正则替换         |
| `collate()`              | `COLLATE`                       | 指定字符集排序   |

-   说明
    -   `match`、`regexp_match`、`regex_replace` 的支持、行为（方言）、结果依赖数据库后端

### *Selectable*

-   *Selectable* 可选对象：可从中选择的行的任何对象
    -   由 `FromClause` 演变而来（类似 *Table*）

> - *SQLAlchemy 1.4 Selectables API*：<https://docs.sqlalchemy.org/en/14/core/selectable.html>
> - *SQLAlchemy 1.4* 可选择对象：<https://www.osgeo.cn/sqlalchemy/core/selectable.html>

####    可选对象基础构造函数

| 基础构建器                         | *SQL* 关键字    | 描述                                 |
|------------------------------------|-----------------|--------------------------------------|
| `select(*args,**kw)`               | `SELECT`        | 创建 `SELECT` 子句                   |
| `table(name,*columns,**kw)`        | `TABLE`         | 创建 `TableClause`                   |
| `values(*columns,**kw)`            | `VALUES`        | 创建 `VALUES` 子句                   |
| `exists(*args,**kwargs)`           | `EXISTS`        | 存在，可直接被调用创建 `EXISTS` 子句 |
| `except_(*selects,**kwargs)`       | `EXCEPT`        | 差集                                 |
| `except_all(*selects,**kwargs)`    | `EXCEPT ALL`    |                                      |
| `intersect(*selects,**kargs)`      | `INTERSECT`     | 交集                                 |
| `intersect_all(*selects,**kwargs)` | `INTERSECT ALL` |                                      |
| `union(*select,**kwargs)`          | `UNION`         | 并集                                 |
| `union_all(*select,**kwargs)`      | `UNION ALL`     |                                      |

####    可选对象修饰构造函数

| 修饰构建器                                 | *SQL* 关键字 | 返回值        | 描述 |
|--------------------------------------------|--------------|---------------|------|
| `alias(selectable[,name,flat])`            | `AS`         | `Alias`       | 别名 |
| `cte(selectable[,name,recursive])`         |              | `CTE`         |      |
| `join(left,right[,onclause,isouter,...])`  | `JOIN`       | `Join`        |      |
| `lateral(selectable[,name])`               |              | `Lateral`     |      |
| `outerjoin(left,right[,onclause,full])`    | `OUTER JOIN` |               |      |
| `tablesample(select,sampling[,name,seed])` |              | `TableSample` |      |

####    可选对象类

| 可选择对象类                                | 描述                              | 基类                                 |
|---------------------------------------------|-----------------------------------|--------------------------------------|
| `ReturnRows`                                | 包含列、表示行的最基类            | `ClauseElement`                      |
| `Selectable`                                | 可选择标记                        | `ReturnRows`                         |
| `FromClause`                                | `FROM` 字句中可使用标记           | `Selectable`                         |
| `ScalarSelect`                              | 标量子查询                        |                                      |
| `SelectBase`                                | `SELECT` 语句基类                 | `Selectable`、`Executable`、`HasCTE` |
| `GenerativeSelect`                          | `SELECT` 语句基类                 | `SelectBase`                         |
| `TextualSelect`                             | `TextClause` 的 `SelectBase` 封装 | `SelectBase`                         |
| `Select`                                    | `SELECT` 语句                     | `GenerativeSelect`                   |
| `CompoundSelect(keyword,*selects,**kwargs)` | 基于 `SELECT` 操作的基础          | `GenerativeSelect`                   |
| `Join`                                      | `FromClause` 间 `Join`            | `FromClause`                         |
| `TableClause`                               | 最小 *Table* 概念                 | `FromClause`                         |
| `AliasedReturnsRows(*args,**kw)`            | 别名类基类                        | `FromClause`                         |
| `Alias(*arg,**kw)`                          | 别名                              | `AliasedReturnsRows`                 |
| `CTE`                                       | *Common Table Expression*         | `AliasedReturnsRows`                 |
| `Lateral`                                   | `LATERAL` 子查询                  | `AliasedReturnsRows`                 |
| `Subquery`                                  | 子查询                            | `AliasedReturnsRows`                 |
| `TableSample`                               | `TABLESAMPLE` 子句                | `AliasedReturnsRows`                 |
| `TableValueAlias`                           | `table valued` 函数               | `Alias`                              |
| `Values`                                    | 可作为 `FROM` 子句尾的 `VALUES`   | `FromClause`                         |
| `Exists`                                    | `EXISTS` 子句                     | `UnaryExpression`                    |
| `HasCTE`                                    | 包含 *CTE* 支持标记               |                                      |
| `HasPrefixes`                               |                                   |                                      |
| `HasSuffixes`                               |                                   |                                      |
| `Executable`                                | 可执行标记                        |                                      |

####    `Select` 语句

| `Executable` 部分方法、属性       | 描述                       |
|-----------------------------------|----------------------------|
| `execute`(*multiparams,**params)` | 编译、执行（2.0 将被移除） |
| `execution_options`(**kw)`        | 设置非 *SQL* 选项          |
| `get_execution_options()`         | 获取非 *SQL* 选项          |

| `SelectBase` 部分方法、属性 | 描述       |
|-----------------------------|------------|
| `add_cte(cte)`              | 添加 *CTE* |
| `subquery([name])`          | 查询别名   |
| `exists()`                  |            |
| `label(name)`               |            |
| `lateral([name])`           |            |
| `selected_columns`          | 结果集中列 |

| `GenerativeSelect` 部分方法、属性  | 描述     |
|------------------------------------|----------|
| `group_by(*clauses)`               | 分组     |
| `limit(limit)`                     | 限制     |
| `offset(offset)`                   | 偏移     |
| `order_by(*clauses)`               | 排序     |
| `set_label_style(style)`           | 标签样式 |
| `slice(start,stop)`                | 切片     |
| `fetch(count[,with_ties,percent])` | 获取     |

| `Select` 部分方法、属性                               | 描述                      |
|-------------------------------------------------------|---------------------------|
| `add_columns(*columns)`                               | 添加列                    |
| `correlate(*fromclauses)`                             |                           |
| `correlate_except(*fromclauses)`                      |                           |
| `distinct(*expr)`                                     | 唯一值                    |
| `except_(other,**kwargs)`                             | `EXCEPT`                  |
| `except_all(other,**kwargs)`                          | `EXCEPT ALL`              |
| `filter(*criteria)`                                   | `WHERE`                   |
| `filter_by(**kwargs)`                                 | 筛选条件作为 `WHERE` 子句 |
| `where(*whereclause)`                                 | `WHERE`                   |
| `from_statement(statement)`                           | `FROM`                    |
| `select_from(*froms)`                                 | `FROM`                    |
| `having(having)`                                      | `HAVING`                  |
| `intersect(other,**kwargs)`                           | `INTERSECT`               |
| `intersect_all(other,**kwargs)`                       | `INTERSECT ALL`           |
| `union(other,**kwargs)`                               | `UNION`                   |
| `union_all(other,**kwargs)`                           | `UNION ALL`               |
| `join(target[,onclause,isouter,full)`                 | 连接                      |
| `outerjoin(target[,onclause,full)`                    | 左外连接                  |
| `join_from(from_,target[,onclause,isouter,full)`      | 连接并返回                |
| `outerjoin_from(from_,target[,onclause,isouter,full)` | 左外连接并返回            |
| `reduce_columns([only_synonyms])`                     | 剔除同名列                |

### *DML* 插入、更新、删除

> - *SQLAlchemy 1.4 DML API*：<https://docs.sqlalchemy.org/en/14/core/dml.html>
> - *SQLAlchemy 1.4* 插入、删除、更新：<https://www.osgeo.cn/sqlalchemy/core/dml.html>

####    *DML* 基础构建器

| *DML* 基础构建器                                              | *SQL* 关键字 | 描述 |
|---------------------------------------------------------------|--------------|------|
| `delete(table[,whereclause,bind,returning,...],**dialect_kw)` | `DELETE`     |      |
| `insert(table[,values,inline,bind,...],**dialect_kw)`         | `INSERT`     |      |
| `update(table[,whereclause,values,inline,...],**dialect_kw)`  | `UPDATE`     |      |

####    *DML* 类

| *DML* 构建器类                    | 描述                                    |
|-----------------------------------|-----------------------------------------|
| `Delete(table[,whereclause,...])` | `DELETE`                                |
| `Insert()`                        | `INSERT`                                |
| `Update()`                        | `UPDATE`                                |
| `UpdateBase()`                    | `INSERT`、`UPDATE`、`DELETE` 语句基础   |
| `ValuesBase`                      | `INSERT`、`UPDATE` 中 `VALUES` 子句支持 |

### *SQL* 函数

> - *SQLAlchemy 1.4 SQL and Generic API*：<https://docs.sqlalchemy.org/en/14/core/functions.html>
> - *SQLAlchemy 1.4* SQL 通用函数：<https://www.osgeo.cn/sqlalchemy/core/functions.html>

####    基类、工具

| 基类、工具                                   | 描述                   |
|----------------------------------------------|------------------------|
| `FunctionElement`                            | *SQL* 函数基类         |
| `Function`                                   | 具名 *SQL* 函数        |
| `GenericFunction`                            | 通用函数类             |
| `AnsiFunction`                               | *ANSI* 格式函数类      |
| `register_function(identifier,fn[,package])` | 关联函数名与可调用对象 |
| `func`                                       | 访问函数句柄           |

-   说明
    -   上述 4 类按顺序存在继承关系
    -   *SQL* 函数常通过访问 `func` 对象属性方式调用
        -   对已注册函数，`func` 自动处理特殊行为规则
            -   *ANSI* 格式函数类不添加括号
        -   对未注册函数，`func` 原样生成函数名
    -   此外，`sa.sql.functions` 模块下有部分常用 *SQL* 函数预定义实现
        -   除预定义实现外，*SQLAlchemy* 不会检查 *SQL* 函数调用限制

####    预定义函数类

| 预定义 *SQL* 函数   | *SQL* 函数          | 描述                           |
|---------------------|---------------------|--------------------------------|
| `array_agg`         | `ARRAY_AGG`         | 聚合元素，返回 `sa.ARRAY` 类型 |
| `max`               | `MAX`               |                                |
| `min`               | `MIN`               |                                |
| `count`             | `COUNT`             | 缺省即 `COUNT(*)`              |
| `char_length`       | `CHAR_LENGTH`       |                                |
| `concat`            | `CONCAT`            | 字符串连接                     |
| `grouping_sets`     | `GTOUPING SETS`     | 创建多个分组集                 |
| `cube`              | `CUBE`              | 生成幂集作为分组集             |
| `next_value`        |                     | 下个值，需 `Sequence` 作为参数 |
| `now`               | `now`               |                                |
| `random`            | `RANDOM`            |                                |
| `rollup`            | `ROLLUP`            |                                |
| `session_user`      | `SESSION_USER`      |                                |
| `sum`               | `SUM`               |                                |
| `localtime`         | `localtime`         |                                |
| `localtimestamp`    | `localtimestamp`    |                                |
| `current_date`      | `CURRENT_DATE`      |                                |
| `current_time`      | `CURRENT_TIME`      |                                |
| `current_timestamp` | `CURRENT_TIMESTAMP` |                                |
| `sysdate`           | `SYSDATE`           |                                |
| `user`              | `USER`              |                                |
| `current_user`      | `CURRENT_USER`      |                                |
| `coalesce`          |                     |                                |

| 预定义聚合函数      | 假设聚合函数                | 描述                   |
|---------------------|-----------------------------|------------------------|
| `rank`              | 假设聚合函数 `rank`         | 在各组中位置，不       |
| `dense_rank`        | 假设聚合函数 `dense_rank`   |                        |
| `percent_rank`      | 假设聚合函数 `percent_rank` | 在各组中百分比位置     |
| `percentile_cont`   | 假设聚合函数 `percent_cont` |                        |
| `percentile_disc`   | 假设聚合函数 `percent_disc` |                        |
| `mode`              | `mode`                      |                        |
| `cume_dist`         | `cume_dist`                 | 返回 `sa.Numeric` 类型 |

##  *Core* 数据库模式

-   数据库元数据：描述、检查数据库模式的综合系统
    -   除创建类实例外，数据库元数据可通过反射机制从数据库引擎附加元信息
    -   `sa.schema` 模块内容

> - *SQLALchemy 1.4 Core* 模式定义语言：<https://www.osgeo.cn/sqlalchemy/core/schema.html>

###  描述数据库

| 数据库元数据描述                            | 描述               |
|---------------------------------------------|--------------------|
| `SchemaItem`                                | 数据库架构项基类   |
| `MetaData()`                                | 维护表、表关联架构 |
| `Table(*,table_name,metadata[,column,...])` | 数据库表           |
| `Column(*,column_name[,type,...])`          | 数据库表列         |
| `BLANK_SCHEMA`                              |                    |

> - *SQLALchemy 1.4 Core* 元数据描述数据库：<https://www.osgeo.cn/sqlalchemy/core/schema.html>

####    `Table` 数据库表

| `Table` 部分方法、属性                     | 描述                   |
|--------------------------------------------|------------------------|
| `add_is_dependent_on(table)`               | 添加依赖项             |
| `alias([name, flat])`                      | 别名                   |
| `append_column(column[,replace_existing])` | 追加列                 |
| `append_constraint(constraint)`            | 追加依赖               |
| `compare(other,**kw)`                      | 比较语句               |
| `compile([bind,dialect,**kw])`             | 编译                   |
| `corresponding_column(column,...)`         |                        |
| `get_children([omit_attrs,**kw])`          |                        |
| `is_derived_from(fromclause)`              |                        |
| `tometadata(metadata[,schema,...])`        | 在其他元数据中创建副本 |
| `create([bind,checkfirst])`                | 创建表                 |
| `drop([bind,checkfirst])`                  | 删除表                 |
| `update([whereclause,vlaues,...])`         | 更新                   |
| `insert([values,inline,kwargs])`           | 插入                   |
| `delete([whereclause,**kwargs])`           | 删除                   |
| `select([whereclause,**kwargs])`           | 查询                   |
| `join(right[,onclause,isouter,full])`      | 连接                   |
| `outerjoin(right[,onclause,full])`         | 外连接                 |
| `constraints`                              | 约束项                 |
| `primary_key`                              | 主键                   |
| `foreign_key_constraints`                  | 外键约束项             |
| `foreign_kyes`                             | 外键                   |
| `bind`                                     | 可连接项               |
| `c`                                        | 列集合                 |
| `columns`                                  | 列集合                 |
| `exported_columns`                         | 导出的列               |
| `description`                              | 描述                   |
| `dialect_kwargs`                           | 方言关键字选项         |
| `dialect_options`                          | 方言关键字选项         |

-   说明
    -   表依赖通常通过外键确定，也可以手动 `add_is_dependent_on` 创建

### 约束、索引


| 约束、索引                         | 描述                        |
|------------------------------------|-----------------------------|
| `ForeignKey(key_name)`             | 外键关系                    |
| `Constraint[name,deferrable,....]` | 表级约束                    |
| `ColumnCollectionMixin`            |                             |
| `Index([name,*expression,...])`    | 表级索引                    |
| `ColumnCollectionConstraint`       | `ColumnCollection` 约束代理 |
| `CheckConstraint`                  | 表级、列级检查约束          |
| `ForeignKeyConstraint`             | 外键约束                    |
| `PrimaryKeyConstraint`             | 主键约束                    |
| `UniqueConstraint`                 | 唯一值约束                  |
| `conv`                             | 标记列名称由命名约定转换    |

-   说明
    -   继承关系
        -   `ForeignKey` 类继承自 `schema.SchemaItem` 类
        -   `Constraint` 类继承自 `schema.SchemaItem` 类，并衍生其他约束类
        -   `Index` 类继承自 `schema.SchemeItem` 类

> - *SQLALchemy 1.4 Core* 约束和索引：<https://www.osgeo.cn/sqlalchemy/core/constraints.html>

### *DDL* 


##  *Core* 数据类型

-   *SQLAlchemy* 为常见数据类型提供抽象，并允许自定义数据类型
    -   数据类型使用 Python 类代表，均继承自 `TypeEngine` 类
        -   *CamelCase* 驼峰（命名）数据类型：所有数据库后端行为预期最大抽象
            -   *SQLAlchemy* 根据后端自动渲染对数据类型
        -   *UpperCase* 大写（命名）数据类型：数据库后端的具体数据类型
            -   被渲染为同名数据库数据类型，无论数据
            -   往往继承特定的驼峰数据类型
    -   *SQLAlchemy* 将负责数据类型转换

> - *SQLALchemy 1.4 Core* 数据类型：<https://www.osgeo.cn/sqlalchemy/core/types.html>
> - *SQLALchemy 2.0 Core Datatype*：<https://docs.sqlalchemy.org/en/20/core/types.html>

### 基本数据类型

| 驼峰数据类型                              | Python3 类型         | SQL 类型              | 描述                  |
|-------------------------------------------|----------------------|-----------------------|-----------------------|
| `SchemaType([name,schema,...])`           |                      |                       | 自定义类型            |
| `Integer`                                 | `int`                | `INT`                 |                       |
| `BigInteger`                              | `int`                | `BIGINT`              | 大整形                |
| `SmallInteger`                            | `int`                | `SMALLINT`            |                       |
| `Float([precision,asdecimal])`            | `float`              | `FLOAT`、`REAL`       | 浮点                  |
| `Numeric([precision,scale,...])`          | `decimal.Decimal`    | `NUMERIC`、`DECIMAL`  | 固定精度              |
| `String([length,collation,...])`          | `str`                | `VARCHAR`             | 字符串、字符基类      |
| `Text([length,collation,...])`            | `str`                | `CLOB`                | 变长字符串            |
| `Unicode([length,**kwargs])`              | `str`                | `NVARCHAR`            | 变长 Unicode 字符串   |
| `UnicodeText([length,**kwargs])`          | `str`                | `NCLOB`、`NTEXT`      | 无限长 Unicode 字符串 |
| `LargeBinary([length])`                   | `byte`               | `BLOB`、`BYTEA`       |                       |
| `PickleType([protocol,pickler,...])`      |                      |                       | 可 Pickle 序列化对象  |
| `Boolean([create_constraint,...])`        | `bool`               | `BOOLEAN`、`SMALLINT` |                       |
| `MatchType([create_constraint,...])`      |                      |                       |                       |
| `Enum(*enums,**kw)`                       | `enum.Enum`          | `ENUM`、`VARCHAR`     | 枚举基类              |
| `DateTime([timezone])`                    | `datetime.datetime`  | `TIME`、`TIMESTAMP`   | 时间戳                |
| `Date`                                    | `datetime.date`      |                       |                       |
| `Time([timezone])`                        | `datetime.time`      |                       |                       |
| `Interval([native,second_precision,...])` | `datetime.timedelta` |                       | 时间片                |

-   说明
    -   `Float.asdecimal`、`Numeric.asdemcimal` 决定 `Float`、`Numeric` 转换的数据类型
    -   `String` 在某些数据库后端上支持不指定长度的初始化

| 大写数据类型 |
|--------------|
| `ARRAY`      |
| `BIGINT`     |
| `BINARY`     |
| `BLOB`       |
| `BOOLEAN`    |
| `CHAR`       |
| `CLOB`       |
| `DATE`       |
| `DATETIME`   |
| `DECIMAL`    |
| `FLOAT`      |
| `INT`        |
| `INTEGER`    |
| `JSON`       |
| `NCHAR`      |
| `NUMERIC`    |
| `NVARCHAR`   |
| `REAL`       |
| `SMALLINT`   |
| `TEXT`       |
| `TIMESTAMP`  |
| `VARBINARY`  |
| `VARCHAR`    |

> - *SQLALchemy 1.4 Core* 基本数据类型：<https://www.osgeo.cn/sqlalchemy/core/type_basics.html>
> - *SQLALchemy 2.0 Core Basic Datatype*：<https://docs.sqlalchemy.org/en/20/core/type_basics.html>

### 自定义数据类型

| 数据类型基类    | 描述                                     |
|-----------------|------------------------------------------|
| `TypeEngine`    | 数据类型基类                             |
| `Concatenable`  | 串联类型 *Mixin*                         |
| `Indexable`     | 支持索引类型 *Mixin*                     |
| `NullType`      | 未知类型                                 |
| `Variant`       | 包装类型，根据数据库方言在各种实现中选择 |
| `TypeDecorator` | 向类型附加功能                           |

> - *SQLALchemy 1.4 Core* 数据类型基类：<https://www.osgeo.cn/sqlalchemy/core/type_api.html>
> - *SQLALchemy 1.4 Core* 自定义类型：<https://www.osgeo.cn/sqlalchemy/core/custom_types.html>
