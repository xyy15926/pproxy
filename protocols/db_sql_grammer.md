---
title: SQL语法
categories:
  - Database
  - SQL DB
tags:
  - Database
  - SQL DB
  - Grammer
  - CUAD
date: 2019-07-10 00:48:33
updated: 2023-08-31 17:18:28
toc: true
mathjax: true
comments: true
description: SQL语法
---

##  SQL 语法

-   *Structured Query Language* 结构化查询语言
    -   *Data Definition Language* 数据定义语言：创建、删除数据库、表、视图、索引
        -   `create`：创建数据库、表
        -   `drop`：删除数据库、表
        -   `alter`：修改数据库、表
    -   *Data Manipulation Language* 数据操纵语言：查询、变更表中记录
        -   `select`：查询表中记录
        -   `insert`：向表中插入记录
        -   `update`：更新表中记录
        -   `delete`：删除表中记录
    -   *Data Control Language* 数据控制语言：确认、取消数据变更，用户权限设置
        -   `commit`：确认数据变更
        -   `rollback`：取消数据变更
        -   `grant`：赋予用户权限
        -   `revoke`：取消用户权限
    -   其他语法
        -   关键字、函数、字段名等大小写不敏感
        -   单行注释：`--`
        -   多行注释：`/**/`
        -   数字字段：反引号 `` ` `` 扩起
        -   字符串：单引号 `'` 扩起（大部分方言也支持 `"` 双引号）

> - SQL 有 SQL92、SQL99 两个 *ANSI* 标准，但未有完全标准实现，各数据库实现均有自身方言

### 数据类型

| ANSI SQL | MySQL          | Oracle          | 描述 |
|----------|----------------|-----------------|------|
| TinyInt  | `TINYINT`      |                 |      |
| SmallInt | `SMALLINT`     |                 |      |
|          | `MEDIUMINT`    |                 |      |
| Integer  | `INT`          |                 |      |
| BigInt   | `BIGINT`       | `LONG`          |      |
| FLOAT    | `FLOAT`        | `BINARY_FLOAT`  |      |
| DOUBLE   | `DOUBLE`       | `BINARY_DOUBLE` |      |
| Real     | `DOUBLE`       | `FLOAT(63)`     |      |
| Decimal  | `DECIMAL(M,D)` | `NUMBER(p,s)`   |      |
| Numeric  | `DECIMAL(M,D)` | `NUMBER(s,s)`   |      |

-   说明
    -   MySQL 中数据类型可用 `unsigned` 修饰表示无符号数
    -   Oracle 中原生数据类型 `NUMBER` 为 10 进制精度，`BINARY_FLOAT`、`BINARY_DOUBLE`、`LONG` 为 2 进制进度
        -   但很多其他库中数据类型对转换为 Oracle 数据类型时，2 进制精度数据类型被映射为 `NUMBER(38)`

> - Oracle 数据类型：<https://docs.oracle.com/cd/B19306_01/server.102/b14220/datatype.htm#i16209>
> - Oracle 数据类型：<https://docs.oracle.com/cd/B28359_01/server.111/b28286/sql_elements001.htm#SQLRF50950>

| MySQL      | Oracle                      | 描述                  |
|------------|-----------------------------|-----------------------|
| `DATE`     | `DATE`                      | `yyyy-MM-dd`          |
| `TIME`     |                             | `HH:mm:ss`            |
| `YEAR`     |                             | `yyyy`                |
| `DATETIME` |                             | `yyyy-MM-dd HH:mm:ss` |
| `TIMSTAMP` | `TIMESTAMP(p)`              |                       |
|            | `INTERVAL YEAR(p) TO MONTH` | 年、月时间段          |
|            | `INTERVAL DAY(p) TO SECOND` | 日、秒时间段          |

-   说明
    -   MySQL 中 `timestamp` 类型为 4B 记录 UTC 秒，最多支持到 2038 年

| 类型         | MySQL               | Oracle                                      | MSSQL | 描述 |
|--------------|---------------------|---------------------------------------------|-------|------|
| 定长字符串   | `CHAR(n)`,255B      | `CHAR(n BYTE,CHAR)`、`NCHAR(n)`,2KB         |       |      |
| 变长字符串   | `VARCHAR(n)`,64KB   | `VARCHAR2(n BYTE,CHAR)`、`NVARCHAR2(n)`,4KB |       |      |
| 定长字节串   | `BINARY(n)`,255B    | `RAW(n)`,2KB                                |       |      |
| 变长字节串   | `VARBINARY(n)`,255B | `VARCHAR2(n [BYTE])`,4KB                    |       |      |
| 外置字符文本 | `TINYTEXT`,255B     |                                             |       |      |
| 外置字符文本 | `TEXT`,64KB         |                                             |       |      |
| 外置字符文本 | `MEDIUMITEXT`,16MB  |                                             |       |      |
| 外置字符文本 | `LONGTEXT`,4GB      | `CLOB`、`NCLOB`,4GB                         |       |      |
| 外置字节串   | `TINYBOLB`,255B     |                                             |       |      |
| 外置字节串   | `BLOB`,64KB         |                                             |       |      |
| 外置字节串   | `MEDIUMBLOB`,16MB   |                                             |       |      |
| 外置字节串   | `LONGBLOB`,4GB      | `BLOB`,4GB                                  |       |      |
| 枚举         | `ENUM(<val_list>)`  |                                             |       |      |
| 集合         | `SET`               |                                             |       |      |


-   MySQL 说明
    -   MYSQL 中各类 `CHAR`、`TEXT` 中语义单元为字符，各类 `BINARY`、`BLOB` 语义单元为字节
        -   各类 `CHAR`、`TEXT` 存储内容长度限制与编码类型有关（UTF-8 等变长编码会被限制为固定长度）
        -   `CHAR` 定长会用空格补足，故存储内容结尾空格会被截断
    -   MYSQL 中 `VAR` 表示变长，长度记录需占据存储空间
        -   长度小于等于 255 时 1B 记录长度
        -   长度大于 255 时 2B 记录长度
        -   `VARCHAR` 在长度设置超限时会自动转换为相应类型 `TEXT`，即 `VARCHAR(255)` 以上与 `TEXT` 无差别
    -   MYSQL 中 `TEXT`、`BLOB` 为表外存储，速度较慢
-   ORACLE 说明
    -   Oracle 中 `N` 表 *national* 针对特定语言，用于指定针对特定语言的字符编码方案
        -   `NCHAR`、`CHAR` 均表示以字符为语义单元计数，但二者对应编码方案不同
            -   `CHAR` 使用默认编码方案 `NLS_CHARACTERSET`
            -   `NCHAR` 使用默认编码方案 `NLS_NCHAR_CHARACTERSET`，仅能选择 `AL16UTF16`、`AL32UTF8`

> - MySQL 中文本类型：<https://www.cnblogs.com/goloving/p/14847408.html>
> - Oracle 中字符编码形如 `AL32UTF8`，其中 `AL` 表 *all* 对所有语言、`32` 表长度
> - Oracle 字符集：<https://www.cnblogs.com/rootq/articles/2049324.html>
> - Oracle `NCHAR` 类型：<https://www.yiibai.com/oracle/oracle-nchar.html>

## DDL

```sql
CREATE DATABASE <db_name>;
CREATE TABLE <tbl>(
    <field>     <dtype>,
    <field>     <dtype>     <CSTRT>,                    -- MSSQL、ORACLE
    <CSTRT>     (<field>,),                             -- MySQL
    CONSTRAINT [<cstrt_name>] <cstrt> (<field>,)        -- common
)
CREATE INDEX <index_name>
ON <tbl> (<field>);
CREATE VIEW <view_name> AS
<select_expr>;

DROP INDEX <tbl>.<index_name>;                          -- MSSQL
DROP INDEX <index_name>;                                -- ORACLE
ALTER TABLE <tbl>
DROP INDEX <index_name>;                                -- MySQL
DROP TABLE <tbl>;                                       -- 丢弃表
DROP DATABASE <db_name>;                                -- 丢弃数据库
DROP VIEW <view>;                                       -- 丢弃视图

ALTER TABLE <tbl> ADD <field> <dtype>;                  -- 添加列
ALTER TABLE <tbl> DROP COLUMN <field>;                  -- 删除列
ALTER TABLE <tbl>
ALTER COLUMN <field> <dtype>;                           -- 修改列类型
```

### 约束

-   添加、删除约束
    -   约束说明
        -   `not null`、`default` 一般不视为 `constraint` 约束
        -   `unique` 在 MySQL 中视为 `index` 约束，在 Oracle、MSSQL 中视为 `constraint` 约束
        -   `primary key`、`foreign key ... reference ...`、`check` 总可设置具名约束
    -   各数据库实现区别
        -   Oracle、MSSQL 中需使用 `column` 关键字引导列名
        -   MySQL 在添加、删除约束时常可省略 `constraint` 关键字

####   `NOT NULL`

```sql
<field> <dtype> NOT NULL                                    -- 建表
```

####    `DEFAULT`

```sql
<field> <dtype> DEFAULT <value>                             -- 建表

ALTER TABLE <tbl>
ALTER <field> SET DEFAULT <value>;                          -- MySQL 已有表添加
ALTER TABLE <tbl>
ALTER COLUMN <field> SET DEFAULT <value>;                   -- MSSQL、ORACLE 已有表添加

ALTER TABLE <tbl>
ALTER <field> DROP DEFAULT;                                 -- MySQL 删除
ALTER TABLE <tbl>
ALTER COLUMN <field> DROP DEFAULT;                          -- MSSQL、ORACLE 删除
```

####    `UNIQUE`

```sql
CONSTRAINT [<cstrt_name>] UNIQUE (<field>,)                 -- MySQL、MSSQL、ORACLE 建表
UNIQUE [KEY] [<cstrt_name>] (<field>)                       -- MySQL 建表
<field> <dtype> UNIQUE                                      -- MSSQL、ORACLE 建表

ALTER TABLE <tbl>
ADD UNIQUE(<field>);
ALTER TABLE <tbl>
ADD CONSTRAINT <cstrt_name> UNIQUE(<field>,);               -- MySQL、MSSQL、ORACLE 已有表添加

ALTER TABLE <tbl>
DROP INDEX <cstrt_name>;                                    -- MySQL 删除
ALTER TABlE <tbl>
DROP CONSTRAINT <cstrt_name>;                               -- MSSQL、ORACLE 删除
```

####    `PRIMARY KEY`

```sql
CONSTRAINT [<cstrt_name>] PRIMARY KEY (<field>,)            -- MySQL、MSSQL、ORACLE 建表
PRIMARY KEY (<field>,)                                      -- MYSQL 建表
<field> <dtype> PRIMARY KEY                                 -- MSSQL、ORACLE 建表


ALTER TABLE <tbl>
ADD PRIMARY KEY (<field>,);
ALTER TABLE <tbl>
ADD CONSTRAINT <cstrt_name> PRIMARY KEY (<field>,);         -- MySQL、MSSQL、ORACLE 已有表添加

ALTER TABLE <tbl>
DROP PRIMARY KEY;                                           -- MySQL 删除
ALTER TABLE <tbl>
DROP CONSTRAINT <cstrt_name>;                               -- MSSQL、ORACLE
```

####    `FOREIGN KEY`

```sql
CONSTRAINT [<cstrt_name>] FOREIGN KEY (<field>,)
REFERENCES <tbl>(<field>,)                                  -- MySQL、MSSQL、ORACLE 建表
FOREIGN KEY (<field>,)
REFERENCES <tbl>(<field>,)                                  -- MYSQL 建表
<field> <dtype> FOREIGN KEY
REFERENCES <tbl>(<field>,)                                  -- MSSQL、ORACLE 建表


ALTER TABLE <tbl>
ADD FOREIGN KEY (<field>,)
REFERENCES <tbl>(<field>,);
ALTER TABLE <tbl>
ADD CONSTRAINT <cstrt_name> FOREIGN KEY (<field>,)
REFERENCES <tbl>(<field>);                                  -- MySQL、MSSQL、ORACLE 已有表添加

ALTER TABLE <tbl>
DROP FOREIGN KEY <cstrt_name>;                              -- MySQL 删除
ALTER TABLE <tbl>
DROP CONSTRAINT <cstrt_name>;                               -- MSSQL、ORACLE 删除
```

####    `CHECK`

```sql
CONSTRAINT [<cstrt_name>] CHECK(<condition>)                -- MySQL、MSSQL、ORACLE 建表
CHECK (condition)                                           -- MySQL 建表
<field> <dtype> CHECK(<condition>)                          -- MSSQL、ORACLE 建表

ALTER TABLE <tbl>
ADD CHECK (condition);
ALTER TABLE <tbl>
ADD CONSTRAINT <cstrt_name> CHECK (condition);              -- MySQL、MSSQL、ORACLE 已有表添加

ALTER TABLE <tbl>
DROP CHECK <cstrt_name>;                                    -- MySQL 删除
ALTER TABLE <tbl>
DROP CONSTRAINT <cstrt_name>;                               -- MSSQL、ORACLE 删除
```

## DML

```sql
SELECT          <field>, DISTINCT <field>
INTO            <new_tbl> [IN <other_db>]
FROM            <tbl>
WHERE           <field> <OP> <value>/<field>
ORDER BY        <field> [ASC/DESC];

INSERT INTO     <tbl>[<field>]
VALUES          (<value>);

UPDATE          <tbl>
SET             <field> = <value>
WHERE           <field> <OP> <value>/<field>;

DELETE
FROM            <tbl>
WHERE           <field> <OP> <value>/<field>;

TRUNCATE TABLE <tbl>;                                       -- 丢弃数据
```

### 关键字

| 功能           | MySQL              | Oracle               | MSSQL            |
|----------------|--------------------|----------------------|------------------|
| 分页           | `limit`            | `rownum <=`          | `top`、`percent` |
| 内连接         | `inner join`       | `inner join`         |                  |
| 左连接         | `left join`        | `left [outer ]join`  |                  |
| 右连接         | `right join`       | `right [outer ]join` |                  |
| 外连接         | 左、右并           | `full [outer ]join`  |                  |
| 笛卡尔积       | `cross join`       | `cross join`         |                  |
| 自然连接       | `natural`          | `nutural`            |                  |
| 并集去重、排序 | `union[ distinct]` | `union`              |                  |
| 并集全部       | `union all`        | `union all`          |                  |
| 升序           | `asc`              | `asc`                |                  |
| 降序           | `desc`             | `desc`               |                  |
| 分组           | `group by`         | `group by`           |                  |
| 分组汇总       | `with rollup`      | `rollup()`           |                  |

| 比较     | MySQL             | Oracle          |
|----------|-------------------|-----------------|
| 为空     | `is NULL`         | `is NULL`       |
| 非空     | `is not NULL`     | `is not NULL`   |
| 等于     | `=`               | 同              |
| 安全等于 | `<=>`             | 无              |
| 不等于   | `<>`、`!=`        | 同              |
| 小于等于 | `<`、`<=`         | 同              |
| 大于等于 | `>`、`>=`         | 同              |
| 介于     | `between and`     | 同              |
| 属于     | `in`              | 同              |
| 加减     | `+`、`-`          | 同              |
| 乘除     | `*`、`/`、        | `*`、`/`        |
| 整除     | `div`             | `trunc()`+`/`   |
| 模       | `%`、`mod`        | `mod()`         |
| 模糊匹配 | `like`            | 同              |
| 正则匹配 | `regexp`、`rlike` | `regexp_like()` |
| 逻辑与   | `and`             | `and`           |
| 逻辑或   | `or`              | `or`            |
| 逻辑非   | `not`             | `not`           |

##  内建函数

> - MySQL 内建函数：<https://www.runoob.com/mysql/mysql-functions.html>

### 字符串

| MYSQL                                 | 描述                        | 说明          |
|---------------------------------------|-----------------------------|---------------|
| `ascii(s)`                            | 返回首个字符 ASCII 码值     |               |
| `char_length(s)`                      | 字符数                      |               |
| `concat(s1,s2,...,sn)`                |                             |               |
| `concat_ws(x,s1,...,sn)`              | 带分隔符连接                |               |
| `format(x,n)`                         | 数字格式化                  |               |
| `insert(s1,x,len,s2)`                 | 替换 `s1[x:x+len]` 为 `s2`  | `x` 从 1 开始 |
| `locate(s1,s)`、`position(s1 in s)`   | 获取起始位置                |               |
| `lcase(s)`、`lower`                   |                             |               |
| `ucase(s)`、`upper`                   |
| `left(s,n)`                           | 左截取                      |               |
| `right(s,n)`                          | 右截取                      |               |
| `mid(s,n,len)`、`substr`、`substring` | 中截取                      |               |
| `lpad(s1,len,s2)`                     | `s1` 左填充 `s2` 至长 `len` |               |
| `rpad(s1,len,s2)`                     | `s1` 右填充 `s2` 至长 `len` |               |
| `ltrim(s)`                            | 左去空格                    |               |
| `rtrim(s)`                            | 右去空格                    |               |
| `trim(s)`                             | 去空格                      |               |
| `repeat(s,n)`                         | 重复                        |               |
| `reverse(s)`                          | 逆序                        |               |
| `strcmp(s1,s2)`                       | 比较                        |               |
| `substring_index(s,delimiter,number)` | 第 `number` 分隔符后子串    |               |
| `binary(s)`                           | 转换为位串                  |               |
| `charset(s)`                          | 返回编码方案                |               |
| `convert(s USING cs)`                 | 转换编码                    |               |

### 数值

| MYSQL                | 描述           | 说明 |
|----------------------|----------------|------|
| `mod(x,y)`           | 模             |      |
| `sign(x)`            | 符号函数       |      |
| `abs(x)`             |                |      |
| `ceil(x)`、`ceiling` | 上取整         |      |
| `floor(x)`           | 下取整         |      |
| `round(x[,y])`       | 就近取整       |      |
| `truncate(x,y)`      | 截断取整       |      |
| `cos(x)`             |                |      |
| `sin(x)`             |                |      |
| `tan(x)`             |                |      |
| `cot(x)`             |                |      |
| `acos(x)`            | 反余弦         |      |
| `asin(x)`            |                |      |
| `atan(x)`            |                |      |
| `atan2(n,m)`         | 直角边反正切   |      |
| `pi()`               |                |      |
| `degrees(x)`         | 弧度转换为角度 |      |
| `radians(x)`         | 角度转弧度     |      |
| `exp(x)`             | 自然指数       |      |
| `pow(x,y)`、`power`  | 指数           |      |
| `ln(x)`              | 自然对数       |      |
| `log([base,]x)`      | 对数，缺省自然 |      |
| `log10(x)`           |                |      |
| `log2(x)`            |                |      |
| `sqrt(x)`            | 平方根         |      |
| `bin(x)`             | 二进制编码     |      |
| `conv(x,f1,f2)`      | 进制转换       |      |


### 聚集

| MYSQL               | 描述     | 说明 |
|---------------------|----------|------|
| `avg(expression)`   | 均值     |      |
| `count(expression)` |          |      |
| `max(expression)`   |          |      |
| `min(expression)`   |          |      |
| `sum(expression)`   |          |      |
| `row_number()`      | 分配行号 |      |
| `rank()`            | 分配排名 |      |

### 容器

| MYSQL                          | 描述                          | 说明 |
|--------------------------------|-------------------------------|------|
| `field(s,s1,s2,...)`           | `s` 在列表 `[s1,s2,...]` 位置 |      |
| `find_in_set(s1,s2)`           | `s1` 在字符串 `s2` 中位置     |      |
| `greatest(expr1, expr2,...)`   | 最大者                        |      |
| `least(expr1,expr2,...)`       | 最小者                        |      |
| `cast(x AS type)`              | 转换数据类型                  |      |
| `coalesce(expr1,expr2,...)`    | 首个非空值                    |      |
| `json_object(k1,v2,k2,v2,...)` | 键值对转换为 JSON 对象        |      |
| `json_array(v1,v2,...)`        | 转换为 JSON 数组              |      |
| `json_extract(obj,path)`       | 提取指定字段                  |      |
| `json_contains(obj,path,ele)`  | 包含                          |      |

### 日期

| MySQL                                                   | Oracle       | 描述                       | 说明           |
|---------------------------------------------------------|--------------|----------------------------|----------------|
| `now()`、`localtime()`、`localtimestamp()`、`sysdate()` |              |                            | 当前日期、时间 |  |
| `curdate()`、`current_date`                             | `getdate()`  |                            |                |
| `curtime()`                                             |              |                            |                |
| `current_timestamp()`                                   |              |                            |                |
| `date(x)`                                               |              | 转换为日期                 |                |
| `extract(type FROM d)`                                  |              | 从日期中提取指定值         |                |
| `microsecond(t)`                                        |              | 提取毫秒                   |                |
| `second(t)`                                             |              | 提取秒                     |                |
| `minute(t)`                                             |              | 提取分                     |                |
| `hour(t)`                                               |              | 提取时                     |                |
| `time(expression)`                                      |              | 提取时、分、秒             |                |
| `day(d)`、`dayofmonth`                                  |              | 提取日                     |                |
| `week(d)`、`weekofyear`                                 |              | 提取周                     |                |
| `month(d)`                                              |              | 提取月                     | 整型           |
| `quarter(d)`                                            |              | 提取季度                   | 整型           |
| `year(d)`                                               |              | 提取年                     |                |
| `dayname(d)`                                            |              | 提取周几名                 | 字符串         |
| `monthname(d)`                                          |              | 提取月份名                 | 字符串         |
| `dayofweek(d)`、`weekday`                               |              | 周内第几天                 | 整型           |
| `dayofyear(d)`                                          |              | 年内第几天                 |                |
| `sec_to_time(s)`                                        |              | 秒转为时分秒               |                |
| `time_to_sec(t)`                                        |              | 时分秒转为秒               |                |
| `timediff(t1,t2)`                                       |              | 时间差                     |                |
| `last_day(d)`                                           |              | 日期所属月最后一日         |                |
| `from_days(n)`                                          |              | 距 0000-01-01 天数转为日期 |                |
| `to_days(d)`                                            |              | 日期转为距 0000-01-01 天数 |                |
| `makedate(year,dayofyear)`                              |              | 年、日转为日期             |                |
| `maketime(hour,minute,second)`                          |              | 时、分、秒转为时间         |                |
| `date_add(d,INTERVAL expr type)`                        | `dateadd()`  | 按指定单位推迟             |                |
| `date_diff(d1,d2)`                                      | `datediff()` | 日差                       |                |
| `date_sub(d,INTERVAL expr type)`                        |              | 按指定单位提前             |                |
| `period_add(period,number)`                             |              | 年月后推月份               |                |
| `period_diff(period1,period2)`                          |              | 月份差                     |                |
| `date_format(d,f)`                                      |              | 日期格式化输出             |                |
| `time_format(t,f)`                                      |              | 时间格式化输出             |                |
| `str_to_date(string,format_date)`                       |              | 日期格式化读取             |                |
|                                                         | `convert()`  |                            |                |
|                                                         | `datepart()` |                            |                |
| `adddate(d,n)`                                          |              | 按日推迟                   |                |
| `subdate(d,n)`                                          |              | 按日提前                   |                |
| `subtime(t,n)`                                          |              | 按秒提前                   |                |
| `addtime(t,n)`                                          |              | 按秒推迟                   |                |
| `timestamp(expr,interval)`                              |              | 返回时间、加总时间         |                |
| `timestampdiff(unit,datetime_expr1,datetime_expr2)`     |              | 时间差                     |                |
|

### 逻辑判断

| MySQL                            | Oracle  | 描述                     | 说明 |
|----------------------------------|---------|--------------------------|------|
| `case ... when ... then ... end` |         | 条件判断                 |      |
| `if(expr,v1,v2)`                 |         | 判断后返回值             |      |
| `ifnull(v1,v2)`                  | `nvl()` | 空则返回替代值           |      |
| `isnull(expr)`                   |         | 是否为空                 |      |
| `nullif(expr1,expr2)`            |         | 相等返回空，否则返回前者 |      |

```sql
CASE expr1
    WHEN cond1 THEN ret1
    WHEN cond2 THEN ret2
    ...
END
```

### 其他

| MySQL                                                   | 描述                     | 说明 |
|---------------------------------------------------------|--------------------------|------|
| `version()`                                             | 数据库版本               |      |
| `connection_id()`                                       | 唯一连接 ID              |      |
| `current_user()`、`session_user`、`system_user`、`user` | 当前用户名               |      |
| `datebase()`                                            | 当前用户名               |      |
| `last_insert_id()`                                      | 最近 *AUTO_INCREMENT* 值 |      |



