---
title: PyEcharts
categories:
  - 
tags:
  - 
date: 2024-09-21 17:16:05
updated: 2024-11-30 20:25:31
toc: true
mathjax: true
description: 
---

##  *PyEcharts*

-   *PyEcharts*
    -   *PyEcharts* 本质上即将 *Echarts* 配置项由 *Python* `dict` 序列化为 *JSON* 格式
        -   即，其支持的数据格式取决于 *JSON* 支持的数据类型
        -   则，数据传入 *PyEchart* 前需自行将数据转换为 **Python 原生数据格式**（包括 `np.int64` 等）

-   *PyEcharts* 中配置项基本采用 `XXXOpts`、`XXXItems` 或 `dict` 形式，两种形式完全等价
    -   配置项项类归属 `options` 模块中
        -   全局配置项 `options.global_options`：配置某图表全局
            -   可通过 `charts.charts.chart.set_global_options` 方法设置
        -   系列配置项 `options.series_options`：配置图表中某个系列数据
            -   可通过 `charts.charts.chart.set_series_options` 方法设置
    -   `set_global_options`、`set_series_options` 方法中
        -   对应配置项 `XXXOpts`、`XXXItems` 参数名一般为 `xxx_opts`、`xxx_items`
            -   除 `xaxis_opts`、`yaxis_opts` 等

> - *PyEcharts* 数据格式：<https://pyecharts.org/#/zh-cn/data_format>
> - *PyEcharts* 参数配置：<https://pyecharts.org/#/zh-cn/parameters>

### 全局配置项

| 配置项类                                       | 含义                           |
|------------------------------------------------|--------------------------------|
| `global_options.InitOpts`                      | 初始化配置项                   |
| `global_options.AnimationOpts`                 | 动画                           |
| `global_options.RenderOpts`                    | 渲染                           |
| `global_options.TooltipOpts`                   | 提示框                         |
| `global_options.TitleOpts`                     | 标题                           |
| `global_options.LegendOpts`                    | 图例                           |
| `global_options.ArialLabelOpts`                | 无障碍标签                     |
| `global_options.AriaDecalOpts`                 | 无障碍贴花                     |
| `global_options.ToolBoxFeatureSaveAsImageOpts` | 工具箱保存图片                 |
| `global_options.ToolBoxFeatureRestoreOpts`     | 工具箱还原配置                 |
| `global_options.ToolBoxFeatureDataViewOpts`    | 工具箱数据视图工具             |
| `global_options.ToolBoxFeatureDataZoomOpts`    | 工具箱区域缩放                 |
| `global_options.ToolBoxFeatureMagicTypeOpts`   | 工具箱动态类型切换             |
| `global_options.ToolBoxFeatureBrushOpts`       | 工具箱选框组件                 |
| `global_options.ToolBoxFeatureOpts`            | 工具箱工具配置                 |
| `global_options.ToolBoxOpts`                   | 工具箱配置项                   |
| `global_options.BrushOpts`                     | 区域组件选择组件               |
| `global_options.DataZoomOpts`                  | 区域缩放                       |
| `global_options.VisualMapOpts`                 | 视觉区块                       |
| `global_options.AxisLineOpts`                  | 坐标轴线                       |
| `global_options.AxisTickOpts`                  | 坐标轴刻度                     |
| `global_options.AxisPointerOpts`               | 坐标轴指示器                   |
| `global_options.AxisOpts`                      | 坐标轴                         |
| `global_options.SingleAxisOpts`                | 单轴                           |
| `global_options.PolarOpts`                     | 极坐标系                       |
| `global_options.DataSetTransformOpts`          | 数据集转换                     |
| `global_options.EmphasisOpts`                  | 高亮状态下多边形和标签样式     |
| `global_options.Emphasis3DOpts`                | 3D图高亮状态下多边形和标签样式 |
| `global_options.BlurOpts`                      | 淡出状态下多边形和标签样式     |
| `global_options.SelectOpts`                    | 选中状态下多边形和标签样式     |
| `global_options.TreeLeavesOpts`                | *Tree Leaves* 组件配置         |

| 原生图形配置项类                     | 含义                 |
|--------------------------------------|----------------------|
| `charts_options.GraphicGroup`        | 原生图形元素组件     |
| `charts_options.GraphicItem`         | 原生图形             |
| `charts_options.GraphBasicStyleOpts` | 原生图形基础配置     |
| `charts_options.GraphShapeOpts`      | 原生图形形状配置     |
| `charts_options.GraphImage`          | 原生图形图片配置     |
| `charts_options.GraphImageStyleOpts` | 原生图形图片样式     |
| `charts_options.GraphText`           | 原生图形文本配置     |
| `charts_options.GraphTextStyleOpts`  | 原生图形文本样式配置 |
| `charts_options.GraphicRect`         | 原生图形矩形配置     |

> - 全局配置项：<https://pyecharts.org/#/zh-cn/global_options>

### 系列配置项

| 配置项类                                | 含义                     |
|-----------------------------------------|--------------------------|
| `series_options.ItemStyleOpts`          | 图元样式                 |
| `series_options.TextStyleOpts`          | 文字样式                 |
| `series_options.LabelOpts`              | 标签                     |
| `series_options.LineStyle`              | 线样式                   |
| `series_options.SplitLineOpts`          | 分割线配置               |
| `series_options.SplitAreaOpts`          | 分隔区域配置             |
| `series_options.MarkPointItem`          | 标记点数据项             |
| `series_options.MarkPointOpts`          | 标记点数据项             |
| `series_options.MarkLineItem`           | 标记线数据项             |
| `series_options.MarkLineOpts`           | 标记线样式               |
| `series_options.MarkAreaItem`           | 标记区域数据项           |
| `series_options.MarkAreaOpts`           | 标记区域样式             |
| `series_options.MinorTickOpts`          | 次级刻度配置             |
| `series_options.MinorSplitLineOpts`     | 次级分割线配置           |
| `series_options.Line3DEffectOpts`       | 3D样式                   |
| `series_options.GraphGLForceAltas2Opts` | *GraphGL Atlas* 算法配置 |

> - 系列配置项：<https://pyecharts.org/#/zh-cn/series_options>

## 图表支持

| 方法                                   | 含义                                                         |
|----------------------------------------|--------------------------------------------------------------|
| `Base.add_js_funcs(*fns)`              | 新增 *JS* 代码，将渲染进 *HTML* 中执行                       |
| `Base.add_js_events(*fns)`             | 新增 *JS* 事件函数，将被渲染在 `setOption` 后执行            |
| `Base.set_colors(colors)`              | 设置全局 `Label` 颜色                                        |
| `Base.get_options()`                   | 获取全局 `options` 字典                                      |
| `Base.dump_options()`                  | 获取全局 `options` *JSON*                                    |
| `Base.dump_options_with_quotes()`      | 获取全局 `options` *JSON*（*JS* 函数带引号）                 |
| `Base.render(path,template_name,env)`  | 渲染为 *HTML* 文件                                           |
| `Base.render_embed(template_name,env)` | 渲染为 *HTML* 字符串                                         |
| `Base.render_notebook()`               | 渲染至 *Notebook*                                            |
| `Base.load_javascript()`               | 加载 *JS* 资源（仅在 *JupyterLab* 环境中需在首次渲染前加载） |

-   `pyecharts.Base` 是所有图表的基类

###    直角坐标系图

| 方法                                                     | 含义            |
|----------------------------------------------------------|-----------------|
| `RectChart.extend_axis(xaxis_data,xaxis,yaxis)`          | 扩展*X/Y* 轴    |
| `RectChart.add_xaxis(xaxis_data)`                        | 新增 *X* 轴数据 |
| `RectChart.reversal_axis()`                              | 翻转 *X/Y* 轴   |
| `RectChart.overlap(chart)`                               | 层叠多图        |
| `RectChart.add_dataset(source,dimensions,source_header)` | 添加数据集      |

-   直角坐标系图表均继承自 `charts.RectChart`
    -   `RectChart.overlap` 方法似乎是合并 Y 轴数据、绘制
        -   `overlap` 返回结果为覆盖图表
        -   被重叠图表的全局配置整体被覆盖，同覆盖图表

| 图表       | 类                                |
|------------|-----------------------------------|
| 条形图     | `charts.Bar(init_opts)`           |
| 象形条形图 | `charts.PictoriaBar(init_opts)`   |
| 散点图     | `charts.Scatter(init_opts)`       |
| 涟漪散点图 | `charts.EffectScatter(init_opts)` |
| 折线图     | `charts.Line(init_opts)`          |
| 箱线图     | `charts.Boxplot(init_opts)`       |
| *K* 线图   | `charts.Kline(init_opts)`         |
| 热力图     | `charts.HeatMap(init_opts)`       |

> - *PyEcharts* 直角坐标系图：<https://pyecharts.org/#/zh-cn/rectangular_charts>

###    其他基本图表

| 图表       | 类                             |
|------------|--------------------------------|
| 日历图     | `charts.Calender(init_opts)`   |
| 漏斗图     | `charts.Funnel(init_opts)`     |
| 仪表盘     | `charts.Gauge(init_opts)`      |
| 关系图     | `charts.Graph(init_opts)`      |
| 水球图     | `charts.Liquid(init_opts)`     |
| 平行竖线图 | `charts.Parallel(init_opts)`   |
| 饼图       | `charts.Pie(init_opts)`        |
| 极坐标图   | `charts.Polar(init_opts)`      |
| 雷达图     | `charts.Radar(init_opts)`      |
| 桑基图     | `charts.Sankey(init_opts)`     |
| 旭日图     | `charts.Sunburst(init_opts)`   |
| 河流图     | `charts.ThemeRiver(init_opts)` |
| 词云图     | `charts.WordCloud(init_opts)`  |
| 树图       | `charts.Tree(init_opts)`       |
| 矩形树图   | `charts.TreeMap(init_opts)`    |

> - *PyEcharts* 基本图表：<https://pyecharts.org/#/zh-cn/basic_charts>
> - *PyEcharts* 树形图：<https://pyecharts.org/#/zh-cn/tree_charts>

###    地理图表

| 图表       | 类                                                  |
|------------|-----------------------------------------------------|
| 地理坐标系 | `charts.Geo(init_opts,is_ignore_nonexistent_coord)` |
| 地图       | `charts.Map(init_opts)`                             |
| 百度地图   | `charts.BMap(init_opts)`                            |

> - *PyEcharts* 地理图表：<https://pyecharts.org/#/zh-cn/geography_charts>

###    *3D* 图表

| 图表        | 类                            |
|-------------|-------------------------------|
| *3D* 柱状图 | `charts.Bar3D(init_opts)`     |
| *3D* 折线图 | `charts.Line3D(init_opts)`    |
| *3D* 散点图 | `charts.Scatter3D(init_opts)` |
| *3D* 曲面图 | `charts.Surface3D(init_opts)` |
| *3D* 路径图 | `charts.Lines3D(init_opts)`   |
| 三维地图    | `charts.Map3D(init_opts)`     |
| *GL* 关系图 | `charts.GraphGL(init_opts)`   |

> - *PyEcharts 3D* 图表：<https://pyecharts.org/#/zh-cn/3d_charts>

###    组合图表

| 组合方式 | 类                                                |
|----------|---------------------------------------------------|
| 网格组合 | `charts.Grid(init_opts)`                          |
| 顺序分页 | `charts.Page(page_title,js_host,interval,layout)` |
| 选项卡   | `charts.Tab(page_title,js_host)`                  |
| 顺序轮播 | `charts.Timeline(init_opts)`                      |

-   组合图表整体数据组合渲染
    -   组合中多个图表的的坐标共用坐标轴序号，绑定数据时注意全局考虑序号

| 配置项类                        | 含义              |
|---------------------------------|-------------------|
| `global_options.GridOpts`       | `Grid` 配置项     |
| `charts_options.PageLayoutOpts` | `Page` 布局配置项 |

> - *PyEcharts* 组合图表：<https://pyecharts.org/#/zh-cn/composite_charts>

###    网页组件

| 组件 | 类                                     |
|------|----------------------------------------|
| 表格 | `components.Table(page_title,js_host)` |
| 图像 | `components.Image(page_title,js_host)` |


> - 当前 `Table`、`Image` 与 `Page` 不兼容
> - *PyEcharts* 网页组件：<https://pyecharts.org/#/zh-cn/html_components>

