---
title: 
categories:
  - 
tags:
  - 
date: 2024-11-15 15:57:17
updated: 2024-11-19 07:22:08
toc: true
mathjax: true
description: 
---

##  金融指数

> - 内置指标：<https://cn.tradingview.com/support/folders/43000587405/>
> - FMLabs Indicator Reference：<https://www.fmlabs.com/reference/default.htm>，指标逻辑查询
> - talib指标公式及释义整理：<https://www.cnblogs.com/forest128/p/13823649.html>，简略解释

### *Overlap Studies*

| `talib`                                | 指标                                | 名称                 | 描述                                        |
|----------------------------------------|-------------------------------------|----------------------|---------------------------------------------|
| `ta.MA(close,timerperiod=30,matype=0)` | *Moving Average*                    | 移动平均             | 通用函数名 `matype` 取不同值即对应不同均线  |
| `ta.SMA(close,timeperiod=30)`          | *Simple Moving Average*             | 简单移动平均         |                                             |
| `ta.EMA(close,timeperiod=30)`          | *Exponential Moving Average*        | 指数移动平均         |                                             |
| `ta.WMA(close,timeperiod=30)`          | *Weighted Moving Average*           | 加权移动平均         |                                             |
| `ta.DEMA(close,timeperiod=30)`         | *Double Exponential Moving Average* | 双重指数平均         | `2 * EMA - EMA(EMA)`，较 `EMA` 更接近价格线 |
| `ta.TEMA(close,timeperiod=30)`         | *Double Exponential Moving Average* | 三重指数平均         | `2 * EMA - EMA(EMA)`，较 `EMA` 更接近价格线 |
| `ta.TRIMA(close,timeperiod=30)`        | *Double Exponential Moving Average* | 三角移动平均         | `SMA(SMA(n//2), n//2 + 1)`，较 `EMA` 更接近价格线 |
| `ta.KAMA(close,timeperiod=30)`         | *Kaufmand Adaptive Moving Average*  | 考夫曼自适应移动平均 |                                             |
| `ta.MAMA(close,timeperiod=30)`         | *MESA Adaptive Moving Average*      |                      |                                             |
| `ta.T3(close,timeperiod-30,vfactor=0)` | *Triple Exponential Moving Average* | 三重指数平均         |                                             |

$$\begin{align*}
SMA_t = \frac {\sum_{i=t-n}^t close_i} n \\
\end{align*}$$

-   重叠指标用途说明
    -   移动平均线指标
        -   确认、跟踪和判断趋势，提示买入、卖出信号
        -   在单边市场行情中把握市场机会、规避风险
    -   函数说明
        -   `ta.MA`：移动均线通用函数，`matype` 参数取值对应不同移动平均线（按上序 `0-8`，缺省即 `ta.SMA`）
        -   `ta.EMA`: 近期数据权重指数增加，对数据变化更敏感
        -   `ta.DEMA`：`2 * EMA - EMA(EMA)`，较 `EMA` 对数据变化更敏感
        -   `ta.DEMA`：`3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))`，对数据变化敏感介于 `EMA`、`DMEA` 之间
        -   `ta.TRRIMA`：权重按三角形分配、中间权重更高的加权均线
        -   `ta.MAVP`：为每个取值配置独立移动窗口


| `talib`                                                      | 指标                                         | 名称             | 描述                       |
|--------------------------------------------------------------|----------------------------------------------|------------------|----------------------------|
| `ta.BBANDS(close,timeperiod=5,nbdevup=2,nbdevdn=2,matype=0)` | *Bollinger Bands*                            | 布林线           | 移动平均值、标准差置信区间 |
| `ta.MAVP(close,periods,minperiod=2,maxperiod=30,matype=0)`   | *Moving Average with Variable Period*        | 可变周期移动平均 |                            |
| `ta.MIDPOINT`                                                | *Midpoint Over Period                        |                  |                            |
| `ta.MIDPrice`                                                | *Midpoint Price Over Period*                 |                  |                            |
| `ta.SAR`                                                     | *Parabolic SAR*                              | 抛物线指标       |                            |
| `ta.SAREXT`                                                  | *Parabolic SAR - Extended*                   |                  |                            |
| `ta.HT_TRENDLINE`                                            | *Hilbert Transformer - Instaneous Trendline* | 希尔伯特瞬时变换 |

> - 重叠指标：<https://blog.csdn.net/weixin_43420026/article/details/118462233>，包含指标计算逻辑

### *Monmentum Indicators*

| `talib`       | 指标                                                     | 名称 | 描述 |
|---------------|----------------------------------------------------------|------|------|
| `ta.ADX`      | *Average Directional Movement Index*                     |
| `ta.ADXR`     | *Average Directional Movement Index Rating*              |
| `ta.APO`      | *Absolute Price Oscillator*                              |
| `ta.AROON`    | *Aroon*                                                  |
| `ta.AROONOSC` | *Aroon Oscillator*                                       |
| `ta.BOP`      | *Balance Of Power*                                       |
| `ta.CCI`      | *Commodity Channel Index*                                |
| `ta.CMO`      | *Chande Momentum Oscillator*                             |
| `ta.DX`       | *Directional Movement Index*                             |
| `ta.MACD`     | *Moving Average Convergence/Divergence*                  |
| `ta.MACDEXT`  | *MACD with controllable MA type*                         |
| `ta.MACDFIX`  | *Moving Average Convergence/Divergence Fix 12/26*        |
| `ta.MFI`      | *Money Flow Index*                                       |
| `ta.MINUS_DI` | *Minus Directional Indicator*                            |
| `ta.MINUS_DM` | *Minus Directional Movement*                             |
| `ta.MOM`      | *Momentum*                                               |
| `ta.PLUS_DI`  | *Plus Directional Indicator*                             |
| `ta.PLUS_DM`  | *Plus Directional Movement*                              |
| `ta.PPO`      | *Percentage Price Oscillator*                            |
| `ta.ROC`      | *Rate of change : ((price/prevPrice)-1)*100*             |
| `ta.ROCP`     | *Rate of change Percentage: (price-prevPrice)/prevPrice* |
| `ta.ROCR`     | *Rate of change ratio: (price/prevPrice)*                |
| `ta.ROCR100`  | *Rate of change ratio 100 scale: (price/prevPrice)*100*  |
| `ta.RSI`      | *Relative Strength Index*                                |
| `ta.STOCH`    | *Stochastic*                                             |
| `ta.STOCHF`   | *Stochastic Fast*                                        |
| `ta.STOCHRSI` | *Stochastic Relative Strength Index*                     |
| `ta.TRIX`     | *1-day Rate-Of-Change (ROC) of a Triple Smooth EMA*      |
| `ta.ULTOSC`   | *Ultimate Oscillator*                                    |
| `ta.WILLR`    | *Williams' %R*                                           |

### *Volume Indicators*

| `talib` | 指标                     | 名称 | 描述 |
|---------|--------------------------|------|------|
| `AD`    | *Chaikin A/D Line*       |
| `ADOSC` | *Chaikin A/D Oscillator* |
| `OBV`   | *On Balance Volume*      |

### *Price Transform*

| `talib`    | 指标                   | 名称 | 描述 |
|------------|------------------------|------|------|
| `AVGPRICE` | *Average Price*        |
| `MEDPRICE` | *Median Price*         |
| `TYPPRICE` | *Typical Price*        |
| `WCLPRICE` | *Weighted Close Price* |

### *Cycle Indicators*

| `talib`        | 指标                                        | 名称 | 描述 |
|----------------|---------------------------------------------|------|------|
| `HT_DCPERIOD`  | *Hilbert Transform - Dominant Cycle Period* |
| `HT_DCPHASE`   | *Hilbert Transform - Dominant Cycle Phase*  |
| `HT_PHASOR`    | *Hilbert Transform - Phasor Components*     |
| `HT_SINE`      | *Hilbert Transform - SineWave*              |
| `HT_TRENDMODE` | *Hilbert Transform - Trend vs Cycle Mode*   |

### *Pattern Recognition*

| `talib`               | 指标                                                    | 名称 | 描述 |
|-----------------------|---------------------------------------------------------|------|------|
| `CDL2CROWS`           | *Two Crows*                                             |
| `CDL3BLACKCROWS`      | *Three Black Crows*                                     |
| `CDL3INSIDE`          | *Three Inside Up/Down*                                  |
| `CDL3LINESTRIKE`      | *Three-Line Strike*                                     |
| `CDL3OUTSIDE`         | *Three Outside Up/Down*                                 |
| `CDL3STARSINSOUTH`    | *Three Stars In The South*                              |
| `CDL3WHITESOLDIERS`   | *Three Advancing White Soldiers*                        |
| `CDLABANDONEDBABY`    | *Abandoned Baby*                                        |
| `CDLADVANCEBLOCK`     | *Advance Block*                                         |
| `CDLBELTHOLD`         | *Belt-hold*                                             |
| `CDLBREAKAWAY`        | *Breakaway*                                             |
| `CDLCLOSINGMARUBOZU`  | *Closing Marubozu*                                      |
| `CDLCONCEALBABYSWALL` | *Concealing Baby Swallow*                               |
| `CDLCOUNTERATTACK`    | *Counterattack*                                         |
| `CDLDARKCLOUDCOVER`   | *Dark Cloud Cover*                                      |
| `CDLDOJI`             | *Doji*                                                  |
| `CDLDOJISTAR`         | *Doji Star*                                             |
| `CDLDRAGONFLYDOJI`    | *Dragonfly Doji*                                        |
| `CDLENGULFING`        | *Engulfing Pattern*                                     |
| `CDLEVENINGDOJISTAR`  | *Evening Doji Star*                                     |
| `CDLEVENINGSTAR`      | *Evening Star*                                          |
| `CDLGAPSIDESIDEWHITE` | *Up/Down-gap side-by-side white lines*                  |
| `CDLGRAVESTONEDOJI`   | *Gravestone Doji*                                       |
| `CDLHAMMER`           | *Hammer*                                                |
| `CDLHANGINGMAN`       | *Hanging Man*                                           |
| `CDLHARAMI`           | *Harami Pattern*                                        |
| `CDLHARAMICROSS`      | *Harami Cross Pattern*                                  |
| `CDLHIGHWAVE`         | *High-Wave Candle*                                      |
| `CDLHIKKAKE`          | *Hikkake Pattern*                                       |
| `CDLHIKKAKEMOD`       | *Modified Hikkake Pattern*                              |
| `CDLHOMINGPIGEON`     | *Homing Pigeon*                                         |
| `CDLIDENTICAL3CROWS`  | *Identical Three Crows*                                 |
| `CDLINNECK`           | *In-Neck Pattern*                                       |
| `CDLINVERTEDHAMMER`   | *Inverted Hammer*                                       |
| `CDLKICKING`          | *Kicking*                                               |
| `CDLKICKINGBYLENGTH`  | *Kicking - bull/bear determined by the longer marubozu* |
| `CDLLADDERBOTTOM`     | *Ladder Bottom*                                         |
| `CDLLONGLEGGEDDOJI`   | *Long Legged Doji*                                      |
| `CDLLONGLINE`         | *Long Line Candle*                                      |
| `CDLMARUBOZU`         | *Marubozu*                                              |
| `CDLMATCHINGLOW`      | *Matching Low*                                          |
| `CDLMATHOLD`          | *Mat Hold*                                              |
| `CDLMORNINGDOJISTAR`  | *Morning Doji Star*                                     |
| `CDLMORNINGSTAR`      | *Morning Star*                                          |
| `CDLONNECK`           | *On-Neck Pattern*                                       |
| `CDLPIERCING`         | *Piercing Pattern*                                      |
| `CDLRICKSHAWMAN`      | *Rickshaw Man*                                          |
| `CDLRISEFALL3METHODS` | *Rising/Falling Three Methods*                          |
| `CDLSEPARATINGLINES`  | *Separating Lines*                                      |
| `CDLSHOOTINGSTAR`     | *Shooting Star*                                         |
| `CDLSHORTLINE`        | *Short Line Candle*                                     |
| `CDLSPINNINGTOP`      | *Spinning Top*                                          |
| `CDLSTALLEDPATTERN`   | *Stalled Pattern*                                       |
| `CDLSTICKSANDWICH`    | *Stick Sandwich*                                        |
| `CDLTAKURI`           | *Takuri (Dragonfly Doji with very long lower shadow)*   |
| `CDLTASUKIGAP`        | *Tasuki Gap*                                            |
| `CDLTHRUSTING`        | *Thrusting Pattern*                                     |
| `CDLTRISTAR`          | *Tristar Pattern*                                       |
| `CDLUNIQUE3RIVER`     | *Unique 3 River*                                        |
| `CDLUPSIDEGAP2CROWS`  | *Upside Gap Two Crows*                                  |
| `CDLXSIDEGAP3METHODS` | *Upside/Downside Gap Three Methods*                     |

### *Statistic Function*

| `talib`               | 指标                                    | 名称 | 描述 |
|-----------------------|-----------------------------------------|------|------|
| `BETA`                | *Beta*                                  |
| `CORREL`              | *Pearson's Correlation Coefficient (r)* |
| `LINEARREG`           | *Linear Regression*                     |
| `LINEARREG_ANGLE`     | *Linear Regression Angle*               |
| `LINEARREG_INTERCEPT` | *Linear Regression Intercept*           |
| `LINEARREG_SLOPE`     | *Linear Regression Slope*               |
| `STDDEV`              | *Standard Deviation*                    |
| `TSF`                 | *Time Series Forecast*                  |
| `VAR`                 | *Variance*                              |
