---
title: Scipy
categories:
  - Python
  - Scipy
tags:
  - Python
  - Data Analysis
  - Data Visualization
  - Numpy Array
  - Scipy
date: 2022-12-19 14:07:33
updated: 2023-12-30 16:02:20
toc: true
mathjax: true
description: 
---

##  *Scipy*

| 顶层                                         | 描述               |
|----------------------------------------------|--------------------|
| `LowLevelCallable(function[,user_data,...])` | 低层回调函数       |
| `show_config([mode])`                        | *Scipy* 依赖       |
| `test`                                       | 为命名空间执行测试 |

| 子模块                     | 描述                   |
|----------------------------|------------------------|
| `cluster`                  | 聚类                   |
| `cluster.vq`               | 向量量化（*K-means*）  |
| `cluster.hierarchy`        | 层次聚类               |
| `constants`                | 数学、物理常量、单位   |
| `datesets`                 | 数据集                 |
| `fft`                      | 离散傅里叶、变换       |
| `fftpack`                  | （传统）离散傅里叶变换 |
| `integrate`                | 数值积分               |
| `interpolate`              | 插值                   |
| `io`                       | 科学数据读写           |
| `io.arff`                  | *Weka* 数据            |
| `io.matlab`                | *MatLab* 数据          |
| `io.wavfile`               | *WAV* 数据             |
| `linalg`                   | 线代                   |
| `linalg.blas`              | 低层 *BLAS* 函数       |
| `linalg.cython_blas`       | *Cython BLAS*          |
| `linalg.lapack`            | 底层 *LAPACK* 函数     |
| `linalg.cython_lapack`     | *Cython LAPACK*        |
| `linalg.interpolative`     | 插值分解               |
| `misc`                     | 功能模块               |
| `ndimage`                  | N 维图像处理、插值     |
| `odr`                      | 正交距离回归           |
| `optimize`                 | 数值优化               |
| `optimize.cython_optimize` | *Cython* 数值优化      |
| `signal`                   | 信号处理               |
| `signal.windows`           | 信号窗口               |
| `sparse`                   | 稀疏矩阵、线代、图算法 |
| `sparse.linalg`            | 稀疏线代               |
| `sparse.csgraph`           | 压缩稀疏图算法         |
| `spatial`                  | 空间数据结构、算法     |
| `spatial.distance`         | 空间距离               |
| `spatial.transform`        | 空间变换               |
| `special`                  | 特殊函数               |
| `stats`                    | 统计函数               |
| `stats.contingency`        | 列联表                 |
| `stats.distributions`      | 分布                   |
| `stats.mstats`             | 支持遮盖数组的统计函数 |
| `stats.qmc`                | 模拟蒙特卡洛算法       |
| `stats.sampling`           | 随机数生成             |

> - *Vector Quantization* 向量量化：将向量空间中点使用其中有限子集编码
> - *Scipy Reference*：<https://docs.scipy.org/doc/scipy/reference/>

##  Cluster

| `cluster.vq` 函数                          | 描述                |
|--------------------------------------------|---------------------|
| `whiten(obs[,check_finite])`               | 逐特征正则化        |
| `vq(obs,code_book[,check_finite])`         | 向量量化            |
| `kmeans(obs,k_or_guess[,iter,thresh,...])` | 执行 *K-means* 聚类 |
| `kmeans2(data,k[,iter,thresh,minit,...])`  | *K-means* 分类      |

| `cluster.hierarchy` 函数                    | 描述           |
|---------------------------------------------|----------------|
| `fcluster(Z,t[,criterion,depth,R,monocrit)` |                |
| `fclusterdata(X,t[,criterion,metric,...])`  | 聚类           |
| `leader(Z,T)`                               | 层次聚类根节点 |

> - *K-means Clustering and Vector Quantization*：<https://docs.scipy.org/doc/scipy/reference/cluster.vq.html>
> - *Hierarchical Clustering*：<https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>

##  Stats

| `stats` 函数                              | 描述                 |
|-------------------------------------------|----------------------|
| `rv_continuous([momtype,a,b,xtol,...])`   | 连续随机变量通用类   |
| `rv_discrete([a,b,name,badvalue,...])`    | 离散随即变量通用类   |
| `rv_histogram([histgram,*args[,density])` | 直方图描述随机变量类 |

-   说明
    -   模块中随机变量为 `rv_continuous`、`rv_discrete` 类子类实例

> - 随机抽样算法TODO：<https://zhuanlan.zhihu.com/p/37121528>
> - 拒绝抽样：<https://gaolei786.github.io/statistics/reject.html>

### 连续分布

| 单变量连续分布        | 描述             |
|-----------------------|------------------|
| `alpha`               | *alpha* 分布实例 |
| `beta`                | *beta* 分布实例  |
| `betaprime`           |                  |
| `anglit`              |                  |
| `arcsine`             |                  |
| `argus`               |                  |
| `bradford`            |                  |
| `burr`                |                  |
| `burr12`              |                  |
| `cauchy`              |                  |
| `halfcauchy`          |                  |
| `foldcauchy`          |                  |
| `wrapcauchy`          |                  |
| `skewcauchy`          |                  |
| `studentized_range`   |                  |
| `t`                   |                  |
| `chi`                 |                  |
| `chi2`                |                  |
| `f`                   |                  |
| `consine`             |                  |
| `crystalball`         |                  |
| `erlang`              |                  |
| `expon`               |                  |
| `exponnorm`           |                  |
| `exponpow`            |                  |
| `genexpon`            |                  |
| `truncexpon`          |                  |
| `fatiguelife`         |                  |
| `fisk`                |                  |
| `genextreme`          |                  |
| `gausshyper`          |                  |
| `geninvgauss`         |                  |
| `invgauss`            |                  |
| `recipinvgauss`       |                  |
| `gamma`               |                  |
| `dgamma`              |                  |
| `gengamma`            |                  |
| `invgamma`            |                  |
| `loggamma`            |                  |
| `genhyperbolic`       |                  |
| `gibrat`              |                  |
| `gompertz`            |                  |
| `gumbel_r`            |                  |
| `gumbel_l`            |                  |
| `hypsecant`           |                  |
| `johnsonsb`           |                  |
| `johnsonsu`           |                  |
| `kappa4`              |                  |
| `kappa3`              |                  |
| `ksone`               |                  |
| `kstwo`               |                  |
| `kstwobign`           |                  |
| `laplace`             |                  |
| `laplace_asynmmetric` |                  |
| `loglaplace`          |                  |
| `levy`                |                  |
| `levy_l`              |                  |
| `levy_stable`         |                  |
| `logistic`            |                  |
| `halflogistic`        |                  |
| `genlogistic`         |                  |
| `genhalflogistic`     |                  |
| `loguniform`          |                  |
| `lomax`               |                  |
| `maxwell`             |                  |
| `mielke`              |                  |
| `moyal`               |                  |
| `nakagami`            |                  |
| `ncx2`                |                  |
| `ncf`                 |                  |
| `nct`                 |                  |
| `norm`                |                  |
| `truncnorm`           |                  |
| `norminvgauss`        |                  |
| `halfnorm`            |                  |
| `halfgennorm`         |                  |
| `foldnorm`            |                  |
| `gennorm`             |                  |
| `skewnorm`            |                  |
| `lognorm`             |                  |
| `pareto`              |                  |
| `genpareto`           |                  |
| `truncpareto`         |                  |
| `pearson3`            |                  |
| `powerlaw`            |                  |
| `powerlognorm`        |                  |
| `powernorm`           |                  |
| `rdist`               |                  |
| `rayleigh`            |                  |
| `rel_breitwigner`     |                  |
| `rice`                |                  |
| `semicircular`        |                  |
| `trapezoid`           |                  |
| `triang`              |                  |
| `tukeylambda`         |                  |
| `uniform`             |                  |
| `vonmises`            |                  |
| `vonmises_line`       |                  |
| `wald`                |                  |
| `weibull_min`         |                  |
| `weibull_max`         |                  |
| `truncweibull_min`    |                  |
| `dweibull`            |                  |
| `invweibull`          |                  |
| `exponweib`           |                  |

| 分布方法                                                | 描述                            |
|---------------------------------------------------------|---------------------------------|
| `rvs(a,b[,loc,scale,size,random_state])`                | 随机变量                        |
| `pdf(x,a,b[,loc,scale])`                                | 概率密度                        |
| `logpdf(x,a,b[,loc,scale])`                             | 对数概率密度                    |
| `cdf(x,a,b[,loc,scale])`                                | 概率分布                        |
| `logcdf(x,a,b[,loc,scale])`                             | 对数概率分布                    |
| `sf(x,a,b[,loc,scale])`                                 | 生存函数 `1 - cdf`              |
| `logsf(x,a,b[,loc,scale])`                              | 对数生存函数                    |
| `ppf(q,a,b[,loc,scale])`                                | 分位数点（分布函数逆）          |
| `ppf(q,a,b[,loc,scale])`                                | 生存函数逆                      |
| `moment(order,a,b[,loc,scale])`                         | 非中心矩                        |
| `stat(a,b[,loc,scale,moments])`                         | 前4阶矩：均值、方差、偏差、峰度 |
| `entropy(a,b[,loc,scale])`                              | 信息熵                          |
| `fit(data)`                                             | 参数拟合                        |
| `expect(func,args[,loc,scale,lb,ub,conditional,**kwds)` | 函数期望                        |
| `median(a,b[,loc,scale])`                               | 中位数                          |
| `mean(a,b[,loc,scale])`                                 | 均值                            |
| `var(a,b[,loc,scale])`                                  | 方差                            |
| `std(a,b[,loc,scale])`                                  | 标准差                          |
| `interval(confidence,a,b[,loc,scale])`                  | 中位数双边置信区间              |

-   说明
    -   分布实例为由对应分布类实例化而来，可通过调用其方法获取所需数据
        -   部分分布实例中部分方法可能不存在
    -   分布实例方法中参数
        -   `a,b` 为分布超参，数量、含义取决于具体分布
        -   其余参数为通用参数，但对部分分布实例可能不存在
    -   其他
        -   `rv_continious.pdf` 方法效率很低，比较通过概率密度函数计算有 50 倍差距

### 单变量离散分布

| 单变量离散分布          | 描述 |
|-------------------------|------|
| `bernoulli`             |      |
| `betabinom`             |      |
| `binom`                 |      |
| `boltzmann`             |      |
| `dlaplace`              |      |
| `geom`                  |      |
| `hypergeom`             |      |
| `logser`                |      |
| `nbinom`                |      |
| `nchypergeom_fisher`    |      |
| `nchypergeom_wallenius` |      |
| `nhypergeom`            |      |
| `planck`                |      |
| `poisson`               |      |
| `randint`               |      |
| `skellam`               |      |
| `yulesimon`             |      |
| `zipf`                  |      |
| `zipfian`               |      |

-   说明
    -   分布实例为由对应分布类实例化而来，可通过调用其方法获取所需数据
        -   分布方法及参数说明同单变量连续分布
        -   部分分布实例中部分方法可能不存在

### 多变量分布

| 多变量连续分布           | 描述 |
|--------------------------|------|
| `multivariate_normal`    |      |
| `matrix_normal`          |      |
| `dirichlet`              |      |
| `dirichlet_multinominal` |      |
| `wishart`                |      |
| `invwishart`             |      |
| `multinominal`           |      |
| `special_ortho_group`    |      |
| `ortho_group`            |      |
| `unitary_group`          |      |
| `random_correlation`     |      |
| `multivariate_t`         |      |
| `multivariate_hypergeom` |      |
| `random_table`           |      |
| `uniform_direction`      |      |
| `vonmises_fisher`        |      |

| 分布方法                         | 描述         |
|----------------------------------|--------------|
| `rvs(alpha[,size,random_state])` | 随机变量     |
| `pmf(x,alpha)`                   | 概率质量     |
| `logpmf(x,alpha)`                | 对数概率质量 |
| `pdf(x,alpha)`                   | 概率密度     |
| `logpdf(x,alpha)`                | 对数概率密度 |
| `cdf(x,alpha)`                   | 概率分布     |
| `logcdf(x,alpha)`                | 对数概率分布 |
| `mean(alpha)`                    | 均值         |
| `var(alpha)`                     | 方差         |
| `entropy(alpha)`                 | 信息熵       |

-   说明
    -   分布实例为由对应分布类实例化而来，可通过调用其方法获取所需数据
        -   部分分布实例中部分方法可能不存在
    -   分布实例方法中参数
        -   `alpha` 为分布超参，数量、含义取决于具体分布
        -   其余参数为通用参数，但对部分分布实例可能不存在

### 归纳统计

| 归纳统计                                       | 描述               |
|------------------------------------------------|--------------------|
| `describe(a[,axis,ddof,bias,nan_policy])`      | 多角度描述         |
| `gmean(a[,axis,dtype,weights,nan_policy,...])` | 几何均值           |
| `hmean(a[,axis,dtype,weights,nan_policy,...])` | 调和均值           |
| `pmean(a[,axis,dtype,weights,nan_policy,...])` | 幂均值             |
| `skew(a[,axis,bias,nan_policy,keepdims)`       | 偏度               |
| `kurtosis(a[,axis,fisher,bias,...])`           | 峰度               |
| `mode(a[,axis,nan_policy,keepdims])`           | 众数               |
| `variation(a[,axis,nan_policy,ddof,keepdims])` | 变异系数           |
| `expectile(a[,alpha,weights])`                 | 经验分布分位数     |
| `moment(a[,moment,axis,nan_policy,...])`       | 矩                 |
| `kstat(a[,axis,bias,nan_policy,keepdims)`      | *K* 统计量         |
| `kstatvar(a[,axis,bias,nan_policy,keepdims)`   | *K* 统计量方差     |
| `tmean(a[,limits,inclusive,axis])`             | 裁剪均值           |
| `tvar(a[,limits,inclusive,axis,ddof])`         | 裁剪方差           |
| `tstd(a[,limits,inclusive,axis,ddof])`         | 裁剪标准差         |
| `tsem(a[,limits,inclusive,axis,ddof])`         | 裁剪标准误         |
| `tmin(a[,lowerlimit,axis,inclusive,...])`      | 裁剪最小           |
| `tmax(a[,upperlimit,axis,inclusive,...])`      | 裁剪最大           |
| `find_repeats(arr)`                            | 重复值、次数       |
| `rankdata(a[,method,axis,nan_policy])`         | 秩（序数）         |
| `tiecorrect(rankvals)`                         |                    |
| `trim_mean(a,proportiontocut[,axis])`          | 比例裁剪均值       |
| `gstd(a[,axis,ddof])`                          | 几何标准差         |
| `iqr(x[,axis,rng,scale,nan_policy,...])`       | 四分位间距         |
| `sem(a[,axis,ddof,nan_policy,keepdims])`       | 标准误             |
| `bayes_mvs(data[,alpha])`                      | 均值贝叶斯置信水平 |
| `mvsdist(data)`                                |                    |
| `entropy(pk[,qk,base,axis])`                   | 信息熵、*KL* 散度  |
| `differential_entropy(values,*[,...])`         | 微分熵             |
| `median_abs_deviation(x[,axis,center,...])`    | 中值绝对离差       |

| 频率统计                                         | 描述           |
|--------------------------------------------------|----------------|
| `cumfreq(a[,numbins,defaultreallimits,weights])` | 累计频数       |
| `percentileofscore(a,score[,kind,nan_policy])`   | 百分位置       |
| `scoreatpercentile(a,per[,limit,...])`           | 百分位值       |
| `relfreq(a[,numbins,defaultreallimits,weights])` | 频率           |
| `binned_statistic(x,values[,statistic,...])`     | 按分箱统计     |
| `binned_statistic_2d(x,y,values[,...])`          | 二维按分箱统计 |
| `binned_statistic_dd(smaple,values[,...])`       | 多维按分箱统计 |

### 假设检验

| 单、成对样本假设检验                            | 描述                            |
|-------------------------------------------------|---------------------------------|
| `ttest_1samp(a,popmean[,axis,nan_policy,...])`  | *T* 检验                        |
| `ttest_rel(a,b[,axis,nan_policy,...])`          | 两组 *T* 检验                   |
| `binomtest(k,n[,p,alternative])`                | 概率检验                        |
| `skewtest(a[,axis,nan_policy,alternative])`     | 正态偏度检验                    |
| `kurtosistest(a[,axis,nan_policy,alternative])` | 正态峰度检验                    |
| `normaltest(a[,axis,nan_policy])`               | 正态分布检验                    |
| `jarque_bera(x,*[,axis,nan_policy,keepdims])`   | *JB* 拟合优度检验               |
| `shapiro(x)`                                    | *Shapiro-Wilk* 正态检验         |
| `anderson(x[,dist])`                            | *Anderson-Darling* 指定分布检验 |
| `cramervonmises(rvs,cdf[,args])`                | *CM* 拟合优度检验               |
| `ks_1samp(x,cdf[,args,alternative,method])`     | *KS* 拟合优度检验               |
| `goodness_of_fit(dist,data,*[,...])`            | 拟合优度检验                    |
| `chisquare(f_obsp[,f_exp,ddof,axis])`           | 卡方检验                        |
| `wilcoxon(x[,y,zero_method,corretion,...])`     | *Wilcoxon* 符号秩检验           |
| `power_divergence(f_obs[,fexp,ddof,axis,...])`  | *CR* 幂散度                     |

| 相关系数、相关性检验                              | 描述                                  |
|---------------------------------------------------|---------------------------------------|
| `linregress(x[,y,alternative])`                   | 线性最小平方回归                      |
| `pearsonr(x,y,*[,alternative,method])`            | *Pearson* 相关系数检验                |
| `spearmanr(a[,b,axis,nan_policy,alternative])`    | *Spearman* 相关检验                   |
| `pointbiserialr(x,y)`                             | 点双列相关系数检验                    |
| `kendalltau(x,y[,inital_lexosrt,...])`            | *Kendall tau* 相关检验                |
| `weightedtau(x,y[,rank,weighter,additive])`       | 加权 *Kendall tau* 相关系数检验       |
| `somersd(x[,y,alternative])`                      | *Somers' D* 序相关检验                |
| `siegelslopes(y[,x,method])`                      | *Siegel* 估计                         |
| `theilslopes(y[,x,method])`                       | *Theil-Sen* 估计                      |
| `page_trend_test(data[,ranked,...])`              | *Page's* 检验                         |
| `mulitscale_graphcorr(x,y[,...])`                 | *MGC* 检验                            |
| `chi2_contingency(observed[,correction,lambda_])` | 列联表卡方检验                        |
| `fisher_exact(table[,alternative])`               | *Fisher* 正确概率检验（22列联表）     |
| `barnard_exact(table[,alternative])`              | *Barnard* 正确概率检验（22列联表）    |
| `boschloo_exact(table[,alternative])`             | *Boschloo's* 正确概率检验（22列联表） |

| 成对样本独立性检验                              | 描述                           |
|-------------------------------------------------|--------------------------------|
| `ttest_ind(a,b[,axis,equal_var,...])`           | *T* 检验                       |
| `ttest_ind_from_stats(means1,std1,nobs1[,...])` | *T* 检验（通过描述统计值）     |
| `poisson_means_test(k1,n1,k2,n2,*[,...])`       | *Poisson* 均值检验（*E* 检验） |
| `mannwhitneyu(x,y[,use_continuity,...])`        | *Mann-Whitney U* 检验          |
| `ranksums(x,y[,alternative,axis,...])`          | *Wilcoxon* 秩和检验            |
| `brunnermunzel(x,y[,alternative,...])`          | *Bruner-Munzel* 检验           |
| `mood(x,y[,axis,alternative,...])`              | *Mood's* 检验                  |
| `ansari(x,y[,alternative])`                     | *Ansari-Bradley* 检验          |
| `cramervonmises_2samp(x,y[,method])`            | *CM* 拟合优度检验              |
| `ks_2samp(data1,data2[,alternative,method])`    | *KS* 检验                      |
| `kstest(rvs,cdf[,args,N,alternative,method])`   | *KS* 检验                      |

| 多组样本独立性检验                             | 描述                            |
|------------------------------------------------|---------------------------------|
| `f_oneway(*samples[,axis])`                    | 方差检验                        |
| `tukey_hsd(*args)`                             | *Tukey's HSD* 均值检验          |
| `dunnet(*samples,control[,alternative,...])`   | *Dunnett's* 均值检验            |
| `kruskal(*samples[,nan_policy,axis,keepdims])` | *Kraskal-Wallis H* 独立样本检验 |
| `alexandergovern(*sample[,nan_policy])`        | *Alexander Govern* 检验         |
| `fligner(*samples[,center,proportiontocut])`   | *Fligner-Killeen* 同方差检验    |
| `levene(*samples[,center,proportiontocut])`    | *Levene* 同方差检验             |
| `bartlett(*sample)`                            | *Bartlett* 同方差检验           |
| `median_test(*samples[,ties,corretion,...])`   | *Mood's* 中位数检验             |
| `friedmanchisqure(*samples)`                   | *Friedman* 重复值检验           |
| `anderson_ksamp(samples[,midrank])`            | *Anderson-Darling* 检验         |

| *Monte Carlo* 方法                             | 描述                       |
|------------------------------------------------|----------------------------|
| `monte_carlo_test(data,rvs,statistic,*[,...])` | *Monte Carlo* 假设检验     |
| `permutation_test(data,statistic,*[,...])`     | 全排列检验                 |
| `boostrap(data,statistic,*[,n_resamples,...])` | 双边自举置信区间           |
| `MonteCarloMethod([n_resamples,batch,rvs])`    | *Monte Carlo* 假设检验配置 |
| `PermutationMethod([n_resamples,batch,...])`   | 全排列假设检验配置         |
| `BoostrapMethod([n_resamples,batch,...])`      | 自举假设检验配置           |

| 多重假设检验                                 | 描述                   |
|----------------------------------------------|------------------------|
| `combine_pvalues(pvalues[,method,weights])`  | 多重假设检验 *P* 值    |
| `false_dicovery_control(ps,*[,axis,method])` | 根据 *FDR* 调整 *P* 值 |

> - *Statistical Functions*：<https://docs.scipy.org/doc/scipy/reference/stats.html>

### `stats.contingency` 列联

| `stats.contingency` 函数                            | 描述             |
|-----------------------------------------------------|------------------|
| `chi2_contingency(observed[,correction,lambda...])` | 列联表卡方检验   |
| `relative_risk(exposed_cases,exposed_total,...])`   | 2*2 列联相对风险 |
| `odds_ratio(table,*[,kind])`                        | 2*2 列联表几率   |
| `crosstab(*args[,levels,sparse])`                   | 列联表           |
| `association(observed[,method,corretion,...])`      | 名义变量相关系数 |
| `expected_freq(observed)`                           | 预期频数         |
| `margins(a)`                                        | 边际分布         |

> - *Contingency Table*：<https://docs.scipy.org/doc/scipy/reference/stats.contingency.html>

### `stats.qmc` 蒙特卡洛

| *Quasi-Monte Carlo*                              | 描述                    |
|--------------------------------------------------|-------------------------|
| `QMCEngine(d,*[,optimization,seed])`             | 拟 *Monte Carlo* 抽样类 |
| `Sobol(d,*[,scramble,bits,seed,optimization])`   | *Sobol* 序列            |
| `Halton(d,*[,scramble,optimization,seed])`       | *Halton* 序列           |
| `LatinHypercube(d,*[,centered,scramble,...])`    | *Latin Hypercube* 采样  |
| `PoissonDisk(d,*[,radius,hypersphere,...])`      | 泊松云盘采样            |
| `MultinomialQMC(pvals,n_trials,*[,engine,...])`  | 多项分布 *QMC* 采样     |
| `MultivariateNormalQMC(mean[,cov,cov_root,...])` | 多项正态分布 *QMC* 采样 |
| `discrepancy(sample,*[,iterative,method,...])`   | 样本差异                |
| `update_discrepancy(x_new,sample,initail_disc)`  | 更新中心差异            |
| `scale(sample,I_bounds,u_bounds,*[,reverse])`    | 采样放缩                |

### 其他统计函数

| 变换函数                                       | 描述                       |
|------------------------------------------------|----------------------------|
| `boxcox(x[,lambda,alpha,optimizer])`           | *Box-Cox* 变换             |
| `boxcox_normmax(x[,brack,method,optimizer])`   | 输入最优 *Box-Cox* 变换    |
| `boxcox_llf(lmb,data)`                         | *Box-Cox* 对数似然函数     |
| `yeojohnson(x[,lambda])`                       | *Yoe-Johnson* 幂变换       |
| `yeojohnson_normmax(x[,lambda])`               | 输入 *Yoe-Johnson* 幂变换  |
| `yeojohnson_llf(x[,lambda])`                   | *Yoe-Johnson* 对数似然函数 |
| `obrientransform(*samples)`                    | *O'Brien* 变换             |
| `sigmaclip(a[,low,high])`                      | 迭代方差裁剪               |
| `trimboth(a,proportiontocut[,axis])`           | 双边裁剪                   |
| `trim1(a,proportiontocut[,tail,axis])`         | 单边裁剪                   |
| `zmap(scores,compares[,axis,ddof,nan_policy])` | 相对 *Z-score*             |
| `zscore(a[,axis,ddof,nan_policy])`             | *Z-score*                  |
| `gzscore(a,*[,axis,ddof,nan_policy])`          | 几何标准差                 |

| 统计距离                                        | 描述                   |
|-------------------------------------------------|------------------------|
| `wasserstain_distance(u_values,v_values[,...])` | *Wasserstein* 距离     |
| `energy_distance(u_values,v_values[,...])`      | 能量距离               |
| `rvs_ratio_uniforms(pdf,umax,vmin,vmax[,...])`  | 概率密度随机数生成     |
| `fit(dist,data[,bounds,guess,method,...])`      | 拟合数据分布           |
| `ecdf(sample)`                                  | 经验累计分布           |
| `logrank(x,y[,alternative])`                    | *logrank* 检验生存比较 |

| 方向统计                                        | 描述               |
|-------------------------------------------------|--------------------|
| `directional_stats(samples,*[,axis,normalize])` | 平均方向、平均长度 |
| `circmean(samples[,high,low,axis,nan_policy])`  | （求和后）取模均值 |
| `circvar(samples[,high,low,axis,nan_policy)`    | 取模方差           |
| `circstd(samples[,high,low,axis,nan_policy)`    | 取模标准差         |

| 敏感度分析                                   | 描述                 |
|----------------------------------------------|----------------------|
| `sobol_indices(*,func,n[,dists,method,...])` | *Sobol* 全局敏感性   |
| `gaussian_kde(dataset[,bw_method,weights])`  | 高斯核高斯核密度估计 |

##  `scipy.optimize`

| 通用                                | 描述       |
|-------------------------------------|------------|
| `show_option([solver,method,disp])` | 可选项文档 |
| `OptimizeResult`                    | 优化结果类 |
| `OptimizeWarning`                   |            |

| 标量函数优化                                  | 描述           |
|-----------------------------------------------|----------------|
| `minimize_scalar(fun,bracket,bounds,...])`    | 单变量函数极小 |
| `minimize(fun,x0[,args,method,jac,hess,...])` | 多变量函数极小 |

| 限制类                                          | 描述         |
|-------------------------------------------------|--------------|
| `NonlinearConstraint(fun,lb,ub[,jac,...])`      | 非线性限制   |
| `linearConstraint(A[,lb,ub,keep_feasible,...])` | 线性限制     |
| `Bounds([lb,ub,keep_feasible])`                 | 变量取值范围 |

| 全局优化                                         | 描述 |
|--------------------------------------------------|------|
| `basinshopping(func,x0[,iter,T,stepsize,...])`   |      |
| `brute(func,ranges[,args,Ns,full_output,...])`   | 蛮力 |
| `differential_evolution(func,bounds[,args,...])` |      |
| `shgo(func,bounds[,args,constraints,n,...])`     |      |
| `dual_annealing(func,bounds[,args,...])`         |      |
| `direct(func,bounds,*[,args,eps,maxfun,...])`    |      |

| 最小二乘                                  | 描述                 |
|-------------------------------------------|----------------------|
| `least_squares(fun,x0[,jac,bounds,...])`  | 非线性最小二乘       |
| `nnls(A,b[,maxiter])`                     | 线性回归最小二乘求解 |
| `lsq_linear(A,b[,bounds,method,tol,...])` | 带限制线性最小二乘   |
| `curve_fit(f,xdata,ydata[,p0,sigma,...])` | 非线性最小二乘拟合   |


| 优化方法       | 描述 |
|----------------|------|
| `brent`        |      |
| `bounded`      |      |
| `golden`       |      |
| `Nelder-Mead`  |      |
| `Powell`       |      |
| `CG`           |      |
| `BFGS`         |      |
| `Newton-CG`    |      |
| `L-BFGS-B`     |      |
| `TNC`          |      |
| `COBYLA`       |      |
| `SLSQP`        |      |
| `trust-constr` |      |
| `dogleg`       |      |
| `trust-ncg`    |      |
| `trust-krylov` |      |
| `truct-exact`  |      |

| 标量求根                                       | 描述                            |
|------------------------------------------------|---------------------------------|
| `root_scalar(f[,args,method,bracket,...])`     | 标量函数求根                    |
| `brentq(f,a,b[,args,xtol,rtol,maxiter,...])`   | *Brent* 法求区间内根            |
| `brenth(f,a,b[,args,xtol,rtol,maxiter,...])`   | 双曲线外推 *Brent* 法求区间内根 |
| `ridder(f,a,b[,args,xtol,rtol,maxiter,...])`   | *Ridder's* 法求区间内根         |
| `bisect(f,a,b[,args,xtol,rtol,maxiter,...])`   | 二分法求区间内根                |
| `newton(func,x0[,fprime,args,tol,...])`        | 牛顿法求区间内根                |
| `toms748(f,a,b[,args,k,xtol,rtol,...])`        | *TOMS 748* 算法求根             |
| `RootResult(root,iterations,...)`              | 求根结果                        |
| `fixed_point(func,x0[,args,xtol,maxiter,...])` | 函数定点                        |

| 求根方法  | 描述 |
|-----------|------|
| `brentq`  |      |
| `brenth`  |      |
| `bisect`  |      |
| `ridder`  |      |
| `newton`  |      |
| `toms748` |      |
| `secant`  |      |
| `hally`   |      |

| 多维求根                                 | 描述         |
|------------------------------------------|--------------|
| `root(fun,x0[,args,method,jac,tol,...])` | 向量函数求根 |

| 求根方法         | 描述 |
|------------------|------|
| `hybr`           |      |
| `lm`             |      |
| `broyden1`       |      |
| `broyden2`       |      |
| `anderson`       |      |
| `linearmixing`   |      |
| `diagbroyden`    |      |
| `excitingmixing` |      |
| `krylov`         |      |
| `df-sane`        |      |

| 线性规划                                      | 描述             |
|-----------------------------------------------|------------------|
| `milp(c,*[,integrality,bounds,...])`          | 整数混合线性规划 |
| `linprog(c[,A_ub,b_ub,A_eq,b_eq,bounds,...])` | 线性规划         |

| 求解方法          | 描述 |
|-------------------|------|
| `simplex`         |      |
| `interir-point`   |      |
| `revised simplex` |      |
| `highs-ipm`       |      |
| `highs-ds`        |      |
| `highs`           |      |

| 赋值问题                                     | 描述 |
|----------------------------------------------|------|
| `linear_sum_assignment`                      |      |
| `quadratic_assignment(A,B[,method,options])` |      |

| 求解方法 | 描述 |
|----------|------|
| `faq`    |      |
| `2opt`   |      |

| 工具方法                                       | 描述 |
|------------------------------------------------|------|
| `approx_fprime(xk,f[,epsilon])`                |      |
| `check_grad(func,grad,x0,*args[,epsilon,...])` |      |
| `bracket(func[,xa,xb,args,grow_limit,...])`    |      |
| `line_search(f,myfprime,xk,pk[,gfk,...])`      |      |
| `LbfgsInvHessProduct(*args,**kwargs)`          |      |
| `HessianUpdateStrategy()`                      |      |
| `rosen(x)`                                     |      |
| `rosen_der(x)`                                 |      |
| `rosen_hess(x)`                                |      |
| `rosen_hess_prod(x,p)`                         |      |

##  `scipy.linalg`

| 基本函数                                        | 描述                                  |
|-------------------------------------------------|---------------------------------------|
| `inv(a[,overwrite_a,check_finite])`             | 逆矩阵                                |
| `solve(a,b[,lower,overwrite_a,...])`            | 方阵求解                              |
| `solve_banded(I_and_u,ab,b[,overwrite_ab,...])` |                                       |
| `solveh_banded(ab,b[,overwrite_ab,...])`        | 矩阵求解                              |
| `solve_circulant(c,b[,singular,tol,...])`       |                                       |
| `solve_triangular(a,b[,trans,lower,...])`       |                                       |
| `solve_toeplitz(c_or_cr,b[,check_finite])`      |                                       |
| `matmul_toeplitz(c_or_cr,x[,check_finite,...])` |                                       |
| `det(a[,overwrite_a,check_finite])`             |                                       |
| `norm(a[,ord,axis,keepdims,check_finite])`      |                                       |
| `lstsq(a,b[,cond,overwrite_a,...])`             | 最小二乘求解矩阵                      |
| `pinv(a[,atol,rtol,return_rank,...])`           | *Moore-Penrose* 伪逆                  |
| `pinvh(a[,atol,rtol,lower,return_rank,...])`    | *Hermitian* 矩阵 *Moore-Penrose* 伪逆 |
| `kron(a,b)`                                     | *Kronecker* 积                        |
| `khatri_rao(a,b)`                               | *Khatri-rao* 积                       |
| `orthogonal_procrustes(A,B[,check_finite])`     |                                       |
| `matrix_balance(A[,permute,scale,...])`         |                                       |
| `subspace_angles(A,B)`                          |                                       |
| `bandwidth(a)`                                  |                                       |
| `issymmetric(a[,atol,rtol])`                    |                                       |
| `ishermitian(a[,atol,rtol])`                    |                                       |

| 特征值问题                                    | 描述       |
|-----------------------------------------------|------------|
| `eig(a[,b,left,right,overwrite_a,...])`       | 方阵特征值 |
| `eigvals(a[,b,overwrite_a,check_finite,...])` |            |
| `eigh(a[,b,lower,eigvals_only,...])`          |            |
| `eigvalsh(a[,b,lower,overwrite_a,...])`       |            |
| `eig_banded(a_band[,lower,eigvals_only,...])` |            |
| `eigh_tridiagnol(d,e[,eigvals_only,...])`     |            |
| `eigvalsh_tridiagonal(d,e[,select,...])`      |            |

| 矩阵分解                                          | 描述      |
|---------------------------------------------------|-----------|
| `lu(a[,permute_l,overwrite_a,...])`               | *LU* 分解 |
| `lu_factor(a[,overwrite_a,check_finite])`         |           |
| `lu_solve(lu_and_piv,b[,trans,...])`              |           |
| `svd(a[,full_matrices,compute_uv,...])`           |           |
| `svdvals(a[,overwrite_a,check_finite])`           |           |
| `diagsvd(s,M,N)`                                  |           |
| `orth(A[,rcond])`                                 |           |
| `null_space(A[,rcond])`                           |           |
| `ldl(A[,lower,hermitian,overwrite_a,...])`        |           |
| `cholesky(a[,lower,overwrite_a,check_finite])`    |           |
| `cholesky_banded(ab[,overwrite_ab,lower,...])`    |           |
| `cho_factor(a[,alower,overwrite_a,check_finite])` |           |
| `cho_solve(c_and_lower,b[,overwrite_b,...])`      |           |
| `cho_solve_banded(cb_and_lower,b[,...])`          |           |
| `polar(a[,side])`                                 |           |
| `qr(a[,overwrite_a,lwork,mode,pivoting,...])`     |           |
| `qr_multiply(a,c[,mode,pivoting,...])`            |           |
| `qr_update(Q,R,u,v[,overwrite_qruv,...])`         |           |
| `qr_delete(Q,R,k[,p,which,...])`                  |           |
| `qr_insert(Q,R,u,k[,which,rcond,...])`            |           |
| `rq(a[,overwrite_a,lwork,mode,check_finite])`     |           |
| `qz(A,B[,output,lwork,sort,overwrite_a,...])`     |           |
| `ordqz(A,B[,sort,output,overwrite_a,...])`        |           |
| `schur(a[,output,lwork,overwrite_a,sort,...])`    |           |
| `rsf2csf(T,Z[,check_finite])`                     |           |
| `hessenberg(a[,calc_q,overwrite_a,...])`          |           |
| `cdf2rdf(w,v)`                                    |           |
| `cossin(X[,p,q,seperate,swap_sign,...])`          |           |

| 矩阵函数                                      | 描述 |
|-----------------------------------------------|------|
| `expm(A)`                                     |      |
| `logm(A[,disp])`                              |      |
| `cosm(A)`                                     |      |
| `sinm(A)`                                     |      |
| `tanm(A)`                                     |      |
| `coshm(A)`                                    |      |
| `sinhm(A)`                                    |      |
| `tanhm(A)`                                    |      |
| `signm(A[,disp])`                             |      |
| `sqrtm(A[,disp,blocksize])`                   |      |
| `funm(A,func[,disp])`                         |      |
| `expm_frechet(A,E[,method,compute_expm,...])` |      |
| `expm_cond(A[,check_finite])`                 |      |
| `fractional_matrix_power(A,t)`                |      |

| 矩阵等式求解                                 | 描述 |
|----------------------------------------------|------|
| `solve_sylvester(a,b,q)`                     |      |
| `solve_continuous_are(a,b,q,r[,e,s,...])`    |      |
| `solve_discrete_are(a,b,q,r[,e,s,balanced])` |      |
| `solve_continous_lyapunov(a,q)`              |      |
| `solve_discrete_lyapunov(a,q[,method])`      |      |

| 随机投影                                        | 描述                     |
|-------------------------------------------------|--------------------------|
| `clarkson_woodruff_transform(input_matrix,...)` | *Clarkson-Woodruff* 变换 |

| 特别                                           | 描述 |
|------------------------------------------------|------|
| `block_diag(*arrs)`                            |      |
| `circulant(c)`                                 |      |
| `companion(a)`                                 |      |
| `convolution_matrix(a,n[,model])`              |      |
| `dft(n[,scale])`                               |      |
| `fiedler(a)`                                   |      |
| `fiedler_companion(a)`                         |      |
| `hadamard(n[,dtype])`                          |      |
| `hankel(c[,r])`                                |      |
| `helmert(n[,full])`                            |      |
| `hilbert(n)`                                   |      |
| `invhilbert(n[,exact])`                        |      |
| `leslie(f,s)`                                  |      |
| `pascal(n[,kind,exact])`                       |      |
| `invpascal(n[,kind,exact])`                    |      |
| `toeplitz(c[,r])`                              |      |
| `tri(N[,M,k,dtype])`                           |      |
| `get_blas_funcs(names[,arrays,dtype,ilp64])`   |      |
| `get_lapack_funcs(names[,arrays,dtype,ilp64])` |      |
| `find_best_blas_type([arrays,dtype])`          |      |
