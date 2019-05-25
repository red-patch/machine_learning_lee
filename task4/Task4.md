## Task4

## 朴素贝叶斯模型：

- 简介：

  贝叶斯分类是一类分类算法的总称，这类算法均以贝叶斯定理为基础，故统称为贝叶斯分类。本章首先介绍贝叶斯分类算法的基础——贝叶斯定理。最后，我们通过实例来讨论贝叶斯分类的中最简单的一种: 朴素贝叶斯分类。

- 模型假设：

  我们假设特征之间  **相互条件独立** 。换句话来说就是特征向量中一个特征的取值并不影响其他特征的取值。所谓 独立(independence) 指的是统计意义上的独立，即一个特征或者单词出现的可能性与它和其他单词相邻没有关系，比如说，“我们”中的“我”和“们”出现的概率与这两个字相邻没有任何关系。这个假设正是朴素贝叶斯分类器中 朴素(naive) 一词的含义。朴素贝叶斯分类器中的另一个假设是，每个特征同等重要。

  

- 模型思想：

  通过数据调整先验分布，得到每个样本属于每个类的条件概率，预测值为最大的条件概率所代表的的类。

- 模型公式：
  $$
  P(c | \boldsymbol{x})=\frac{P(c) P(\boldsymbol{x} | c)}{P(\boldsymbol{x})}=\frac{P(c)}{P(\boldsymbol{x})} \prod_{i=1}^{d} P\left(x_{i} | c\right)
  $$

  $$
  h_{n b}(\boldsymbol{x})=\underset{c \in \mathcal{Y}}{\arg \max } P(c) \prod_{i=1}^{d} P\left(x_{i} | c\right)
  $$

  $$
  P(c)=\frac{\left|D_{\mathrm{c}}\right|}{|D|}
  $$

  $$
  P\left(x_{i} | c\right)=\frac{\left|D_{c, x_{i}}\right|}{\left|D_{c}\right|}
  $$

  假定$p\left(x_{i} | c\right) \sim \mathcal{N}\left(\mu_{c, i}, \sigma_{c, i}^{2}\right)$
  $$
  p\left(x_{i} | c\right)=\frac{1}{\sqrt{2 \pi} \sigma_{c, i}} \exp \left(-\frac{\left(x_{i}-\mu_{c, i}\right)^{2}}{2 \sigma_{c, i}^{2}}\right)
  $$

- 优化方法：

  按步骤计算。

- 优缺点

  优点: 在数据较少的情况下仍然有效，可以处理多类别问题。
  缺点: 对于输入数据的准备方式较为敏感。
  适用数据类型: 标称型数据。

## 先验概率和后验概率：

在朴素贝叶斯的例子中，贝叶斯定理：
$$
P(c | \boldsymbol{x})=\frac{P(c) P(\boldsymbol{x} | c)}{P(\boldsymbol{x})}=\frac{P(c)}{P(\boldsymbol{x})} \prod_{i=1}^{d} P\left(x_{i} | c\right)
$$
其中，$P(c)$是类先验概率；是样本$x$ 对于类标记$c$的类条件概率(class-chonditional probability)，或称为“似然”(likelihood);$P(x)$适用于归一化“证据”的因子。对给定样本$x$，证据因子$P(x)$与类标记无关，因此估计$P(c|x)$的问题转化为如何基于训练数据$D$来估计先验$P(c)$和似然$P(x|c)$。

由于朴素贝叶斯假假设属性间条件独立，所以似然函数表示为每个属性的似然函数乘积。

**先验**（**A priori**；又译：**先天**）在[拉丁文](http://zh.wikipedia.org/wiki/拉丁文)中指“来自先前的东西”，或稍稍引申指“在[经验](http://zh.wikipedia.org/wiki/經驗)之前”。[近代](http://zh.wikipedia.org/wiki/近代)[西方](http://zh.wikipedia.org/wiki/西方)传统中，认为先验指无需经验或先于经验获得的[知识](http://zh.wikipedia.org/wiki/知识)。它通常与[后验](http://zh.wikipedia.org/w/index.php?title=后验&action=edit&redlink=1)知识相比较，后验意指“在经验之后”，需要经验。这一区分来自于中世纪逻辑所区分的两种论证，从原因到结果的论证称为“先验的”，而从结果到原因的论证称为“后验的”。

​    先验概率是指根据以往经验和分析得到的概率，如全概率公式 中的 ，它往往作为“由因求果”问题中的“因”出现。后验概率是指在得到“结果”的信息后重新修正的概率，是“执果寻因”问题中的“因” 。后验概率是基于新的信息，修正原来的先验概率后所获得的更接近实际情况的概率估计。先验概率和后验概率是相对的。如果以后还有新的信息引入，更新了现在所谓的后验概率，得到了新的概率值，那么这个新的概率值被称为后验概率。

先验概率的分类：

1. 利用过去历史资料计算得到的先验概率，称为客观先验概率；
2. 当历史资料无从取得或资料不完全时，凭人们的主观经验来判断而得到的先验概率，称为主观先验概率。

后验概率是指通过调查或其它方式获取新的附加信息，利用[贝叶斯公式](http://wiki.mbalib.com/wiki/贝叶斯公式)对先验概率进行修正，而后得到的概率。

先验概率和后验概率的区别：

先验概率不是根据有关自然状态的全部资料测定的，而只是利用现有的材料(主要是历史资料)计算的；后验概率使用了有关自然状态更加全面的资料，既有先验概率资料，也有补充资料；

先验概率的计算比较简单，没有使用[贝叶斯公式](http://wiki.mbalib.com/wiki/贝叶斯公式)；而后验概率的计算，要使用贝叶斯公式，而且在利用[样本](http://wiki.mbalib.com/wiki/样本)资料计算逻辑概率时，还要使用理论概率分布，需要更多的[数理统计](http://wiki.mbalib.com/wiki/数理统计)知识。

## LR和OLS的区别和联系

联系：

1. 都是对于变量X的线性组合参数进行估计
2. 都使用了极大似然估计

区别：

1. 逻辑回归的极大似然估计等价于交叉熵损失，而线性回归的极大似然估计等价于最小二乘估计
2. LR属于广义线性模型，$g(\mu) = \frac{\mu}{1-\mu}$

## 推导sigmoid function公式

$$
\begin{align}  
 g ( z ) &= \frac { 1 } { 1 + e ^ { - z } }  \\ 
  g ^ { \prime } ( z ) &= \left( \frac { 1 } { 1 + e ^ { - z } } \right) ^ { \prime } \\
 &= \frac { e ^ { - z } } { \left( 1 + e ^ { - z } \right) ^ { 2 } }  \\ 
 &= \frac { 1 } { 1 + e ^ { - z } } \cdot \frac { e ^ { - z } } { 1 + e ^ { - z } } \\
 &= \frac { 1 } { 1 + e ^ { - z } } \cdot \left( 1 - \frac { 1 } { 1 + e ^ { - z } } \right)  \\ 
  &= g ( z ) \cdot ( 1 - g ( z ) ) 
 \end{align}
$$

> 参考：https://github.com/thisisreallife/Hungyi_Lee_ML2017/blob/master/task_and_answer/task4/task4_ans.md
>
> https://zhuanlan.zhihu.com/p/26262151
>
> [朴素贝叶斯分类器]([https://zh.wikipedia.org/wiki/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%88%86%E7%B1%BB%E5%99%A8](https://zh.wikipedia.org/wiki/朴素贝叶斯分类器))
>
> [CodingLabs](https://www.cnblogs.com/leoo2sk/)

