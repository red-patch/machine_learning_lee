#task1

---

# 目录

## 0、了解什么是Machine learning

## 1、中心极限定理

> **中心极限定理**是[概率论](https://zh.wikipedia.org/wiki/概率论)中的一组定理。中心极限定理说明，在适当的条件下，大量相互独立[随机变量](https://zh.wikipedia.org/wiki/随机变量)的均值经适当标准化后[依分布收敛](https://zh.wikipedia.org/wiki/依分布收敛)于[正态分布](https://zh.wikipedia.org/wiki/正态分布)。这组定理是[数理统计学](https://zh.wikipedia.org/wiki/数理统计学)和误差分析的理论基础，指出了大量随机变量之和近似服从正态分布的条件。

### 1.1 棣莫佛－拉普拉斯定理

> 棣莫佛－拉普拉斯（de Moivre - Laplace）定理是中央极限定理的最初版本，讨论了服从[二项分布](https://zh.wikipedia.org/wiki/二项分布)的随机变量序列。它指出，参数为*n*, *p*的二项分布以*np*为均值、*np(1-p)*为方差的正态分布为极限

$$
若 {\displaystyle X\sim B(n,p)} 是 {\displaystyle n}次伯努利实验中事件 A 出现的次数，每次试验成功的概率为 {\displaystyle p} p，且 {\displaystyle q=1-p}，则对任意有限区间 {\displaystyle [a,b]}：
令 {\displaystyle x_{k}\equiv {\frac {k-np}{\sqrt {npq}}}}，当 {\displaystyle n\to {\infty }}时
$$


$$
\begin{array}{l}{\text { (i) } P(X=k) \rightarrow \frac{1}{\sqrt{n p q}} \cdot \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2} x_{\mu_{n}}^{2}}} \\ {\text { (ii) } P\left(a \leq \frac{X-n p}{\sqrt{n p q}} \leq b\right) \rightarrow \int_{a}^{b} \varphi(x) d x, \# \Psi \varphi(x)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{x^{2}}{2}}(-\infty<x<\infty)}\end{array}
$$

### 1.2 林德伯格－列维定理

> 林德伯格－列维（Lindeberg-Levy）定理，是棣莫佛－拉普拉斯定理的扩展，讨论独立同分布随机变量序列的中央极限定理。它表明，独立同分布(iid independent and indentically distributed)、且数学期望和方差有限的随机变量序列的标准化和以标准正态分布为极限

$$
设随机变量 {\displaystyle X_{1},X_{2},\cdots ,X_{n}} X_{1},X_{2},\cdots ,X_{n}独立同分布， 且具有有限的数学期望和方差 {\displaystyle E(X_{i})=\mu } E(X_{i})=\mu ， {\displaystyle D(X_{i})=\sigma ^{2}\neq 0(i=1,2,\cdots ,n)} D(X_{i})=\sigma ^{2}\neq 0(i=1,2,\cdots ,n)，记
$$

$$
\overline{X}=\frac{1}{n} \sum_{i=1}^{n} X_{i}, \zeta_{n}=\frac{\overline{X}-\mu}{\sigma / \sqrt{n}}, \quad \underset{n \rightarrow \infty}{\lim } P\left(\zeta_{n} \leq z\right)=\Phi(z)
$$

$$
其中 {\displaystyle \Phi (z)} \Phi (z)是标准正态分布的分布函数
$$

## 2、正态分布

> **正态分布**（英语：normal distribution）又名**高斯分布**（英语：**Gaussian distribution**），是一个非常常见的[连续概率分布](https://zh.wikipedia.org/wiki/概率分布)。正态分布在[统计学](https://zh.wikipedia.org/wiki/统计学)上十分重要，经常用在[自然](https://zh.wikipedia.org/wiki/自然科学)和[社会科学](https://zh.wikipedia.org/wiki/社会科学)来代表一个不明的随机变量

$$
若随机变量 {\displaystyle X} 服从一个位置参数为 {\displaystyle \mu }、尺度参数为 {\displaystyle \sigma } 的正态分布，记为：
$$

$$
X \sim N\left(\mu, \sigma^{2}\right)
$$

$$
则其概率密度函数为
$$

$$
f(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}}
$$



## 3、最大似然估计

> 在[统计学](https://zh.wikipedia.org/wiki/统计学)中，**最大似然估计**（英语：maximum likelihood estimation，缩写为MLE），也称**最大概似估计**，是用来[估计](https://zh.wikipedia.org/wiki/估计函数)一个[概率模型](https://zh.wikipedia.org/wiki/概率模型)的参数的一种方法。

$$
给定一个概率分布 {\displaystyle D} ，已知其概率密度函数（连续分布）或概率质量函数（离散分布）为 {\displaystyle f_{D}}，以及一个分布参数 {\displaystyle \theta } ，我们可以从这个分布中抽出一个具有 {\displaystyle n} 个值的采样 {\displaystyle X_{1},X_{2},\ldots ,X_{n}}，利用 {\displaystyle f_{D}}计算出其似然函数：

{\displaystyle {\mbox{L}}(\theta \mid x_{1},\dots ,x_{n})=f_{\theta }(x_{1},\dots ,x_{n}).}
$$

$$
若 {\displaystyle D}是离散分布， {\displaystyle f_{\theta }}即是在参数为 {\displaystyle \theta }时观测到这一采样的概率。若其是连续分布， {\displaystyle f_{\theta }} 则为 {\displaystyle X_{1},X_{2},\ldots ,X_{n}}, X_n联合分布的概率密度函数在观测值处的取值。一旦我们获得 {\displaystyle X_{1},X_{2},\ldots ,X_{n}} ，我们就能求得一个关于 {\displaystyle \theta }的估计。最大似然估计会寻找关于 {\displaystyle \theta }的最可能的值（即，在所有可能的 {\displaystyle \theta }  取值中，寻找一个值使这个采样的“可能性”最大化）。从数学上来说，我们可以在 {\displaystyle \theta } 的所有可能取值中寻找一个值使得似然函数取到最大值。这个使可能性最大的 {\displaystyle {\widehat {\theta }}} 值即称为 {\displaystyle \theta }的最大似然估计。由定义，最大似然估计是样本的函数。
$$

## 4、泰勒公式



- 推导回归Loss function

- 学习损失函数与凸函数之间的关系

- 了解全局最优和局部最优

- 学习导数，泰勒展开

- 推导梯度下降公式

- 写出梯度下降的代码

- 学习L2-Norm，L1-Norm，L0-Norm

- 推导正则化公式

- 说明为什么用L1-Norm代替L0-Norm

- 学习为什么只对w/Θ做限制，不对b做限制