## Task7

### 信息熵 (information entropy)

> 一条信息的信息量大小和它的不确定性有直接的关系。我们需要搞清楚一件非常非常不确定的事，或者是我们一无所知的事，就需要了解大量的信息。相反，如果我们对某件事已经有了较多的了解，我们就不需要太多的信息就能把它搞清楚。所以，从这个角度，我们可以认为，信息量的度量就等于不确定性的多少。比如，有人说广东下雪了。对于这句话，我们是十分不确定的。因为广东几十年来下雪的次数寥寥无几。为了搞清楚，我们就要去看天气预报，新闻，询问在广东的朋友，而这就需要大量的信息，信息熵很高。再比如，中国男足进军2022年卡塔尔世界杯决赛圈。对于这句话，因为确定性很高，几乎不需要引入信息，信息熵很低。
>
> 考虑一个**离散的随机变量** x，由上面两个例子可知，信息的量度应该依赖于概率分布 p(x)，因此我们想要寻找一个函数 I(x)，它是概率 p(x) 的单调函数，表达了信息的内容。怎么寻找呢？如果我们有两个不相关的事件 x 和 y，那么观察两个事件同时发生时获得的信息量应该等于观察到事件各自发生时获得的信息之和，即
> $$
> I(x, y)=I(x)+I(y)
> $$
> 因为两个事件是独立不相关的，因此 p(x,y)=p(x)p(y)。根据这两个关系，很容易看出 I(x)一定与 p(x) 的对数有关 (因为对数的运算法则是
> $$
> \log _{a}(m n)=\log _{a} m+\log _{a} n 
> $$
> )。因此，我们有
> $$
> I(x)=-\log p(x)
> $$
> 其中负号是用来保证信息量是正数或者零。而 log 函数基的选择是任意的（信息论中基常常选择为2，因此信息的单位为比特bits；而机器学习中基常常选择为自然常数，因此单位常常被称为奈特nats），I(x) 也被称为随机变量 x 的自信息 (self-information)，描述的是随机变量的某个事件发生所带来的信息量。图像如图：
> ![1](img/1.png)
>
> 
>
> 最后，我们正式引出信息熵。 现在假设一个发送者想传送一个随机变量的值给接收者。那么在这个过程中，他们传输的平均信息量可以通过求
> $$
> I(x)=-\log p(x)
> $$
> 关于概率分布 p(x)p(x) 的期望得到，即：
> $$
> H(X)=-\sum_{x} p(x) \log p(x)=-\sum_{i=1}^{n} p\left(x_{i}\right) \log p\left(x_{i}\right)
> $$
> H(X) 就被称为随机变量 x 的熵,它是表示随机变量不确定的度量，是对所有可能发生的事件产生的信息量的期望。 从公式可得，随机变量的取值个数越多，状态数也就越多，信息熵就越大，混乱程度就越大。当随机分布为均匀分布时，熵最大，且
> $$
> 0 \leq H(X) \leq \log n
> $$
> 将一维随机变量分布推广到多维随机变量分布，则其**联合熵 (Joint entropy)** 为：
> $$
> H(X, Y)=-\sum_{x, y} p(x, y) \log p(x, y)=-\sum_{i=1}^{n} \sum_{j=1}^{m} p\left(x_{i}, y_{i}\right) \log p\left(x_{i}, y_{i}\right)
> $$
> **注意点：**
>
> 1、熵只依赖于随机变量的分布,与随机变量取值无关，所以也可以将 X 的熵记作H(p)。
>
> 2、令0log0=0(因为某个取值概率可能为0)。

### 相对熵 (Relative entropy)

> 相对熵 (Relative entropy)也称KL散度 (Kullback–Leibler divergence)
>
> 设 p(x)、q(x) 是 离散随机变量 X 中取值的两个概率分布，则 p 对 q 的相对熵是：
> $$
> D_{K L}(p \| q)=\sum_{x} p(x) \log \frac{p(x)}{q(x)}=E_{p(x)} \log \frac{p(x)}{q(x)}
> $$
> **性质：**
>
> 1、如果 p(x) 和 q(x) 两个分布相同，那么相对熵等于0
>
> 2、
> $$
> D_{K L}(p \| q) \neq D_{K L}(q \| p)
> $$
> ,相对熵具有不对称性
>
> 3、
> $$
> D_{K L}(p \| q) \geq 0
> $$
> 证明如下：
> $$
> \begin{aligned} D_{K L}(p | q) &=\sum_{x} p(x) \log \frac{p(x)}{q(x)} \\ &=-\sum_{x} p(x) \log \frac{q(x)}{p(x)} \\ &=-E_{p(x)}\left(\log \frac{q(x)}{p(x)}\right) \\ & \geq-\log E_{p(x)}\left(\frac{q(x)}{p(x)}\right) \\ &=-\log \sum_{x} p(x) \frac{q(x)}{p(x)} \\ &=-\log \sum_{x}^{x} q(x) \end{aligned}
> $$
> 因为：
> $$
> \sum_{x} p(x)=1
> $$
> 所以：
> $$
> D_{K L}(p \| q) \geq 0
> $$
> **总结：相对熵可以用来衡量两个概率分布之间的差异，上面公式的意义就是求 pp 与 qq之间的对数差在 pp 上的期望值**

### 交叉熵 (Cross entropy)

> $$
> D_{K L}(p \| q)=\sum_{x} p(x) \log \frac{p(x)}{q(x)}=\sum_{x} p(x) \log p(x)-p(x) \log q(x)
> $$
>
> 由于：
> $$
> H(p)=-\sum_{x} p(x) \log p(x)
> $$
>
> $$
> H(p, q)=\sum_{x} p(x) \log \frac{1}{q(x)}=-\sum_{x} p(x) \log q(x)
> $$
>
> 所以：
> $$
> D_{K L}(p \| q)=H(p, q)-H(p)
> $$
> 当用非真实分布 q(x) 得到的平均码长比真实分布 p(x) 得到的平均码长多出的比特数就是相对熵）
>
> 又因为 ：
> $$
> D_{K L}(p \| q) \geq 0
> $$
> 所以 H(p,q)≥H(p)（当 p(x)=q(x) 时取等号，此时交叉熵等于信息熵）
>
> 并且当 H(p) 为常量时（注：在机器学习中，训练数据分布是固定的），最小化相对熵 
> $$
> D_{K L}(p \| q) \geq 0
> $$
> 等价于最小化交叉熵 H(p,q) 也等价于最大化似然估计.
>
> **在机器学习中，我们希望在训练数据上模型学到的分布 P(model) 和真实数据的分布 P(real) 越接近越好，所以我们可以使其相对熵最小。但是我们没有真实数据的分布，所以只能希望模型学到的分布 P(model) 和训练数据的分布 P(train) 尽量相同。假设训练数据是从总体中独立同分布采样的，那么我们可以通过最小化训练数据的经验误差来降低模型的泛化误差。即：**
>
> 1、希望学到的模型的分布和真实分布一致，
> $$
> P(\text { model }) \simeq P(\text { real })
> $$
> 2、但是真实分布不可知，假设训练数据是从真实数据中独立同分布采样的，
> $$
> P(\text {train}) \simeq P(\text {real})
> $$
> 3、我们希望学到的模型分布至少和训练数据的分布一致，
> $$
> P(\text {train}) \simeq P(\text {model})
> $$
> 最小化训练数据上的分布  P(train)P(train) 与最小化模型分布 P(model)P(model) 的差异等价于最小化相对熵，
> $$
> D_{K L}(P(\text {train}) \| P(\text {model}))
> $$
> 此时， P(train) 就是
> $$
> D_{K L}(p \| q)
> $$
> 中的 p，即真实分布，P(model) 就是 q。又因为训练数据的分布 p 是给定的，所以求DKL(p||q) 等价于求 H(p,q)，所以**交叉熵可以用来计算学习模型分布与训练分布之间的差异**。

### 代码

```python
import numpy as np
import pandas as pd
```


```python
data=pd.read_csv('watermelon_3a.csv')
data
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Idx</th>
      <th>color</th>
      <th>root</th>
      <th>knocks</th>
      <th>texture</th>
      <th>navel</th>
      <th>touch</th>
      <th>density</th>
      <th>sugar_ratio</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>dark_green</td>
      <td>curl_up</td>
      <td>little_heavily</td>
      <td>distinct</td>
      <td>sinking</td>
      <td>hard_smooth</td>
      <td>0.697</td>
      <td>0.460</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>black</td>
      <td>curl_up</td>
      <td>heavily</td>
      <td>distinct</td>
      <td>sinking</td>
      <td>hard_smooth</td>
      <td>0.774</td>
      <td>0.376</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>black</td>
      <td>curl_up</td>
      <td>little_heavily</td>
      <td>distinct</td>
      <td>sinking</td>
      <td>hard_smooth</td>
      <td>0.634</td>
      <td>0.264</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>dark_green</td>
      <td>curl_up</td>
      <td>heavily</td>
      <td>distinct</td>
      <td>sinking</td>
      <td>hard_smooth</td>
      <td>0.608</td>
      <td>0.318</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>light_white</td>
      <td>curl_up</td>
      <td>little_heavily</td>
      <td>distinct</td>
      <td>sinking</td>
      <td>hard_smooth</td>
      <td>0.556</td>
      <td>0.215</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>dark_green</td>
      <td>little_curl_up</td>
      <td>little_heavily</td>
      <td>distinct</td>
      <td>little_sinking</td>
      <td>soft_stick</td>
      <td>0.403</td>
      <td>0.237</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>black</td>
      <td>little_curl_up</td>
      <td>little_heavily</td>
      <td>little_blur</td>
      <td>little_sinking</td>
      <td>soft_stick</td>
      <td>0.481</td>
      <td>0.149</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>black</td>
      <td>little_curl_up</td>
      <td>little_heavily</td>
      <td>distinct</td>
      <td>little_sinking</td>
      <td>hard_smooth</td>
      <td>0.437</td>
      <td>0.211</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>black</td>
      <td>little_curl_up</td>
      <td>heavily</td>
      <td>little_blur</td>
      <td>little_sinking</td>
      <td>hard_smooth</td>
      <td>0.666</td>
      <td>0.091</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>dark_green</td>
      <td>stiff</td>
      <td>clear</td>
      <td>distinct</td>
      <td>even</td>
      <td>soft_stick</td>
      <td>0.243</td>
      <td>0.267</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>light_white</td>
      <td>stiff</td>
      <td>clear</td>
      <td>blur</td>
      <td>even</td>
      <td>hard_smooth</td>
      <td>0.245</td>
      <td>0.057</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>light_white</td>
      <td>curl_up</td>
      <td>little_heavily</td>
      <td>blur</td>
      <td>even</td>
      <td>soft_stick</td>
      <td>0.343</td>
      <td>0.099</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>dark_green</td>
      <td>little_curl_up</td>
      <td>little_heavily</td>
      <td>little_blur</td>
      <td>sinking</td>
      <td>hard_smooth</td>
      <td>0.639</td>
      <td>0.161</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>light_white</td>
      <td>little_curl_up</td>
      <td>heavily</td>
      <td>little_blur</td>
      <td>sinking</td>
      <td>hard_smooth</td>
      <td>0.657</td>
      <td>0.198</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>black</td>
      <td>little_curl_up</td>
      <td>little_heavily</td>
      <td>distinct</td>
      <td>little_sinking</td>
      <td>soft_stick</td>
      <td>0.360</td>
      <td>0.370</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>light_white</td>
      <td>curl_up</td>
      <td>little_heavily</td>
      <td>blur</td>
      <td>even</td>
      <td>hard_smooth</td>
      <td>0.593</td>
      <td>0.042</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>dark_green</td>
      <td>curl_up</td>
      <td>heavily</td>
      <td>little_blur</td>
      <td>little_sinking</td>
      <td>hard_smooth</td>
      <td>0.719</td>
      <td>0.103</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


```python
def cancShannonEnt(dataSet,featIndex=-1):
        numEntries = len(dataSet)  # 获取数据的行数
        labelCounts = {}  # 设置字典数据格式，想要存储的数据格式为：类别：频数
        for featVec in dataSet: # 获取数据集每一行的数据
            currentLabel = featVec[featIndex]  # 获取特征向量的最后一列
            # 检查字典中key是否存在
            # 如果key不存在
            if currentLabel not in labelCounts.keys():
                # 将当前的标签存于字典中，并将频数置为0
                labelCounts[currentLabel] = 0
            # 如果key存在，在当前的键值上+1
            labelCounts[currentLabel] += 1
        # 数据已准备好，计算熵
        shannonEnt = 0.0  # 初始化信息熵
        for key in labelCounts:  # 遍历出数据中所的类别
            pro = float(labelCounts[key]) / numEntries
            print(pro)
            shannonEnt -= pro * np.log2(pro)  # 计算信息熵
        return shannonEnt
```


```python
cancShannonEnt(np.array(data),-2)
```

    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705
    0.058823529411764705

    4.08746284125034

### 参考

> https://www.cnblogs.com/kyrieng/p/8694705.html
>
> [https://datawhalechina.github.io/Leeml-Book/#/AdditionalReferences/Entropy?id=%E8%AE%A1%E7%AE%97%E7%BB%99%E5%AE%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84%E9%A6%99%E5%86%9C%E7%86%B5%EF%BC%9A](https://datawhalechina.github.io/Leeml-Book/#/AdditionalReferences/Entropy?id=计算给定数据集的香农熵：)