# Task3

> 作业1：预测PM2.5的值
> 在这个作业中，我们将用梯度下降方法预测PM2.5的值
> hw1要求：
>      1、要求python3.5+
>      2、只能用（1）numpy（2）scipy（3）pandas
>      3、请用梯度下降手写线性回归
>      4、最好的公共简单基线
>      5、对于想加载模型而并不想运行整个训练过程的人：
> 	请上传训练代码并命名成 train.py
> 	只要用梯度下降的代码就行了
> hw_best要求：
>      1、要求python3.5+ 
>      2、任何库都可以用
>      3、在kaggle上获得你选择的更高的分
>
> 数据介绍：
> 本次作業使用豐原站的觀測記錄，分成train set跟test set，train set是豐原站每個月的前20天所有資料test set則是從豐原站剩下的資料中取樣出來。
> train.csv:每個月前20天每個小時的氣象資料(每小時有18種測資)。共12個月。
> test.csv:從剩下的資料當中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。一共取出240筆不重複的test data，請根據feauure預測這240筆的PM2.5。
>
> 请完成之后参考以下资料：
> Sample_code:https://ntumlta.github.io/2017fall-ml-hw1/code.html
> Supplementary_Slide:https://docs.google.com/presentation/d/1WwIQAVI0RRA6tpcieynPVoYDuMmuVKGvVNF_DSKIiDI/edit#slide=id.g1ef6d808f1_2_0
> 答案参考answer.csv

```python
#!/usr/bin/env python 3.7
# -*- coding:utf-8 -*-
# @File    : LR_Gradient.py
# @Date    : 2019-03-03
# @Author  : 黑桃
# @Software: PyCharm 

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取文件


path = "./week1/"
train = pd.read_csv(path + 'train.csv', engine='python', encoding='gbk')
test = pd.read_csv(path + 'test.csv', engine='python', encoding='gbk')

# 数据预处理删去不要的特征

train = train[train['observation'] == 'PM2.5']
test = test[test['AMB_TEMP'] == 'PM2.5']
train = train.drop(['Date', 'stations', 'observation'], axis=1)
test_x = test.iloc[:, 2:]

# 使用训练集构成一个大的训练集
# 每连续的9列作为一组特征，后面的一列作为label，
# 原始数据就可以构造出15组这样的特征+label，最后拼接起来就是3600*9


train_x = []
train_y = []
for i in range(15):
    x = train.iloc[:, i:i + 9]
    x.columns = np.array(
        range(9))  # notice if we don't set columns name, it will have different columns name in each iteration
    y = train.iloc[:, i + 9]
    y.columns = np.array(range(1))
    train_x.append(x)
    train_y.append(y)

# 拼接

train_x = pd.concat(train_x)
train_y = pd.concat(train_y)

#
train_y = np.array(train_y, float)
test_x = np.array(test_x, float)
print(train_x.shape, train_y.shape)

# 数据归一化，若不归一化，数据收敛特别慢
ss = StandardScaler()
ss.fit(train_x)
train_x = ss.transform(train_x)

ss.fit(test_x)
test_x = ss.transform(test_x)


# 定义一个评价函数
def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的MSE"""
    MSE = np.sum((y_true - y_predict) ** 2) / len(y_true)
    """计算y_true和y_predict之间的R Square"""
    return 1 - MSE / np.var(y_true)


# 线性回归，【参考某大佬的代码】

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        '''

        :param X_train: 训练集
        :param y_train: label
        :param eta: 学习率
        :param n_iters: 迭代次数
        :return: theta
        '''

        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        '''定义一个损失函数'''

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        '''对损失函数求导'''

        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            '''

            :param X_b:
            :param y: lebel
            :param initial_theta: 初始theta值
            :param eta: 学习率
            :param n_iters: 迭代次数
            :param epsilon: theta更新变化值
            :return:
            '''
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])

        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LR()"


# 模型训练+评分+预测
LR = LinearRegression().fit_gd(train_x, train_y)

LR.score(train_x, train_y)
result = LR.predict(test_x)

# 保存结果
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', engine='python', encoding='gbk')
sampleSubmission['value'] = result
sampleSubmission.to_csv(path + 'result.csv')

```

> [李宏毅《机器学习》笔记(leeml-book)](https://datawhalechina.github.io/Leeml-Book/#/?id=李宏毅《机器学习》笔记leeml-book)

