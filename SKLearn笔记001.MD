sklearn笔记001

# 机器学习概括包含内容:
[机器学习](https://morvanzhou.github.io/tutorials/machine-learning/sklearn/1-1-A-ML/)
通常来说, 机器学习的方法包括:

- 监督学习 supervised learning;(提供数据 & 数据对应的值,指引试学习)
- 非监督学习 unsupervised learning;(只有数据没有对应值)
- 半监督学习 semi-supervised learning;
- 强化学习 reinforcement learning;(让计算机完全陌生的环境自我适应,从经验总结)
- 遗传算法 genetic algorithm.(模拟进化理论, 适者生存)
---
# 为什么用SKlearn?
[Why Sklearn?](https://morvanzhou.github.io/tutorials/machine-learning/sklearn/1-1-why/)
[scikit-learn官网: machine learning in Python — scikit-learn 0.16.1 documentation](http://scikit-learn.org/)
Sklearn 包含了很多种机器学习的方式: (监督 & 非监督学习都能做)

- Classification 分类
- Regression 回归
- Clustering 非监督分类
- Dimensionality reduction 数据降维
- Model Selection 模型选择
- Preprocessing 数据预处理

安装保证:
- Python (>=2.6 或 >=3.3 版本)
- Numpy (>=1.6.1)
- Scipy (>=0.9)
---
# 机器学习方法的选择
Sklearn 官网提供了一个[`流程图`](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)， 蓝色圆圈内是判断条件，绿色方框内是可以选择的算法：
- 从 START 开始，首先看数据的样本是否 >50，小于则需要收集更多的数据。
- 由图中，可以看到算法有四类，分类，回归，聚类，降维。
- 其中 分类和回归是监督式学习，即每个数据对应一个 label。 聚类 是非监督式学习，即没有 label。 另外一类是 降维，当数据集有很多很多属性的时候，可以通过 降维 算法把属性归纳起来。例如 20 个属性只变成 2 个，注意，这不是挑出 2 个，而是压缩成为 2 个，它们集合了 20 个属性的所有特征，相当于把重要的信息提取的更好，不重要的信息就不要了。
- 然后看问题属于哪一类问题，是分类还是回归，还是聚类，就选择相应的算法。 当然还要考虑数据的大小，例如 100K 是一个阈值。
- 可以发现有些方法是既可以作为分类，也可以作为回归，例如 SGD。

![SKlearn_machine_learn_path](https://scikit-learn.org/stable/_static/ml_map.png)

---
# 通用学习模式
[链接n](https://morvanzhou.github.io/tutorials/machine-learning/sklearn/2-2-general-pattern/)
sklearn 数据库带着很多数据集合
iris-鸢尾花 就是其中之一.笔记代码 见jupyter notebook中

---

# sklearn的数据库介绍
[sklearn带的数据集网址介绍](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets)
- 有: 鸢尾花 \ 波士顿房价 \ 糖尿病 \ 数字
- 可以: 生成虚拟数据[API Reference — scikit-learn 0.21.3 documentation](https://scikit-learn.org/stable/modules/classes.html#samples-generator)

LinearRegression案例 见jupyternotebook中 房价预测和1特征1分类案例

---

# sklearn model 的属性与功能
见jupyter部分笔记:
`model.coef_ `和 `model.intercept_ `属于 Model 的属性， 例如对于 LinearRegressor 这个模型，这两个属性分别输出模型的斜率和截距（与y轴的交点）。
`model.get_params() `也是功能，它可以取出之前定义的参数。
`model.score(data_X, data_y)` 它可以对 Model 用` R^2` 的方式进行打分，输出精确度。关于 R^2 coefficient of determination 可以查看[ wiki](https://en.wikipedia.org/wiki/Coefficient_of_determination)
```python
上边见notebook演示,下边属性如print后备注
# model的属性
print('model.coef_ : \n',model.coef_)  #y=0.1x + 0.3  ,zhege coef_输出0.1
print('model.intercept_ : \n',model.intercept_) # imtercept 输出的就是 b
print('看看参数 : ', model.get_params())  # 拿出之前定义的参数,看看用没用
print('精确度score : ', model.score(data_X , data_y)) # R^2 打分 coefficient of determination

```

---
# normalization 标准化
[教程视频网址](https://morvanzhou.github.io/tutorials/machine-learning/sklearn/3-1-normalization/)
[特征标准化介绍 (Feature Normalization)](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/3-02-normalization/)

- feature scaling 指属性数据的范围
- X , Y的跨度最好接近,最好在0-1之间

--- 
# 检验\评价 神经网络 
[视频地址](https://morvanzhou.github.io/tutorials/machine-learning/sklearn/3-2-A-Evaluate-NN/)
## 训练集 & 测试集
为了检验,评价神经网络, 避免和改善这些问题, 我们通常会把收集到的数据分为`训练数据` 和 `测试数据`, 一般用于训练的数据可以是所有数据的70%, 剩下的30%可以拿来测试学习结果.如果你想问为什么要分开成两批, 那就想想我们读书时的日子, 训练多,考试少.

## 误差曲线
对于神经网络的评价基本上是基于这30%的测试数据. 想想期末考试虽然花的时间少, 但是占得总成绩肯定要比你平时作业的分多吧. 所以说这30%虽然少, 但是很重要. 

然后, 我们就可以开始画图啦! 评价机器学习可以从`误差`这个值开始, 随着训练时间的变长, 优秀的神经网络能预测到更为精准的答案, 预测误差也会越少 . 到最后能够提升的空间变小, 曲线也趋于水平 . 

班上的差生, 从不及格到80分已经不容易啦, 再往上冲刺100分, 就变成了更难的事了. 机器学习也一样. 所以, 如果你的机器学习的误差曲线是这样一条曲线, 那就已经是很不错的学习成果啦.

## 准确度曲线
除了误差曲线, 我们可以看他的精确度曲线. 最好的精度是趋向于100%精确. 比如在神经网络的分类问题中, 100个样本中, 我有90张样本分类正确, 那就是说我的预测精确度是90%.

不过, 不知道大家有没有想过对于回归的问题呢? 怎样看预测值是连续数字的精确度? 这时, 我们可以引用 `R2 `分数在测量回归问题的精度 . 

R2给出的最大精度也是100%, 所以分类和回归就都有的统一的精度标准. 

除了这些评分标准, 我们还有很多其他的标准, 比如 `F1 `分数 

## 正规化
有时候, 意外是猝不及防的, 比如有时候我们明明每一道作业习题都会做, 可是考试分数为什么总是比作业分数低许多?

原来, 我们只复习了作业题,并没有深入, 拓展研究作业反映出来的知识. 这件事情发生在机器学习中, 我们就叫做`过拟合`. 我们在回到误差曲线, 不过这时我们也把训练误差画出来. 红色的是训练误差, 黑色的是测试误差. 训练时的误差比测试的误差小, 神经网络虽然学习到了知识, 但是对于平时作业太过依赖, 到了考试的时候, 却不能随机应变, 没有成功的把作业的知识扩展开来. 

在机器学习中, 解决过拟合也有很多方法 , 比如 `l1 \ l2 \ 正规化 \ dropout` 方法.


## 交叉验证
神经网络也有很多参数, 我们怎么确定哪样的参数能够更有效的解决现有的问题呢? 这时, 交叉验证 就是最好的途径了. 

交叉验证不仅仅可以用于神经网络的调参, 还能用于其他机器学习方法的调参. 同样是选择你想观看的误差值或者是精确度, 不过横坐标不再是学习时间, 而是你要测试的某一参数 (比如说神经网络层数) . 我们逐渐增加神经层, 然后对于每一个不同层结构的神经网络求出最终的误差或精度, 画在图中.

我们知道, 神经层越多, 计算机所需要消耗的时间和资源就越多, 所以我们只需要找到那个能满足误差要求, 有节约资源的层结构. 

比如说误差在0.005一下都能接受 , 那我们就可以采用30层的神经网络结构 .

---
# 交叉验证 1 Cross-validation
[视频链接](https://morvanzhou.github.io/tutorials/machine-learning/sklearn/3-2-cross-validation1/)

> Sklearn 中的 Cross Validation (交叉验证)对于我们选择正确的 Model 和 Model 的参数是非常有帮助的， 有了他的帮助，我们能直观的看出不同 Model 或者参数对结构准确度的影响。

# 交叉验证2 
[网页链接](https://morvanzhou.github.io/tutorials/machine-learning/sklearn/3-3-cross-validation2/)
`sklearn.model_selection `中的 `learning_curve`可以很直观的看出我们的 model 学习的进度, 对比发现有没有 overfitting 的问题. 然后我们可以对我们的 model 进行调整, 克服 overfitting 的问题.

# 交叉验证3

sklearn 中我们用到了`sklearn.model_selection`当中的另外一种, 叫做`validation_curve`,用这一种曲线我们就能更加直观看出改变模型中的参数的时候有没有过拟合(overfitting)的问题了. 这也是可以让我们更好的选择参数的方法.[API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve)

# 如何存储
[视频链接](https://morvanzhou.github.io/tutorials/machine-learning/sklearn/3-5-save/)
- 主要介绍两种保存Model的模块pickle与joblib。
- 详细代码见 jupyternotebook. 

- [什么是pickle](https://morvanzhou.github.io/tutorials/python-basic/basic/13-08-pickle/)
```python
# pickle写入
import pickle

a_dict = {'da': 111, 2: [23,1,4], '23': {1:2,'d':'sad'}}

# pickle a variable to a file
file = open('pickle_example.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()
```

```python
# pickle读取
with open('pickle_example.pickle', 'rb') as file:
    a_dict1 =pickle.load(file)

print(a_dict1)
```
