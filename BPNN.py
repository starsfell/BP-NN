# -*- coding: utf-8 -*-
'''
---------------------------------------------------------------------
--导入数据集
---------------------------------------------------------------------
'''
import numpy as np
from matplotlib import pyplot
import matplotlib as plt
from sklearn.datasets import make_moons #数据集
from sklearn import linear_model

# 显示数据集，并plot显示
np.random.seed(0)
X, y = make_moons(200, noise=0.2) # numpy ndarray
pyplot.scatter(X[:,0], X[:,1], s=40, c=y, cmap=pyplot.cm.Spectral)


# 生成了两类数据集，分别用红点和蓝点表示。你可以把蓝点想象成男性病人，红点想象成女性病人，把x轴和y轴想象成药物治疗剂量。
# 我们希望通过训练使得机器学习分类器能够在给定的x轴y轴坐标上预测正确的分类情况。我们无法用直线就把数据划分，可见这些数据样本呈非线性。那么，除非你手动构造非线性功能（例如多项式），否则，诸如逻辑回归（Logistic Regression）这类线性分类器将无法适用于这个案例。
# 事实上，这也正是神经网络的一大主要优势。神经网络的隐藏层会为你去学习特征，所以你不需要为构造特征这件事去操心。

# 为了证明（学习特征）这点，让我们来训练一个逻辑回归分类器吧。以x轴，y轴的值为输入，它将输出预测的类（0或1）。为了简单起见，这儿我们将直接使用scikit-learn里面的逻辑回归分类器。

'''
---------------------------------------------------------------------
--逻辑回归显示
---------------------------------------------------------------------
'''
## 逻辑回归分类器
clf = linear_model.LogisticRegressionCV()
clf.fit(X, y)

 # Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    pyplot.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    pyplot.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)   
    
    
    
    
    
    
## 绘制边界
plot_decision_boundary(lambda x: clf.predict(x))


# 图表向我们展示了逻辑回归分类器经过学习最终得到的决策边界。
# 尽管它尽可能地将数据区分为两类，却不能捕获到数据呈“月亮形状”的特性。


'''
---------------------------------------------------------------------
--训练神经网络
---------------------------------------------------------------------
'''

# 现在，我们搭建由一个输入层，一个隐藏层，一个输出层组成的三层神经网络。
# 输入层中的节点数由数据的维度来决定，也就是2个。
# 相应的，输出层的节点数则是由类的数量来决定，也是2个。（因为我们只有一个预测0和1的输出节点，所以我们只有两类输出，实际中，两个输出节点将更易于在后期进行扩展从而获得更多类别的输出）。
# 以x，y坐标作为输入，输出的则是两种概率，一种是0（代表女），另一种是1（代表男）。结果如下：

# 我们可以选择隐藏层的维度。放进去的节点越多，实现的功能就可以越复杂。
# 但是维度过高也是会有代价的。首先，更多的预测以及学习网络参数意味着更高的计算强度，更多的参数也会带来过拟合的风险。

# 那么该如何判断隐藏层的规模呢？尽管总会有许多通用性很好的引导和推荐，但问题的差异性也不该被忽视。在我看来，选择规模这件事绝不仅仅是门科学，它更像是一门艺术。通过待会儿的演示，我们可以看到隐藏层里的节点数是怎么影响我们的输出的。

# 另外，我们还需要为隐藏层选择激活函数（activation function）。
# 激活函数会将输入转化成输出。非线性的激活函数可以帮助我们处理非线性的假设。通常选用的激活函数有tanh, the sigmoid function, ReLUs。
# 在这里我们将使用tanh这样一个适用性很好地函数。这些函数有一个优点，就是通过原始的函数值便可以计算出它们的导数。例如tanh的导数就是1-tanh2x。这让我们可以在推算出tanh⁡x一次后就重复利用这个得到导数值。
# 鉴于我们希望我们的网络输出的值为概率，所以我们将使用softmax作为输出层的激活函数，这个函数可以将原始的数值转化为概率。如果你很熟悉逻辑回归函数，你可以把它当做是逻辑回归的一般形式。


'实现'
# 接下来我们就要实现这个三层的神经网络了。首先，我们需要定义一些对用于梯度下降法的变量和参数。
num_examples = len(X)  # 训练集的规模
nn_input_dim = 2       # 输入层的维度
nn_output_dim = 2      # 输出层的维度

# 自定义的梯度下降参数
epsilon = 0.01
reg_lambda = 0.01

# 首先，我们先实现之前定义的损失函数，这将用来评估我们的模型。
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss


    
# 我们还要实现一个用于计算输出的辅助函数。它会通过定义好的前向传播方法来返回拥有最大概率的类别。
# 用于预测输出的辅助函数
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)# softmax的分子部分
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
        
    
# 最后是训练神经网络的函数。它会使用我们之前找到的后向传播导数来进行批量梯度下降运算。
# 该函数将学习神经网络的参数，返回模型

# nn_hdim：隐藏层中的节点数
# num_passes： number of passes through the training data for GD
# print_loss: 如果返回True，每一千次迭代打印一次损失
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    
    # 用随机值初始化函数，我们需要学习这些值，找出最适合的值
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1,nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    
    # 这个用于最终反馈结果的变量
    model = {}

    # 批量梯度下降
    for i in range(0, num_passes):
        # 向前传播
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2) # softmax的分子部分
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # softmax functions
        
        # 向后传播
        delta3 = probs  # prob = yhat - y
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)  # W2
        db2 = np.sum(delta3, axis=0, keepdims=True) # b2
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)   # W1
        db1 = np.sum(delta2, axis = 0)# b1
        
        # add regulatization term, b1 and b2 dont have regularization terms
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
        
        # 梯度下降参数更新
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # 将新的参数赋在model里
        model = {'W1':W1,'b1':b1,'W2':W2,'b2':b2 }

        # 打印LF
        # 这个操作代价较大，因为要用到整个数据集，所以我们会降低使用频率
        if print_loss and i % 1000 == 0:
            print('Loss after interation %i: %f' %(i, calculate_loss(model)))
    return model

# 一个隐藏层规模为3的网络
# 让我们看看训练一个隐藏层规模为3的网络会发生什么。
model = build_model(3, print_loss=True)

# 绘制决策边界
plot_decision_boundary(lambda x:predict(model, x))
pyplot.title("Decision Boundary for hidden layer size 3")


# 在刚刚的示例中，我们选择了一个隐藏层规模为3的网络，现在我们来看看不同规模的隐藏层会带来什么样的效果。
pyplot.figure(figsize=(16,32))
hidden_layer_dimensions = [1,2,3,4,5,20,50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    pyplot.subplot(5,2, i+1)
    pyplot.title('Hidden Layer Size %d' % nn_hdim)
    model = build_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x))
pyplot.show()
pyplot.savefig()

# 我们可以看到，低维度的隐藏层很好地抓住了数据的整体趋势。
# 高维度的隐藏层则显现出过拟合的状态。
# 相对于整体性适应，它们更倾向于精确记录各个点。
# 如果我们要在一个分散的数据集上进行测试（你也应该这么做），那么隐藏层规模较小的模型会因为更好的通用性从而获得更好的表现。
# 虽然我们可以通过强化规范化来抵消过拟合，但选择正确的隐藏层规模相对来说会更“经济实惠”一点。 

# 将图片保存为PDF文件

import datetime

starttime = datetime.datetime.now()

pyplot.figure(figsize=(16,32))
hidden_layer_dimensions = [1,2,3,4,5,20,50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    pyplot.subplot(5,2, i+1)
    pyplot.title('Hidden Layer Size %d' % nn_hdim)
    model = build_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x))
pyplot.savefig('test.png')
pyplot.show()


endtime = datetime.datetime.now()

print (endtime - starttime).seconds

















