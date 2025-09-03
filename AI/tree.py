import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')

# 经典数据集
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
# 加载数据集
iris = load_iris()
# 挑选特征，python数组语法，第二列之后的所有列
X = iris.data[:,2:]
# 取出标签
y = iris.target
# 设置最大深度
tree_clf = DecisionTreeClassifier(max_depth = 2)
# 训练
tree_clf.fit(X,y)
# 可视化决策树
export_graphviz(
    # 训练好的模型
    tree_clf,
    # 文件名称
    out_file="iris_tree.dot",
    # 特征名称
    feature_names=iris.feature_names[2:],
    # 标签名称
    class_names=iris.target_names,
    rounded=True,
    filled=True
)


from matplotlib.colors import ListedColormap

# 绘制决策边界的函数
def plot_decision_boundary(clf,X,y,axes=[0,7,0,3],iris=True,legend=False,plot_training=True):

    # X是二维特征的数据
    # y是每个样本的标签
    # axes是绘图范围
    # plot_training是否将训练数据点一起画在图上

    # 构建坐标棋盘,也就是构建网格的坐标
    # 等距选100个点
    x1s = np.linspace(axes[0],axes[1],100)
    # 等距选100个点
    x2s = np.linspace(axes[2],axes[3],100)

    # 把一维坐标轴数组变成二维网格坐标矩阵
    x1,x2 = np.meshgrid(x1s,x2s)
    # ravel把数组拉成一维,再连起来
    X_new = np.c_[x1.ravel(),x2.ravel()]

    # 对每个网格点做分类然后把预测向量重塑回二维
    y_pred = clf.predict(X_new).reshape(x1.shape)

    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

    #创建离散颜色图
    plt.contourf(x1,x2,y_pred,alpha=0.3,cmap=custom_cmap)

    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contourf(x1,x2,y_pred,cmap=custom_cmap2,alpha=0.8)
    # 画散点和折线
    if plot_training:
        plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    # 坐标标记
    if iris:
        plt.xlabel("Petal length",fontsize=14)
        plt.ylabel("Petal width",fontsize=14)
    else:
        plt.xlabel(r"$x_1$",fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18,rotation=0)
    # 图例
    if legend:
        plt.legend(loc="lower right",fontsize=14)

# 绘制决策边界
plt.figure(figsize=(8,4))
plot_decision_boundary(tree_clf,X,y)
# 绘制决策边界的切割线
plt.plot([2.45,2.45],[0,3],"k-",linewidth=2)
plt.plot([2.45,7.5],[1.75,1.75],"k--",linewidth=2)
plt.plot([4.95,4.95],[0,1.75],"k:",linewidth=2)
plt.plot([4.85,4.85],[1.75,3],"k:",linewidth=2)
# 绘制深度
plt.text(1,1.0,"Depth=1",fontsize=15)
plt.text(3.2,2,"Depth=2",fontsize=13)
plt.text(3.2,0.5,"Depth=2",fontsize=13)
plt.title("Decision Tree decision boundaries")

plt.show()
# 选一个大一点的数据集
from sklearn.datasets import make_moons
# 构建数据集
X,y = make_moons(n_samples=100,noise=0.25,random_state=43)

# 构建决策树
tree_clf1 = DecisionTreeClassifier(random_state=6)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5,random_state=16)
tree_clf1.fit(X,y)
tree_clf2.fit(X,y)

# 画图展示绘制决策边界
plt.figure(figsize=(12,4))
plt.subplot(121)
plot_decision_boundary(tree_clf1,X,y,axes=[-1.5,2.5,-1,1.5],iris=False)
plt.title("No Restrictions")
plt.subplot(122)
plt.show()
plot_decision_boundary(tree_clf2,X,y,axes=[-1.5,2.5,-1,1.5],iris=False)
plt.title("min_samples_leaf = 5")
plt.show()

# 探究树模型对数据的敏感程度
# 构建随机测试数据
np.random.seed(6)
Xs = np.random.rand(100,2) - 0.5
ys = (Xs[:,0] > 0).astype(np.float32) * 2
# 定义数据的旋转角度
angle = np.pi / 4
# 旋转数据矩阵
rotation_matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
Xsr = Xs.dot(rotation_matrix)
# 构建分类器
tree_clf_s = DecisionTreeClassifier(random_state=42)
tree_clf_s.fit(Xs,ys)

tree_clf_sr = DecisionTreeClassifier(random_state=42)
tree_clf_sr.fit(Xsr,ys)

plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf_s,Xs,ys,axes=[-0.7,0.7,-0.7,0.7],iris=False)
plt.title('Sensitivity to training set rotation')

plt.figure(122)
plot_decision_boundary(tree_clf_sr,Xsr,ys,axes=[-0.7,0.7,-0.7,0.7],iris=False)
plt.title('Sensitivity to training set rotation')
plt.show()

