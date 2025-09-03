import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[1.0,2.0],[2.0,3.0],[3.0,4.0]])
y = np.array([0,1,0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 逻辑回归
lr_model = LogisticRegression()
# K-近邻
knn_model = KNeighborsClassifier(n_neighbors=3)
# 支持向量机SVM
svc_model = SVC(kernel='linear')
# 决策树
dt_model = DecisionTreeClassifier()
# 随机森林
rf_model = RandomForestClassifier(n_estimators=100)

#线性回归
lr2_model = LinearRegression()
# 岭回归
r_model = Ridge(alpha=1.0)
# Lasso回归
l_model = Lasso(alpha=0.1)


