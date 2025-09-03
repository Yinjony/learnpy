import sklearn
import numpy as np
import pandas as pd
X = np.array([[1.0,2.0],[2.0,3.0],[3.0,4.0]])
y = np.array([0,1,0])
#分割数据集，分成训练集和测试集
X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.3,random_state=42)

#决策树进行分类任务，实现监督学习
clf = sklearn.tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

kmeans = sklearn.cluster.KMeans(n_clusters=3)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

#数据预处理
#数据标准化
scaler = sklearn.preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

#归一化
scaler = sklearn.preprocessing.MinMaxScaler()
X_normalized = scaler.fit_transform(X)

#数据评估
#交叉验证
scores = sklearn.model_selection.cross_val_score(clf,X,y,cv=5)
print(scores)

#准确率
print(sklearn.metrics.accuracy_score(y_test,y_pred))
#分类报告
print(sklearn.metrics.classification_report(y_test,y_pred))
#均方误差
print(sklearn.metrics.mean_squared_error(y_test,y_pred))
#决定系数
print(sklearn.metrics.r2_score(y_test,y_pred))

# 模型选择与调优
#网格搜索
param_grid = {'max_depth':[3,5,7],'min_samples_split':[2,5,10]}
grid_search = sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeClassifier(),param_grid,cv=5)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
# sklearn.model_selection.RandomizedSearchCV 随机搜索

#缺失值处理
df = pd.array()
df_cleaned = df.dropna()
imputer = sklearn.impute.SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df)

#标签编码
label_encoder = sklearn.preprocessing.LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#独热编码
encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

#特征选择
clf = sklearn.ensemble.RandomForestClassifier()#训练一个随机森林模型
clf.fit(X_train,y_train)
#获取特征的重要性
importances = clf.feature_importances_
#递归特征消除
rfe = sklearn.feature_selection.RFE(clf,n_features_to_select=3)
X_rfe = rfe.fit_transform(X_train,y_train)

#特征提取
pca = sklearn.decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X)

lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X,y)
