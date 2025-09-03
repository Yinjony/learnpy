import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 加载数据集
dataset = load_iris()
# print("数据集内容")
# print(dataset)
# print(dataset.feature_names)

# 转换为DataFrame
dataFrame = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
# print("初步转换后")
# print(dataFrame)
dataFrame['target'] = dataset.target
dataFrame['species'] = dataFrame['target'].apply(lambda x: dataset.target_names[x])
# print("完全转换后")
# print(dataFrame)

# 获取特征和标签
X = dataFrame.drop(columns=['target','species'])
y = dataFrame['target']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X_scaled)

# 数据集分割
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# 使用决策树模型
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

# 得出预测结果
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"准确率:{accuracy:.4f}")
print("\n模型评估报告:")
print(classification_report(y_test,y_pred))
print("\n混淆矩阵")
print(confusion_matrix(y_test,y_pred))

# 定义网格超参数
param_grid = {
    'max_depth':[3,5,10,None],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}
# 使用5次交叉验证进行网格超参数搜索
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),param_grid=param_grid,cv=5)
grid_search.fit(X_train,y_train)

print("\n获取的最佳参数")
print(grid_search.best_params_)
# 获取最佳模型
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)

accuracy_optimized = accuracy_score(y_test,y_pred_optimized)
print(f"优化后准确率:{accuracy_optimized:.4f}")

cross_val = cross_val_score(best_model,X_scaled,y,cv=5)
print("\n交叉验证结果:")
print(cross_val)
print(f"CV准确率:{cross_val.mean():.4f}")