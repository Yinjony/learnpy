import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 准备数据
data = {
    'area': [70, 85, 100, 120, 60, 150, 200, 80, 95, 110],
    'rooms': [2, 3, 3, 4, 2, 5, 6, 3, 3, 4],
    'floor': [5, 2, 8, 10, 3, 15, 18, 7, 9, 11],
    'year_built': [2005, 2010, 2012, 2015, 2000, 2018, 2020, 2008, 2011, 2016],
    'location': ['Chaoyang', 'Haidian', 'Chaoyang', 'Dongcheng', 'Fengtai', 'Haidian', 'Chaoyang', 'Fengtai', 'Dongcheng', 'Haidian'],
    'price': [5000000, 6000000, 6500000, 7000000, 4500000, 10000000, 12000000, 5500000, 6200000, 7500000]  # 房价（目标变量）
}

# 便于展示
df = pd.DataFrame(data)
# 数据预览
print(df.head())

# 获取特征和标签
X = df[['area','rooms','floor','year_built','location']]
y = df['price']

# 分割数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 对数值数据进行标准化，对类别数据进行独热编码
numeric_features = ['area','rooms','floor','year_built']
categorical_features = ['location']

numeric_transformer = Pipeline(steps=[
    ('scaler',StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numeric_features),
        ('cat',categorical_transformer,categorical_features)
    ]
)

# 建立模型
model = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('regressor',LinearRegression())
])

# 进行训练
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("\n预测结果")
print(y_pred)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("\n模型评估：")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"决定系数 (R²): {r2:.2f}")

# 网格搜索参数
param_grid = {
    'regressor__fit_intercept':[True,False],
}

grid_search = GridSearchCV(model,param_grid,cv=5,scoring='neg_mean_squared_error',verbose=1)
grid_search.fit(X_train,y_train)

print("\n最佳参数")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)

mse_opt = mean_squared_error(y_test,y_pred_optimized)
r2_opt = r2_score(y_test,y_pred_optimized)

print("\n优化后的模型评估：")
print(f"均方误差 (MSE): {mse_opt:.2f}")
print(f"决定系数 (R²): {r2_opt:.2f}")