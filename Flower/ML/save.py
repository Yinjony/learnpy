import time

import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train,y_train)
# 保存模型，适用于大型数据
timestamp = time.strftime("%Y%m%d-%H%M%S")
# 带保存时间戳的
joblib.dump(model, f'svm_model_{timestamp}.joblib')
joblib.dump(model,'svm_model.joblib')
# 加载保存的模型
loaded_model = joblib.load('svm_model.joblib')
# 使用加载的模型进行预测
y_pred = loaded_model.predict(X_test)

# 利用pickle加载模型，适用于一般情况
with open('svm_model.pkl','wb') as f:
    pickle.dump(model,f)

with open('svm_model.pkl','rb') as f:
    loaded_model = pickle.load(f)

y_pred = loaded_model.predict(X_test)
print(y_pred)

# 还可以保存管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear'))
])

# 训练管道
pipeline.fit(X_train, y_train)
# 保存管道到文件
joblib.dump(pipeline, 'pipeline_model.joblib')