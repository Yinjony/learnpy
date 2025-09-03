import numpy as np
# 使用基类
from sklearn.base import BaseEstimator,TransformerMixin

class CustomScaler(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.mean_ = np.mean(X,axis=0)
        self.std_ = np.std(X,axis=0)
        return self
    def transform(self,X):
        return (X - self.mean_) / self.std_

# 测试自定义转换器
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = CustomScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

