from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

data = load_iris()
X,y = data.data,data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([('scaler',StandardScaler()),('svc',SVC())])

pipeline.fit(X_train,y_train)

param_grid = {'svc__C':[0.1,1,10],
              'svc__kernel':['linear','rbf']}
grid_search = GridSearchCV(pipeline,param_grid,cv=5)

grid_search.fit(X_train,y_train)

cv_scores = cross_val_score(pipeline,X,y,cv=5)