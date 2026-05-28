from keras.datasets import boston_housing
from keras import models,layers,regularizers
from sklearn.model_selection import KFold

(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

# 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# 建立模型
def build_model():
    model = models.Sequential([
        # 优化模型之深层网络
        # layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)),

        # 优化模型之正则化
        layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001),input_shape=(train_data.shape[1],)),
        layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)),

        # layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model

model = build_model()
model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
history = model.fit(train_data,train_targets,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2)

test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
print(f"测试集MAE:{test_mae_score}")

# 优化模型之交叉验证
# k = 4
# kf = KFold(n_splits=k)
# for train_index, val_index in kf.split(train_data):
#     # 划分训练集和验证集
#     partial_train_data = train_data[train_index]
#     partial_train_targets = train_targets[train_index]
#     val_data = train_data[val_index]
#     val_targets = train_targets[val_index]
#
#     # 训练和评估模型
#     model = build_model()
#     model.compile(optimizer='rmsprop',
#                   loss='mse',
#                   metrics=['mae'])
#     model.fit(partial_train_data, partial_train_targets,
#               epochs=100, batch_size=16, verbose=0)
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     print(f"验证MAE: {val_mae}")


sample = test_data[0]
pred = model.predict(sample.reshape(1,-1))
print(f"预测价格:{pred[0][0]},实际价格:{test_targets[0]}")