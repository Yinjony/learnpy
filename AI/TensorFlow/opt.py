import keras
import keras_tuner as kt
from numpy.ma.core import min_val

# 动态学习率调整
initial_lr = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True,
)

optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

# 查找最佳学习率
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(10))

    hp_lr = hp.Choice('learning_rate',values=[1e-2,1e-3,1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_lr),loss='mse')
    return model
# 超参数搜索器
# 参数分别是代表前面定义的模型，搜索以验证集损失最小作为优化目标，以及做多尝试5组不同的超参数组合
tuner = kt.RandomSearch(build_model,objective='val_loss',max_trials=5)

# 查找最佳的层大小
def build_model2(hp):
    model = keras.Sequential()
    hp_units = hp.Int('units',min_value=32,max_value=512,step=32)
    model.add(keras.layers.Dense(units=hp_units,activation='relu'))
    model.add(keras.layers.Dense(10))
    model.compile(optimizer='adam',loss='mse')
    return model

# 正则化技术
# l2正则
keras.layers.Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))

# 使用早停法，放到callback回调函数这里
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)