from keras.datasets import mnist
from keras.src.layers.regularization.dropout import Dropout
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# 获取数据集
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# 图像识别，转成灰度图
x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255
x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255
# 转换成独热编码
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential([
    # 二维卷积层，正常的神经网络
    Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
    Conv2D(64,kernel_size=(3,3),activation='relu'),
    # 取窗口内最大值下采样特征图
    MaxPooling2D(pool_size=(2,2)),
    # 在训练过程中随机将25%神经元输出设为0，防止过拟合。
    Dropout(0.25),
    # 把3D特征图展平成1D向量
    Flatten(),
    # 再进行组合学习
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')
])

# 进行编译
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 进行训练
model.fit(x_train,y_train,
          batch_size=128,
          epochs=12,
          # 控制日志显示的参数
          verbose=1,
          validation_data=(x_test,y_test))

# 然后评分
score = model.evaluate(x_test,y_test,verbose=0)
print("损失值:",score[0])
print("准确度:",score[1])

