import keras.losses
from keras.datasets import cifar10
from keras import layers,models
import matplotlib.pyplot as plt
import numpy as np
# 图像分类加载数据
(train_img,train_labels),(test_img,test_labels) = cifar10.load_data()
# 标签对应类名
class_names = ['飞机', '汽车', '鸟', '猫', '鹿',
               '狗', '青蛙', '马', '船', '卡车']
# 归一化
train_img = train_img / 255.0
test_img = test_img / 255.0

# 设置模型
model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10)
])

# 编译模型,损失函数选择true再在之后添加softmax层确保数据准确
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 返回记录的各种值包括['loss', 'val_loss', 'accuracy', 'val_accuracy']
history = model.fit(train_img,train_labels,epochs=10,validation_data=(test_img,test_labels))

# 画图展示
plt.plot(history.history['accuracy'],label='训练的准确率')
plt.plot(history.history['val_accuracy'],label='验证准确率')
plt.xlabel('轮数')
plt.ylabel('准确率')
plt.ylim([0,1])
plt.legend(loc='lower right')
plt.show()

# 模型评估
test_loss,test_acc = model.evaluate(test_img,test_labels,verbose=2)
print(f"\n测试准确率:{test_acc}")

# 进行分类
p_model = keras.Sequential([model,layers.Softmax()])
pred = p_model.predict(test_img[:5])

for i in range(5):
    pred_label = np.argmax(pred[i])
    true_label = test_labels[i][0]
    print(f"预测结果：{class_names[pred_label]} | 实际:{class_names[true_label]}")