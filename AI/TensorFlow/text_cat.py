import keras
import tensorflow as tf
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 文本二分类
imdb = keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
# 文本向量化
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequences in enumerate(sequences):
        results[i,sequences] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 转换成数据
y_train = np.array(train_labels).astype('float32')
y_test = np.array(test_labels).astype('float32')

# 设置神经网络
model = keras.Sequential([
    layers.Dense(16,activation='relu',input_shape=(10000,)),
    layers.Dense(16,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])
# 二分类
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_test,y_test))

# 保存和加载模型
model.save("text_category_model.keras")
# loaded_model = keras.models.load_model('text_category_model.keras')

# 保存和加载权重（就是保存参数而已）
model.save_weights("text_category_model_weights.keras")
# new_model.load_weights('text_category_model_weights.keras')

# 保存检查点，用于中断恢复
# checkpoint_path = "/cp.ckpt"
# cp_callback = keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=True,
#     verbose=1
# )


history_dict = history.history

# # 绘制训练损失和验证损失
# plt.plot(history_dict['loss'], 'bo', label='Training loss')
# plt.plot(history_dict['val_loss'], 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # 绘制训练准确率和验证准确率
# plt.plot(history_dict['accuracy'], 'bo', label='Training acc')
# plt.plot(history_dict['val_accuracy'], 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

results = model.evaluate(x_test,y_test)
print("测试的损失和准确率:",results)

pred = model.predict(x_test)
print("第一条评论的预测概率:",pred[1])



