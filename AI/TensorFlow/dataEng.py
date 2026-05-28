import os

import keras
import tensorflow as tf
from keras import layers

# 根据图片路径加载数据
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img,channels=3)
    return tf.image.resize(img,[256,256])

# 对数据进行归一化
def normalize(image):
    return image / 255.0

# 数据增强
def augment(img):
    # 随机水平翻转
    img = tf.image.random_flip_left_right(img)
    # 随机调整亮度
    img = tf.image.random_brightness(img,max_delta=0.2)
    return img

# 建立数据预处理的管道流程
def build_pipline(img_dir,batch=32,is_training=True):
    dataset = tf.data.Dataset.list_files(f"{img_dir}/*/*.jpg")

    # 获得图片和对应标签
    def process_path(file_path):
        label = tf.strings.split(file_path,os.sep)[-2]
        img = load_img(file_path)
        return img,label

    # 使用map，把dataset里每个元素传给里面的函数，然后返回结果替换
    # num_parallel_calls = tf.data.AUTOTUNE 并行化，加速数据加载
    dataset = dataset.map(process_path,num_parallel_calls=tf.data.AUTOTUNE)

    # 是否进行数据增强
    if is_training:
        dataset = dataset.map(
            lambda x,y: (augment(x),y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # 打包成小批次
    dataset = dataset.batch(batch)
    # 预取，减少等待时间
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# 使用keras高级库进行预处理
augmenter = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.Rescaling(1./255)
])

model = keras.Sequential([
    augmenter,
    layers.Conv2D(32,3,activation='relu'),
    ...
])