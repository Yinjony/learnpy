import tensorflow as tf
import keras

tensor_callback = keras.callbacks.TensorBoard(
    log_dir=&#39;./logs&#39;,
)