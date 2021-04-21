import tensorflow as tf
from tensorflow.keras import layers, models

def create_simple_cnn():
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.Input(shape=(*image_size, 3)),
    tf.keras.layers.Flatten(input_shape=(64,64)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

 if __name__ == __main__