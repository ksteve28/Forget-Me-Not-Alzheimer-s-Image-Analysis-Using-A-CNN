import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
AUTOTUNE = tf.data.experimental.AUTOTUNE 

AUTOTUNE = tf.data.experimental.AUTOTUNE
image_size = [64,64]
batch_size = 32
epoch = 100

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    '../Alz_data/train/',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    '../Alz_data/train',
        validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(64,64),
    batch_size=batch_size,
)

class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
train_data.class_names = class_names
val_data.class_names = class_names

NUM_CLASSES = len(class_names)

def one_hot_label(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

train_data = train_data.map(one_hot_label, num_parallel_calls=AUTOTUNE)
val_data = val_data.map(one_hot_label, num_parallel_calls=AUTOTUNE)


