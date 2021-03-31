import numpy as np
import pandas as pd
import tensorflow as transform
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 

AUTOTUNE = tf.data.experimental.AUTOTUNE 


#Create dataset and define parameters

image_size = [176, 208]
batch_size = 32
epochs = 10

def define_parameters(image_size, batch_size, epochs):
    image_size = image_size
    batch_size = batch_size
    epochs = epochs

#Initiate dataset
def init_data(name, path, subset, seed, image_size, batch_size):
    name = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset=subset,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size)


val_data = tf.keras.preprocessing.image_dataset_from_directory(
    '../Alzheimer_s Dataset/train',
        validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
)
#declare classes 
def class_names(class1, class2):
    class_names = [class1, class2]



train_data.class_names = class_names
val_data.class_names = class_names 
num_classes = len(class_names)


def one_hot(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)


"""
Drop out is one of the most popular techniques to reduce overfitting in deep learning.

"""
def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )
    
    return block


def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block