import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
AUTOTUNE = tf.data.experimental.AUTOTUNE 

    """[summary]

    Returns:
        [type]: [description]
    """

AUTOTUNE = tf.data.experimental.AUTOTUNE
image_size = [64,64]
batch_size = 32
epoch = 100


def plot_images():
    plt.figure(figsize=(10,10))
    for images, labels in train_data.take(1):
        for i in range(9):
            ax = plt.subplot(3,3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(train_data.class_names[labels[i]])
            plt.axis('off')

def one_hot_label(image, label):
    label = tf.one_hot(label, NUM_CLASSES


NUM_CLASSES = len(class_names)

def one_hot_label(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

train_data = train_data.map(one_hot_label, num_parallel_calls=AUTOTUNE)
val_data = val_data.map(one_hot_label, num_parallel_calls=AUTOTUNE)


def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ])
    
    return block


def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block

def build_model():
    model = tf.keras.Sequential([
#         tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.Input(shape=(*image_size, 1)),
        
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        
        conv_block(32),
        conv_block(64),
        
        conv_block(128),
        tf.keras.layers.Dropout(0.5),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),

        
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    return model


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.15 **(epoch / s)
    return exponential_decay_fn

def image_generator_train():


if __name__ == '__main__':
    #import the data
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


    train_data = train_data.map(one_hot_label, num_parallel_calls=AUTOTUNE)
    val_data = val_data.map(one_hot_label, num_parallel_calls=AUTOTUNE)

    model = build_model()

    METRICS = [
        tf.keras.metrics.AUC(name='AUC'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    model.compile(
            optimizer='adam',
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=METRICS
        )

    exponential_decay_fn = exponential_decay(0.001, 20)

    checkpoint_filepath = 'src/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                        restore_best_weights=True)
    history = model.fit(
        train_data,
        validation_data=val_data,
        callbacks=[early_stopping_cb, lr_scheduler, model_checkpoint_callback],
        epochs=epoch)