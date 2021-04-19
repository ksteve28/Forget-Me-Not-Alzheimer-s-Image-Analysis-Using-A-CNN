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




"""
Drop out is one of the most popular techniques to reduce overfitting in deep learning.

"""
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
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.Input(shape=(*image_size, 3)),
        #First Layer
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        #Second Layer
        conv_block(32),
        #Third Layer
        conv_block(64),
        #Fourth Layer
        conv_block(128),
        tf.keras.layers.Dropout(0.2),

        #Full Connected Layers 
        tf.keras.layers.Flatten(),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        #Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


  
    def model_summary(model):
        return model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
#     callbacks=[lr_scheduler],
    epochs=epoch
)

#Easy to combine the graphs, although for visualization purpose they are split

def graph_performance():
    acc = history.history['accuracy']
    recall = history.history['recall']
    precision = history.history['precision']

    epochs_range = range(epoch)

    plt.figure(figsize=(8,8))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, recall, label='Training Recall')
    plt.plot(epochs_range, precision, label='Training Precision')
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage Increased')
    plt.title('Training Performance Metrics')
    plt.savefig('../images/performance.jpg')
    plt.tight_layout()
    plt.show()


def graph_loss():
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epoch)
    plt.figure(figsize=(8,8))
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss') 
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage Decreased')
    plt.title('Loss')
    plt.savefig('../images/loss.jpg')
    plt.tight_layout()
    plt.show()

def graph_false_negs(): 
    #False negatives label may change due to rerunning the code.   
    false_neg = history.history['false_negatives_7']
    epochs_range = range(epoch)

    plt.figure(figsize=(8,8))
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, false_neg, label='False Negative Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Number of False Negative Loss')
    plt.title('FN Loss')
    plt.savefig('../images/false_negs.jpg')
    plt.tight_layout()

    plt.show()





 if __name__ == "__main__":



    val_data = tf.keras.preprocessing.image_dataset_from_directory(
                '../Alzheimer_s Dataset/train',
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=image_size,
                batch_size=batch_size,
                )
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
                '../Alz_data/train/',
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            )



    test_data = tf.keras.preprocessing.image_dataset_from_directory(
                '../Alz_data/test/',
                mage_size=image_size,
                batch_size=batch_size,
                )




    model = build_model()

    METRICS = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'), 
            tf.keras.metrics.FalseNegatives()
            ]
    
    model.compile(
            optimizer='adam',
            loss=tf.losses.BinaryCrossentropy(),
            metrics=METRICS
            )
        
    history = model.fit(
                    train_data,
                validation_data=val_data,
                epochs=epoch
                )

    test_loss, test_acc, test_precision, test_recall, test_false = model.evaluate(test_data)

    print('\nTest accuracy: {:.2f}'.format(test_acc),
                '\nTest recall: {:.2f}'.format(test_recall),
                '\nTest precision: {:.2f}'.format(test_precision),
                '\nTest False Negatives: {:.2f}'.format(test_false),
                '\nTest loss: {:.2f}'.format(test_loss))




                train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
        '../Alz_data/train',
        target_size=(64,64),
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        '../Alz_data/train',
        target_size=(64,64),
        batch_size=32,
        class_mode='categorical')

        img_hist = model.fit(
        train_generator,
        validation_data=validation_generator,
        callbacks=[early_stopping_cb, lr_scheduler], 
        epochs=epoch)