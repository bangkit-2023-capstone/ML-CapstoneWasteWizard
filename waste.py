import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

val_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        "/Users/wrath/Downloads/ML-CapstoneWasteWizard/data/DATASET/TRAIN",
        target_size = (150, 150),
        batch_size = 64,
        class_mode = 'categorical')

val_generator = val_datagen.flow_from_directory(
        "/Users/wrath/Downloads/ML-CapstoneWasteWizard/data/DATASET/VALIDATION",
        target_size = (150, 150),
        batch_size = 64,
        class_mode = 'categorical')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(loss = "categorical_crossentropy",
              optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001),
              metrics = ["accuracy"])

history = model.fit(
      train_generator,
      steps_per_epoch = 176, # 22,564 images = batch_size * steps
      epochs = 10,
      verbose = 1,
      validation_data = val_generator, # 4,513 images = batch_size * steps
      validation_steps = 36)

def plot_loss_acc(history):
        '''Plots the training and validation loss and accuracy from a history object'''
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'bo', label='Training_accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation loss')

        plt.show()

plot_loss_acc(history)

model.save('model/model_v1.h5')