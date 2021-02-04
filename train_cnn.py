
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import keras.backend as K
import tensorflow as tf
import numpy as np
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50

K.tensorflow_backend._get_available_gpus()

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True)


def get_gen(x): return datagen.flow_from_directory(
    '/itet-stor/himeva/net_scratch/fullres_data/{}'.format(x),
    target_size=(320, 320),
    batch_size=32,
    #color_mode="grayscale",
    class_mode='binary'
)


# generator objects
train_generator = get_gen('train')
val_generator = get_gen('val')
test_generator = get_gen('test')

# Initialising the CNN
# model = Sequential()
# # Create convolutional layer. There are 3 dimensions for input shape
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(320, 320, 1)))
# # Pooling layer
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.5))

# # Adding a second convolutional layer with 64 filters
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# # Second pooling layer
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.5))
# # Adding a third convolutional layer with 128 filters
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# # Third pooling layer
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.BatchNormalization())

# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# # Third pooling layer
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.5))


# # Flattening
# model.add(layers.Flatten())
# # Full connection
# model.add(layers.Dense(units=512, activation='relu'))
# model.add(layers.Dense(units=1, activation='sigmoid'))

# print(model.summary())

# model.compile(optimizer="Adam",
#               loss='binary_crossentropy',
#               metrics=['accuracy', keras.metrics.Precision()])

# # Define the callbacks for early stopping of model based on val loss change.

# early_stopping = EarlyStopping(
#     monitor='val_loss', patience=8, verbose=1)
# checkpoint = ModelCheckpoint('model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
#                              verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

# reduce_lr_loss = ReduceLROnPlateau(
#     monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4)


# history = model.fit(train_generator,
#                     steps_per_epoch=30,
#                     epochs=50,
#                     verbose=1,
#                     callbacks =[early_stopping, checkpoint, reduce_lr_loss],
#                     validation_data=val_generator,
#                     validation_steps=8)
# model.save("cnn_model.h5")
input_tensor = layers.Input(shape=(320, 320, 3))

base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = layers.Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = layers.Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='binary_crossentropy')

history = model.fit(train_generator,
                                        steps_per_epoch=30,
                                        epochs=50,
                                        verbose=1,
                                        validation_data=val_generator,
                                        validation_steps=8)
model.save("cnn_model.h5")
# train the model on the new data for a few epochs
model.evaluate(val_generator)
STEP_SIZE_TEST = val_generator.n//val_generator.batch_size
val_generator.reset()
preds = model.predict(val_generator,
                      verbose=1)

fpr, tpr, _ = roc_curve(val_generator.classes, preds)

roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("plot2.png")
