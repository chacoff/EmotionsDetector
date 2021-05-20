import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Activation, Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
sns.set_theme(style="white")


def get_gpus():
    local_device_protos = device_lib.list_local_devices()
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def loader(data, KerasSplit=True):
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for index, row in data.iterrows():

        image_bw = np.asarray([int(x) for x in row['pixels'].split()])
        image_bw = image_bw.reshape(48, 48).astype('uint8') / 255  # /255 normalized
        image_bw = cv2.flip(image_bw, 0)

        if KerasSplit:
            train_data.append(image_bw)
            train_label.append(row['emotion'])
        else:
            if row['Usage'] == 'Training':
                train_data.append(image_bw)
                train_label.append(row['emotion'])
            else:
                test_data.append(image_bw)
                test_label.append(row['emotion'])

    test_data = np.array(np.expand_dims(test_data, -1))
    test_label = np.array(to_categorical(test_label, num_classes=7))
    train_data = np.array(np.expand_dims(train_data, -1))
    train_label = np.array(to_categorical(train_label, num_classes=7))
    print(f'[info] \ntraining data: {len(train_data)} '
          f'\ntraining labels: {len(train_label)} '
          f'\nvalidation data: {len(test_data)}')

    return test_data, test_label, train_data, train_label


# Hyperparameters of exp1
KerasSplit = True
epochs = 150
batch_size = 32
learning_rate = 0.0001
lr_factor = 0.9  # 0.9 - learning rate factor in plateau
lr_patience = 6  # 6 - learning rate patience
val = 0.20  # validation split
l2_reg = 0.01  # regularization L2
stop_patience = 20
IM_SIZE = 48
pesos = 'models/exp2/weights.hd5'
modelo_json = 'models/exp2/model.json'
modelo_h5 = 'models/exp2/model.h5'

# Data exploration Training 28709 Test 7178 (public+private)
source = 'fer2013/fer2013.csv'
data = pd.read_csv(source)
test_data, test_label, train_data, train_label = loader(data, KerasSplit=KerasSplit)

# data augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(IM_SIZE, IM_SIZE, 1)),
    # layers.experimental.preprocessing.RandomRotation(0.3),
    layers.experimental.preprocessing.RandomZoom(0.5),
])

# model architecture
model = Sequential([
    data_augmentation,
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_reg)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Dropout(0.5),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.5),

    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.5),

    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.5),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax'),  # 7 = number of classes
])

model.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience)
early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=stop_patience, mode='auto')
checkpointer = ModelCheckpoint(pesos, monitor='val_loss', verbose=1, save_best_only=True)

model.fit(train_data,
          train_label,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[lr_reducer, checkpointer, early_stopper],
          validation_split=val,
          shuffle=True,
          verbose=1)

model_json = model.to_json()
with open(modelo_json, "w") as json_file:
    json_file.write(model_json)
model.save_weights(modelo_h5)
print('Saved model to disk')
