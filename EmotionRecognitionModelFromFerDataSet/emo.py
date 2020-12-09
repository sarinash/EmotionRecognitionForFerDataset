import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.utils import np_utils
import sys
import os
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# initiate parameters
batch_size = 64
epochs = 35
width, height = 48, 48
num_features = 64
num_labels = 7

###############################____________________________________________________###########################3
df = pd.read_csv("fer2013.csv")
X_train, train_y, X_test, test_y = [], [], [], []
for index, row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print("error")
#### to see a part of data
# print  (f"X_train samole data:{X_train[0:2]}")
# to numpy array  and normalize it ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')
train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)
X_test /= np.std(X_test, axis=0)
X_test -= np.mean(X_test, axis=0)
X_train /= np.std(X_train, axis=0)
X_train -= np.mean(X_train, axis=0)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# reshape data from array to keras format 48=> pixels width and hight
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# design model conv2d as pictures are 2D
model = Sequential()
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(num_features,kernel_size= (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(num_features, (3, 3), activation='relu'))
model.add(Conv2D(num_features, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(2*num_features, (3, 3), activation='relu'))
model.add(Conv2D(2*num_features, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
######################## addding dense layers ******************************************
model.add(Dense(16*num_features, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16*num_features, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_labels, activation='softmax'))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, test_y), shuffle= True)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# to json
emo_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(emo_json)
model.save_weights("emo.h5")
