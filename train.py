import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def get_data(data_dir, labels, resize=False, size=64):
    data = [] 

    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)

        for img in os.listdir(path):
            try:
                #convert BGR to RGB format
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] 

                if(resize):
                    # Reshaping images to preferred size
                    img_arr = cv2.resize(img_arr, (img_size, img_size))

                data.append([img_arr, class_num])

            except Exception as e:
                print(e)

    return np.array(data)


labels = ['circle', 'rectangle']
img_size = 64

train_data = get_data("./dataset/toy_train", labels)
test_data = get_data("./dataset/toy_val", labels)

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train_data:
  x_train.append(feature)
  y_train.append(label)

for feature, label in test_data:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

model = Sequential()
model.add(Convolution2D(32,3, padding="same", activation="relu", input_shape=(64,64,3)))
model.add(MaxPooling2D())

model.add(Convolution2D(32, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = Adam(learning_rate=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

history = model.fit(x_train,y_train,epochs = 500 , validation_data = (x_val, y_val))

model.save('trained_model')
print("model saved to disk")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = np.argmax(model.predict(x_val),axis=1)

print(classification_report(y_val, predictions, target_names = ['Rugby (Class 0)','Soccer (Class 1)']))

