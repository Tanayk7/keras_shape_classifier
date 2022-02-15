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

# load the test data from dataset
test_data = get_data("./dataset/toy_val", labels)

x_val = []
y_val = []

# separate the features and labels
for feature, label in test_data:
  x_val.append(feature)
  y_val.append(label)

# normalize
x_val = np.array(x_val) / 255

# reshape data for model
x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

# load the trained model
model = tf.keras.models.load_model("trained_model")

# make predictions using trained model
predictions = np.argmax(model.predict(x_val),axis=1)

total = len(predictions)
correct = 0

for i in range(len(y_val)):
    if(y_val[i] == predictions[i]):
        correct += 1
    print('actual: ', y_val[i], ' predicted: ', predictions[i])

accuracy = (correct/total) * 100

print("accuracy: ", accuracy, "%")

# generate the classification report
print(classification_report(y_val, predictions, target_names = ['circle (Class 0)','rectangle (Class 1)']))