# Title:        Cat and Dog classification
# Author:       Ahmed Muhammed Abdelgaber
# Data:         Fri 8, 2024 
# email:         ahmedmuhammedza1998@gmail.com 
# code version: 0.0.0
#
# Copyright 2024 Ahmed Muhammed 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This model trained using limited dataset
# and for improving performance just use any available dataset on 'kaggle'

from tensorflow.keras import models, layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
file_path_list= []
labels = []
for root, dirs, files in os.walk("D:\\pydata\\data"):
    for file in files:
        filepath = os.path.join(root, file)
        file_path_list.append(filepath)
        sfilePath = filepath.split("\\")
        if "cat" in sfilePath:
            labels.append([0,1])
        elif "dog" in sfilePath:
            labels.append([1,0])

        # print(filepath)
# print(labels)
arrayOfImages = []        
for img in file_path_list:
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_LINEAR)
    
    arrayOfImages.append(img)
    cv2.imshow("test", img)
    cv2.waitKey(0)


cv2.destroyAllWindows()


# print(arrayOfImages[0].shape)
class_names = ['cat','dog']
images = np.stack(arrayOfImages).reshape(-1, 28, 28, 1)

# print(file_path_list)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
                loss = tf.keras.losses.binary_crossentropy,
                metrics=['accuracy'])

history = model.fit(tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.int32), epochs=14)
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()

img= "D:\\pydata\\test\\testDog.PNG"
img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_LINEAR)
img = np.reshape(img, (-1, 28, 28, 1))
print(model.predict(tf.convert_to_tensor(img, dtype= tf.float32)))
