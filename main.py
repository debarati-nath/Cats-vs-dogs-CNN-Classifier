import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Input
import pickle
from tensorflow.keras.models import load_model
import h5py
import datetime
import time
import random

#Load images from the personal computer
DATADIR="C:/Users/Tuli/Downloads/kagglecatsanddogs_3367a/PetImages"

CATAGORIES =["Dog","Cat"]
for category in CATAGORIES:
    path=os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
        break

print(img_array.shape)  #Display the shape of the images

#Rearrange the image
IMG_SIZE=70
new_array=cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap="gray")
plt.show()

#Create training data
training_data=[]
def create_training_data():
    for category in CATAGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATAGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as exc:
                pass

create_training_data()
print(len(training_data))

random.shuffle(training_data)
for sample in training_data:
    print(sample[1])

#Save training and testng data
X=[]
y=[]
for features , label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
pickle_out= open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out= open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
pickle_in=open("X.pickle","rb")
X=pickle.load(pickle_in)
pickle_in=open("y.pickle","rb")
y=pickle.load(pickle_in)

#Load the data
X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))

#Normalize the training data
X=X/255.0

#Model
model=Sequential()
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics= ['accuracy'])
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3)

#Save the model
model.save('my_model.h5')

#Load the model
model_load=keras.models.load_model('my_model.h5')
model_load.summary()

