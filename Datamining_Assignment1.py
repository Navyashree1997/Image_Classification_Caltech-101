#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split

import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from prettytable import PrettyTable


# In[5]:


path = './caltech101_classification/'

list_of_list = []

for folder in os.listdir(path):
    for image in os.listdir(path + folder):
        list_of_list.append([path + folder + '/' + image, folder])

df = pd.DataFrame(list_of_list, columns = ['image_path', 'object_name'])
df['label'] = df['object_name'].replace(['Motorbikes', 'airplanes', 'schooner'], ['0', '1', '2'])


df.head()


# In[6]:


shapes = [cv2.imread(df.iloc[index, 0]).shape for index in df.index]

shapes


# In[7]:


# shapes of images are different, so resize to 128 x 128 x 3

Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3
batch_size = 16


# In[13]:


train, test = train_test_split(df, test_size = .2, stratify = df['label'], random_state = 0)

train.shape, test.shape


# In[14]:


train_datagen = ImageDataGenerator(rotation_range = 15, 
                                   rescale = 1./255, 
                                   shear_range = 0.1, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True, 
                                   width_shift_range = 0.1, 
                                   height_shift_range = 0.1)
train_generator = train_datagen.flow_from_dataframe(train, 
                                                    x_col = 'image_path', 
                                                    y_col = 'label', 
                                                    target_size = Image_Size, 
                                                    class_mode = 'sparse', 
                                                    batch_size = batch_size)

test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_dataframe(test, 
                                              x_col = 'image_path', 
                                              y_col = 'label', 
                                              target_size = Image_Size, 
                                              class_mode = 'sparse', 
                                              batch_size = batch_size)


# In[15]:


total_train = train.shape[0]
total_test = test.shape[0]

total_train, total_test


# In[22]:


reduce_lr = ReduceLROnPlateau(monitor = "val_accuracy", 
                              factor = .4642, 
                              patience = 3, 
                              verbose = 1, 
                              min_delta = 0.001, 
                              mode = "max")
earlystop = EarlyStopping(monitor = "val_accuracy", 
                          patience = 10, 
                          verbose = 1, 
                          mode = "max", 
                          restore_best_weights = True
                          )


# # model2
# 
# conv2d + relu activation + max pool + dense layer + dropout

# In[21]:


def model_2():
    tf.keras.backend.clear_session()
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape = (Image_Width, Image_Height, Image_Channels)))
  #model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  #model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3)))
  #model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  #model.add(Dropout(0.25))

  model.add(Conv2D(128, (3, 3)))
  #model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  #model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(64))
  model.add(Dropout(0.5))
  model.add(Dense(3, activation = 'softmax'))

  model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model


# In[23]:


model2 = model_2()
model2.summary()


# In[24]:


callbacks_list = [reduce_lr, earlystop]
epochs = 150
history = model2.fit(train_generator, 
                    epochs = epochs, 
                    validation_data = test_generator, 
                    validation_steps = total_test//batch_size, 
                    steps_per_epoch = total_train//batch_size, 
                    callbacks = callbacks_list)


# In[25]:


plt.plot(history.history['accuracy'], label = 'train')
plt.plot(history.history['val_accuracy'], label = 'cv')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.show()


# # model1
# 
# conv2d + batchnormalization + relu activation + max pool + dropout + output dense layer 

# In[32]:


def model_4():
  tf.keras.backend.clear_session()
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape = (Image_Width, Image_Height, Image_Channels)))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3)))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, (3, 3)))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Dropout(0.25))
  
  model.add(Conv2D(3, (14, 14)))
  model.add(Activation("relu"))
  model.add(Flatten())
  #model.add(Dense(64))
  #model.add(Dropout(0.5))
  model.add(Dense(3, activation = 'softmax'))

  model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model


# In[33]:


model4 = model_4()
model4.summary()


# In[34]:


callbacks_list = [reduce_lr, earlystop]
epochs = 150
history = model4.fit(train_generator, 
                    epochs = epochs, 
                    validation_data = test_generator, 
                    validation_steps = total_test//batch_size, 
                    steps_per_epoch = total_train//batch_size, 
                    callbacks = callbacks_list)


# In[35]:


plt.plot(history.history['accuracy'], label = 'train')
plt.plot(history.history['val_accuracy'], label = 'cv')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.show()

