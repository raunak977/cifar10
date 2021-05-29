#!/usr/bin/env python
# coding: utf-8

# In[10]:


#import tensorflow
#import numpy
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import OrdinalEncoder
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


# In[11]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# In[12]:


m = Sequential()
m.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
m.add(Dropout(0.2))
m.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
m.add(MaxPooling2D())
#m.add(Flatten())
m.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
m.add(Dropout(0.2))
m.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
m.add(MaxPooling2D())
m.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
m.add(Dropout(0.2))
m.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
m.add(MaxPooling2D())
m.add(Flatten())
m.add(Dropout(0.2))
m.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
m.add(Dropout(0.2))
m.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
m.add(Dropout(0.2))
m.add(Dense(num_classes, activation='softmax'))


# In[13]:


epochs = 10
m.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, decay=0.01/epochs, nesterov=False), metrics=['accuracy'])


# In[14]:


# Fit the model
history=m.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = m.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]))


# In[16]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5,1])
plt.legend(loc = 'lower right')


# In[15]:


from tensorflow.keras.models import load_model
m.save("project.h5")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




