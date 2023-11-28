#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !wget https://github.com/alexeygrigorev/large-datasets/releases/download/wasps-bees/bees-wasps.h5


# In[2]:


# pip install tensorflow


# In[4]:


import tensorflow as tf
from tensorflow import keras


# In[5]:


model = keras.models.load_model('bees-wasps.h5')


# #### Question-1: Now convert this model from Keras to TF-Lite format. What's the size of the converted model?

# In[6]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('bees-wasps.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


# In[7]:


ls -lh


# In[8]:


ls -lh bees-wasps.tflite


# #### Question-2: To be able to use this model, we need to know the index of the input and the index of the output. What's the output index for this model?

# In[9]:


import tensorflow.lite as tflite


# In[10]:


interpreter = tflite.Interpreter(model_path='bees-wasps.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# In[11]:


output_index


# ## Prepare the image

# In[12]:


pip install pillow


# In[13]:


from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


# In[14]:


img = download_image('https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg')
img = prepare_image(img, target_size=(150,150))
img


# #### Question-3: Now we need to turn the image into numpy array and pre-process it. Tip: Check the previous homework. What was the pre-processing we did there? After the pre-processing, what's the value in the first pixel, the R channel

# In[15]:


import numpy as np


# In[16]:


def prepare_input(x):
    return x * (1/255.0)


# In[17]:


x = np.array(img, dtype='float32')
X = np.array([x])
X = prepare_input(X)
X


# In[18]:


X[0, 0, 0, 0]


# #### Question-4: Now let's apply this model to this image. What's the output of the model?

# In[19]:


interpreter.set_tensor(input_index, X)

interpreter.invoke()

preds = interpreter.get_tensor(output_index)


# In[20]:


preds


# ##### the probability we have a bee is 66%

# In[ ]:




