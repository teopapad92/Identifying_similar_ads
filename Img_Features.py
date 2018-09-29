
# coding: utf-8

# In[4]:


import tensorflow as tf
from keras.applications import vgg19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle

model = vgg19.VGG19(weights='imagenet', include_top=True)
model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)

# start_time = time.time()
# print("Calculating Features")

def Get_Features(Img_List):
    fc2_feature_list = []
    #load images
    for f1 in Img_List:
        img = image.load_img(f1, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        #Extract image features
        fc2_features = model_extractfeatures.predict(img_data)
        #Reshape 
        fc2_features = fc2_features.reshape((4096,))
        fc2_feature_list.append(fc2_features)
    fc2_feature_list_np = np.array(fc2_feature_list)
    return fc2_feature_list_np

# #Print how long the extraction lasted
# print("--- %s seconds ---" % (time.time() - start_time))

# # Uncomment For offline Feature extraction
# text_files = pd.read_csv('Text_With_Img.csv')
# titles = text_files['img'][0:1000]

# feats = Get_Features(titles)
# display(feats.shape)
# display(fe.shape)

# np.save('C:/Users/teo/Desktop/deeplab/dataset/features2',feats)

