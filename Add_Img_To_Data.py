
# coding: utf-8

# In[3]:


import os
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gensim
import glob

# #display max columns (for debugging)
# pd.set_option('display.max_colwidth', -1)

# Enter Directory of all images
img_dir = "C:/Users/teo/Desktop/deeplab/dataset/images2"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)

text_files = pd.read_csv('data.txt',delimiter = ';',error_bad_lines=False)
text_files.set_index('cc', inplace=True,drop=False)

img = []

for cc in text_files['cc']:
    photoID = '_'+str(cc)+'_'
    for f1 in files:
        if (photoID in f1):
            img.append(f1)


text_files.insert(loc=3, column='img', value=img)

filename_write = os.path.join("Text_With_Img.csv")
text_files.to_csv(filename_write,index=False)
print("File Created")

