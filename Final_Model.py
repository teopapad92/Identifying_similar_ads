
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from itertools import combinations
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import concatenate
from keras.layers.merge import Add
from keras.layers.merge import Subtract
from keras.layers.merge import Multiply
from keras import metrics
from keras.utils import plot_model
from keras.applications import vgg19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from Img_Features import Get_Features

# #display max columns (for debugging)
# pd.set_option('display.max_colwidth', -1)

#Read data
text_files = pd.read_csv('Text_With_Img.csv')


#Create Tokenizer and fit on all titles
titles = text_files['title'].astype(str)

t = Tokenizer()
t.fit_on_texts(titles)
vocab_size = len(t.word_index) + 1
max_length = 15

print("Loading glove.6B.100d weigths...\n")
embeddings_index = dict()
f = open('C:/Users/teo/Desktop/deeplab/dataset/glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector




#Tokenize function for titles
def Text_Vectorize(List_of_Txt):
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(List_of_Txt)
    # pad documents to a max length of words
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs

def imgfeature(Mylist):
    img = []
    for cc in Mylist:
        img.append(Imagefeatures[cc])
    return img

print("Loading Ready\n")


# In[2]:


#How far in data to go
how_far = 500
length = text_files[0:how_far]

# #Calculate Image Features Live
# print("Calculating Img Features...")
# Imagefeatures = Get_Features(length['img'])
# print("Image features are ready!")

#Load Image Features from disk
path_to_features = 'C:/Users/teo/Desktop/deeplab/dataset'
data_path = os.path.join(path_to_features,'features.npy')
features = np.load(data_path)
Imagefeatures = features[0:how_far]
print("Image features loaded from disk!")

#Create Unique Pairs
print("Creating Unique Pairs...")
title_pairs = [ comb for comb in combinations(length['cc'], 2)]

#Create data for Network
title1 = []
title2 = []
img1 = []
img2 = []
IsSame = []

for cc in title_pairs:
    title1.append(text_files['title'][cc[0]])
    title2.append(text_files['title'][cc[1]])
    img1.append(cc[0])
    img2.append(cc[1])
    if (text_files['label'][cc[0]] == text_files['label'][cc[1]]):
        IsSame.append(1)
    else:
        IsSame.append(0)

print(len(title_pairs)," Pairs Created!\n")

#Create Dataset
final_data = pd.DataFrame({'Truth': IsSame})
final_data['textVectors1'] = [title for title in title1]
final_data['img1'] = [img for img in img1]
final_data['textVectors2'] = [title for title in title2]
final_data['img2'] = [img for img in img2]

# display(final_data['img1'])

attributes = ['textVectors1','img1','textVectors2','img2']
x_train, x_test, y_train, y_test = train_test_split(final_data[attributes],
                                                    final_data['Truth'], test_size=0.20, random_state=0)

x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train, test_size=0.20, random_state=0)

print('Train:', len(y_train),'Val:',len(y_test), 'Test:', len(y_test))


# Vectorize Text Before feed to network
print("Creating Text Vectors...")
TrainTextVectors1 = np.array(Text_Vectorize(x_train['textVectors1']))
TrainTextVectors2 = np.array(Text_Vectorize(x_train['textVectors2']))
ValTextVectors1 = np.array(Text_Vectorize(x_val['textVectors1']))
ValTextVectors2 = np.array(Text_Vectorize(x_val['textVectors2']))
TestTextVectors1 = np.array(Text_Vectorize(x_test['textVectors1']))
TestTextVectors2 = np.array(Text_Vectorize(x_test['textVectors2']))

Trainimg1 = np.array(imgfeature(x_train['img1']))
Trainimg2 = np.array(imgfeature(x_train['img2']))
Valimg1 = np.array(imgfeature(x_val['img1']))
Valimg2 = np.array(imgfeature(x_val['img2']))
Testimg1 = np.array(imgfeature(x_test['img1']))
Testimg2 = np.array(imgfeature(x_test['img2']))

input_size = TrainTextVectors1.shape[1]
print("Data ready for training!")


# In[3]:


#First AD

#First text 
InputText1 = Input(shape=(input_size,) )
Embed1 = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=True)(InputText1)
flat1 = Flatten()(Embed1)

#First Image 
Image1 = Input(shape=(4096,))

#merge first text with first image
merge1= concatenate([flat1,Image1])

#Second AD

#Second text 
InputText2 = Input(shape=(input_size,) )
Embed2 = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=True)(InputText2)
flat2 = Flatten()(Embed2)

#Second Image 
Image2 = Input(shape=(4096,))

#merge second text with second image
merge2 = concatenate([flat2,Image2])


#Connect two pairs
merge3 = Subtract()([merge1,merge2])
output = Dense(1, activation='sigmoid')(merge3)

final_model = Model(inputs=[InputText1,Image1,InputText2,Image2], outputs=output)


# print(final_model.summary())        

# plot
plot_model(final_model, to_file='final_model_layers.png',show_shapes=True)

final_model.compile(loss='binary_crossentropy',
                    optimizer=tf.train.AdamOptimizer(),
                   )


history = final_model.fit( [TrainTextVectors1,Trainimg1,TrainTextVectors2,Trainimg2], y_train, 
                          epochs=3,
                          batch_size=256,
                          validation_data=([ValTextVectors1,Valimg1,ValTextVectors2,Valimg2], y_val),
                          verbose=1
                        )

y_score = final_model.predict([TestTextVectors1,Testimg1,TestTextVectors2,Testimg2])
ROC_score = roc_auc_score(y_test,y_score)
print("ROC_score: ",ROC_score)

#Create Pyplot for roc curve
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_score)
auc_rf = auc(fpr_rf, tpr_rf)


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

