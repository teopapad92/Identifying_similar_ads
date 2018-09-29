
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
from keras.layers.merge import Add
from keras.layers.merge import Multiply
from keras.layers.merge import concatenate
from keras.layers.merge import Concatenate
from keras.layers.merge import Subtract
from keras import metrics
from keras.utils import plot_model
from tensorflow import metrics

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


# In[2]:


#How far in data to go
how_far=500
length = text_files[0:how_far]

#Create Unique Pairs
print("Creating Unique Pairs...")
title_pairs = [ comb for comb in combinations(length['cc'], 2)]


#Create data for Network
title1 = []
title2 = []
IsSame = []

for cc in title_pairs:
    title1.append(text_files['title'][cc[0]])
    title2.append(text_files['title'][cc[1]])
    if (text_files['label'][cc[0]] == text_files['label'][cc[1]]):
        IsSame.append(1)
    else:
        IsSame.append(0)

print(len(title_pairs)," Pairs Created!\n")

#Create Dataset
final_data = pd.DataFrame({'Truth': IsSame})
final_data['textVectors1'] = [title for title in title1]
final_data['textVectors2'] = [title for title in title2]


attributes = ['textVectors1','textVectors2']
x_train, x_test, y_train, y_test = train_test_split(final_data[attributes],
                                                    final_data['Truth'], test_size=0.20, random_state=0)

x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train, test_size=0.20, random_state=0)

# print('Train:', len(y_train),'Val:', len(y_val), 'Test:', len(y_test))
    
# Vectorize Text Before feed to network
print("Creating Text Features...")
TrainTextVectors1 = Text_Vectorize(x_train['textVectors1'])
TrainTextVectors2 = Text_Vectorize(x_train['textVectors2'])
ValTextVectors1 = Text_Vectorize(x_val['textVectors1'])
ValTextVectors2 = Text_Vectorize(x_val['textVectors2'])
TestTextVectors1 = Text_Vectorize(x_test['textVectors1'])
TestTextVectors2 = Text_Vectorize(x_test['textVectors2'])

input_size = TrainTextVectors1.shape[1]
print("Data ready for training!")


# In[3]:


#First input label
visible1 = Input(shape=(input_size,) )
Embed1 = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)(visible1)
flat1 = Flatten()(Embed1)

#Second input label
visible2 = Input(shape=(input_size,)) 
Embed2 = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)(visible2)
flat2 = Flatten()(Embed2)

#merge inputs
# merge = concatenate([flat1,flat2])
merge = Subtract()([flat1,flat2])
output = Dense(1, activation='sigmoid')(merge)

text_model = Model(inputs=[visible1,visible2], outputs=output)

# print(text_model.summary())        

# plot
os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/Library/bin/Graphviz2.38/bin'
plot_model(text_model, to_file='Text_only_Model.png',show_shapes=True)

text_model.compile(loss='binary_crossentropy',
                   optimizer=tf.train.AdamOptimizer(),
                  )


history = text_model.fit( [TrainTextVectors1,TrainTextVectors2], y_train, 
                         epochs=3,
                         batch_size=256,
                         validation_data=([ValTextVectors1,ValTextVectors2], y_val),
                         verbose=1
                        )

# results = text_model.evaluate( [TestTextVectors1,TestTextVectors1], y_test)

# print("Results: ",results)


y_score = text_model.predict([TestTextVectors1,TestTextVectors2])
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

