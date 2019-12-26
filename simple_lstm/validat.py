# built on python3.7

import numpy as np
import pandas as pd
import bz2

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json

import re, pickle
import tensorflow as tf

graph = tf.Graph()
def clean_and_process_file(file):
    max_len = 0
    labels = []
    sentences = []
    for line in bz2.BZ2File(file).readlines():
        decoded_line = line.decode('utf-8').split(" ")
        if len(decoded_line)-1 > max_len:
            max_len = len(decoded_line)-1 
        
        labels.append(decoded_line[0].split("__")[2])
        sentence = re.sub('\d','0'," ".join(decoded_line[1:]).lower())
        if 'www.' in sentence or 'http:' in sentence or 'https:' in sentence or '.com' in sentence:
            sentence = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", sentence)
        sentences.append(sentence)
    return max_len,labels,sentences


max_len, test_labels,test_sentences = clean_and_process_file('../data/test.ft.txt.bz2')
# data prepration/feature extraction/feature building
tokenizer = pickle.load(open("tokenizer.pkl","rb"))
max_len = pickle.load(open("max_text_length.pkl","rb"))

X = tokenizer.texts_to_sequences(test_sentences)
X = pad_sequences(X,max_len)
Y = pd.get_dummies(test_labels).values

batch_size = 32
pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

with graph.as_default():
    model = model_from_json(open("model.json","r").read())
    print (model.summary())
    model.load_weights('model.h5')

    for x in range(len(X)):
        
        result = model.predict(X[x].reshape(1,X.shape[1]),batch_size=1)[0]
       
        if np.argmax(result) == np.argmax(Y[x]):
            if np.argmax(Y[x]) == 1:
                neg_correct += 1
            else:
                pos_correct += 1
           
        if np.argmax(Y[x]) == 1:
            neg_cnt += 1
        else:
            pos_cnt += 1



print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")
    
