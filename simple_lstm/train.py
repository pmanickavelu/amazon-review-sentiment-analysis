import pandas as pd 
import bz2
import re, pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

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


max_len, train_labels,train_sentences = clean_and_process_file('../data/train.ft.txt.bz2')

max_features = 20000


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_sentences)

X = tokenizer.texts_to_sequences(train_sentences)
X = pad_sequences(X,max_len)
Y = pd.get_dummies(train_labels).values


graph = tf.Graph()
embed_dim = 8
lstm_out = 8
batch_size = 128
epochs = 7
with graph.as_default():
    model = Sequential()
    model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(Y.shape[1],activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.fit(X, Y, epochs = epochs, batch_size=batch_size)
# saveing the model
    model.save_weights('model.h5')
with open("model.json","w") as f:
    f.write(model.to_json())
pickle.dump(tokenizer,open("tokenizer.pkl","wb"))
pickle.dump(X.shape[1],open("max_text_length.pkl","wb"))




