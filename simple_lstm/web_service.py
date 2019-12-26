#!flask/bin/python
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

hello = tf.constant('Hello, TensorFlow!')
graph = tf.Graph()
sess = tf.compat.v1.Session(graph=graph)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json

import re, pickle, os

with sess.as_default():
    with graph.as_default():
        model = model_from_json(open("model.json","r").read())
        model.load_weights('model.h5')

tokenizer = pickle.load(open("tokenizer.pkl","rb"))
max_text_length = pickle.load(open("max_text_length.pkl","rb"))

app = Flask(__name__)
@app.route('/', methods=['POST'])
def index():

    o_texts, texts = clean_data(request.get_json()['text'])
    predicts = make_prediction(texts)
    results = build_results(predicts,o_texts,texts)
    
    return jsonify(results)
def clean_data(text):
    o_texts = []
    if type(text) == str:
        o_texts = [text]
    elif type(text) == list:
        o_texts = text
    texts = o_texts
    for i in range(len(texts)):
        texts[i] = texts[i].lower()
        texts[i] = re.sub('[^a-zA-z0-9\s]','',texts[i])
        texts[i] = texts[i].replace('rt',' ')
    return o_texts,texts
def make_prediction(texts):
    global sess
    global graph
    global model
    with sess.as_default():
        with graph.as_default():
            return model.predict(pad_sequences(tokenizer.texts_to_sequences(texts),max_text_length).reshape(len(texts),max_text_length),batch_size=1)
def build_results(predicts,o_texts,texts):
    results = []
    for i in range(len(texts)):
        sentiment = "Positive"
        if np.argmax(predicts[i]) == 0:
            sentiment = "Negative"
        elif np.argmax(predicts[i]) == 1:
            sentiment = "Neutral"
        
        results.append({
                "sentiment": sentiment,
                "text": o_texts[i],
                "score":float(predicts[i][np.argmax(predicts[i])])
                })
    return results
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=os.environ.get("FLAST_DEBUG_MODE"True))


