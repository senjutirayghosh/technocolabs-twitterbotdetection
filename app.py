from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, preprocessing, ensemble, metrics
#from sklearn.externals import joblib

#Loading the model
filename = 'twitterbot_detection.pkl'
lr = pickle.load(open(filename, 'rb'))
tfidf_vect_char = pickle.load(open('transform.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('webpage.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        #print(data)
        vec = tfidf_vect_char.transform(data).toarray()
        my_pred = lr.predict(vec)
        #print(my_pred)
    return render_template('webpage.html', prediction = my_pred)


if __name__  == '__main__':
    app.run(debug = True)
