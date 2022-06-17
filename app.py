import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
import re
from scipy import spatial

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def cleaned_doc(text):
    #remove punctuation
    stripped_text = re.sub(r'[^\w\s]','',text)
    #tokenize
    word_tokens = word_tokenize(stripped_text)
    word_tokens = [word.lower() for word in word_tokens]
    #remove words that are not alphabetic
    alpha_words = [word for word in word_tokens if word.isalpha()]
    #remove stopwords
    cleaned_words = [word for word in alpha_words if not word in stopwords.words('english')]
    return cleaned_words

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/score', methods = ['POST'])
def score():
    #for rendering result in html GUI
    text1 = request.form['document 1']
    text2 = request.form['document 2']
    vec1 = model.infer_vector(cleaned_doc(text1))
    vec2 = model.infer_vector(cleaned_doc(text2))
    cos_distance = spatial.distance.cosine(vec1, vec2)
    cosine_sim = 1 - cos_distance

    return render_template('homepage.html', calculated_score = 'Similarity Score : {} '.format(cosine_sim))


if __name__ == "__main__":
    app.run(host='0.0.0.0')
