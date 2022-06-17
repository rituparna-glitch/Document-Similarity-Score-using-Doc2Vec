import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy import spatial
import pickle

#load the dataset
df = pd.read_csv('Precily_Text_Similarity.csv')

#merge the two columns into one
df_new = pd.concat([df['text1'], df['text2']], ignore_index=True)

def cleaned_doc(text):
    #remove punctuation
    stripped_text = re.sub(r'[^\w\s]','',text)
    #tokenize
    word_tokens = word_tokenize(stripped_text)
    word_tokens = [word.lower() for word in word_tokens]
    #remove words that are not alphabetic
    alpha_words = [word for word in word_tokens if word.isalpha()]
    #remove stopwords
    #cleaned_words = [word for word in alpha_words if not word in stopwords.words('english')]
    #return cleaned_words
    return alpha_words

#clean the paragraphs
text_docs = []
for text in df_new:
    cleaned_document = cleaned_doc(text)
    text_docs.append(cleaned_document)

#creating tagged documents
tagged_data = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(text_docs)]

#model building

max_epochs = 100
vec_size = 300
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)


#saving model to disk
with open('model.pkl', 'wb') as files:
    pickle.dump(model, files)
print("Model Saved")

#load model to predict score
model_saved = pickle.load(open('model.pkl','rb'))
score = model_saved.similarity_unseen_docs('I am a girl'.split(), 'I am a boy'.split())
print(score)
