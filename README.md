# Document-Similarity-Score

Link to the web app : https://similarity-score-api.herokuapp.com/

## Procedure
### For the model building part: 
I developed the model and trained it. I have used doc2vec and cosine similarity
for estimating the similarity score between two documents.
1. First I loaded the dataset and merged the columns using pandas.
2. I have used nltk library to Tokenize the words in each paragraph. 
3. Used regex for punctuation removal for each paragraph.
4. Converted the text in Lowercase for each paragraph. 
5. Removed the words which are not alphabetical for each paragraph. 
6. Used TaggedDocument from gensim library for tagging the documents. 
7. Used Doc2Vec model from gensim library for embedding of the paragraphs. 
8. Trained the Doc2Vec model with the tagged documents. 
9. Saved the model in disk using pickle library. 

### For flask api part: 
In this file, I have used the flask web framework to handle the POST requests. 
I have used Heroku for deploying the app. 
1. Used Flask for creating the web application object. 
2. Used pickle library to load the trained model into the model. 
3. In this file I have used two methods home() and score().
    i) The home() method renders homepage.html page and gives the home page interface. 
    ii) The score() method is for rendering requests in html GUI. It gets the text
    data from the request.form. The text data is then cleaned and preprocessed
    by a defined function. The cleaned texts are vectorized using the model.infer_vector
    attribute. Then I calculate the cosine similarity using scipy.spatial. 
    the score method is called when someone clicks the similarity_score button in order
    to get the result. 
4. Created Procfile for deploying the web application. 
5. Created requirements.txt for listing out all the version of the packages I have 
    used in my app. 
6. Pushed all the files to github repository. 
7. Created a Heroku Application. 
7. Deployed the app by linking the github repository to Heroku application. 
