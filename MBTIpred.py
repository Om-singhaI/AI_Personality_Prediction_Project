import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Reading the file
data=pd.read_csv('mbti_1.csv')

# Splits training data and testing data into 2 different splits
# Training data: 80%
# Testing data: 20%
train_data,test_data=train_test_split(data,test_size=0.2,random_state=35,stratify=data.type)


# Use simple function to clean up 'https?://[^\s<>"]+|www\.[^\s<>"]+'
def clear_text(data):
    data_length=[]
    lemmatizer=WordNetLemmatizer()
    cleaned_text=[]
    for sentence in tqdm(data.posts):
        sentence=sentence.lower()

#        removing links from text data
        sentence=re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+',' ',sentence)

#        removing other symbols
        sentence=re.sub('[^0-9a-z]',' ',sentence)


        data_length.append(len(sentence.split()))
        cleaned_text.append(sentence)
    return cleaned_text,data_length


# Using the function on train_data.posts
train_data.posts,train_length=clear_text(train_data)
test_data.posts,test_length=clear_text(test_data)

# TOKENIZING THE POSTS
# - Breaks words down into smaller units called tokens
# - Will be using Lemmatization:
# Sentence: "The quick brown fox jumps over the lazy dog."
# Lemmatization: "the", "quick", "brown", "fox", "jump", "over", "the", "lazy", "dog"

import nltk
nltk.download('wordnet')
# Setting up Lemmatizer
class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word)>2]
    
# Vectorize Data
# Fitting data to a vectorizer object
vectorizer=TfidfVectorizer( max_features=5000,stop_words='english',tokenizer=Lemmatizer())
vectorizer.fit(train_data.posts)

# Vectorizing the training posts and the testing posts
train_post=vectorizer.transform(train_data.posts).toarray()
test_post=vectorizer.transform(test_data.posts).toarray()
train_post.shape

target_encoder=LabelEncoder()
train_target=target_encoder.fit_transform(train_data.type)
test_target=target_encoder.fit_transform(test_data.type)






# import the library
from appJar import gui
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
import re
import string

# Replace this part with your actual train_data


class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.lemmatizer.lemmatize(word) for word in doc.split()]

# Replace ... with your actual TfidfVectorizer and Logistic Regression model
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', tokenizer=Lemmatizer())
model_log = LogisticRegression(max_iter=3000, C=0.5, n_jobs=-1)

# Assuming train_data is your training dataset
vectorizer.fit(train_data.posts)
train_post = vectorizer.transform(train_data.posts)

target_encoder = LabelEncoder()
train_target = target_encoder.fit_transform(train_data.type)

model_log.fit(train_post, train_target)

def preprocess_text(text):
    # Function to preprocess input text (similar to what was done during training)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

def predict_mbti(text):
    # Function to predict MBTI from input text
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    probabilities = model_log.predict_proba(vectorized_text)[0]
    prediction = model_log.predict(vectorized_text)[0]
    confidence_percentage = probabilities[prediction] * 100
    return target_encoder.inverse_transform([prediction])[0], confidence_percentage

# handle button events
def press(button):
    if button == "Cancel":
        app.stop()
    else:
        input_text = app.getEntry("Entry")
        prediction, confidence = predict_mbti(input_text)
        app.clearLabel("Result")
        app.setLabel("Result", f"Predicted MBTI: {prediction}\nConfidence: {confidence:.2f}%")

# create a GUI variable called app
app = gui("Personality Prediction", "400x300")
app.setBg("black")
app.setFont(18)

# add & configure widgets
app.addFlashLabel("title", "Welcome to the AI Personality Predictor")
app.setLabelBg("title", "gray")
app.setLabelFg("title", "white")
app.setLabelFont("title", 20)

app.addLabelEntry("Entry")
app.setLabelBg("Entry", "gray")
app.setLabelFg("Entry", "white")

app.addLabel("Result", "")
app.setLabelBg("Result", "gray")
app.setLabelFg("Result", "white")

app.addButtons(["Submit", "Cancel"], press)

app.setFocus("Entry")

# start the GUI
app.go()