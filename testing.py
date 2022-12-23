import streamlit as st
import joblib as jb
import nltk
from nltk.corpus import stopwords
import re 

model = jb.load('Text_Spam_detection_Randomforestmodel')
vectorizer = jb.load('Text_Spam_detection_transformation')

def sentence_processing(sentence):
    filtered_sentence = re.sub('[^a-zA-z]',' ',sentence).lower().split()
    filtered_sentence = [word for word in filtered_sentence if word not in stopwords.words('english')]
    filtered_sentence = ' '.join(filtered_sentence)
    
    vector = vectorizer.transform([filtered_sentence])
    return vector


def predict(sentence):   

    y_pred = sentence_processing(sentence)    
    y_pred = model.predict(y_pred)

    return y_pred

testing = 'Hello how are you'

print(predict(testing))
print('nnh')