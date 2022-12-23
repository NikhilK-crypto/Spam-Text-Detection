import streamlit as st
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re 

model = joblib.load('Text_Spam_detection_Xgboostmodel')
vectorizer = joblib.load('Text_Spam_detection_transformation')
stemming=PorterStemmer()

def sentence_processing(sentence):
    filtered_sentence = re.sub('[^a-zA-z]',' ',sentence).lower().split()
    filtered_sentence = [stemming.stem(word) for word in filtered_sentence if word not in stopwords.words('english')]
    filtered_sentence = ' '.join(filtered_sentence)
    
    vector = vectorizer.transform([filtered_sentence])
    return vector


def predict(sentence):   

    y_pred = sentence_processing(sentence)    
    y_pred = model.predict(y_pred)

    return y_pred


st.title("Spam Text Detection")

text = st.text_input("Please Enter the Message you want to check","")
st.markdown(f"input is :{text}")

prediction = predict(text)

if prediction[0] == 1:
    pred = 'Message is Spam'
    st.success(pred)


else:
    pred = 'Message is not spam'
    st.error(pred)


