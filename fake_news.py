import streamlit as st
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd


with open(r"D:\final project\fake_news\ETC_model.pkl", "rb") as f:
    model_class = pickle.load(f)


with open(r'D:\final project\fake_news\vector.pkl', "rb") as j:
        vector_form = pickle.load(j)

def wordopt(text):
    text = text.lower()  
    # remove urls:
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    # remove html tags:
    text = re.sub(r'<.*?>', '', text)
    # remove punctation:
    text = re.sub(r'[^\w\s]', "", text)
    # remove digits:
    text = re.sub(r'\d', "", text)
    # remove newline characters:
    text = re.sub(r'\n', " ", text)
    return text

    
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pD

def fake_news(new):
    news=wordopt(new)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = model_class.predict(vector_form1)
    return prediction



st.title('Fake News Classification app ')
st.subheader("Input the News content below")
sentence = st.text_area("Enter your news content here", "",height=200)
predict_btt = st.button("predict")
if predict_btt:
    prediction_class=fake_news(sentence)
    print(prediction_class)
    
    if prediction_class == [0]:
        st.success('Reliable')
    if prediction_class == [1]:
        st.warning('Unreliable')