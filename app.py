import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from PIL import Image



st.set_page_config(page_title="Spam Detection app")


# Displaying images using streamlit
image = Image.open('spam.jpeg')

st.image(image, width=500)

# function defining
def transform_text(text):
    #converting to lower case
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # removing the special character
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # removing stop words
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    #applying stemming
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Spam Classifier", max_chars=256)

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")        
    else:
        st.header("Not Spam")
        