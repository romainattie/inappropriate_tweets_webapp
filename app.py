import streamlit as st

import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Tweet harassement detection",  # => Quick reference - Streamlit
    page_icon="🤖",
    layout="centered",  # wide
    initial_sidebar_state="auto",
    menu_items={"About": 'lol'})  # collapsed


st.markdown("""
            # Harassement tweets detector
            ##       --O-O--
            This is text""")

# Tweet input

txt = st.text_area(
    'Text to analyze', '''
    Enter the tweet here...
    ''')

st.write('Length:', len(txt))
st.text('')

# Model input

@st.cache
def get_select_box_data():

    return pd.DataFrame({
        'Pipeline': ['LogReg', 'XGBoost', 'Bernoulli']
    })


df = get_select_box_data()

option = st.selectbox('Select a model', df['Pipeline'])
st.write('The model chosen is', option)
st.text('')

# Enter Button

if st.button('Predict'):
    # print is visible in the server output, not in the page
    print('button clicked!')
    st.write("Let's predict!")
else:
    st.write('Click to predict!')


# Image Background

import base64

@st.cache
def load_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return encoded


def background_image_style(path):
    encoded = load_image(path)
    style = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    '''
    return style


image_path = '/home/romain/code/romainattie/inappropriate_tweets_webapp/images/twit.jpeg'

st.write(background_image_style(image_path), unsafe_allow_html=True)
