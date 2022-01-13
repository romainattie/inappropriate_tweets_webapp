from os import write
import streamlit as st

import numpy as np
import pandas as pd
import requests
from streamlit.elements.legacy_data_frame import CSSStyle
from pytwitter import Api
from PIL import Image
from st_aggrid import AgGrid
from keys import keys

api = Api(
    **keys
)

# URL_API = 'http://127.0.0.1:8000'
URL_API = 'https://tweet-image-ej5pl4pgra-ew.a.run.app'


st.set_page_config(
    page_title="Tweet harassment detection",  # => Quick reference - Streamlit
    page_icon="🤖",
    layout="wide",  # wide
    initial_sidebar_state="auto",
    menu_items={"About": 'lol'})  # collapsed

image_path = 'images/twit_2.jpeg'

CSS = """
h1 {
    color: black;
    text-shadow: 1.5px 1.5px 2px;
    font-family: 'Oxygen';
}
.stApp {
    background-image: {image_path};
    background-size: cover;
}
body {
    font-family: 'Oxygen';
}
"""
st.write('''<style>
@import url('https://fonts.googleapis.com/css2?family=Mochiy+Pop+P+One&family=Oxygen:wght@300;400;700&display=swap');
</style>''',
         unsafe_allow_html=True)

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


image_path = 'images/twit_2.jpeg'


# Title
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.write(background_image_style(image_path), unsafe_allow_html=True)

st.title('Hatred tweets detector')
st.markdown('''### Choose an option to retrieve tweet(s)''')


# Twitter API

columns = st.columns(3)

# Username input

username = columns[0].text_input('Enter an username', '''''')

def get_tweets(username, tweets_number=10):
    user_id = api.get_user(username=username).data.id
    tweets = api.get_timelines(user_id=user_id, max_results=tweets_number)
    tweets_list = []
    for tw in tweets.data:
        tweets_list.append(tw.text)
    return tweets_list


# User ID input

userid = columns[1].text_input('Enter an user ID', '''''')

def get_tweets_id(userid, tweets_number=10):
    tweets = api.get_timelines(user_id=userid, max_results=tweets_number)
    tweets_list = []
    for tw in tweets.data:
        tweets_list.append(tw.text)
    return tweets_list


# Keyword Input

keyword = columns[2].text_input('Enter a keyword', '''''')

def get_tweet_keyword(st):
    tweets = api.search_tweets(st)
    tweets_list = []
    for tw in tweets.data:
        tweets_list.append(tw.text)
    return tweets_list

center_button = st.columns(5)

# Button enter
predi = []

if center_button[2].button('Enter'):
    # print is visible in the server output, not in the page
    print('button clicked!')
    if len(username) > 0:
        tweets_list = get_tweets(username)
        st.write(tweets_list)
    elif len(userid) > 0:
        tweets_list = get_tweets_id(userid)
        st.write(tweets_list)
    elif len(keyword) > 0:
        tweets_list = get_tweet_keyword(keyword)
        st.write(tweets_list)
    else :
        tweets_list = []
        st.text('Please, fill in a field.')

    for tx in tweets_list:
        requete = URL_API + f'/predict?text={tx}'
        response = requests.get(requete).json()
        predi.append(response['Result'])


    f = {'Tweets': tweets_list, 'Prediction': predi}
    full = pd.DataFrame(f)
    df = AgGrid(full, height=500, fit_columns_on_grid_load=True)
    st.write(df)

    # for pred in predi:
    #     if predi > 0.5:
    #         st.write('Hatred : ' + predi)
    #     else:
    #         st.write('No hatred : ' + predi)

else:
    pass


columns_2 = st.columns(2)

# Tweet input

txt = columns_2[0].text_area(
    'Text to analyze', '''Enter the tweet here...''')

st.write('Length:', len(txt))
# st.text('')

# Model input

model = columns_2[1].radio('Select a model', ('TfidfVectorizer + Logistic Regression', 'Sentence Transformer + Neuron Network'))

st.write(model)

if model == 'TfidfVectorizer + Logistic Regression':
    columns_2[1].write('▶ TfidfVectorizer + Logistic Regression')
    requete = URL_API + f'/predict?text={txt}'
else :
    columns_2[1].write('▶ Sentence Transformer + Neuron Network')
    requete = URL_API + f'/predict_deep?text={txt}'


# Enter Button

if st.button('Predict'):
    # print is visible in the server output, not in the page
    response = requests.get(requete).json()
    print('button clicked!')
    st.write(response['Result'])
else:
    st.write('Click to predict!')


# # Image Background

# import base64

# @st.cache
# def load_image(path):
#     with open(path, 'rb') as f:
#         data = f.read()
#     encoded = base64.b64encode(data).decode()
#     return encoded


# st.image(
#     '/home/romain/code/romainattie/inappropriate_tweets_webapp/images/noir_3.jpg',
#     use_column_width=True, clamp=True
# )


# def background_image_style(path):
#     encoded = load_image(path)
#     style = f'''
#     <style>
#     .stApp {{
#         background-image: url("data:image/png;base64,{encoded}");
#         background-size: cover;
#     }}
#     </style>
#     '''
#     return style


# def background_image_style_2(path):
#     encoded = load_image(path)
#     style = f'''
#     <style>
#     .stApp {{
#         background-image: url("data:image/png;base64,{encoded}");
#         background-size: cover; opacity:0.2;
#     }}
#     </style>
#     '''
#     return style



# image_path = '/home/romain/code/romainattie/inappropriate_tweets_webapp/images/twit.jpeg'
# image_noir_path = '/home/romain/code/romainattie/inappropriate_tweets_webapp/images/noir_2.jpg'
# image_noir = Image.open('/home/romain/code/romainattie/inappropriate_tweets_webapp/images/noir_2.jpg')


# st.write(background_image_style(image_path), unsafe_allow_html=True)
# st.write(background_image_style_2(image_noir_path), unsafe_allow_html=True)



# st.write(background_image_style(image_noir_path), unsafe_allow_html=True, )
