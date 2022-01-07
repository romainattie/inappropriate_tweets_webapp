import streamlit as st

import numpy as np
import pandas as pd

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
