import streamlit as st
import pickle
import numpy as np
import sklearn

st.title("House Price Predicition")
df = pickle.load(open('data.pkl', 'rb'))
model = pickle.load(open('hpp_model.pkl', 'rb'))
loc = pickle.load(open('locations.pkl', 'rb'))
locations = st.selectbox("ENTER LOCATION ::", loc)

bhk = st.text_input("ENTER BHK ::")
bath = st.text_input("ENTER BATHROOM ::")
total_sqft = st.text_input("ENTER TOTAL SQFT ::")

def predict_price(location, sqft, baths, bhks):
    loc_index = np.where(df.columns == location)[0][0]
    x = np.zeros(len(df.columns))
    x[0] = sqft
    x[1] = baths
    x[2] = bhks
    if loc_index >= 0:
        x[loc_index] = 1
    return round(model.predict([x])[0], 2)

if st.button('PREDICT'):
    pred = predict_price(locations, total_sqft, bath, bhk)
    st.header("Rs. {} Lakhs".format(abs(pred)))

