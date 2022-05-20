import streamlit as st
import pickle
import pandas as pd

with open('src/models/votingRegressor.pkl','rb') as f:
    vr = pickle.load(f)

def predictSpeed(vr,user_query):
    return vr.predict(query)[0]
    

st.write("""
# speedEST
Vehicle speed estimation at impact with a steel road barrier using Machine Learning
""")

mass = st.number_input('Enter the mass of the vehicle', min_value=900.0, max_value=1800.0, step=1.0)
angle = st.slider('Choose the impact angle [in degrees]', min_value=0, max_value=30, value=15, step=1)
fdisp = st.number_input('Enter the final displacement of the barrier [in mm]', min_value=0.0, max_value=1400.0,step=1.0)
npoles = st.slider('Choose the number of damaged guardrail poles', min_value=0, max_value=11, step=1)
nsegments = st.slider('Choose the number of damaged segments of the road barrier', min_value=0, max_value=24, step=1)

feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']
query = pd.DataFrame([[mass,angle,fdisp/1000,npoles,nsegments]],columns=feats)

clicked = st.button('Estimate vehicle speed')


if clicked:
    speed = predictSpeed(vr,query)
    st.write('Estimated vehicle speed is ', speed, ' km/h' )
