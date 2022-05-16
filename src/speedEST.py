import streamlit as st

st.write("""
# speedEST
Vehicle speed estimation at impact with a steel road barrier using Machine Learning
""")

mass = st.number_input('Enter the mass of the vehicle', min_value=1100.0, max_value=2000.0)
st.write("Current mass set to ", mass)