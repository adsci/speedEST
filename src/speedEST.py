import streamlit as st
import webbrowser
from PIL import Image
from pathlib import Path
import utils

vr = utils.loadVRModel()
mlp = utils.loadMLPModel()
projdesc_md = utils.splitMarkdown("src/project.md")


with st.sidebar:
    st.write("# speedEST")
    selected_page = st.selectbox('Select Page ',
        ('Speed estimator', 'About the data', 'Voting Regressor model', 'Multilayer Perception model', 'About the project'),
        index=0)

    if st.button('Github Page'):
        webbrowser.open_new_tab('https://github.com/adsci/speedEST')

    if st.button('CACM website'):
        webbrowser.open_new_tab('https://wilis.pg.edu.pl/en/cacm')


if selected_page == 'About the project':
    img = Image.open('src'+projdesc_md[1][0])
    st.image(img)
    for paragraph in projdesc_md[0]:
        st.markdown(paragraph, unsafe_allow_html=True)
elif selected_page == 'Speed estimator':
    st.write("""
        ### Vehicle speed estimation at impact with a steel road barrier using Machine Learning
        """)

    mass = st.number_input('Enter the mass of the vehicle', min_value=900.0, max_value=1800.0, value=1300., step=1.0)
    angle = st.slider('Choose the impact angle [in degrees]', min_value=4, max_value=30, value=15, step=1)
    fdisp = st.number_input('Enter the final displacement of the barrier [in mm]', min_value=6.0, max_value=1400.0, value=100.0, step=10.0)
    npoles = st.slider('Choose the number of damaged guardrail poles', min_value=0, max_value=11, step=1)
    nsegments = st.slider('Choose the number of damaged segments of the road barrier', min_value=0, max_value=6, step=1)

    usr_query = [mass,angle,fdisp/1000,npoles,nsegments*4]

    clicked = st.button('Estimate vehicle speed')

    if clicked:
        speedVR = utils.predictSpeedVR(usr_query, vr)
        speedMLP = utils.predictSpeedMLP(usr_query, mlp)
        st.markdown(" ### Vehicle speed at impact was")
        st.markdown(f"&emsp; :green[{speedVR:.2f}] km/h, according to :orange[**_Voting Regressor_**] model")
        st.markdown(f"&emsp; :green[{speedMLP:.2f}] km/h, according to :orange[**_Multilayer Perceptron_**] model")