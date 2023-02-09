import streamlit as st
import webbrowser
from PIL import Image
from pathlib import Path
import text
import models

def v_spacer(height) -> None:
    for _ in range(height):
        st.write('\n')

with st.sidebar:
    st.write("# speedEST")
    selected_page = st.selectbox('Select Page ',
        ('Speed estimator', 'Data', 'Voting Regressor model', 'Multilayer Perception model', 'About the project'),
        index=0)

    if st.button('Source'):
        webbrowser.open_new_tab('https://github.com/adsci/speedEST')

    if st.button('CACM website'):
        webbrowser.open_new_tab('https://wilis.pg.edu.pl/en/cacm')
    
    if st.button('KWM website'):
        webbrowser.open_new_tab('https://wilis.pg.edu.pl/en/department-mechanics-materials-and-structures')


    v_spacer(20)
    st.text("Copyright (c) 2023 \n Gdańsk University of Technology")
    st.text("Data acquisition by\n Dawid Bruski\n Łukasz Pachocki")
    st.text("Models and app built by\n Adam Sciegaj")
    

if selected_page == 'About the project':
    st.image('src' + text.projdesc_md[1][0])
    for paragraph in text.projdesc_md[0]:
        st.markdown(paragraph, unsafe_allow_html=True)
elif selected_page == 'Voting Regressor model':
    st.markdown(text.vrdesc_md[0][0], unsafe_allow_html=True)
    st.image('src' + text.vrdesc_md[1][0])
    st.markdown(text.vrdesc_md[0][1], unsafe_allow_html=True)
    st.table(models.vr_metrics.style.format("{:.2f}"))
    st.markdown(text.vrdesc_md[0][2], unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Histogram", "Density"])
    with tab1:
        st.altair_chart(models.vr_res, use_container_width=True)
    with tab2:
        st.altair_chart(models.vr_pdf, use_container_width=True)
    st.markdown(text.vrdesc_md[0][3], unsafe_allow_html=True)
elif selected_page == 'Multilayer Perception model':
    st.markdown(text.mlpdesc_md[0][0], unsafe_allow_html=True)
    st.image('src' + text.mlpdesc_md[1][0])
    st.markdown(text.mlpdesc_md[0][1], unsafe_allow_html=True)
    st.image('src' + text.mlpdesc_md[1][1])
    st.markdown(text.mlpdesc_md[0][2], unsafe_allow_html=True)
    tab1, tab2 = st.tabs(['Loss', 'Mean Absolute Error'])
    with tab1:
        st.altair_chart(models.mlp_loss, use_container_width=True)
    with tab2:
        st.altair_chart(models.mlp_mae, use_container_width=True)
    st.markdown(text.mlpdesc_md[0][3], unsafe_allow_html=True)
    st.table(models.mlp_metrics.style.format("{:.2f}"))
    st.markdown(text.mlpdesc_md[0][4], unsafe_allow_html=True)
    tab3, tab4 = st.tabs(["Histogram", "Density"])
    with tab3:
        st.altair_chart(models.mlp_res, use_container_width=True)
    with tab4:
        st.altair_chart(models.mlp_pdf, use_container_width=True)
    st.markdown(text.mlpdesc_md[0][5], unsafe_allow_html=True)
elif selected_page == 'Speed estimator':
    st.write("""
        ### Vehicle speed estimation at impact with a steel road barrier using Machine Learning
        """)

    mass = st.number_input('Enter the mass of the vehicle', min_value=900.0, max_value=1800.0, value=1300., step=1.0)
    angle = st.slider('Choose the impact angle [in degrees]', min_value=4, max_value=30, value=15, step=1)
    fdisp = st.number_input('Enter the final displacement of the barrier [in mm]', min_value=6.0, max_value=1400.0, value=100.0, step=10.0)
    npoles = st.slider('Choose the number of damaged guardrail poles', min_value=0, max_value=11, step=1, value=2)
    nsegments = st.slider('Choose the number of damaged segments of the road barrier', min_value=0, max_value=6, step=1, value = 1)

    usr_query = [mass,angle,fdisp/1000,npoles,nsegments*4]

    clicked = st.button('Estimate vehicle speed')

    if clicked:
        speedVR = models.predictSpeedVR(usr_query, models.vr)
        speedMLP = models.predictSpeedMLP(usr_query, models.mlp)
        st.markdown(" ### Vehicle speed at impact was")
        st.markdown(f"&emsp; :green[{speedVR:.2f}] km/h, according to :orange[**_Voting Regressor_**] model")
        st.markdown(f"&emsp; :green[{speedMLP:.2f}] km/h, according to :orange[**_Multilayer Perceptron_**] model")