import streamlit as st
from pathlib import Path
import text
import models

def v_spacer(height) -> None:
    for _ in range(height):
        st.write('\n')

def link_button(name,url):
    html = f'''
        <a target="_self" href={url}>
            <button style="background-color: #969696; border: None; text-font: monospace">
                {name}
            </button>
        </a>
    '''
    return html

with st.sidebar:
    st.write("# speedEST")
    selected_page = st.selectbox('Select Page ',
        ('Speed estimator', 'Data', 'Tree Ensemble', 'Multilayer Perception', 'Final Voting Ensemble', 'About the project'),
        index=0)

    v_spacer(1)
    st.write(link_button('Source','https://github.com/adsci/speedEST'),unsafe_allow_html=True)
    v_spacer(1)
    st.write(link_button('CACM website','https://wilis.pg.edu.pl/en/cacm'),unsafe_allow_html=True)
    v_spacer(1)
    st.write(link_button('KWM website','https://wilis.pg.edu.pl/en/department-mechanics-materials-and-structures'),unsafe_allow_html=True)

    v_spacer(20)
    st.text("Copyright (c) 2023 \n Gdańsk University of Technology")
    st.text("Data acquisition by\n Dawid Bruski\n Łukasz Pachocki")
    st.text("Models and app by\n Adam Sciegaj")
    

if selected_page == 'About the project':
    st.image('src' + text.projdesc_md[1][0])
    for paragraph in text.projdesc_md[0]:
        st.markdown(paragraph, unsafe_allow_html=True)
elif selected_page == 'Data':
    st.markdown(text.datadesc_md[0][0], unsafe_allow_html=True)
    st.image('src' + text.datadesc_md[1][0])
    st.markdown(text.datadesc_md[0][1], unsafe_allow_html=True)
    st.image('src' + text.datadesc_md[1][1])
    st.markdown(text.datadesc_md[0][2], unsafe_allow_html=True)
    st.image('src' + text.datadesc_md[1][2])
    st.markdown(text.datadesc_md[0][3], unsafe_allow_html=True)
    st.image('src' + text.datadesc_md[1][3])
    st.markdown(text.datadesc_md[0][4], unsafe_allow_html=True)
    st.image('src' + text.datadesc_md[1][4])
    st.markdown(text.datadesc_md[0][5], unsafe_allow_html=True)
elif selected_page == 'Tree Ensemble':
    st.markdown(text.tredesc_md[0][0], unsafe_allow_html=True)
    st.image('src' + text.tredesc_md[1][0])
    st.markdown(text.tredesc_md[0][1], unsafe_allow_html=True)
    st.dataframe(models.tre.getMetrics().style.format("{:.3f}"))
    st.markdown(text.tredesc_md[0][2], unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Histogram", "Density"])
    with tab1:
        st.altair_chart(models.tre.getResidualHist(), use_container_width=True)
    with tab2:
        st.altair_chart(models.tre.getResidualPDF(), use_container_width=True)
    st.markdown(text.tredesc_md[0][3], unsafe_allow_html=True)
elif selected_page == 'Multilayer Perception':
    st.markdown(text.mlpdesc_md[0][0], unsafe_allow_html=True)
    st.image('src' + text.mlpdesc_md[1][0])
    st.markdown(text.mlpdesc_md[0][1], unsafe_allow_html=True)
    st.image('src' + text.mlpdesc_md[1][1])
    st.markdown(text.mlpdesc_md[0][2], unsafe_allow_html=True)
    tab1, tab2 = st.tabs(['Loss', 'Mean Absolute Error'])
    with tab1:
        st.altair_chart(models.mlp.getLoss(), use_container_width=True)
    with tab2:
        st.altair_chart(models.mlp.getMAE(), use_container_width=True)
    st.markdown(text.mlpdesc_md[0][3], unsafe_allow_html=True)
    st.dataframe(models.mlp.getMetricsFull().style.format("{:.3f}"))
    st.markdown(text.mlpdesc_md[0][4], unsafe_allow_html=True)
    tab3, tab4 = st.tabs(["Histogram", "Density"])
    with tab3:
        st.altair_chart(models.mlp.getResidualHist(), use_container_width=True)
    with tab4:
        st.altair_chart(models.mlp.getResidualPDF(), use_container_width=True)
    st.markdown(text.mlpdesc_md[0][5], unsafe_allow_html=True)
elif selected_page == 'Final Voting Ensemble':
    st.markdown(text.fvedesc_md[0][0], unsafe_allow_html=True)
    st.dataframe(models.fve.getBaseEstCV().style.format("{:.3f}"))
    st.markdown(text.fvedesc_md[0][1], unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Histogram","Density"])
    with tab1:
        st.altair_chart(models.fve.getResidualHist(), use_container_width=True)
    with tab2:
        st.altair_chart(models.fve.getResidualPDF(), use_container_width=True)
    st.markdown(text.fvedesc_md[0][2], unsafe_allow_html=True)
    st.dataframe(models.fve.getMetrics().style.format("{:.3f}"))
elif selected_page == 'Speed estimator':
    st.write("""
        ### Vehicle speed estimation at impact with a steel road barrier using Machine Learning
        """)

    mass = st.number_input('Enter the mass of the vehicle, including the mass of occupants, [kg]', min_value=900.0, max_value=1800.0, value=1300., step=1.0)
    angle = st.slider('Choose the impact angle [degrees]', min_value=4, max_value=30, value=15, step=1)
    fdisp = st.number_input('Enter the final lateral displacement of the barrier (static working width) [mm]', min_value=6.0, max_value=1400.0, value=100.0, step=10.0)
    npoles = st.slider('Choose the number of damaged guardrail posts', min_value=0, max_value=11, step=1, value=2)
    nsegments = st.slider('Choose the number of damaged segments of the W-beam guardrails', min_value=0, max_value=6, step=1, value = 1)

    usr_query = [mass,angle,fdisp/1000,npoles,nsegments*4]

    clicked = st.button('Estimate vehicle speed')

    if clicked:
        col1, col2 = st.columns(2)
        basepreds, speedFVE = models.fve.predict(usr_query)
        col1.markdown(" ### Vehicle speed at impact was")
        col1.markdown(f"&emsp; :green[{speedFVE:.2f}] km/h (:orange[**_Final Voting Ensemble_**])")
        col2.dataframe(basepreds.style.format("{:.2f}"))

