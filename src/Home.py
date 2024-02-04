import streamlit as st
import utils
import pandas as pd
import models

st.set_page_config(
    page_title="speedEST",
    layout="wide"
)

utils.make_sidebar()

top_cols = st.columns([0.2, 0.4, 0.2, 0.2], gap="large")
with top_cols[0]:
    st.write("""
                ### Welcome to speedEST!
                """)

    st.success("""
            __speedEST__ is an online app showcasing a data-driven approach
            to estimating vehicle speed at impact with a steel road barrier.
            
            speedEST comprises a collection of machine learning models ready
            for prediction. 
            
            """)

    utils.v_spacer(2)
    st.info("""
            To get a prediction, input the required parameters on the right 
            and click on the "Estimate vehicle speed" button.
            """)

    utils.v_spacer(2)
    st.warning("""
                To get more information about the data, machine learning models and the project, 
                choose a page from the side menu.
                """)

with top_cols[1]:
    vMass = st.number_input('Enter the mass of the vehicle, including the mass of occupants [kg]', min_value=900.0,
                            max_value=1800.0, value=1300., step=1.0)
    iAng = st.slider('Choose the impact angle [degrees]', min_value=4, max_value=30, value=15, step=1)
    fDisp = st.number_input('Enter the final lateral displacement of the barrier (static working width) [mm]',
                            min_value=6.0, max_value=1400.0, value=100.0, step=10.0)
    nP = st.slider('Choose the number of damaged guardrail posts', min_value=0, max_value=11, step=1, value=2)
    dSeg = st.slider('Choose the number of damaged segments of the W-beam guardrails', min_value=0, max_value=6, step=1,
                     value=1)

    feats = ['vehicleMass', 'impactAngle', 'finalDisp', 'nPoles', 'nSegments']
    query = pd.DataFrame([map(float, [vMass, iAng, fDisp, nP, dSeg])], columns=feats)

    clicked = st.button('Estimate vehicle speed')

    if clicked:
        col1, col2 = st.columns(2)
        base_preds, ensemble_speed = models.fve.ensemble_predict(query)
        col1.subheader("Vehicle speed at impact was")
        col1.markdown(f"&emsp; :green[{ensemble_speed:.2f}] km/h (:orange[**_Final Voting Ensemble_**])")
        col2.dataframe(base_preds.style.format("{:.2f}"))

with top_cols[3]:
    st.write("### Resources")
    st.link_button("Source code", "https://github.com/adsci/speedEST")
    st.link_button("Publication", "https://doi.org/10.1016/j.advengsoft.2023.103502")

    st.write("### Miscellaneous")
    st.link_button("CACM", "https://wilis.pg.edu.pl/en/cacm")
    st.link_button("KWM", "https://wilis.pg.edu.pl/en/department-mechanics-materials-and-structures")
