import pandas as pd
import streamlit as st


def v_spacer(height) -> None:
    for _ in range(height):
        st.write("\n")


def read_version(path="VERSION"):
    with open(path, "r") as f:
        return f.read().strip()


def make_sidebar(theme="light"):
    with st.sidebar:
        if theme == "light":
            st.image("src/img/logo_small_light.png")
        elif theme == "dark":
            st.image("src/img/logo_small_dark.png")
        v_spacer(1)
        st.page_link("Home.py", label="Home")
        st.page_link("pages/data.py", label=":blue[__Data__]")
        st.page_link("pages/ml_models.py", label=":blue[__Models__]")
        st.page_link("pages/tre.py", label=":orange[\1 Tree Ensemble]")
        st.page_link("pages/mlp.py", label=":orange[\2 Multilayer Perceptron]")
        st.page_link("pages/rle.py", label=":orange[\3 Regularized Linear Ensemble]")
        st.page_link("pages/sve.py", label=":orange[\4 Support Vector Ensemble]")
        st.page_link("pages/fve.py", label=":orange[\5 Final Voting Ensemble]")
        st.page_link("pages/about_project.py", label=":blue[__About the project__]")
        v_spacer(5)
        st.text(f"speedEST v{read_version()}")
        st.text("Copyright (c) 2022-2024 \n Gdańsk University of Technology")
        st.text("Models and dashboard by\n Adam Ścięgaj")
        st.text("Data acquisition by\n Dawid Bruski\n Łukasz Pachocki")


def print_welcome_info():
    st.write(
        """
        ### Welcome to speedEST!
        """
    )
    st.success(
        """
        __speedEST__ is an online app showcasing a data-driven approach
        to estimating vehicle speed at impact with a steel road barrier.
        The dashboard comprises a collection of machine learning models ready
        for prediction. 
        """
    )
    v_spacer(2)
    st.info(
        """
        To get a prediction, input the required parameters on the right 
        and click on the "Estimate vehicle speed" button.
        """
    )
    v_spacer(2)
    st.warning(
        """
        To get more information about the data, machine learning models and the project, 
        choose a page from the side menu.
        """
    )


def print_additional_info():
    st.write("### Resources")
    st.link_button("Source code", "https://github.com/adsci/speedEST")
    st.link_button("Publication", "https://doi.org/10.1016/j.advengsoft.2023.103502")
    v_spacer(2)
    st.write("### Miscellaneous")
    st.link_button("CACM", "https://wilis.pg.edu.pl/en/cacm")
    st.link_button(
        "KWM",
        "https://wilis.pg.edu.pl/en/department-mechanics-materials-and-structures",
    )


def get_query() -> pd.DataFrame:
    st.subheader("Define impact parameters")
    vMass = st.slider(
        "Enter the mass of the vehicle, including the mass of occupants [kg]",
        min_value=900.0,
        max_value=1800.0,
        value=1300.0,
        step=50.0,
    )
    iAng = st.slider(
        "Choose the impact angle [degrees]", min_value=4, max_value=30, value=15, step=1
    )
    fDisp = st.number_input(
        "Enter the final lateral displacement of the barrier (static working width) [mm]",
        min_value=6.0,
        max_value=1400.0,
        value=100.0,
        step=10.0,
    )
    nP = st.slider(
        "Choose the number of damaged guardrail posts",
        min_value=0,
        max_value=11,
        step=1,
        value=2,
    )
    nSeg = st.slider(
        "Choose the number of damaged segments of the W-beam guardrails",
        min_value=0,
        max_value=6,
        step=1,
        value=1,
    )

    feats = ["vehicleMass", "impactAngle", "finalDisp", "nPoles", "nSegments"]
    query = pd.DataFrame([map(float, [vMass, iAng, fDisp, nP, nSeg])], columns=feats)

    return query


def print_prediction(base_predictions, ensemble_speed):
    st.subheader("Model prediction is")
    st.subheader(f"&emsp; :green[{ensemble_speed:.2f}] km/h", anchor=False)
    v_spacer(2)
    st.subheader("Base estimators", anchor=False)
    st.dataframe(base_predictions)
