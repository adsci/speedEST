import pandas as pd
import streamlit as st
from typing import Tuple
import math

LOGO_SIDEBAR_DARK = "src/img/logo/logo_small_dark.png"
LOGO_SIDEBAR_LIGHT = "src/img/logo/logo_small_light.png"
LOGO_DARK = "src/img/logo/logo_large_dark.png"
LOGO_LIGHT = "src/img/logo/logo_large_light.png"
LOGO_ICON_LIGHT = "src/img/logo/logo_icon_light.png"
LOGO_ICON_DARK = "src/img/logo/logo_icon_dark.png"

def v_spacer(height) -> None:
    for _ in range(height):
        st.write("\n")


def read_version(path="VERSION"):
    with open(path, "r") as f:
        return f.read().strip()


def get_segment_bounds(n_poles: int) -> Tuple[int, int]:
    lower = math.ceil((n_poles-3)/2)
    upper = math.floor((n_poles+3)/2)

    return max(0, lower),  upper


def make_sidebar(theme="light"):
    with st.sidebar:
        if theme == "light":
            st.logo(LOGO_SIDEBAR_LIGHT, icon_image=LOGO_ICON_LIGHT)
        elif theme == "dark":
            st.logo(LOGO_SIDEBAR_DARK, icon_image=LOGO_ICON_DARK)
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
    """
    Reads user input and returns the corresponding model query
    """
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
        step=50.0,
    )
    nP = st.slider(
        "Choose the number of damaged guardrail posts",
        min_value=0,
        max_value=11,
        step=1,
        value=2,
    )
    nSeg_lower, nSeg_upper = get_segment_bounds(nP)
    nSeg = st.slider(
        "Choose the number of damaged segments of the W-beam guardrails",
        min_value=nSeg_lower,
        max_value=nSeg_upper,
        step=1,
        value=int((nSeg_lower + nSeg_upper)/2),
    )

    return preprocess_raw_input(vMass, iAng, fDisp, nP, nSeg)


def preprocess_raw_input(v_mass: float, i_ang: int, f_disp: float, n_posts: int, n_seg: int) -> pd.DataFrame:
    """
    Transforms user input into pandas DataFrame format, which is required by the models.

    :param v_mass: Vehicle mass [kg]
    :param i_ang: Impact angle [degrees]
    :param f_disp: Final laterral displacement of the guardrail [mm]
    :param n_posts: Number of damaged guardrail posts
    :param n_seg: Number of dagamed guardrail segments

    :return query: Query dataframe accepted by the models
    """
    feats = ["vehicleMass", "impactAngle", "finalDisp", "nPoles", "nSegments"]
    query = pd.DataFrame([map(float, [v_mass, i_ang, f_disp, n_posts, n_seg])], columns=feats)
    return query


def print_prediction(base_predictions, ensemble_speed):
    st.subheader("Model prediction is")
    st.subheader(f"&emsp; :green[{ensemble_speed:.2f}] km/h", anchor=False)
    v_spacer(2)
    st.subheader("Base estimators", anchor=False)
    st.dataframe(base_predictions)
