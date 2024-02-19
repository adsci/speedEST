import streamlit as st

from utils import make_sidebar, v_spacer

WIDTH = 350
st.set_page_config(page_title="speedEST - Models", layout="wide")

make_sidebar()

top_cols = st.columns([0.3, 0.3, 0.3], gap="large")

with top_cols[0]:
    st.page_link("pages/tre.py", label=":blue[__Tree Ensemble__]")
    st.image("src/img/models/tre_logo.png", width=WIDTH)

    st.page_link("pages/rle.py", label=":blue[__Regularized Linear Ensemble__]")
    st.image("src/img/models/rle_logo.png", width=WIDTH)

with top_cols[1]:
    st.page_link("pages/mlp.py", label=":blue[__Multilayer Perceptron__]")
    st.image("src/img/models/mlp_logo.png", width=WIDTH)

    st.page_link("pages/sve.py", label=":blue[__Support Vector Ensemble__]")
    st.image("src/img/models/sve_logo.png", width=WIDTH)

with top_cols[2]:
    v_spacer(12)
    st.page_link("pages/fve.py", label=":blue[__Final Voting Ensemble__]")
    st.image("src/img/models/fve_logo.png", width=WIDTH)
