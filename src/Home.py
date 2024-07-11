import warnings

import streamlit as st
from streamlit_theme import st_theme

import utils
from models import fve

# hack for KeyError: 'warnings' on rerun
warnings.filterwarnings("ignore")

st.set_page_config(page_title="speedEST - Home", layout="wide")

theme_dict = st_theme()
if theme_dict:
    theme = theme_dict["base"]
else:
    theme = "light"
utils.make_sidebar(theme)

top_cols = st.columns([0.2, 0.4, 0.4], gap="large")
with top_cols[0]:
    utils.print_welcome_info()

with top_cols[1]:
    query = utils.get_query()
    clicked = st.button("Estimate vehicle speed")

    if clicked:
        col1, col2 = st.columns(2)
        base_preds, ensemble_speed = fve.ensemble_predict(query)

with top_cols[2]:
    if clicked:
        utils.print_prediction(base_preds, ensemble_speed)
