import streamlit as st

import utils
from models import fve

st.set_page_config(page_title="speedEST - Home", layout="wide")

utils.make_sidebar()

top_cols = st.columns([0.2, 0.4, 0.2, 0.2], gap="large")
with top_cols[0]:
    utils.print_welcome_info()

with top_cols[1]:
    query = utils.get_query()
    clicked = st.button("Estimate vehicle speed")

    if clicked:
        col1, col2 = st.columns(2)
        base_preds, ensemble_speed = fve.ensemble_predict(query)
        col1.subheader("Vehicle speed at impact was")
        col1.markdown(
            f"&emsp; :green[{ensemble_speed:.2f}] km/h (:orange[**_Final Voting Ensemble_**])"
        )
        col2.dataframe(base_preds.style.format("{:.2f}"))

with top_cols[3]:
    utils.print_additional_info()
