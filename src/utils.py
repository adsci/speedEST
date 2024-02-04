import streamlit as st
import re
import dill
import pickle
import pandas as pd
import numpy as np
import base64
from pathlib import Path


def v_spacer(height) -> None:
    for _ in range(height):
        st.write('\n')


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, width):
    img_html = "<img src='data:image/png;base64,{}' width='{}' class='img-fluid'>".format(img_to_bytes(img_path), width)
    return img_html


def load_pickle(path, format='pickle'):
    with open(path, 'rb') as f:
        if format == 'dill':
            return dill.load(f)
        return pickle.load(f)


def split_markdown(path):
    imgregex = re.compile(r"!\[Split\]\(..(.+)\)")
    with open(path, 'r') as f:
        lines = f.read()

    imgpaths = re.findall(imgregex, lines)
    parts = lines.split("![Split")

    mdparts = []
    for part in parts:
        for imgpath in imgpaths:
            part = part.replace('](..' + imgpath + ')\n', "")
        part = part.replace('Here]()\n\n', "")
        mdparts.append(part)

    return mdparts, imgpaths


def load_and_flatten_residuals(path):
    resdata = load_pickle(path)
    encoding = {'residuals': 'Speed residual'}
    for k, v in resdata.items():
        if not isinstance(v, pd.DataFrame):
            res_flattened = np.array(resdata['residuals']).flatten()
            resdata[k] = pd.DataFrame(res_flattened, columns=[encoding[k]])
    return resdata


def read_version(path='VERSION'):
    with open(path, 'r') as f:
        return f.read().strip()

def make_sidebar():
    with st.sidebar:
        st.image("src/img/logo.png")

        v_spacer(3)
        st.page_link("Home.py", label="Home")
        st.page_link("pages/speedEST.py", label=":blue[__Data__]")
        st.page_link("pages/speedEST.py", label=":blue[__Models__]")
        st.page_link("pages/speedEST.py", label=":orange[\1 Tree Ensemble]")
        st.page_link("pages/speedEST.py", label=":orange[\2 Multilayer Perceptron]")
        st.page_link("pages/speedEST.py", label=":orange[\3 Regularized Linear Ensemble]")
        st.page_link("pages/speedEST.py", label=":orange[\4 Support Vector Ensemble]")
        st.page_link("pages/speedEST.py", label=":orange[\5 Final Voting Ensemble]")
        st.page_link("pages/speedEST.py", label=":blue[__About the project__]")

        v_spacer(15)
        st.text(f"speedEST v{read_version()}")
        st.text("Copyright (c) 2022-2024 \n Gdańsk University of Technology")
        st.text("Data acquisition by\n Dawid Bruski\n Łukasz Pachocki")
        st.text("Models and dashboard by\n Adam Ścięgaj")
