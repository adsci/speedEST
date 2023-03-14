import re
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path,width):
    img_html = "<img src='data:image/png;base64,{}' width='{}' class='img-fluid'>".format(img_to_bytes(img_path),width)
    return img_html

def link_button(name,url):
    html = f'''
        <a target="_self" href={url}>
            <button style="background-color: #969696; border: None; text-font: monospace">
                {name}
            </button>
        </a>
    '''
    return html

def loadPickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def loadPandasPickle(path):
    return pd.read_pickle(path)

def splitMarkdown(path):
    imgregex = re.compile(r"!\[Split\]\(..(.+)\)")
    with open(path,'r') as f:
        lines = f.read()

    imgpaths = re.findall(imgregex,lines)
    parts = lines.split("![Split")

    mdparts = []
    for part in parts:
        for imgpath in imgpaths:
            part = part.replace('](..' + imgpath+')\n',"")
        part = part.replace('Here]()\n\n',"")
        mdparts.append(part)

    return mdparts, imgpaths

def loadAndFlattenResiduals(path):
    resdata = loadPickle(path)
    encoding = {'residuals': 'Speed residual'}
    for k, v in resdata.items():
        if not isinstance(v, pd.DataFrame):
            res_flattened = np.array(resdata['residuals']).flatten()
            resdata[k] = pd.DataFrame(res_flattened, columns=[encoding[k]])
    return resdata