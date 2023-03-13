import re
import pickle
import pandas as pd
import numpy as np
import torch

def loadPickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def loadPandasPickle(path):
    return pd.read_pickle(path)


def loadRLEModel():
    return loadPickle('src/models/regularizedLinearEnsemble.pkl')

def loadSVEModel():
    return loadPickle('src/models/supportVectorEnsemble.pkl')

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