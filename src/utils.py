import re
import pickle
import pandas as pd
import torch

def loadPickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def loadPandasPickle(path):
    return pd.read_pickle(path)

def loadVRModel():
    return loadPickle('src/models/votingRegressor.pkl')

def loadMLPModel():
    mlp = torch.jit.load('src/models/multilayerPerceptron.pt')
    mlp.eval()
    return mlp

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
