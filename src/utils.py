import re
import pickle
import pandas as pd
import torch

def loadVRModel():
    with open('src/models/votingRegressor.pkl','rb') as f:
        vr = pickle.load(f)
    return vr

def predictSpeedVR(query, vrModel, feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']):
    query = pd.DataFrame([query],columns=feats)
    return vrModel.predict(query)[0]

def loadMLPModel():
    mlp = torch.jit.load('src/models/multilayerPerceptron.pt')
    mlp.eval()
    return mlp

def predictSpeedMLP(query, mlpModel):
    query = torch.tensor(query)
    normquery = mlpModel.normalizeFeatures(query)
    with torch.inference_mode():
        pred = mlpModel(normquery)

    return mlpModel.recoverTargets(pred).item()

def splitMarkdown(path):
    imgregex = re.compile(r"!\[Split\]\(.(.+)\)")
    with open(path,'r') as f:
        lines = f.read()

    imgpaths = re.findall(imgregex,lines)
    parts = lines.split("![Split](.")

    mdparts = []
    for part in parts:
        for imgpath in imgpaths:
            part = part.replace(imgpath+')',"")
        mdparts.append(part)

    return mdparts, imgpaths
