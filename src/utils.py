import re
import pickle

def loadVRModel():
    with open('src/models/votingRegressor.pkl','rb') as f:
        vr = pickle.load(f)
    return vr

def predictSpeedVR(vr,query):
    return vr.predict(query)[0]

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
