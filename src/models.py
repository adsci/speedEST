import pandas as pd
import torch
import altair as alt
import utils

def predictSpeedVR(query, vrModel, feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']):
    query = pd.DataFrame([query],columns=feats)
    return vrModel.predict(query)[0]

def predictSpeedMLP(query, mlpModel):
    query = torch.tensor(query)
    normquery = mlpModel.normalizeFeatures(query)
    with torch.inference_mode():
        pred = mlpModel(normquery)

    return mlpModel.recoverTargets(pred).item()


vr = utils.loadVRModel()
mlp = utils.loadMLPModel()

vr_metrics = utils.loadPandasPickle("src/models/vr_metrics.pkl")
vr_residuals = utils.loadPickle("src/models/vr_residuals.pkl")


vr_res = alt.Chart(vr_residuals['residuals']).mark_bar().encode(
        alt.X("Speed residual", bin=alt.Bin(extent=[-16,16],step=4)),
        y='count()').properties(title='Voting Regressor model')


vr_pdf = alt.Chart(vr_residuals['pdf_data']).mark_line().encode(x='Speed residual',y='Density').properties(title='Voting Regressor model')