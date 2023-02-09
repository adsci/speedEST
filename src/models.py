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


### Voting Regressor model
vr = utils.loadVRModel()

vr_metrics = utils.loadPandasPickle("src/models/vr_metrics.pkl")
vr_residuals = utils.loadPickle("src/models/vr_residuals.pkl")


vr_res = alt.Chart(vr_residuals['residuals']).mark_bar().encode(
        alt.X("Speed residual", bin=alt.Bin(extent=[-16,16],step=4)),
        y='count()').properties(title='Voting Regressor model')


vr_pdf = alt.Chart(vr_residuals['pdf_data']).mark_line().encode(x='Speed residual',y='Density').properties(title='Voting Regressor model')

### Multilayer Perceptron model
mlp = utils.loadMLPModel()

mlp_metrics = utils.loadPandasPickle("src/models/mlp_metrics.pkl")
mlp_perf = utils.loadPickle("src/models/mlp_perf.pkl")
mlp_residuals = utils.loadPickle("src/models/mlp_residuals.pkl")

mlp_res = alt.Chart(mlp_residuals['residuals']).mark_bar().encode(
        alt.X("Speed residual", bin=alt.Bin(extent=[-16,16],step=4)),
        y='count()').properties(title='Multilayer Perceptron model')


mlp_pdf = alt.Chart(mlp_residuals['pdf_data']).mark_line().encode(x='Speed residual',y='Density').properties(title='Multilayer Perceptron model')
mlp_loss = alt.Chart(mlp_perf['loss']).mark_line().encode(x='Epoch',y='Loss',color='Dataset').properties(title='Multilayer Perceptron model')
mlp_mae = alt.Chart(mlp_perf['mae']).mark_line().encode(x='Epoch',y='Mean Absolute Error',color='Dataset').properties(title='Multilayer Perceptron model')
