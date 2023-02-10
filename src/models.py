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


model_names = ['Voting Regressor', 'Multilayer Perceptron']

### Voting Regressor model
vr = utils.loadVRModel()

vr_metrics_all = utils.loadPandasPickle("src/models/vr_metrics.pkl")
vr_residuals = utils.loadPickle("src/models/vr_residuals.pkl")


vr_res = alt.Chart(vr_residuals['residuals']).mark_bar().encode(
        alt.X("Speed residual", bin=alt.Bin(extent=[-16,16],step=2)),
        y='count()').properties(title=model_names[0])


vr_pdf = alt.Chart(vr_residuals['pdf_data']).mark_line().encode(
        x='Speed residual',y='Density').properties(title=model_names[0])

### Multilayer Perceptron model
mlp = utils.loadMLPModel()

mlp_metrics = utils.loadPandasPickle("src/models/mlp_metrics.pkl")
mlp_perf = utils.loadPickle("src/models/mlp_perf.pkl")
mlp_residuals = utils.loadPickle("src/models/mlp_residuals.pkl")

mlp_loss = alt.Chart(mlp_perf['loss']).mark_line().encode(x='Epoch',y='Loss',
                    color=alt.Color("Dataset", scale=alt.Scale(domain=['Training','Validation'],range=['#4d96d9','#ff6b6b']))).properties(title=model_names[1])
mlp_mae = alt.Chart(mlp_perf['mae']).mark_line().encode(x='Epoch',y='Mean Absolute Error',
                    color=alt.Color("Dataset", scale=alt.Scale(domain=['Training','Validation'],range=['#4d96d9','#ff6b6b']))).properties(title=model_names[1])

mlp_res = alt.Chart(mlp_residuals['residuals']).mark_bar().encode(
                    alt.X("Speed residual", bin=alt.Bin(extent=[-16,16],step=2)),
                    y='count()').properties(title=model_names[1])


mlp_pdf = alt.Chart(mlp_residuals['pdf_data']).mark_line().encode(x='Speed residual',y='Density').properties(title=model_names[1])

### Performance summary - metrics
cols = ['Train MAE', 'Val MAE', 'Test MAE', 'Test R2']
metrics = pd.DataFrame(columns=cols)

vr_metrics=metrics.copy()
vr_metrics.loc['Voting Regressor'] = vr_metrics_all.T.loc['VotingRegressor']

metrics_df = [vr_metrics, mlp_metrics]
for i, df in enumerate(metrics_df):
    metrics = pd.concat(metrics_df)


### Performance summary - residual distribution
residuals_df = [vr_residuals['residuals'], mlp_residuals['residuals']]
for i, df in enumerate(residuals_df):
    residuals_df[i]['Model'] = model_names[i]

pdf_df = [vr_residuals['pdf_data'], mlp_residuals['pdf_data']]
for i, df in enumerate(pdf_df):
    pdf_df[i]['Model'] = model_names[i]

residuals = pd.concat(residuals_df, axis=0)
pdfs = pd.concat(pdf_df, axis=0)

res_summary = alt.Chart(residuals).mark_bar(opacity=0.5).encode(
                alt.X("Speed residual", bin=alt.Bin(extent=[-16,16],step=2)),
                alt.Y('count()', stack=None), 
                alt.Color("Model", scale=alt.Scale(domain=model_names, range=['#4d96d9','#ff6b6b'])))

pdf_summary = alt.Chart(pdfs).mark_line().encode(x='Speed residual',y='Density',
                color=alt.Color("Model", scale=alt.Scale(domain=model_names, range=['#4d96d9','#ff6b6b']))).properties(title=model_names[1])

