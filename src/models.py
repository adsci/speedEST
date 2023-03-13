import pandas as pd
import torch
import altair as alt
import utils

def predictSpeedTRE(query, treModel, feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']):
    query = pd.DataFrame([query],columns=feats)
    return treModel.predict(query)[0]

def predictSpeedMLP(query, mlpModel):
    query = torch.tensor(query)
    normquery = mlpModel.normalizeFeatures(query)
    with torch.inference_mode():
        pred = mlpModel(normquery)

    return mlpModel.recoverTargets(pred).item()


model_names = ['Tree Ensemble', 'Multilayer Perceptron']

### Tree Ensemble model
tre = utils.loadTREModel()

tre_metrics = utils.loadPandasPickle("src/models/tre_metrics.pkl")
tre_residuals = utils.loadAndFlattenResiduals("src/models/tre_residuals.pkl")


tre_res = alt.Chart(tre_residuals['residuals']).mark_bar().encode(
        alt.X("Speed residual", bin=alt.Bin(extent=[-32,32],step=4)),
        y='count()').properties(title=model_names[0])


tre_pdf = alt.Chart(tre_residuals['pdf_data']).mark_line().encode(
        x='Speed residual',y='Density').properties(title=model_names[0])

### Multilayer Perceptron model
mlp = utils.loadMLPModel()

mlp_metrics = utils.loadPandasPickle("src/models/mlp_metrics.pkl")
mlp_metrics_renamed = mlp_metrics.rename(columns={"Multilayer Perceptron": "MLP"})
mlp_training = utils.loadPickle("src/models/mlp_training.pkl")
mlp_residuals = utils.loadAndFlattenResiduals("src/models/mlp_residuals.pkl")

mlp_loss = alt.Chart(mlp_training['loss']).mark_line().encode(x='Epoch',y='Loss',
                    color=alt.Color("Dataset", scale=alt.Scale(domain=['Training (CV mean)','Validation (CV mean)'],range=['#4d96d9','#ff6b6b']))).properties(title=model_names[1])
mlp_mae = alt.Chart(mlp_training['mae']).mark_line().encode(x='Epoch',y='Mean Absolute Error',
                    color=alt.Color("Dataset", scale=alt.Scale(domain=['Training (CV mean)','Validation (CV mean)'],range=['#4d96d9','#ff6b6b']))).properties(title=model_names[1])

mlp_res = alt.Chart(mlp_residuals['residuals']).mark_bar().encode(
                    alt.X("Speed residual", bin=alt.Bin(extent=[-24,24],step=4)),
                    y='count()').properties(title=model_names[1])


mlp_pdf = alt.Chart(mlp_residuals['pdf_data']).mark_line().encode(x='Speed residual',y='Density').properties(title=model_names[1])

### Performance summary - metrics
metrics = [tre_metrics, mlp_metrics_renamed]
cols = [metric.columns[-1] for metric in metrics]
fve_baseest_cv = pd.DataFrame(columns=cols)
for i, col in enumerate(cols):
    fve_baseest_cv.loc[:, col] = metrics[i].loc[[('MAE','Val'), ('σ MAE','Val'), ('R2','Val')]][col]

fve_metrics = utils.loadPandasPickle("src/models/fve_metrics.pkl")

### Performance summary - residual distribution
residuals_df = [tre_residuals['residuals'], mlp_residuals['residuals']]
for i, df in enumerate(residuals_df):
    residuals_df[i]['Model'] = model_names[i]

pdf_df = [tre_residuals['pdf_data'], mlp_residuals['pdf_data']]
for i, df in enumerate(pdf_df):
    pdf_df[i]['Model'] = model_names[i]

residuals = pd.concat(residuals_df, axis=0)
pdfs = pd.concat(pdf_df, axis=0)

res_summary = alt.Chart(residuals).mark_bar(opacity=0.5).encode(
                alt.X("Speed residual", bin=alt.Bin(extent=[-32,32],step=4)),
                alt.Y('count()', stack=None), 
                alt.Color("Model", scale=alt.Scale(domain=model_names, range=['#4d96d9','#ff6b6b'])))

pdf_summary = alt.Chart(pdfs).mark_line().encode(x='Speed residual',y='Density',
                color=alt.Color("Model", scale=alt.Scale(domain=model_names, range=['#4d96d9','#ff6b6b'])))

