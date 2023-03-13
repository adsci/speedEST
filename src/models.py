import pandas as pd
import torch
import altair as alt
import utils

class MLModel():
    def __init__(self):
        self.name = ""
        self.abbr = ""

    def getName(self):
        return self.name
    
    def getAbbrName(self):
        return self.abbr
class TRE(MLModel):
    def __init__(self, path='src/models/treeEnsemble.pkl'):
        self.name = 'Tree Ensemble'
        self.abbr = 'TRE'
        self.model = utils.loadPickle(path)
        self.metrics = utils.loadPandasPickle("src/models/tre_metrics.pkl")
        self.residuals = utils.loadAndFlattenResiduals("src/models/tre_residuals.pkl")

        self.resHist = alt.Chart(self.residuals['residuals']).mark_bar().encode(alt.X("Speed residual", bin=alt.Bin(extent=[-32,32],step=4)),
            y='count()').properties(title=self.name)
        
        self.resPDF = alt.Chart(self.residuals['pdf_data']).mark_line().encode(x='Speed residual',y='Density').properties(title=self.name)
    
    def getMetrics(self):
        return self.metrics
    
    def getResiduals(self):
        return self.residuals
    
    def getResidualHist(self):
        return self.resHist

    def getResidualPDF(self):
        return self.resPDF
    
    def predict(self, query:list) -> float:
        feats = feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']
        query = pd.DataFrame([query],columns=feats)
        return self.model.predict(query)[0]

class MLP(MLModel):
    def __init__(self, path='src/models/multilayerPerceptron.pt'):
        self.name = 'Multilayer Perceptron'
        self.abbr = 'MLP'
        self.model = torch.jit.load(path)
        self.model.eval()
        self.metrics = utils.loadPandasPickle("src/models/mlp_metrics.pkl")
        self.residuals = utils.loadAndFlattenResiduals("src/models/mlp_residuals.pkl")
        self.training = utils.loadPickle("src/models/mlp_training.pkl")

        self.loss = alt.Chart(self.training['loss']).mark_line().encode(x='Epoch',y='Loss',
                    color=alt.Color("Dataset", scale=alt.Scale(domain=['Training (CV mean)','Validation (CV mean)'],range=['#4d96d9','#ff6b6b']))).properties(title=self.name)
        self.mae = alt.Chart(self.training['mae']).mark_line().encode(x='Epoch',y='Mean Absolute Error',
                            color=alt.Color("Dataset", scale=alt.Scale(domain=['Training (CV mean)','Validation (CV mean)'],range=['#4d96d9','#ff6b6b']))).properties(title=self.name)
        self.resHist = alt.Chart(self.residuals['residuals']).mark_bar().encode(
                            alt.X("Speed residual", bin=alt.Bin(extent=[-24,24],step=4)),
                            y='count()').properties(title=self.name)
        self.resPDF = alt.Chart(self.residuals['pdf_data']).mark_line().encode(x='Speed residual',y='Density').properties(title=self.name)

    def getMetrics(self):
        return self.metrics.rename(columns={self.name: self.abbr})
    
    def getMetricsFull(self):
        return self.metrics
    
    def getResiduals(self):
        return self.residuals
    
    def getTraining(self):
        return self.training
    
    def getLoss(self):
        return self.loss
    
    def getMAE(self):
        return self.mae
    
    def getResHist(self):
        return self.resHist
    
    def getResPDF(self):
        return self.resPDF
    
    def predict(self, query):
        query = torch.tensor(query)
        normquery = self.model.normalizeFeatures(query)
        with torch.inference_mode():
            pred = self.model(normquery)
        return self.model.recoverTargets(pred).item()

### Tree Ensemble model
tre = TRE('src/models/treeEnsemble.pkl')
### Multilayer Perceptron model
mlp = MLP('src/models/multilayerPerceptron.pt')
# ### Regularized Linear Ensemble model
# rle = utils.loadRLEModel()
# ### Support Vector Ensemble model
# sve = utils.loadSVEModel()

models = [tre, mlp]

### Performance summary - metrics
metrics = [m.getMetrics() for m in models]
cols = [metric.columns[-1] for metric in metrics]
fve_baseest_cv = pd.DataFrame(columns=cols)
for i, col in enumerate(cols):
    fve_baseest_cv.loc[:, col] = metrics[i].loc[[('MAE','Val'), ('σ MAE','Val'), ('R2','Val')]][col]

fve_metrics = utils.loadPandasPickle("src/models/fve_metrics.pkl")

### Performance summary - residual distribution
residuals_df = [m.getResiduals()['residuals'] for m in models]
for i, df in enumerate(residuals_df):
    residuals_df[i]['Model'] = models[i].getName()

pdf_df = [tre.getResiduals()['pdf_data'], mlp.getResiduals()['pdf_data']]
for i, df in enumerate(pdf_df):
    pdf_df[i]['Model'] = models[i].getName()

residuals = pd.concat(residuals_df, axis=0)
pdfs = pd.concat(pdf_df, axis=0)

res_summary = alt.Chart(residuals).mark_bar(opacity=0.5).encode(
                alt.X("Speed residual", bin=alt.Bin(extent=[-32,32],step=4)),
                alt.Y('count()', stack=None), 
                alt.Color("Model", scale=alt.Scale(domain=[m.getName() for m in models], range=['#4d96d9','#ff6b6b'])))

pdf_summary = alt.Chart(pdfs).mark_line().encode(x='Speed residual',y='Density',
                color=alt.Color("Model", scale=alt.Scale(domain=[m.getName() for m in models], range=['#4d96d9','#ff6b6b'])))

