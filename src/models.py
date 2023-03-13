import pandas as pd
import torch
import altair as alt
import utils

class MLModel():
    #Abstract class for machine learning models, containing common functions
    def __init__(self):
        self.name = ""
        self.abbr = ""
        self.metrics = []
        self.residuals = []
        self.resHist = []
        self.resPDF = []

    def getName(self):
        return self.name
    
    def getAbbrName(self):
        return self.abbr
    
    def getMetrics(self):
        return self.metrics
    
    def getResiduals(self):
        return self.residuals
    
    def getResidualHist(self):
        return self.resHist

    def getResidualPDF(self):
        return self.resPDF


class TRE(MLModel):
    def __init__(self, path='src/models/treeEnsemble.pkl'):
        self.name = 'Tree Ensemble'
        self.abbr = 'TRE'
        self.model = utils.loadPickle(path)
        self.metrics = utils.loadPandasPickle("src/models/tre_metrics.pkl")
        self.residuals = utils.loadAndFlattenResiduals("src/models/tre_residuals.pkl")
        self.resHist = alt.Chart(self.residuals['residuals']).mark_bar().encode(alt.X("Speed residual",
                            bin=alt.Bin(extent=[-32,32],step=4)),y='count()').properties(title=self.name)
        self.resPDF = alt.Chart(self.residuals['pdf_data']).mark_line().encode(x='Speed residual',
                                                                               y='Density').properties(title=self.name)

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
                            color=alt.Color("Dataset", scale=alt.Scale(domain=['Training (CV mean)',
                                'Validation (CV mean)'],range=['#4d96d9','#ff6b6b']))).properties(title=self.name)
        self.resHist = alt.Chart(self.residuals['residuals']).mark_bar().encode(
                            alt.X("Speed residual", bin=alt.Bin(extent=[-24,24],step=4)),
                            y='count()').properties(title=self.name)
        self.resPDF = alt.Chart(self.residuals['pdf_data']).mark_line().encode(x='Speed residual',
                                                                               y='Density').properties(title=self.name)

    def getMetrics(self):
        return self.metrics.rename(columns={self.name: self.abbr})
    
    def getMetricsFull(self):
        return self.metrics
    
    def getLoss(self):
        return self.loss
    
    def getMAE(self):
        return self.mae
        
    def predict(self, query):
        query = torch.tensor(query)
        normquery = self.model.normalizeFeatures(query)
        with torch.inference_mode():
            pred = self.model(normquery)
        return self.model.recoverTargets(pred).item()


class FVE(MLModel):
    def __init__(self, submodels):
        self.name = "Final Voting Ensemble"
        self.abbr = "FVE"
        self.submodels = submodels
        self.baseest_cv = self.exctractValMetrics()
        self.metrics = utils.loadPandasPickle("src/models/fve_metrics.pkl")
        self.residuals = self.extractResiduals()
        self.pdfs = self.extractPDFs()
        self.resHist = alt.Chart(self.residuals).mark_bar(opacity=0.5).encode(
                alt.X("Speed residual", bin=alt.Bin(extent=[-32,32],step=4)),
                alt.Y('count()', stack=None), 
                alt.Color("Model", scale=alt.Scale(domain=[m.getName() for m in models], range=['#4d96d9','#ff6b6b'])))
        self.resPDF = alt.Chart(self.pdfs).mark_line().encode(x='Speed residual',y='Density',
                color=alt.Color("Model", scale=alt.Scale(domain=[m.getName() for m in models], range=['#4d96d9','#ff6b6b'])))

    def exctractValMetrics(self):
        metrics = [m.getMetrics() for m in self.submodels]
        cols = [metric.columns[-1] for metric in metrics]
        fve_baseest_cv = pd.DataFrame(columns=cols)
        for i, col in enumerate(cols):
            fve_baseest_cv.loc[:, col] = metrics[i].loc[[('MAE','Val'), ('σ MAE','Val'), ('R2','Val')]][col]
        return fve_baseest_cv
    
    def extractResiduals(self):
        residuals_df = [m.getResiduals()['residuals'] for m in self.submodels]
        for i, df in enumerate(residuals_df):
            residuals_df[i]['Model'] = models[i].getName()
        return pd.concat(residuals_df, axis=0)
    
    def extractPDFs(self):
        pdf_df = [m.getResiduals()['pdf_data'] for m in self.submodels]
        for i, df in enumerate(pdf_df):
            pdf_df[i]['Model'] = models[i].getName()
        return pd.concat(pdf_df, axis=0)
    
    def getBaseEstCV(self):
        return self.baseest_cv

    
tre = TRE('src/models/treeEnsemble.pkl')
mlp = MLP('src/models/multilayerPerceptron.pt')
# ### Regularized Linear Ensemble model
# rle = utils.loadRLEModel()
# ### Support Vector Ensemble model
# sve = utils.loadSVEModel()

models = [tre, mlp]
fve = FVE(models)
