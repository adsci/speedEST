import pandas as pd
import torch
import altair as alt
import utils

class MLModel():
    #Abstract class for machine learning models, containing common functions
    def __init__(self, modelpath, metricpath, residualpath, histrange):
        self.name = ""
        self.abbr = ""
        if modelpath.endswith('.pkl'):
            self.model = utils.loadPickle(modelpath)
        elif modelpath.endswith('.pt'):
            self.model = torch.jit.load(modelpath)
            self.model.eval()
        self.metrics = utils.loadPandasPickle(metricpath)
        self.residuals = utils.loadAndFlattenResiduals(residualpath)
        self.resHist = alt.Chart(self.residuals['residuals']).mark_bar().encode(alt.X("Speed residual",
                            bin=alt.Bin(extent=histrange,step=4)),y='count()').properties(title=self.name)
        self.resPDF = alt.Chart(self.residuals['pdf_data']).mark_line().encode(x='Speed residual',
                                                                               y='Density').properties(title=self.name)

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
    def __init__(self, modelpath='src/models/treeEnsemble.pkl', metricpath='src/models/tre_metrics.pkl',
                 residualpath='src/models/tre_residuals.pkl', histrange=[-32,32]):
        super().__init__(modelpath, metricpath, residualpath, histrange)
        self.name = 'Tree Ensemble'
        self.abbr = 'TRE'

    def predict(self, query:list) -> float:
        feats = feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']
        query = pd.DataFrame([query],columns=feats)
        return self.model.predict(query)[0]


class MLP(MLModel):
    def __init__(self, modelpath='src/models/multilayerPerceptron.pt', metricpath='src/models/mlp_metrics.pkl',
                 residualpath='src/models/mlp_residuals.pkl', histrange=[-24,24]):
        super().__init__(modelpath, metricpath, residualpath, histrange)
        self.name = 'Multilayer Perceptron'
        self.abbr = 'MLP'
        self.training = utils.loadPickle("src/models/mlp_training.pkl")
        self.loss = alt.Chart(self.training['loss']).mark_line().encode(x='Epoch',y='Loss',
                    color=alt.Color("Dataset", scale=alt.Scale(domain=['Training (CV mean)','Validation (CV mean)'],range=['#4d96d9','#ff6b6b']))).properties(title=self.name)
        self.mae = alt.Chart(self.training['mae']).mark_line().encode(x='Epoch',y='Mean Absolute Error',
                            color=alt.Color("Dataset", scale=alt.Scale(domain=['Training (CV mean)',
                                'Validation (CV mean)'],range=['#4d96d9','#ff6b6b']))).properties(title=self.name)

    def getMetrics(self):
        return self.metrics.rename(columns={self.name: self.abbr})
    
    def getMetricsFull(self):
        return self.metrics
    
    def getLoss(self):
        return self.loss
    
    def getMAE(self):
        return self.mae
        
    def predict(self, query:list) -> float:
        query = torch.tensor(query)
        normquery = self.model.normalizeFeatures(query)
        with torch.inference_mode():
            pred = self.model(normquery)
        return self.model.recoverTargets(pred).item()


class RLE(MLModel):
    def __init__(self, modelpath='src/models/regularizedLinearEnsemble.pkl', metricpath='src/models/rle_metrics.pkl',
                 residualpath='src/models/rle_residuals.pkl', histrange=[-40,40]):
        super().__init__(modelpath, metricpath, residualpath, histrange)
        self.name = "Regularized Linear Ensemble"
        self.abbr = "RLE"
        self.scaler_x = self.model['scaler_x']
        self.scaler_y = self.model['scaler_y']
        self.model = self.model['rle']
    
    def predict(self, query:list) -> float:
        feats = feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']
        query = pd.DataFrame([query],columns=feats)
        normquery = self.scaler_x.transform(query)
        normpred = self.model.predict(normquery) 
        pred = self.scaler_y.inverse_transform(normpred.reshape(-1,1)).flatten()
        return pred[0]


class SVE(MLModel):
    def __init__(self, modelpath='src/models/supportVectorEnsemble.pkl', metricpath='src/models/sve_metrics.pkl',
                 residualpath='src/models/sve_residuals.pkl', histrange=[-24,24]):
        super().__init__(modelpath, metricpath, residualpath, histrange)
        self.name = "Support Vector Ensemble"
        self.abbr = "SVE"
        self.scaler_x = self.model['scaler_x']
        self.scaler_y = self.model['scaler_y']
        self.model = self.model['sve']
    
    def predict(self, query:list) -> float:
        feats = feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']
        query = pd.DataFrame([query],columns=feats)
        normquery = self.scaler_x.transform(query)
        normpred = self.model.predict(normquery) 
        pred = self.scaler_y.inverse_transform(normpred.reshape(-1,1)).flatten()
        return pred[0]  


class FVE(MLModel):
    def __init__(self, submodels, weights):
        self.submodels = submodels
        self.weights = weights
        self.baseest_cv = self.exctractValMetrics()
        self.metrics = utils.loadPandasPickle("src/models/fve_metrics.pkl")
        self.residuals = self.extractResiduals()
        self.pdfs = self.extractPDFs()
        self.resHist = alt.Chart(self.residuals).mark_bar(opacity=0.5).encode(
                alt.X("Speed residual", bin=alt.Bin(extent=[-40,40],step=4)),
                alt.Y('count()', stack=None), 
                alt.Color("Model", scale=alt.Scale(domain=[m.getName() for m in self.submodels], range=['#4d96d9','#ff6b6b'])))
        self.resPDF = alt.Chart(self.pdfs).mark_line().encode(x='Speed residual',y='Density',
                color=alt.Color("Model", scale=alt.Scale(domain=[m.getName() for m in self.submodels], range=['#4d96d9','#ff6b6b'])))

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
            residuals_df[i]['Model'] = self.submodels[i].getName()
        return pd.concat(residuals_df, axis=0)
    
    def extractPDFs(self):
        pdf_df = [m.getResiduals()['pdf_data'] for m in self.submodels]
        for i, df in enumerate(pdf_df):
            pdf_df[i]['Model'] = self.submodels[i].getName()
        return pd.concat(pdf_df, axis=0)
    
    def getBaseEstCV(self):
        return self.baseest_cv
    
    def predict(self, query):
        b_preds = [m.predict(query) for m in self.submodels]
        base_preds = pd.DataFrame(b_preds,index=[m.getName() for m in self.submodels],columns=['Speed [km/h]'])
        base_preds.index.name = 'Base predictor'
        ens_pred = 0
        for p,w in zip(b_preds, self.weights):
            ens_pred += p*w
        return base_preds, ens_pred


tre, rle, sve, mlp = TRE(), RLE(), SVE(), MLP()
w = [0.15, 0.05, 0.40, 0.40]
fve = FVE([tre, rle, sve, mlp], w)
