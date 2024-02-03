import pandas as pd
import torch
import altair as alt
import utils

class MLModel():
    #Abstract class for machine learning models, containing common functions
    def __init__(self, model_path, metric_path, residual_path, hist_range):
        model_data = utils.load_pickle(model_path, 'dill')
        self.name = ""
        self.abbr = ""
        self.model = model_data['model']
        self.input_transformer = model_data['input_pipeline']
        self.output_transformer = model_data['output_pipeline']
        self.metrics = utils.load_pandas_pickle(metric_path)
        self.residuals = utils.load_and_flatten_residuals(residual_path)
        self.res_hist = alt.Chart(self.residuals['residuals']).mark_bar().encode(alt.X("Speed residual",\
                                    bin=alt.Bin(extent=hist_range, step=4)), y='count()').properties(title=self.name)
        self.res_PDF = alt.Chart(self.residuals['pdf_data']).mark_line().encode(x='Speed residual', \
                                                                                y='Density').properties(title=self.name)

    def get_name(self):
        return self.name
    
    def get_abbr_name(self):
        return self.abbr
    
    def get_metrics(self):
        return self.metrics
    
    def get_residuals(self):
        return self.residuals
    
    def get_residual_hist(self):
        return self.res_hist

    def get_residual_PDF(self):
        return self.res_PDF


class TRE(MLModel):
    def __init__(self, model_path='src/models/treeEnsemble.pkl', metric_path='src/models/metrics/tre_metrics.pkl',
                 residual_path='src/models/residuals/tre_residuals.pkl', hist_range=[-40, 40]):
        super().__init__(model_path, metric_path, residual_path, hist_range)
        self.name = 'Tree Ensemble'
        self.abbr = 'TRE'

    # def predict(self, query:list) -> float:
    #     feats = feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']
    #     query = pd.DataFrame([query],columns=feats)
    #     return self.model.predict(query)[0]


class MLP(MLModel):
    def __init__(self, model_path='src/models/multilayerPerceptron.pkl', metric_path='src/models/metrics/mlp_metrics.pkl',
                 residual_path='src/models/residuals/mlp_residuals.pkl', hist_range=[-40, 40]):
        super().__init__(model_path, metric_path, residual_path, hist_range)
        self.name = 'Multilayer Perceptron'
        self.abbr = 'MLP'
        self.training = utils.load_pickle("src/models/metrics/mlp_training.pkl")
        self.loss = alt.Chart(self.training['loss']).mark_line().encode(x='Epoch',y='Loss', \
                    color=alt.Color("Dataset", scale=alt.Scale(domain=['Training (CV mean)','Validation (CV mean)'],\
                                                               range=['#4d96d9','#ff6b6b']))).properties(title=self.name)
        self.mae = alt.Chart(self.training['mae']).mark_line().encode(x='Epoch',y='Mean Absolute Error', \
                    color=alt.Color("Dataset", scale=alt.Scale(domain=['Training (CV mean)', \
                                'Validation (CV mean)'],range=['#4d96d9','#ff6b6b']))).properties(title=self.name)

    def get_metrics(self):
        return self.metrics.rename(columns={self.name: self.abbr})

    def get_metrics_full(self):
        return self.metrics

    def get_loss(self):
        return self.loss

    def get_MAE(self):
        return self.mae
#
#     def predict(self, query:list) -> float:
#         query = torch.tensor(query)
#         normquery = self.model.normalizeFeatures(query)
#         with torch.inference_mode():
#             pred = self.model(normquery)
#         return self.model.recoverTargets(pred).item()
#
#
class RLE(MLModel):
    def __init__(self, model_path='src/models/regularizedLinearEnsemble.pkl', metric_path='src/models/metrics/rle_metrics.pkl',
                 residual_path='src/models/residuals/rle_residuals.pkl', hist_range=[-40,40]):
        super().__init__(model_path, metric_path, residual_path, hist_range)
        self.name = "Regularized Linear Ensemble"
        self.abbr = "RLE"
#
#     def predict(self, query:list) -> float:
#         feats = feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']
#         query = pd.DataFrame([query],columns=feats)
#         normquery = self.scaler_x.transform(query)
#         normpred = self.model.predict(normquery)
#         pred = self.scaler_y.inverse_transform(normpred.reshape(-1,1)).flatten()
#         return pred[0]
#
#
class SVE(MLModel):
    def __init__(self, model_path='src/models/supportVectorEnsemble.pkl', metric_path='src/models/metrics/sve_metrics.pkl',
                 residual_path='src/models/residuals/sve_residuals.pkl', hist_range=[-40,40]):
        super().__init__(model_path, metric_path, residual_path, hist_range)
        self.name = "Support Vector Ensemble"
        self.abbr = "SVE"
#
#     def predict(self, query:list) -> float:
#         feats = feats=['vehicleMass','impactAngle','finalDisp','nPoles','damageLength']
#         query = pd.DataFrame([query],columns=feats)
#         normquery = self.scaler_x.transform(query)
#         normpred = self.model.predict(normquery)
#         pred = self.scaler_y.inverse_transform(normpred.reshape(-1,1)).flatten()
#         return pred[0]
#
#
class FVE(MLModel):
    def __init__(self, submodels, weights):
        self.name = 'Final Voting Regressor'
        self.abbr = 'FVE'
        self.submodels = submodels
        self.weights = weights
        self.metrics = utils.load_pandas_pickle("src/models/metrics/fve_metrics.pkl")
        self.baseest_cv = self.metrics.loc[[('MAE','Val'), ('σ MAE','Val')]]
        self.residuals_base, self.residuals = self.extract_residuals()
        self.pdfs_base, self.pdfs = self.extract_pdfs()
        self.res_hist_base = alt.Chart(self.residuals_base).mark_bar(opacity=0.8).encode(
                alt.X("Speed residual", bin=alt.Bin(extent=[-40,40],step=4)),
                alt.Y('count()'),
                alt.Color("Model", scale=alt.Scale(domain=[m.get_name() for m in self.submodels],
                                                   range=['#c98b65','#c96567','#9d5a63','#304455'])))
        # colormap=['#c98b65','#c96567','#9d5a63','#644e5b','#304455']
        self.res_PDF_base = alt.Chart(self.pdfs_base).mark_line(opacity=1).encode(x='Speed residual',y='Density',
                color=alt.Color("Model", scale=alt.Scale(domain=[m.get_name() for m in self.submodels],
                                                         range=['#e7ba52','#1f77b4','#9467bd','#ed1717'])))
        self.res_hist = alt.Chart(self.residuals).mark_bar().encode(alt.X("Speed residual",
                            bin=alt.Bin(extent=[-40,40],step=4)),y='count()').properties(title=self.name)
        self.res_PDF = alt.Chart(self.pdfs).mark_line().encode(x='Speed residual',y='Density').properties(title=self.name)

    def extract_residuals(self):
        residuals_df = [m.get_residuals()['residuals'] for m in self.submodels]
        res_fve = residuals_df[0].copy()
        res_fve['Model'] = self.get_name()
        res_fve['Speed residual'] = 0
        for i, df in enumerate(residuals_df):
            residuals_df[i]['Model'] = self.submodels[i].get_name()
            res_fve['Speed residual'] += self.weights[i]*residuals_df[i]['Speed residual']
        return pd.concat(residuals_df, axis=0), res_fve

    def extract_pdfs(self):
        pdf_df = [m.get_residuals()['pdf_data'] for m in self.submodels]
        pdf_fve = pdf_df[0].copy()
        pdf_fve['Model'] = self.get_name()
        pdf_fve['Density'] = 0
        for i, df in enumerate(pdf_df):
            pdf_df[i]['Model'] = self.submodels[i].get_name()
            pdf_fve['Density'] += self.weights[i]*pdf_df[i]['Density']
        return pd.concat(pdf_df, axis=0), pdf_fve

    def get_base_est_cv(self):
        return self.baseest_cv

    def get_base_residual_hist(self):
        return self.res_hist_base

    def get_base_residual_pdf(self):
        return self.res_PDF_base
#
#     def predict(self, query):
#         b_preds = [m.predict(query) for m in self.submodels]
#         base_preds = pd.DataFrame(b_preds,index=[m.getName() for m in self.submodels],columns=['Speed [km/h]'])
#         base_preds.index.name = 'Base predictor'
#         ens_pred = 0
#         for p,w in zip(b_preds, self.weights):
#             ens_pred += p*w
#         return base_preds, ens_pred



tre, rle, sve, mlp = TRE(), RLE(), SVE(), MLP()
w = [0.15, 0.15, 0.35, 0.35]
fve = FVE((tre, rle, sve, mlp), w)
