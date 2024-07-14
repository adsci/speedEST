import pickle

import altair as alt
import dill
import numpy as np
import pandas as pd
import torch


def load_pickle(path, format="pickle"):
    with open(path, "rb") as f:
        if format == "dill":
            return dill.load(f)
        return pickle.load(f)


def load_and_flatten_residuals(path):
    resdata = load_pickle(path)
    encoding = {"residuals": "Speed residual"}
    for k, v in resdata.items():
        if not isinstance(v, pd.DataFrame):
            res_flattened = np.array(resdata["residuals"]).flatten()
            resdata[k] = pd.DataFrame(res_flattened, columns=[encoding[k]])
    return resdata


class MLModel:
    # Abstract class for machine learning models, containing common functions
    def __init__(self, model_path, metric_path, residual_path, hist_range):
        model_data = load_pickle(model_path, "dill")
        self.name = ""
        self.abbr = ""
        self.is_pytorch_model = False
        self.model = model_data["model"]
        self.input_transformer = model_data["input_pipeline"]
        self.output_transformer = model_data["output_pipeline"]
        self.metrics = pd.read_pickle(metric_path)
        self.residuals = load_and_flatten_residuals(residual_path)
        self.res_hist = (
            alt.Chart(self.residuals["residuals"])
            .mark_bar()
            .encode(
                alt.X("Speed residual", bin=alt.Bin(extent=hist_range, step=4)),
                y="count()",
            )
            .properties(title=self.name)
        )
        self.res_PDF = (
            alt.Chart(self.residuals["pdf_data"])
            .mark_line()
            .encode(x="Speed residual", y="Density")
            .properties(title=self.name)
        )

    def transform_input(self, query: pd.DataFrame) -> pd.DataFrame:
        return self.input_transformer.transform(query)

    def transform_output(self, model_output: np.array) -> float:
        return self.output_transformer.inverse_transform(model_output)

    def predict(self, query: pd.DataFrame) -> tuple[float, bool]:
        input = self.transform_input(query)
        if (
            self.is_pytorch_model
        ):  # for PyTorch, convert to tensors and call the model instead
            input = torch.tensor(input.values).to(dtype=torch.float32)
            with torch.inference_mode():
                output = self.model(input).reshape(-1, 1)
        else:
            output = self.model.predict(input).reshape(-1, 1)
        pred = self.transform_output(output).item()
        return (pred, pred > 0)

    def get_name(self):
        return self.name

    def get_metrics(self):
        return self.metrics

    def get_residuals(self):
        return self.residuals

    def get_residual_hist(self):
        return self.res_hist

    def get_residual_PDF(self):
        return self.res_PDF


class TRE(MLModel):
    def __init__(
        self,
        model_path="src/models/treeEnsemble.pkl",
        metric_path="src/models/metrics/tre_metrics.pkl",
        residual_path="src/models/residuals/tre_residuals.pkl",
        hist_range=(-40, 40),
    ):
        super().__init__(model_path, metric_path, residual_path, hist_range)
        self.name = "Tree Ensemble"
        self.abbr = "TRE"


class MLP(MLModel):
    def __init__(
        self,
        model_path="src/models/multilayerPerceptron.pkl",
        metric_path="src/models/metrics/mlp_metrics.pkl",
        residual_path="src/models/residuals/mlp_residuals.pkl",
        hist_range=(-40, 40),
    ):
        super().__init__(model_path, metric_path, residual_path, hist_range)
        self.name = "Multilayer Perceptron"
        self.abbr = "MLP"
        self.is_pytorch_model = True
        self.training = load_pickle("src/models/metrics/mlp_training.pkl")
        self.loss = (
            alt.Chart(self.training["loss"])
            .mark_line()
            .encode(
                x="Epoch",
                y="Loss",
                color=alt.Color(
                    "Dataset",
                    scale=alt.Scale(
                        domain=["Training (CV mean)", "Validation (CV mean)"],
                        range=["#4d96d9", "#ff6b6b"],
                    ),
                ),
            )
            .properties(title=self.name)
        )
        self.mae = (
            alt.Chart(self.training["mae"])
            .mark_line()
            .encode(
                x="Epoch",
                y="Mean Absolute Error",
                color=alt.Color(
                    "Dataset",
                    scale=alt.Scale(
                        domain=["Training (CV mean)", "Validation (CV mean)"],
                        range=["#4d96d9", "#ff6b6b"],
                    ),
                ),
            )
            .properties(title=self.name)
        )

    def get_metrics(self):
        return self.metrics.rename(columns={self.name: self.abbr})

    def get_metrics_full(self):
        return self.metrics

    def get_loss(self):
        return self.loss

    def get_MAE(self):
        return self.mae


class RLE(MLModel):
    def __init__(
        self,
        model_path="src/models/regularizedLinearEnsemble.pkl",
        metric_path="src/models/metrics/rle_metrics.pkl",
        residual_path="src/models/residuals/rle_residuals.pkl",
        hist_range=(-40, 40),
    ):
        super().__init__(model_path, metric_path, residual_path, hist_range)
        self.name = "Regularized Linear Ensemble"
        self.abbr = "RLE"


class SVE(MLModel):
    def __init__(
        self,
        model_path="src/models/supportVectorEnsemble.pkl",
        metric_path="src/models/metrics/sve_metrics.pkl",
        residual_path="src/models/residuals/sve_residuals.pkl",
        hist_range=(-40, 40),
    ):
        super().__init__(model_path, metric_path, residual_path, hist_range)
        self.name = "Support Vector Ensemble"
        self.abbr = "SVE"


class FVE:
    def __init__(self, submodels, weights):
        self.name = "Final Voting Regressor"
        self.abbr = "FVE"
        self.submodels = submodels
        self.weights = weights
        self.metrics = pd.read_pickle("src/models/metrics/fve_metrics.pkl")
        self.baseest_cv = self.metrics.loc[[("MAE", "Val"), ("σ MAE", "Val")]]
        self.residuals_base, self.residuals = self.extract_residuals()
        self.pdfs_base, self.pdfs = self.extract_pdfs()
        self.res_hist_base = (
            alt.Chart(self.residuals_base)
            .mark_bar(opacity=0.8)
            .encode(
                alt.X("Speed residual", bin=alt.Bin(extent=[-40, 40], step=4)),
                alt.Y("count()"),
                alt.Color(
                    "Model",
                    scale=alt.Scale(
                        domain=[m.get_name() for m in self.submodels.values()],
                        range=["#c98b65", "#304455", "#c96567", "#9d5a63"],
                    ),
                ),
            )
        )
        # colormap=['#c98b65','#c96567','#9d5a63','#644e5b','#304455']
        self.res_PDF_base = (
            alt.Chart(self.pdfs_base)
            .mark_line(opacity=1)
            .encode(
                x="Speed residual",
                y="Density",
                color=alt.Color(
                    "Model",
                    scale=alt.Scale(
                        domain=[m.get_name() for m in self.submodels.values()],
                        range=["#e7ba52", "#1f77b4", "#9467bd", "#ed1717"],
                    ),
                ),
            )
        )
        self.res_hist = (
            alt.Chart(self.residuals)
            .mark_bar()
            .encode(
                alt.X("Speed residual", bin=alt.Bin(extent=[-40, 40], step=4)),
                y="count()",
            )
            .properties(title=self.name)
        )
        self.res_PDF = (
            alt.Chart(self.pdfs)
            .mark_line()
            .encode(x="Speed residual", y="Density")
            .properties(title=self.name)
        )

    def ensemble_predict(self, query: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        preds = {}
        full_names = {}
        pred_array = []
        weight_array = []
        # predict with each model,
        for name, ml_model in self.submodels.items():
            model_prediction, status = ml_model.predict(query)
            if status:
                full_names[name] = ml_model.get_name()
                preds[name] = model_prediction
                pred_array.append(model_prediction)
                weight_array.append(self.weights[name])

        # construct numpy arrays and renormalize weights
        pred_array = np.array(pred_array)
        weight_array = np.array(weight_array)
        weight_array /= np.sum(weight_array)
        ens_pred = np.dot(pred_array, weight_array)

        base_preds = pd.DataFrame.from_dict(
            preds, orient="index", columns=["Speed [km/h]"]
        )
        base_preds.index.name = "Base predictor"
        base_preds.index = base_preds.index.map(full_names)
        return base_preds, ens_pred

    def extract_residuals(self):
        model_full_names, model_abbr_names, residuals_df = [], [], []
        for name, model in self.submodels.items():
            model_full_names.append(model.get_name())
            model_abbr_names.append(name)
            residuals_df.append(model.get_residuals()["residuals"])
        res_fve = residuals_df[0].copy()
        res_fve["Model"] = "FVE"
        res_fve["Speed residual"] = 0
        for i, _df in enumerate(residuals_df):
            residuals_df[i]["Model"] = model_full_names[i]
            res_fve["Speed residual"] += (
                self.weights[model_abbr_names[i]] * residuals_df[i]["Speed residual"]
            )
        return pd.concat(residuals_df, axis=0), res_fve

    def extract_pdfs(self):
        model_full_names, model_abbr_names, pdf_df = [], [], []
        for name, model in self.submodels.items():
            model_full_names.append(model.get_name())
            model_abbr_names.append(name)
            pdf_df.append(model.get_residuals()["pdf_data"])
        pdf_fve = pdf_df[0].copy()
        pdf_fve["Model"] = "FVE"
        pdf_fve["Density"] = 0
        for i, _df in enumerate(pdf_df):
            pdf_df[i]["Model"] = model_full_names[i]
            pdf_fve["Density"] += (
                self.weights[model_abbr_names[i]] * pdf_df[i]["Density"]
            )
        return pd.concat(pdf_df, axis=0), pdf_fve

    def get_base_est_cv(self):
        return self.baseest_cv

    def get_base_residual_hist(self):
        return self.res_hist_base

    def get_base_residual_pdf(self):
        return self.res_PDF_base

    def get_residual_hist(self):
        return self.res_hist

    def get_residual_PDF(self):
        return self.res_PDF

    def get_metrics(self):
        return self.metrics


tre, mlp, rle, sve = TRE(), MLP(), RLE(), SVE()
submodels = {"TRE": tre, "MLP": mlp, "RLE": rle, "SVE": sve}
weights = {"TRE": 0.15, "MLP": 0.35, "RLE": 0.15, "SVE": 0.35}
fve = FVE(submodels, weights)
