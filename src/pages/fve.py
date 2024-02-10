import streamlit as st
from utils import make_sidebar
from models import fve

st.set_page_config(
    page_title="speedEST - Final Voting Ensemble",
    layout="centered"
)

make_sidebar()

st.markdown(
    """
    # Final Voting Ensemble model (`FVE`)

    All individual models can be used independently with the given performance metrics. 
    However, as they are already trained, they can also be used as base estimators in 
    a voting ensemble, which will average their predictions. 
    As the models represent a variety of methods, an ensemble created by these models 
    might result in better predictions on new data. In the following, 
    the Final Voting Ensemble model is built (using the `VotingRegressor` class 
    in `scikit-learn`). Having in mind the mean errors of individual models, 
    the following weights are used:
    
    * 15% weight for Tree Ensemble (`TRE`)
    * 35% weight for Multilayer Perceptron (`MLP`)
    * 15% weight for Regularized Linear Ensemble (`RLE`)
    * 35% weight for Support Vector Ensemble (`SVE`)
    
    
    # Performance of base estimators
    
    The performance of the developed base estimators is summarised in the table below, 
    where the mean absolute error (MAE, in km/h), its standard deviation (in km/h) 
    and the $R^2$ score is reported for the validation sets. Note that these are 
    the mean values obtained during the course of cross-validation on 50 random splits 
    of the full training set (189 samples) into training and validation subsets 
    (157 and 32 samples, respectively). For each random split, the base estimator 
    is trained on the training subset, and the performance of the model is evaluated 
    on the validation set. These values give an estimation on the mean error and 
    standard deviation the base estimators will make on unseen data. 
    The scores for the Final Voting Ensemble in an analogous fashion, c
    alculating the model residual as the weighted average of the residuals 
    obtained from the base estimators.
    """
)

st.dataframe(fve.get_base_est_cv().style.format("{:.3f}"))

st.markdown(
    """
    It is noteworthy, that the mean absolute error and its standard deviation 
    (over all validation sets) of the Final Voting Ensemble are smaller than those 
    of any individual base estimators.

    For completeness, the stacked histogram and probability density distribution 
    of the residuals (differences between the true and predicted values) in the 
    validation set are presented below for each base estimator. Note that for each 
    base estimator all residuals were recorded in the course of cross-validation 
    using 50 different random splits of the full training sets (a total of 1600 predictions). 
    In the resulting graph, the histograms of all base estimators are stacked on top 
    of each other (giving a total of 6400 predictions).
    
    Furthermore, histogram and probability density distribution of the residuals 
    obtained with the Final Voting ensemble model on the validation sets are 
    presented in separate graphs.
    """
)

tab1, tab2, tab3, tab4 = st.tabs(["Histogram - Base Estimators","Density - Base Estimators",
                                       "Histogram - Ensemble", "Density - Ensemble"])
with tab1:
    st.altair_chart(fve.get_base_residual_hist(), use_container_width=True)
with tab2:
    st.altair_chart(fve.get_base_residual_pdf(), use_container_width=True)
with tab3:
    st.altair_chart(fve.get_residual_hist(), use_container_width=True)
with tab4:
    st.altair_chart(fve.get_residual_PDF(), use_container_width=True)

st.markdown(
    """
    # Performance of the ensemble

    The performance of the model and base estimators on the training and test sets 
    is presented in the table below.
    """
)

st.dataframe(fve.get_metrics().style.format("{:.3f}"))