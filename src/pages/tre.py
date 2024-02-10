import streamlit as st
from utils import make_sidebar
from models import tre

st.set_page_config(
    page_title="speedEST - Tree Ensemble",
    layout="centered"
)

make_sidebar()

st.markdown(
    """
    # Tree Ensemble model (`TRE`)


    The Tree Ensemble model [1] is an ensemble estimator that uses several base regressors 
    and averages their prediction to form a final prediction. 
    This particular model includes an extremely randomized forest [2,3], 
    gradient boosted regression trees [4,5], and an AdaBoost regressor [6,7,8]. 
    In the following, the base estimators are briefly introduced.
    """
)

st.image("src/img/treModel.png")

st.markdown(
    r"""
    As all voters are based on regression trees, it is noteworthy that the input 
    data does not need to be normalised in order to successfully train this model.

    ## Extremely randomized forest (`ERF`)
    
    The extremely randomized forest model is based on the classic random forest 
    regression model. In random forest, an ensemble of regression trees is 
    trained with bootstrap aggregating, i.e., every predictor is trained on a random 
    subset of the training set, sampled with replacement. Each regression tree is then 
    grown and the training set is split into subsets, so that the mean squared error 
    between the average within each subset and the corresponding targets is minimised. 
    When splitting a node during growth of the trees, classic random forest searches for 
    the best feature among a random subset of features (e.g., one that reduces the variance 
    the most). After all trees have been trained, the forest can make predictions by 
    averaging the predictions from all trees.
    
    Extremely randomized forest takes this approach one step further. 
    Unlike regular decision trees, which search for the best possible threshold 
    for each feature while splitting the node, a random threshold for 
    each feature is considered. This, along with using a random subset of 
    features at node splitting results in even greater regressor diversity, 
    trading higher bias for a lower variance. Furthermore, random thresholds 
    also provide a speed-up in training.
    
    To build the model, the `ExtraTreesRegressor` class in `scikit-learn` was used. 
    An optimal set of model hyperparameters was found using randomized grid 
    search with 5-fold cross-validation on the full training set including 189 training 
    examples (using 20% of the training samples as validation set). In the course of 
    cross-validation, the values of the hyperparameters used to train the model were 
    repeatedly sampled from a predefined distribution.
    In the end, the optimal set was chosen as the one yielding the lowest 
    mean squared error on the validation set, and the model was retrained on the 
    full training set for further use in voting ensemble.
    
    
    ## Gradient boosted regression trees (`GBRT`)
    
    The general idea of boosting methods is to sequentially train predictors, 
    where every following predictor tries to correct its predecessor. 
    In this aspect, the gradient boosting regressor fits a new predictor 
    to the residual error made by previous predictors.
    In this setting, a regression tree is used as the base regressor, 
    and in each training stage, a new regression tree is fit on the negative 
    gradient of the loss function (mean squared error).
    In the end, the ensemble prediction can be obtained as the sum 
    of the prediction of all regressors.
    
    To build the model, the `GradientBoostingRegressor` class in `scikit-learn` was used. 
    An optimal set of model hyperparameters was found using randomized grid search 
    with 5-fold cross-validation on the full training set including 189 training 
    examples (using 20% of the training samples as validation set). In the course of 
    cross-validation, the values of the hyperparameters used to train the model were 
    repeatedly sampled from a predefined distribution.
    In the end, the optimal set was chosen as the one yielding the lowest mean 
    squared error on the validation set, and the model was retrained on the full 
    training set for further use in voting ensemble.
    
    ## Adaptive boosting of regression trees (`AdaBoost`)
    Similar to the previous model, the AdaBoost (short for *Adaptive Boosting*) model 
    trains regression trees sequentially so that each following predictor tries 
    to correct its predecessor.
    In contrast to gradient boosting, AdaBoost adjusts the weights of the 
    training examples according to the error of the current predictions. 
    As a result, subsequent predictors focus more on  examples which are 
    difficult to predict. The final prediction from the regressor is then 
    obtained through a weighted sum of the predictions made by individual regressors.
    
    To build the model, the `AdaBoostRegressor` class in `scikit-learn` was used. 
    An optimal set of model hyperparameters was found using randomized grid search 
    with 5-fold cross-validation on the full training set including 189 training 
    examples (using 20% of the training samples as validation set). In the course 
    of cross-validation, the values of the hyperparameters used to train the 
    model were repeatedly sampled from a predefined distribution.
    In the end, the optimal set was chosen as the one yielding the lowest mean 
    squared error on the validation set, and the model was retrained on the full 
    training set for further use in voting ensemble.

    ## Tree Ensemble (`TRE`)

    After the optimal sets of hyperparameters have been found for all base 
    estimators as a result of cross-validation, the final voting 
    ensemble (the `VotingRegressor` class in `scikit-learn`) can be built. 
    The final prediction of the ensemble is obtained as a weighted average 
    of the predictions of the random forest (15% weight), gradient boosted 
    regression trees (75% weight) and AdaBoost (10% weight).

    ## Performance

    To evaluate model performance, the original training set (189 training examples), 
    was repeatedly split into training and validation sets 
    (157 and 32 examples, respectively). 
    For each random split, all base estimators were trained on the smaller training set, 
    and the voting ensemble of them was formed.
    After that, the models make predictions on the validation set, 
    and suitable metrics are computed. 
    
    For the metrics, the mean absolute error (MAE) between the target and 
    predicted value is considered. 
    It can be computed as
    
    $\textrm{MAE} = \dfrac{\overset{n}{\underset{i=1}{\sum}} | y_i - \hat{y}_i |}{n}$,

    where $y_i$ denotes the target value, and $\hat{y}_i$ is the value predicted 
    by the model. The MAE gives a rough estimate of the mean error 
    on the predicted value given by the model.
    
    Another metric is the coefficient of determination, called $R^2$, 
    which can be computed as

    $R^2 = 1 - \dfrac{\overset{n}{\underset{i=1}{\sum}} ( y_i - \hat{y}_i)^2}{\overset{n}{\underset{i=1}{\sum}} ( y_i - \bar{y})^2}$,

    where $\bar{y} = \dfrac{1}{n} \overset{n}{\underset{i=1}{\sum}} y_i$ is the average 
    target value.
    
    This score provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model. 
    The best possible $R^2$ score is $1.0$.
    
    For each random split of the full training set, the MAE and $R^2$ on the 
    validation sets were recorded. Afterwards, the mean values (and standard deviations) 
    over all validation sets can be used to estimate the value of MAE and $R^2$ 
    on unseen data. As such, model evaluation using random splitting of the 
    full training set was done for 50 random splits.
    
    For a final check, we also compute the MAE and $R^2$ on the test set, 
    which was initially created and set aside for this purpose.
    
    The values of the MAE and $R^2$ for the training, validation and test sets 
    are summarised in the table below. For the validation set, the values represent 
    the mean and standard deviation over all random splits. 
    """
)

st.dataframe(tre.get_metrics().style.format("{:.3f}"))

st.markdown(
    """
    As can be seen in the table, the MAE (in km/h) in the validation set 
    is similar to the one on the test set, meaning this 
    metric (and its standard deviation) can be used to estimate performance 
    of the model on unseen data. The histogram and probability density 
    distribution of the residuals (differences between the true and predicted values) 
    in the validation set are presented below. Note that these are all 
    residuals recorded in the course of cross-validation using 50 different 
    random splits of the full training sets (a total of 1600 predictions). 
    It can be seen that the speed residual has an approximately Gaussian distribution.
    """
)

tab1, tab2 = st.tabs(["Histogram", "Density"])
with tab1:
    st.altair_chart(tre.get_residual_hist(), use_container_width=True)
with tab2:
    st.altair_chart(tre.get_residual_PDF(), use_container_width=True)

st.markdown(
    """
    ## References

    1. `VotingRegressor`. scikit-learn documentation. 
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html
    
    2. Geurts, P. et al. (2006). Extremely Randomized Trees, 
    *Machine Learning* 63(1), pp. 3-42, https://doi.org/10.1007/s10994-006-6226-1
    
    3. `ExtraTreesRegressor`. scikit-learn documentation. 
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
    
    4. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. 
    *Annals of statistics*, 1189-1232, https://doi.org/10.1214/aos/1013203451
    
    5. `GradientBoostingRegressor`. scikit-learn documentation. 
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    
    6. Freund, Y., Schapire, R. E. (1997). A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. 
    *Journal of Computer and System Sciences* 55(1), pp. 119-139, 
    https://doi.org/10.1006/jcss.1997.1504
    
    7. Drucker, H. (1997). Improving Regressors using Boosting Techniques. 
    *International Conference on Machine Learning*.
    
    8. `AdaBoostRegressor`. scikit-learn documentation. 
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    
    9. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow: 
    Concepts, Tools, and Techniques to Build Intelligent Systems. (2nd ed.). O’Reilly. 
    """
)