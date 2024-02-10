import streamlit as st

from models import rle
from utils import make_sidebar, get_query

st.set_page_config(page_title="speedEST - Regularized Linear Ensemble", layout="wide")
top_cols = st.columns([0.3, 0.35, 0.35], gap="large")

make_sidebar()

with top_cols[1]:
    st.markdown(
        """
        # Regularized Linear Ensemble(`RLE`)
    
        The Regularized Linear Ensemble model [1] is an ensemble estimator that uses several 
        base regressors and averages their prediction to form a final prediction. 
        This particular model considers linear models: Lasso [2], Ridge regression [3], 
        and Elastic Net model [4].
        In the following, the base estimators are briefly introduced.
        """
    )

    st.image("src/img/rleModel.png")

    st.markdown(
        r"""
        In order to train a linear model, the *mean squared error* loss function is 
        considered. The loss on a given training set with $n$ examples can be computed as
    
        $\mathcal{L}(\beta) = \dfrac{1}{n} \overset{n}{\underset{i=1}{\sum}} (y_i - x_i \beta)^2 = \dfrac{1}{n} || y - X \beta ||^2_2$,
    
        where $y_i$ is the target variable associated with a feature vector $x_i$. 
        The equation is rewritten in matrix form, where $|| \bullet ||_2^2$ denotes the 
        L2-norm, $y$ are the target variables in vector form, and $X$ is the feature 
        matrix of training examples. Model parameters are grouped within the vector $\beta$. 
        The optimal set of model parameters, $\hat{\beta}$ which are found during training 
        of the model:
    
        $\hat{\beta} = \underset{\beta}{\text{argmin}} (\dfrac{1}{n} || y - X \beta ||^2_2)$
    
        This set of parameters minimizes the loss function.
    
        ## Feature normalisation
    
        As all voters are based on linear regression, it is noteworthy that the input 
        data needs to be normalised in order to successfully train this model. 
        Normalisation ensures that features which originally have different scales, 
        are of comparable values. This, in turn, speeds up the training and reduces 
        the chance to end up in a local minimum while minimising the loss function.
        Because of this, the training and validation datasets are standardised such 
        that each feature has a 0 mean and a standard deviation of 1, i.e., 
        the standardised value (z-score) for feature $i$ is calculated as
    
        $x'_i = \dfrac{x_{i} - \mu}{\sqrt{\sigma^2}}$,
    
        where $x_i$ is the raw value of the feature $i$, $\mu$ is the mean of the population, 
        and $\sigma^2$ is the variance of the population.
        As the mean and variance of the whole population are not known, 
        we use the unbiased estimators on the training set. Assuming that the size 
        of the training set is $n$, we have
    
        $\hat{\mu} = \dfrac{1}{n} \overset{n}{\underset{i=1}{\sum}} x_i$
    
        $\hat{\sigma}^2 = \dfrac{1}{n-1} \overset{n}{\underset{i=1}{\sum}} (x_i - \hat{\mu})^2$
    
        The target values are standardised in an analogous way.
        It is noteworthy that during inference, the validation and test sets 
        are standardised with respect to the means and standard deviations of the *training* set.
    
        ## Lasso regression (`Lasso`)
    
        The Lasso method (Least Absolute Shrinkage and Selection Operator Regression) 
        introduces regularization into the mode by adding the L1-norm of the parameters 
        to the loss function:
    
        $\mathcal{L} = \dfrac{1}{n} || y - X \beta ||^2_2 + \lambda ||\beta||_1$
    
        where $|| \bullet ||_1$ denotes the L1-norm, and the hyperparameter $\lambda$ 
        controls the amount of the regularization. The optimal set of model parameters 
        is then found as:
    
        $\hat{\beta} = \underset{\beta}{\text{argmin}} ( \dfrac{1}{n}|| y - X \beta ||^2_2 + \lambda ||\beta||_1)$
    
        Lasso regression tends to eliminate the weights of the least important 
        features, i.e., it automatically performs feature selection and outputs a sparse 
        model, where a few features are important. 
    
        To build the model, the `Lasso` class in `scikit-learn` was used. 
        An optimal set of model hyperparameters was found using 5-fold cross-validation 
        on the full training set including 189 training examples (using 20% of the 
        training samples as validation set). In the course of cross-validation, 
        the values of the hyperparameters used to train the model were repeatedly sampled 
        from a predefined distribution.
        In the end, the optimal set was chosen as the one yielding the lowest loss 
        function value on the validation set, and the model was retrained on the full 
        training set for further use in voting ensemble.
    
        ## Ridge regression  (`Ridge`)
    
        Ridge regression introduces regularization into the mode by adding the L2-norm 
        of the parameters to the loss function:
    
        $\mathcal{L} = \dfrac{1}{n} || y - X \beta ||^2_2 + \dfrac{\lambda}{2} ||\beta||_2^2$
    
        where $|| \bullet ||_2$ denotes the L2-norm. Regularization forces the 
        learning algorithm to keep the parameters as small as possible. The optimal 
        set of model parameters is then found as:
    
        $\hat{\beta} = \underset{\beta}{\text{argmin}} ( \dfrac{1}{n}|| y - X \beta ||^2_2 + \dfrac{\lambda}{2} ||\beta||_2^2)$
    
        The hyperparameter $\lambda$ controls the amount of the shrinkage. The larger 
        the value of $\lambda$, the greater the amount of shrinkage and the model 
        parameters show more robustness to colinearity.
    
        To build the model, the `Ridge` class in `scikit-learn` was used. An optimal set 
        of model hyperparameters was found using 5-fold cross-validation on the full 
        training set including 189 training examples (using 20% of the training samples 
        as validation set). In the course of cross-validation, the values of the 
        hyperparameters used to train the model were repeatedly sampled from a predefined 
        distribution. In the end, the optimal set was chosen as the one yielding 
        the lowest loss function value on the validation set, and the model was 
        retrained on the full training set for further use in voting ensemble.
    
        ## Elastic Net (`ElasticNet`)
    
        The Elastic Net model is a regression model that is a linear combination of the 
        Lasso and Ridge regression models, i.e., it uses both L1- and L2-norm 
        regularization in the loss function:
    
        $\mathcal{L} = \dfrac{1}{n} || y - X \beta ||^2_2 + r \lambda ||\beta||_1 + \dfrac{1-r}{2}\lambda ||\beta||_2^2$
    
        where $|| \bullet ||_1$ and $|| \bullet||_2$ denote the L1- and L2-norms, respectively. 
        The ratio of the mix is controlled by the parameter $r$ (so that $r=0$ is equivalent 
        to Ridge regression and $r=1$ is equivalent to Lasso regression). 
        The optimal set of model parameters is then found as:
    
        $\hat{\beta} = \underset{\beta}{\text{argmin}} ( \dfrac{1}{n}|| y - X \beta ||^2_2 + r \lambda ||\beta||_1 + \dfrac{1-r}{2}\lambda ||\beta||_2^2)$
    
        The Elastic net is a middle ground between the two regularizations, 
        as it tends to eliminate unimportant feature weights and works well 
        in cases when the number of features is greater than the number of training 
        examples or when several features are strongly correlated.
    
        To build the model, the `ElasticNet` class in `scikit-learn` was used. 
        An optimal set of model hyperparameters was found using 5-fold cross-validation 
        on the full training set including 189 training examples (using 20% of the 
        training samples as validation set). In the course of cross-validation, the values 
        of the hyperparameters used to train the model were repeatedly sampled from 
        a predefined distribution. In the end, the optimal set was chosen as the one 
        yielding the lowest loss function value on the validation set, and the 
        model was retrained on the full training set for further use in voting ensemble.
    
        ## Regularized Linear Ensemble (`RLE`)
    
        After the optimal sets of hyperparameters have been found for all base estimators 
        as a result of cross-validation, the final voting ensemble 
        (the `VotingRegressor` class in `scikit-learn`) can be built. 
        The final prediction of the ensemble is obtained as a weighted average 
        of the predictions of the Lasso (20% weight), Ridge regression (20% weight) 
        and Elastic Net (60% weight).
    
        ## Performance
    
        To evaluate model performance, the original training set (189 training examples), 
        was repeatedly split into training and validation sets (157 and 32 examples, respectively). 
        For each random split, all base estimators were trained on the smaller training set, 
        and the voting ensemble of them was formed.
        After that, the models make predictions on the validation set, 
        and suitable metrics are computed. 
    
        For the metrics, the mean absolute error (MAE) between the target and predicted 
        value is considered. It can be computed as
    
        $\textrm{MAE} = \dfrac{\overset{n}{\underset{i=1}{\sum}} | y_i - \hat{y}_i |}{n}$,
    
        where $y_i$ denotes the target value, and $\hat{y}_i$ is the value predicted by the model. 
        The MAE gives a rough estimate of the mean error on the predicted value given by the model.
    
        Another metric is the coefficient of determination, called $R^2$, which can be computed as
    
        $R^2 = 1 - \dfrac{\overset{n}{\underset{i=1}{\sum}} ( y_i - \hat{y}_i)^2}{\overset{n}{\underset{i=1}{\sum}} ( y_i - \bar{y})^2}$,
    
        where $\bar{y} = \dfrac{1}{n} \overset{n}{\underset{i=1}{\sum}} y_i$ is the average 
        target value.
    
        This score provides a measure of how well observed outcomes are replicated 
        by the model, based on the proportion of total variation of outcomes 
        explained by the model. The best possible $R^2$ score is $1.0$.
    
        For each random split of the full training set, the MAE and $R^2$ on 
        the validation sets were recorded. Afterwards, the mean values 
        (and standard deviations) over all validation sets can be used 
        to estimate the value of MAE and $R^2$ on unseen data. As such, model evaluation 
        using random splitting of the full training set was done for 50 random splits.
    
        For a final check, we also compute the MAE and $R^2$ on the test set, 
        which was initially created and set aside for this purpose.
        The values of the MAE and $R^2$ for the training, validation and test 
        sets are summarised in the table below. For the validation set, 
        the values represent the mean and standard deviation over all random splits. 
        """
    )

    st.dataframe(rle.get_metrics().style.format("{:.3f}"))

    st.markdown(
        """
        As can be seen in the table, the MAE (in km/h) in the validation set is similar 
        to the one on the test set, meaning this metric (and its standard deviation) 
        can be used to estimate performance of the model on unseen data. 
        Moreover, the performance on the training set is also within the same range of results.
        The histogram and probability density distribution of the residuals 
        (differences between the true and predicted values) in the validation set 
        are presented below. Note that these are all residuals recorded in the course 
        of cross-validation using 50 different random splits of the full training 
        sets (a total of 1600 predictions). It can be seen that the speed residual 
        has an approximately Gaussian distribution.
        """
    )

    tab1, tab2 = st.tabs(["Histogram", "Density"])
    with tab1:
        st.altair_chart(rle.get_residual_hist(), use_container_width=True)
    with tab2:
        st.altair_chart(rle.get_residual_PDF(), use_container_width=True)

    st.markdown(
        """
        ## References
    
        1. `VotingRegressor`. scikit-learn documentation. 
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html
        
        2. Tibshirani, R. (1996). Regression Shrinkage and Selection Via the Lasso, 
        *Journal of the Royal Statistical Society: Series B (Methodological)* 58(1), pp. 267–288, 
        https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
        
        3. Hoerl, A. E, Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems, 
        *Technometrics* 12(1), pp. 55-67, 
        https://doi.org/10.1080/00401706.1970.10488634
        
        4. Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. 
        *Journal of the Royal Statistical Society: Series B (Statistical Methodology)* 67, pp. 301-320. 
        https://doi.org/10.1111/j.1467-9868.2005.00503.x
        
        5. `Lasso`. scikit-learn documentation. 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        
        6. `Ridge`. scikit-learn documentation. 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        
        7. `ElasticNet`. scikit-learn documentation. 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
        
        8. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow: 
        Concepts, Tools, and Techniques to Build Intelligent Systems. (2nd ed.). O’Reilly. 
        """
    )

with top_cols[2]:
    st.markdown("# Use the model")
    col1, col2 = st.columns(2)
    with col1:
        query = get_query()
    with col2:
        speed = rle.predict(query)
        col2.subheader(f"{rle.get_name()} predicts", anchor=False)
        col2.subheader(
            f"&emsp; :green[{speed:.2f}] km/h", anchor=False
        )
