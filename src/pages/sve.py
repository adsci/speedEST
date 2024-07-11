import streamlit as st
from streamlit_theme import st_theme

from models import sve
from utils import get_query, make_sidebar

st.set_page_config(page_title="speedEST - Support Vector Ensemble", layout="wide")
top_cols = st.columns([0.5, 0.5], gap="large")

theme_dict = st_theme()
if theme_dict:
    theme = theme_dict["base"]
else:
    theme = "light"
make_sidebar(theme)

with top_cols[0]:
    st.title("Support Vector Ensemble(`SVE`)")

    st.image(
        "src/img/models/sve_logo.png",
        caption="""
               Image representation of a support vector regressor, created by text-to-image deep learning model.
               In this image, the central element is a glowing, meandering river that acts as the regression line,
               cutting through the terrain to represent the SVR's methodology of fitting a model within
               a dataset while minimizing error. The riverbanks are lined with glowing orbs or stones,
               symbolizing the data points, with some nestled within a soft margin on either side of the river
               to depict the epsilon-insensitive loss function used in SVR.
               """,
        width=500
    )

    st.markdown(
        """
        The Support Vector Ensemble model [1] is an ensemble estimator that uses several
        base regressors and averages their prediction to form a final prediction.
        This particular model considers support vector machines (support vector regressors)
        with different kernels. In the following, the base estimators are briefly introduced.
        """
    )

    st.image("src/img/models/sveModel.png")

    st.markdown(
        r"""
        ## Support Vector Machines

        Support Vector Machine (SVM) is versatile machine learning model capable of
        performing classification and regression tasks. In a nutshell, the SVM fits a
        "margin" (or a tube) that separates different classes. Points lying on different
        sides of the margin will be classified as belonging to different classes.
        The margin is said to be *supported* on the instances located at its edges
        (which are also called *support vectors*).
        However, if we impose that all instances must be on the "right" side of the margin
        (*hard margin classification*), the margin (if found) will be susceptible to outlier
        and will likely not generalize very well. Usually, some degree of margin violation
        is allowed, i.e., instances can end up within or on the "wrong" side of the
        margin (*soft margin classification*). Hence, the SVM algorithms are trying to
        maximize the width of the margin, while keeping the amount of margin violations
        small (controlled with hyperparameter $C$ - a low $C$ makes the decision
        surface smoother, while a high $C$ aims at classifying all examples correctly).

        Support Vector Regression (SVR) reverses this objective.
        Instead of trying to fit the largest possible margin between two classes
        (while limiting margin violation), SVR tries to fit a margin, which contains
        as many training instances as possible while limiting margin violations
        (instances outside the margin). The width of the margin is controlled
        with hyperparameter $\varepsilon$, hence this model is often called $\varepsilon$-SVR.

        ## Primal and dual problem

        The $\varepsilon$-SVR solves the following primal problem.
        For given training vectors $x_i$ (where $i=0,1,...,n$) and target values $y_i$:

        $\qquad \qquad\underset{w,b,\zeta,\zeta^*}{\mathrm{min}} \dfrac{1}{2} w^Tw + C\sum_{i=1}^n(\zeta_i + \zeta_i^*)$

        subject to $\quad y_i - w^T \phi(x_i) - b \leq \varepsilon + \zeta_i$,

        $\qquad \qquad \quad w^T \phi(x_i) + b - y_i \leq \varepsilon + \zeta_i^*$,

        $\qquad \qquad \quad \zeta_i, \zeta_i^* \geq 0$.

        The goal is to find the weights $w$ and bias $b$ so that predictions for
        a vector $x$, $w^T x + b$ can be made. In the above, the slack variables
        $\zeta_i$ and $\zeta_i^*$ penalize the objective, depending on whether their
        prediction lies above or below the $\varepsilon$-sized margin.
        Hard and soft margin problems are convex quadratic optimization problems
        with linear constraints, and can usually be solved using Quadratic Programming solvers.

        It is noteworthy that the features $x_i$ are mapped to a (potentially)
        higher-dimensional space using the mapping function $\phi(x_i)$.
        This way, we are able to perform nonlinear classification/regression.
        Depending on the mapping function, this operation can be expensive, adding
        a large number of features. However, it is possible to obtain the same result
        without actually adding any new features, using the kernel trick. Using this trick,
        however, requires another formulation of the problem, i.e., a *dual* form.
        The dual problem is an equivalent formulation of the primal problem,
        which (under certain conditions) will produce the same solution.
        The dual form of the SVR problem is the following:

        $\qquad \qquad\underset{\alpha,\alpha^*}{\mathrm{min}} \dfrac{1}{2} (\alpha - \alpha^*)^T Q (\alpha - \alpha^*) + \varepsilon e^T (\alpha + \alpha^*) - y^T (\alpha - \alpha^*)$

        subject to $\quad e^T (\alpha - \alpha^*) = 0$,

        $\qquad \qquad \quad 0 \leq \alpha_i, \alpha_i^* \leq C$.

        Within this setting, $e$ is a vector of ones, and $Q$ is the matrix of kernels.
        Each entry $Q_{ij} = K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ contains a kernel,
        using implicitly mapping the features into a higher (or infinite) dimensional space.
        Using the kernel trick, it is possible to compute the values of
        $\phi(x)^T\phi(x')$ directly by performing simple operations on the
        dot product of the original vectors $x$ and $x'$, without explicitly
        computing their mappings $\phi(x)$ and $\phi(x')$.

        The optimization algorithm finds the values of the parameters
        $\alpha_i$ and $\alpha_i^*$. All non-zero values of $\alpha_i$ and $\alpha_i^*$
        signify that the sample is a support vector. Using those, it is possible to
        compute the bias $b$. The final prediction for a given feature vector,
        $x$, can be computed as:

        $\underset{i \in SV}{\sum} (\alpha_i - \alpha_i^*) K(x_i, x) + b$,

        where $K$ is the chosen kernel. Note that sum is performed only on the
        support vector set.

        ## Feature normalisation

        As all voters are based on support vector machines, it is noteworthy that
        the input data needs to be normalised in order to successfully train this model.
        Normalisation ensures that features which originally have different scales,
        are of comparable values. This, in turn, speeds up the training and reduces
        the chances to end up in a local minimum while minimising the loss function.
        Because of this, the training and validation datasets are standardised such
        that each feature has a 0 mean and a standard deviation of 1, i.e., the
        standardised value (z-score) for feature $i$ is calculated as

        $x'_i = \dfrac{x_{i} - \mu}{\sqrt{\sigma^2}}$,

        where $x_i$ is the raw value of the feature $i$, $\mu$ is the mean of the population,
        and $\sigma^2$ is the variance of the population.
        As the mean and variance of the whole population are not known, we use
        the unbiased estimators on the training set. Assuming that the size of the
        training set is $n$, we have

        $\hat{\mu} = \dfrac{1}{n} \overset{n}{\underset{i=1}{\sum}} x_i$

        $\hat{\sigma}^2 = \dfrac{1}{n-1} \overset{n}{\underset{i=1}{\sum}} (x_i - \hat{\mu})^2$

        The target values are standardised in an analogous way.
        It is noteworthy that during inference, the validation and test sets
        are standardised with respect to the means and standard deviations of
        the *training* set.

        ## Support Vector Machine with linear kernel (`Linear`)

        The linear kernel is the easiest kernel. It does not introduce any mapping
        into a higher dimensional space. Instead, only the original features
        are considered. For vectors $x$ and $x'$, linear kernel reduces to inner product, i.e.,

        $ K(x, x') = x^Tx'$.

        To build the model, the `SVR` class in `scikit-learn` was used. An optimal set
        of model hyperparameters was found using 5-fold cross-validation on the
        full training set including 189 training examples (using 20% of the training
        samples as validation set). In the course of cross-validation, the values of
        the hyperparameters used to train the model were repeatedly sampled from
        a predefined distribution. In the end, the optimal set was chosen as the
        one yielding the lowest loss function value on the validation set, and
        the model was retrained on the full training set for further use in voting ensemble.

        ## Support Vector Machine with RBF kernel (`RBF`)

        The radial basis function (RBF) kernel is a popular kernel function.
        For vectors $x$ and $x'$, the RBF kernel is defined as

        $ K(x, x') = \mathrm{exp}(-\gamma || x^T - x'||^2)$,

        where $\gamma$ is the scale parameter and $||\bullet||$ is the L2-norm.
        This hyperparameter defines how much influence a single training example
        has, i.e., for a large $\gamma$, other examples must be closer if they are to be affected.
        The RBF kernel can be also interpreted as a similarity measure between
        different samples. It is noteworthy that the feature space of this kernel
        has an infinite number of dimension (which makes explicit mapping $\phi(x)$ impossible!).

        To build the model, the `SVR` class in `scikit-learn` was used.
        An optimal set of model hyperparameters was found using 5-fold cross-validation
        on the full training set including 189 training examples (using 20% of the
        training samples as validation set). In the course of cross-validation,
        the values of the hyperparameters used to train the model were repeatedly
        sampled from a predefined distribution.
        In the end, the optimal set was chosen as the one yielding the lowest
        loss function value on the validation set, and the model was retrained on
        the full training set for further use in voting ensemble.


        ## Support Vector Ensemble (`SVE`)

        After the optimal sets of hyperparameters have been found for all base
        estimators as a result of cross-validation, the final voting ensemble
        (the `VotingRegressor` class in `scikit-learn`) can be built.
        The final prediction of the ensemble is obtained as a weighted average
        of the predictions of the SVR using linear (10% weight) and RBF kernel (90% weight).

        ## Performance

        To evaluate model performance, the original training set (189 training examples),
        was repeatedly split into training and validation sets (157 and 32 examples, respectively).
        For each random split, all base estimators were trained on the smaller training set,
        and the voting ensemble of them was formed. After that, the models make predictions
        on the validation set, and suitable metrics are computed.

        For the metrics, the mean absolute error (MAE) between the target and predicted
        value is considered. It can be computed as

        $\textrm{MAE} = \dfrac{\overset{n}{\underset{i=1}{\sum}} | y_i - \hat{y}_i |}{n}$,

        where $y_i$ denotes the target value, and $\hat{y}_i$ is the value predicted by
        the model. The MAE gives a rough estimate of the mean error on the predicted value
        given by the model.
        Another metric is the coefficient of determination, called $R^2$, which can be computed as

        $R^2 = 1 - \dfrac{\overset{n}{\underset{i=1}{\sum}} ( y_i - \hat{y}_i)^2}{\overset{n}{\underset{i=1}{\sum}} ( y_i - \bar{y})^2}$,

        where $\bar{y} = \dfrac{1}{n} \overset{n}{\underset{i=1}{\sum}} y_i$ is the average
        target value.
        This score provides a measure of how well observed outcomes are replicated
        by the model, based on the proportion of total variation of outcomes explained by the model.
        The best possible $R^2$ score is $1.0$.

        For each random split of the full training set, the MAE and $R^2$ on the
        validation sets were recorded. Afterwards, the mean values (and standard deviations)
        over all validation sets can be used to estimate the value of MAE and $R^2$ on unseen data. As such, model evaluation using random splitting of the full training set was done for 50 random splits.

        For a final check, we also compute the MAE and $R^2$ on the test set,
        which was initially created and set aside for this purpose.
        The values of the MAE and $R^2$ for the training, validation and test sets
        are summarised in the table below. For the validation set, the values
        represent the mean and standard deviation over all random splits.
        """  # noqa: E501
    )

    st.dataframe(sve.get_metrics().style.format("{:.3f}"))

    st.markdown(
        """
        As can be seen in the table, the MAE (in km/h) in the validation set is similar
        to the one on the test set, meaning this metric (and its standard deviation)
        can be used to estimate performance of the model on unseen data.
        The histogram and probability density distribution of the residuals
        (differences between the true and predicted values) in the validation set are
        presented below. Note that these are all residuals recorded in the course
        of cross-validation using 50 different random splits of the full training sets
        (a total of 1600 predictions). It can be seen that the speed residual has an
        approximately Gaussian distribution.
        """
    )

    tab1, tab2 = st.tabs(["Histogram", "Density"])
    with tab1:
        st.altair_chart(sve.get_residual_hist(), use_container_width=True)
    with tab2:
        st.altair_chart(sve.get_residual_PDF(), use_container_width=True)

    st.markdown(
        """
        ## References

        1. `VotingRegressor`. scikit-learn documentation.
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html

        2. `SVR`. scikit-learn documentation.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

        3. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow:
        Concepts, Tools, and Techniques to Build Intelligent Systems. (2nd ed.). O’Reilly.
        """
    )

with top_cols[1]:
    st.markdown("# Use the model")
    col1, col2 = st.columns(2)
    with col1:
        query = get_query()
    with col2:
        speed, status = sve.predict(query)
        col2.subheader(f"{sve.get_name()} predicts", anchor=False)
        if status:
            col2.subheader(f"&emsp; :green[{speed:.2f}] km/h", anchor=False)
        else:
            col2.subheader("&emsp; :red[invalid prediction]", anchor=False)
