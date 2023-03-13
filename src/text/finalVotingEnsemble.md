# Final Voting Ensemble model (`FVE`)

All individual models can be used independently with the given performance metrics. 
However, as they are already trained, they can also be used as base estimators in a voting ensemble, which will average 
their predictions. 
As the models represent a variety of methods, an ensemble created by these models might result in better predictions on new data. In the following, the Final Voting Ensemble model is built (using the `VotingRegressor` class in `scikit-learn`). 
Having in mind the mean errors of individual models, the following weights are used:

* 15% weight for Tree Ensemble (`TRE`)
* 5% weight for Regularized Linear Ensemble (`RLE`)
* 40% weight for Support Vector Ensemble (`SVE`)
* 40% weight for Multilayer Perceptron (`MLP`)

# Performance of base estimators

The performance of the developed base estimators is summarised in the table below, where the mean absolute error (MAE, in km/h), its standard deviation (in km/h) and the $R^2$ score is reported for the validation sets. Note that these are the mean values obtained during the course of cross-validation on 50 random splits of the full training set (189 samples) into training and validation subsets (157 and 32 samples, respectively). For each random split, the base estimator is trained on the training subset, and the performance of the model is evaluated on the validation set. 
These values give an estimation on the mean error and standard deviation the base estimators will make on unseen data. 
As the base estimators used in the Final Voting Ensemble were retrained on the full training set after cross-validation, there are no such scores for that model. 

![SplitHere]()

For completeness, the histogram and probability density distribution of the residuals (differences between the true and predicted values) in the validation set are presented below for each base estimator. Note that these are all residuals recorded in the course of cross-validation using 50 different random splits of the full training sets (a total of 1600 predictions).

![SplitHere]()

# Performance of the ensemble

The performance of the model and base estimators on the training and test sets is presented in the table below.
It is noteworthy that the MAE and $R^2$ metrics of the voting ensemble are better than the individual base estimators.

![SplitHere]()