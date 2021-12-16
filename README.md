# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset that is being trained on represents employment, demographic, and educational data concerning potential customers of a bank. The task of classification is to predict whether or not the potential customer will be likely to subscribe to term deposits with that bank.

The best performing model in the sklearn pipeline was a logistic regression with parameters of C =  0.9362 and maximum iterations = 1000, which performed with 0.91032 accuracy. Logistic regression provides a lightweight model capable of powerful binary classification performance. These parameters represented an optimal combination of the number of iterations allowed for solver convergence and an inverse regularization hyperparameter. The best overall performing model was found in the AutoML pipeline, and was a VotingEnsemble model. The accuracy for this model was slightly higher than the logistic regression model, at 0.9179.

## Scikit-learn Pipeline
The pipeline for model training includes cleaning and One Hot Encoding the data, splitting into test and train sets, conducting random parameter sampling, and using a bandit early termination policy. The model performance is scored via accuracy. The hyperparameter tuner finds optimal values for the C parameter (related to inverse regularization and overfitting) as well as max_iter, or the maximum number of iterations allowed for solver convergence in the logistic model.

Random parameter sampling provides a more resource-conservative way to optimize parameter values. Rather than exhaustively test all possible value combinations, such as a grid search, random search can sample (in this case uniformly) across the entire range of parameter value combinations, allowing the parameter search to cover more or less the same parameter space as grid search at a lower cost. While bayesian search methods can use the score of values to update projected distributions, they can still prove to be more costly and less effective than random sampling. The random parameter sampling was conducted over a uniform distribution from 0.005 to 1 for C, and was a discrete choice distribution between 100, 200, 500, and 1000 for max iterations.

The bandit policy utilized prevents the model training from wasting resources once model performance dips below a certain threshold. This threshold is related to the current best performing score. In this case, with a slack factor of 0.2, any subsequently assessed performance that is less than 1/(1.2) or 83% of the current best performance triggers the termination policy. An evaluation delay of 2 trials was chosen for generating a robust set of initial performance metrics. The evaluation interval was set as 2, so that evaluation would not consume resources every trial, but would still sample trials at a high enough frequency so as to track performance dips.

The best performing logistic regression model had parameters of C = 0.9363 abd max_iter=1000.

## AutoML
The Auto ML pipeline tested the accuracy of over 30 models and pipeline combinations. The best performing model was the Voting Ensemble model, with an accuracy of 0.9179. This model is characterized by a voting algorithm that incorporates multiple outputs from a set of several models, which in turn allows it to achieve better performance.

## Pipeline comparison
The difference in accuracy between HyperDrive and AutoML was negligible. Whereas hyperdrive was limited by the framework put forth in the training script, AutoML samples across a broad arrau of differnt types of models and pipelines. AutoML can provide a broader scope for analyzing which aspects/features of a model pipeline work best for your circumstance. The similarity in accuracy suggests that simpler solutions may be more practical, as a simple logistic regression was able to perform nearly as accurately as a much more complex voting ensemble model.

## Future work
As suggested by the AutoML output, class imbalance handling could improve the model's validity. The minority class only represented about 10% of the total training data, and balancing the classes could allow for more proper assessment of the model's performance. In terms of binary classification, analyzing the recall, precision, or f1-score could additionally improve model assessment. Another advancement could be the inclusion of more hyperparameters for tuning the logistic regression, such as differnt norms for the penalty, intercept scaling, and solvers.

## Proof of cluster clean up

![Screen Shot 2021-12-16 at 6 40 20 PM](https://user-images.githubusercontent.com/87383001/146465129-064a0af4-cce4-4e32-9544-512f92ac5f39.png)
