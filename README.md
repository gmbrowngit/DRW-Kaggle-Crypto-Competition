The data set is bid-ask quantities every minute for a couple years of crypto trades, plus hundreds of DRW's proprietary predictive features
and the labels are the price change in that minute. The dataset is available at the kaggle link.

Preprocessing.ipynb- This contains the cleaning and feature selection of the training data. First we visualize the occurences of values in 
each feature and notice that many features are mostly repeating values. We drop these features. Then we do some correlation heatmaps and find
many features are highly correlated. Among groups of highly correlated features, we only keep the one that is most correlated with the labels.
Finally, we use Lasso regression to further select features (remove those with zero coefficients).

RidgeProcessed.ipynb- Here we apply Ridge Regression with sklearn to the processed data, we tune the ridge coefficient with cross-validation.

RidgeEnsemble.ipynb- Here I tried the ensemble method of bagging, but realized that this is not a good choice for regression on such a large
data set as regression is so stable that it does not reduce the correlation between learners. Perhaps smaller bags, or training on random
subsets of features may be a better approach.
