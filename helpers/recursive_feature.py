from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
import numpy as np
import pandas as pd

def recursive_feature_elimination(X, y):
    """Use Recursive Feature Elimination to filter out features that are closely correlated and features
       that have very low correlation with the target variable"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # Use automatic feature selection instead of manually specifying number of features.
    rfecv = RFECV(estimator=DecisionTreeRegressor())
    model = DecisionTreeRegressor()
    # Pipeline — since we’ll perform some cross-validation. It’s best practice in order to avoid data leakage.
    pipeline = Pipeline(steps=[('Feature Selection', rfecv), ('Model', model)])
    # Evaluate model
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
    n_scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1,
                               error_score='raise')
    # Report performance
    print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    pipeline.fit(X_train, y_train)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print('\n')
    # Find the 'Ranks' of each feature and sort these values by individual 'Rank'.
    # Rank is the order of importance i.e. Rank 1 (most important), Rank 2 (second order of importance) and so on.
    rf_df = pd.DataFrame(rfecv.ranking_, index=X.columns, columns=['Rank']).sort_values(by='Rank', ascending=True)

    return rf_df.head(rfecv.n_features_)
