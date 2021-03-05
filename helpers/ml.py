from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def importance_fn(model):
    """
    Parameters
    ----------
        model : Model that has been fitted on the training data.

    Returns
    -------
        feature_importance: Calculate importance scores of each feature that
        the energy burden depends on.
    """
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    feature_importance = [value for value in importance]

    return feature_importance


def random_forest_regression(X_train, X_test, y_train, y_test):
    """Find the feature importance that random forest predicts.
    Decision Trees are prone to overfitting so Random Forest algorithm was implemented.
    RF has enough capacity to completely memorize the training set
    but it can still generalize well enough due to built-in bagging of random forests.

    Returns
    -------
    rf_feature_importance : Calculate importance scores of each feature that
    the energy burden depends on.
    rf_train_accuracy : Accuracy of the Random Forest model on training data.
    rf_test_accuracy : Accuracy of the Random Forest model on test data.
    """
    model = RandomForestRegressor()
    # fit the model
    model.fit(X_train, y_train)
    rf_train_accuracy = model.score(X_train, y_train)
    rf_test_accuracy = model.score(X_test, y_test)
    print("Random forest train accuracy: %0.3f" % rf_train_accuracy)
    print("Random forest test accuracy: %0.3f" % rf_test_accuracy)

    rf_feature_importance = importance_fn(model)

    return rf_feature_importance, rf_train_accuracy, rf_test_accuracy


def xg_boost(X_train, X_test, y_train, y_test):  ###
    """
    XGBoost for feature importance on a regression problem.
    Just like gradient boosting, XGBoost comprises an ensemble method
    XGBoost comes with an additional custom regularization term in the
    objective function which is why it outperforms other implementations
    of gradient tree boosting.
    Returns
    -------
    xgboost_feature_importance : Calculate importance scores of each feature that
    the energy burden depends on.
    xgboost_train_accuracy : Accuracy of the XGBoost model on training data.
    xgboost_test_accuracy : Accuracy of the XGBoost model on test data.
    """
    # define the model
    model = XGBRegressor()
    # fit the model
    model.fit(X_train, y_train)
    xgboost_train_accuracy = model.score(X_train, y_train)
    xgboost_test_accuracy = model.score(X_test, y_test)
    print("XGBoost train accuracy: %0.3f" % xgboost_train_accuracy)
    print("XGBoost test accuracy: %0.3f" % xgboost_test_accuracy)
    xgboost_feature_importance = importance_fn(model)

    return xgboost_feature_importance, xgboost_train_accuracy, xgboost_test_accuracy
