# DEVELOPED BY SAMI KHAN FOR GREENLINK ANALYTICS
# FEATURE IMPORTANCE PACKAGE
# DATE 02/14/2021

# Problem: Build a model that explains how some of the indicators influence energy burden
# !/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import globals

from helpers import preprocess, visualize, dnn, ml, exploratory_analysis, recursive_feature


def main():
    """Main function"""
    # Making a list of missing value types
    missing_values = ["n/a", "na", "--"]
    # Read csv
    df = pd.read_csv(globals.csv_path, na_values=missing_values)
    # Find the length of csv
    original_data_len = len(df)
    print(f"There are {original_data_len} rows in the data set\n")

    # Select whether to choose a combined data set or state wise data set
    # Combined data set has all county, states combined
    # State data set will only include the corresponding data to that state i.e. Colorado or Georgia
    # Default set to "Combined"
    # ToDo: Currently, only Combined works but need to implement automatic feature selection for individual states
    if globals.dataset_option == "Combined":
        pass
    elif globals.dataset_option == "Colorado":
        df = df[df.state == "CO"]
    elif globals.dataset_option == "Georgia":
        df = df[df.state == "GA"]
    else:
        pass

    # Perform exploratory data analysis
    describe_data = exploratory_analysis.df_stats(df)

    find_missing_values = exploratory_analysis.df_missing(df)
    # Result shows that 85% of asthma_rate are Nans

    visualize.visualize_data(df)
    # The histograms portray some of the standardizations that need to take place:
    # 1. The indicators need to be converted to the same units (decimals).
    # 2. The negative values in average cost columns need to be replaced with NaN.

    # Preprocess data
    prep = preprocess.pre_process_data(df, original_data_len)
    # Visualize correlation plot, default set to True
    if globals.visualize_correlation:
        visualize.visualize_corr_plot(df)

    df = prep.df

    X = df.drop(['energy_burden'], axis=1)
    y = df['energy_burden']
    # Find features through recursive feature elimination, default set to True
    # Change correlation_plot = False in option.ini to True to test it out.

    # Select features predicted by recursive feature elimination
    if globals.feature_selection_method == "rfe":
        top_features = recursive_feature.recursive_feature_elimination(X, y)
        print(top_features)
    # Otherwise pick features manually
    else:
        pass
    # Recursive feature elimination resulted that the optimal number of features is three.
    # However, I will include top 5 features for the sake of comparison. i.e. median_income,
    # avg_cost_electricity, avg_cost_gas, avg_cost_water, percent_black

    # ToDo: Automate the selection process where the features predicted by RFE are directly included as input variables
    # Currently, this is being done manually
    X = df.drop(['percent_renter', 'percent_asian', 'percent_hispanic', 'percent_multifamily_housing',
                 'poverty_rate', 'eviction_rate', 'energy_burden',
                 'geography_ DeKalb County, Georgia', 'geography_ Denver County, Colorado',
                 'geography_ Fulton County, Georgia'], axis=1)
    y = df['energy_burden']
    # Select columns
    cols = [col for col in X.columns]

    # Use preferred algorithm, default set to "average of all algorithms" because the dataset is combined (all states).
    if globals.ml_method == "random_forest":
        X_train, X_test, y_train, y_test = prep.split_data(X, y)
        rf_feature_importance, rf_train_accuracy, rf_test_accuracy = ml.random_forest_regression(X_train, X_test,
                                                                                                 y_train, y_test)
        visualize.visualize_features("Random Forest", cols, rf_feature_importance)

    elif globals.ml_method == "xgboost":
        X_train, X_test, y_train, y_test = prep.split_data(X, y)
        xgboost_feature_importance, xgboost_train_accuracy, xgboost_test_accuracy = ml.xg_boost(X_train, X_test,
                                                                                                y_train, y_test)
        visualize.visualize_features("XGBoost", cols, xgboost_feature_importance)

    elif globals.ml_method == "neural_net":
        X_train, X_test, y_train, y_test = prep.split_data(X, y)
        print(X_train.shape, y_train.shape)
        model = dnn.DNN(X_train, y_train)
        model.build()
        model.compile_and_fit()
        scores = model.compute_scores(X_test, y_test)
        final_score_dnn = [score * 20 for score in scores]
        visualize.visualize_features("Neural Network", cols, final_score_dnn)
        scaler = StandardScaler()
        real_pred = dnn.reverse_target(model.model.predict(scaler.fit_transform(X_test)).ravel(), model.y_train.mean(),
                                       model.y_train.std())
        MAE = mean_absolute_error(y_test, real_pred)
        print("Mean Absolute Error from Neural Net: ", round(MAE, 5))

    elif globals.ml_method == "avg_algorithms":
        X_train, X_test, y_train, y_test = prep.split_data(X, y)
        rf_feature_importance, rf_train_accuracy, rf_test_accuracy = ml.random_forest_regression(X_train, X_test,
                                                                                                 y_train, y_test)
        xgboost_feature_importance, xgboost_train_accuracy, xgboost_test_accuracy = ml.xg_boost(X_train, X_test,
                                                                                                y_train, y_test)
        model = dnn.DNN(X_train, y_train)
        model.build()
        model.compile_and_fit()
        scores = model.compute_scores(X_test, y_test)
        final_score_dnn = [score * 20 for score in scores]
        avg_importance = [np.mean(i) for i in zip(rf_feature_importance, xgboost_feature_importance, final_score_dnn)]
        visualize.visualize_features("Average", cols, avg_importance)
        visualize.visualize_accuracy(rf_train_accuracy, rf_test_accuracy, xgboost_train_accuracy, xgboost_test_accuracy)
        scaler = StandardScaler()
        real_pred = dnn.reverse_target(model.model.predict(scaler.fit_transform(X_test)).ravel(), model.y_train.mean(),
                                       model.y_train.std())
        MAE = mean_absolute_error(y_test, real_pred)
        print("Mean Absolute Error from Neural Net: ", round(MAE, 5))


if __name__ == "__main__":
    main()
