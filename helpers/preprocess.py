import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import globals

avg_values_list = ["avg_cost_electricity", "avg_cost_gas", "avg_cost_water"]


class Preprocess():
    """This class is used to preprocess/wrangle our dataframe"""

    def __init__(self, df):
        self.df = df

    def drop_na(self):
        """Drop null values since not too many"""
        self.df = self.df.dropna()
        return len(self.df)

    def replace_negatives(self):
        """Replace negative numbers with NaN"""
        self.df[avg_values_list] = self.df[avg_values_list].mask(self.df[avg_values_list] < 0)

    def drop_irrelevant(self):
        """
		1. Drop state because it is a sub category of geography,
		2. Drop year because unnecessary
		3. Drop asthma_rate since 85% of are Nans
		"""
        self.df = self.df.drop(['id2', 'state', 'year', 'asthma_rate'], axis=1)

    def extract_cs(self):
        """Feature engineer geography to extract the county, state from it"""
        self.df['geography'] = self.df['geography'].apply(lambda geography: ",".join(geography.split(",")[-2:]))

    def make_categorical_features(self):
        """One hot encode geography since it is a categorical variable"""
        self.df = pd.get_dummies(self.df[[x for x in self.df.columns]])

    # Drop geography column
    # self.df = pd.concat([self.df.drop('geography', axis=1), dummies], axis=1)

    def convert_decimal(self):
        """Convert eviction, poverty rate to decimal"""
        self.df[["eviction_rate",
                 "poverty_rate"]] = \
            self.df[["eviction_rate",
                     "poverty_rate"]] / 100

    def create_energy_burden(self):
        """Create energy burden column i.e. (avg cost of electricity + avg cost of gas) / median income"""
        self.df["energy_burden"] = (self.df.loc[:, [avg_values_list[0],
                                                    avg_values_list[1]]].sum(axis=1)) \
                                   / self.df["median_income"]

    def replace_median(self):
        """Replace median income with log(median income) to break linearity b/w income and energy burden"""
        self.df["median_income"] = np.log10(self.df["median_income"])

    def split_data(self, X, y):
        """Split train/test data"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Number of training data:", y_train.shape[0])
        print("Number of test data:", y_test.shape[0])
        return X_train, X_test, y_train, y_test


def pre_process_data(df, original_data_len):
    """Instantiate the Preprocess class and run required preprocessing"""
    prep = Preprocess(df)

    prep.replace_negatives()
    prep.drop_irrelevant()

    if globals.drop_na:
        new_data_size = prep.drop_na()
        print(f"Number of data points remaining is {new_data_size} "
              f"which is {round(((new_data_size / original_data_len) * 100), 2)} " "percent of the original data set")
        prep.extract_cs()

        prep.make_categorical_features()

        prep.convert_decimal()

        prep.create_energy_burden()

        prep.replace_median()

    return prep
