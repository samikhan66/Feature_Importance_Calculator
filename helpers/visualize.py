import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualize():
    def __init__(self, df):
        self.df = df

    def visualize_attributes(self):
        """
        Visualize the attributes to get a sense of the data
        """
        self.df.hist(bins=50, figsize=(20, 15))
        plt.show()

    def visualize_correlation_plot(self):
        """# Heatmap to visualize the correlated and uncorrelated data"""
        corr_ = self.df.corr()["energy_burden"].sort_values(ascending = False)
        corr_ = self.df.corr()
        # visualise the data with seaborn
        mask = np.triu(np.ones_like(corr_, dtype=bool))
        sns.set_style(style='white')
        f, ax = plt.subplots(figsize=(12, 15))
        cmap = sns.diverging_palette(10, 250, as_cmap=True)
        sns.heatmap(corr_, mask=mask, cmap=cmap,
                    square=True, annot=True,
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.show()
        

def visualize_data(df):
    viz = Visualize(df)
    viz.visualize_attributes()

def visualize_corr_plot(df):
    viz = Visualize(df)
    viz.visualize_correlation_plot()

def visualize_features(name, labels, values):
    plt.figure(figsize=(20, 8))
    plt.title(name + " Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.bar([label for label in labels], [v for v in values])
    plt.show()

def visualize_accuracy(rf_train_accuracy, rf_test_accuracy, xgboost_train_accuracy, xgboost_test_accuracy):
    N = 2
    rf_accuracies = (rf_train_accuracy, rf_test_accuracy)
    xgboost_accuracies = (xgboost_train_accuracy, xgboost_test_accuracy)

    ind = np.arange(N)
    width = 0.1
    plt.bar(ind, rf_accuracies, width, label='Random Forest')
    plt.bar(ind + width, xgboost_accuracies, width, label='XGBoost')

    plt.ylabel('Accuracy')
    plt.title('Accuracies by algorithms')

    plt.xticks(ind + width / 2, ('Train Accuracy', 'Test Accuracy'))
    plt.legend(loc='best')
    plt.show()
    
