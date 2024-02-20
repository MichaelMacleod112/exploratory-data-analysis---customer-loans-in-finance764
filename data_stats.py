import pandas as pd
import seaborn as sns
import scipy.stats as stats

from matplotlib import pyplot as plt

class DataFrameInfo():
    """
    Contains methods to extract some useful stats from a dataframe, which should be passed in at initialisation
    """
    def __init__(self, data: pd.DataFrame()):
        self.data = data
    
    def print_all_info(self):
        self.describe_columns()
        self.extract_statistics()
        self.get_shape
        self.get_null_counts()
    
    def describe_columns(self):
        print("Datatypes:")
        print(self.data.dtypes) 
        # return self.data.dtypes

    def get_shape(self):
        print("Table shape:")
        print(self.data.shape())
        
    def extract_statistics(self):
        print("Overall table statistics")
        print(self.data.describe())
        
    def get_null_counts(self):
        print("Null counts")
        print(self.data.isnull().sum())
        
        
class Plotter():
    """
    Class to store plotting and data visualisation functions
    """
    def __init__(self, df):
        self.df = df
    
    def plot_hist(self, col):
        """Plots histogram of a column col in the dataframe
        """
        plt.figure(figsize=(7,4),dpi=150)
        sns.histplot(self.df[col])
        
        plt.title(f"{col} data distribution", color='white')

    def plot_skew(self, col):
        """
        Displays data distribution histogram alongside a Q-Q plot to show skew

        Args:
            col (): df column to visualise
        """
        data = self.df[col]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{col} distribution and skew", color='white')
        # Plot histogram
        axes[0].hist(data, bins=20, alpha=0.7)
        axes[0].set_title('Histogram', color='white')

        # Plot Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot', color='white')

        plt.tight_layout()
        plt.show()
    
    def plot_correlation(self):
        """
        Visualise data correlation on heatmap
        """
        plt.figure(figsize=(7,4),dpi=150)
        sns.heatmap(self.df.corr(), cmap = 'coolwarm')