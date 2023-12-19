import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin): # NOTE CODE REVIEW - Assuming BaseEstator and TransformerMixin will be used when this class' methods are called, inheritance makes sense
    """Class to extract features from datetime columns. 
    Extracts features to numerical forms but also cyclical features (i.e. month of year features preserved)

    Args:
        BaseEstimator : ensures estimator complies with expectations of sklearn 
        TransformerMixin : inherits fit and transform hooks to create a unified fit_transform function
    """
    
    def __init__(self, year_flag=False, month_flag=False, day_flag=False, dayofweek_flag=False, 
                 quarter_flag=False, cyclic_flag=False):
        """Flags set on which features to extract from datetime cols
        TODO make this more robust - dataframe may have multiple types of datetime column, not sure how feasible this is within timeframe

        """
        self.year_flag = year_flag
        self.month_flag = month_flag
        self.day_flag = day_flag
        self.dayofweek_flag = dayofweek_flag
        self.quarter_flag = quarter_flag
        self.cyclic_flag = cyclic_flag
        
    def fit(self, X:pd.DataFrame, y=None):# NOTE CODE REVIEW - df might be a better variable name rather than X
        return self


    def transform(self, X, y=None): 
        """Transform datetime columns into multiple usable feature columns

        """
        X_copy = X.copy() # NOTE CODE REVIEW - Similarly to previous note, df_copy might be a better option than X_copy, assuming you replace X with df
        for column in X_copy: 
            # dict provides most flexible way to handle flags
            transformations = {
                'year': X_copy[column].dt.year if self.year_flag else None,
                'month': X_copy[column].dt.month if self.month_flag else None,
                'day': X_copy[column].dt.day if self.day_flag else None,
                'dayofweek': X_copy[column].dt.dayofweek if self.dayofweek_flag else None,
                'quarter': X_copy[column].dt.quarter if self.quarter_flag else None,
                'cyclic_sin' : np.sin(2 * np.pi * X_copy['month'] / 12) if self.cyclic_flag else None,
                'cyclic_cos' : np.cos(2 * np.pi * X_copy['month'] / 12) if self.cyclic_flag else None
            }

            try:
                X_copy[column] = pd.to_datetime(X_copy[column])
            except ValueError:
                print(f"Value Error: \"{column}\" column is not compatible with datetime format!")
            except KeyError:
                print(f"Key Error: \"{column}\" not found in dataframe!") # NOTE CODE REVIEW - Great key and value errors

            # impute newly extracted features
            for key, transform in transformations.items():
                if transform is not None:
                    
                    X_copy[column + f'_{key}'] = transform
            # drop the pre-transform column
            X_copy = X_copy.drop(column, axis=1) # NOTE CODE REVIEW - Could replace the old column with the new one but this will also change the name, makes sense
        return X_copy

# NOTE CODE REVIEW - No real improvements in this file just some of my own opinions, amazing class !