import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, classification_report

from preprocessing import DataPreprocessor, DataTransform
from date_time_feature_extractor import DateTimeFeatureExtractor


class RDSRandomForestClassifier(DataPreprocessor):
    """Class to build random forest classifier model for predicting loan status
    Initially creates data table in the same way as previously, before following through with some daditional preprocessing
    - Yeo-Johnson transformation applied on numeric columns
    - Object columns one hot encoded
    - Datetime features extracted
    Contains functions to train, predict based on current/past data, and to display information about the model.
    Predictions can be accessed via .current_data_with_predictions; feature importances via .feature_importances_df (after training and predicting)
    """
    
    def __init__(self, raw_data):
        super().__init__(raw_data)

        print("Initialising Random Forest Classifier")
        self._preprocessing_steps()
        self.set_excluded_cols()
        # TODO allow for external control of model training?
        self.train_model()
        self.predict_for_current_data()
        # self.print_model_info()
        
    def _preprocessing_steps(self):
        """Skewed columns are not transformed by default in parent so done here
        """
        self.transform_skewed_cols()
        # self.__set_col_types()
        return
        
    def predict_for_current_data(self):
        """Uses trained model to make predictions on loan_status for current data
        """
        # Separate current data
        self.current_data = self.data[self.data['loan_status'].isin(['Current',
                                                                     'Late (31-120 days)',
                                                                     'In Grace Period',
                                                                     'Late (16-30 days)'])]
        # Split target and estimators
        self.current_data_target_column = pd.get_dummies(self.current_data['loan_status'])
        self.current_data_estimators = self.current_data.drop(self.excluded_cols,axis=1)
        
        # Make predictions
        try:
            self.current_data_predictions = self.pipeline.predict(self.current_data_estimators)
            
            # Reverse encode predictions and append back to data
            reverse_encoded_predictions = self.label_encoder.inverse_transform(self.current_data_predictions)
            self.current_data['predicted_loan_status'] = reverse_encoded_predictions
        except AttributeError as e:
            print(f"{e}: please ensure the model is trained before making predictions")
        


    def train_model(self, print_model_stats_flag = True):
        """Builds and trains a Random Forest Classifier based on loans which are fully paid or charged off
        Prints model accuracy and classification report
        """
        
        print("Training Random Forest Classifier")
        # Use only past data for training purposes
        self.past_data = self.data[self.data['loan_status'].isin(['Charged Off',
                                                                  'Fully Paid'])]
        # Separate target and predictor cols
        self.past_data_estimators = self.past_data.drop(self.excluded_cols, axis=1)
        self.past_data_target_column = self.past_data['loan_status']

        # Split training and test sets
        self.X_train, X_test, y_train, y_test = train_test_split(self.past_data_estimators, self.past_data_target_column, test_size=0.33, random_state=42)
        
        self.__set_col_types()
        # Target is not yet encoded
        self.label_encoder = LabelEncoder() # label encoder more suitable since binary option
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Build and fit pipeline
        self.build_pipeline()
        self.pipeline.fit(self.X_train, y_train_encoded)
        
        # Predict using the pipeline
        self.predictions = self.pipeline.predict(X_test)
        # Evaluate the model
        if print_model_stats_flag:
            accuracy = self.pipeline.score(X_test, y_test_encoded)
            print(f"Model accuracy: {accuracy:.3f}")
            print(classification_report(y_test_encoded, self.predictions))
            
    def print_model_info(self, plot_flag = False):
        """Prints information about the trained model, plots 10 most important features
        # TODO more metrics and info to print
        """
        if not self.pipeline:
            print("Model has not been trained: please train the model before attempting to access model information")
            return
        print("Model Info:")
        
        # Create dict of feature importances
        transformed_feature_names = []
        for name, trans, columns in self.preprocessor.transformers_:
            if hasattr(trans, 'get_feature_names_out'):
                if isinstance(trans, OneHotEncoder):
                    encoded_cols = trans.get_feature_names_out(input_features=columns)
                    transformed_feature_names.extend(encoded_cols)
                else:
                    transformed_feature_names.extend(columns)  # For non-encoded columns

        # Get feature importances
        feat_importances = self.pipeline.named_steps['classifier'].feature_importances_

        feature_importances_dict = dict(zip(transformed_feature_names, feat_importances))
        self.feature_importances_df = pd.DataFrame.from_dict(feature_importances_dict, orient='index', columns=['Importance'])
        self.feature_importances_df = self.feature_importances_df.sort_values(by='Importance', ascending=False)

        # Plot 10 most important features
        if plot_flag:
            plt.figure(figsize=(10,8), dpi=150)
            self.feature_importances_df.head(10).plot(kind='barh')
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importances')
            plt.show()

# TODO this function can probably live happily in a parent class
    # Get columns and split by type - required for passing into model
    def __set_col_types(self):
        """Create arrays containing columns by their type, excludes data not used for estimation e.g. 
        """
        self.numeric_cols = []
        self.object_cols = []
        self.datetime_cols = []
        self.other_cols = []
        for col in self.past_data_estimators.columns:
            if self.past_data_estimators[col].dtype == 'int64' or self.past_data_estimators[col].dtype == 'float64':
                self.numeric_cols.append(col)
            elif self.past_data_estimators[col].dtype == 'object':
                self.object_cols.append(col)
            elif self.past_data_estimators[col].dtype == 'datetime64[ns]':
                self.datetime_cols.append(col)
            else:
                self.other_cols.append(col)
        # self.object_cols.remove('loan_status') # not needed if getting col names from estimators
        
    def build_pipeline(self):
        # Preprocessing steps for different column types
        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.object_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Custom transformer used to split into month and year features
        self.datetime_transformer = Pipeline([
            ('date_feature_extractor', DateTimeFeatureExtractor(year_flag=True, month_flag=True)),
        ])

        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_transformer, self.numeric_cols),
                ('obj', self.object_transformer, self.object_cols),
                ('datetime', self.datetime_transformer, self.datetime_cols),
            ]
        )
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),('classifier', RandomForestClassifier())])
        
    def set_excluded_cols(self):
        """Set columns to exclude as estimators. See below for reasoning.
        """
        self.excluded_cols = ['loan_status', 'member_id', 'loan_id', 
                              'collection_recovery_fee', 'recoveries',
                              'out_prncp','total_rec_prncp',
                              'out_prncp_inv',
                              'total_payment_inv','total_payment',
                              'last_payment_amount']
        # collection_recovery_fee and recoveries: only non-zero if charge off is TRUE - leak info
        # out_prncp and _inv are 0 for all paid off or charged off loans
        # 'total_payment_inv','total_payment' - in past data these will be strong predictors as loans where payment
        # is less than the loan amount are highly likely charged off - not true for open loans where payment is ongoing
        # last payment amount - seems like a large number of loans are repaid in a single payment, i.e. data present in past loans missing from current loans


if __name__ == "__main__":
    raw_data = pd.read_csv('RDS_data.csv',index_col=0)
    forest_test = RDSRandomForestClassifier(raw_data)
    forest_test.print_model_info(plot_flag=True)