import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import yeojohnson

class DataTransform():
    """Class to clean up RDS data, including:
    - fixing typos in col names
    - fixing data types
    Takes raw data as downloaded as input
    """
    def __init__(self, raw_data):
        print("Processing data...")
        self.data = raw_data
        self.missing_columns = []
          
        self._rename_columns()
        self._change_col_dtypes()


    def _check_cols_exist(self, column_names)->None:
        """Checks if columns exist within raw data, notes any missing columns for reference

        """
        col_intersect = np.intersect1d(self.data.columns, column_names)
        missing_cols = list(set(column_names)-set(col_intersect))
        if missing_cols:
            self.missing_columns.extend(missing_cols)

    
    def _rename_columns(self):
        """Renames several columns for ease of understanding or correct typos
        """
        column_rename_dict = {"id":"loan_id",
                              "grade":"loan_grade",
                              "instalment":"installment",
                              "sub_grade":"loan_sub_grade"}
        self.data.rename(columns=column_rename_dict, inplace=True)
        return
    
    def _change_col_dtypes(self):
        """Corrects a number of data types of various columns based on usage and/or memory
        """
        # set columns to change types and check they exist in the dataframe, then change column dtypes
        self.__set__int_to_obj_cols()
        self.__set__float_to_int_cols()
        self.__set__obj_to_bool_cols()
        self.__set__obj_to_datetime_cols()
        
        if self.missing_columns:
            print("WARNING: expected columns are missing in the loaded data! Missing columns:")
            print(self.missing_columns)
            print("Data table may contain incorrect data types!")
            
            # drop missing columns from columns to change, before changing dtypes
            self.int_to_obj_cols = [col for col in self.int_to_obj_cols if col not in self.missing_columns]
            self.float_to_int_cols = [col for col in self.float_to_int_cols if col not in self.missing_columns]
            self.obj_to_bool_cols = [col for col in self.obj_to_bool_cols if col not in self.missing_columns]
            self.obj_to_datetime_cols = [col for col in self.obj_to_datetime_cols if col not in self.missing_columns]
            
            
        self.data[self.int_to_obj_cols] = self.data[self.int_to_obj_cols].astype(object)

        # needs nan values set to placeholder         
        for col in self.float_to_int_cols:
            self.data[col] = self.data[col].fillna(-1).astype(int)
            self.data[col] = self.data[col].replace(-1, np.nan)
        
        self.data[self.obj_to_bool_cols] = self.data[self.obj_to_bool_cols].astype(bool)
        
        # self.data[self.obj_to_datetime_cols] = self.data[self.obj_to_datetime_cols].apply(lambda col: pd.to_datetime(col, format='%Y-%m-%d', errors='coerce'))
        self.data[self.obj_to_datetime_cols] = self.data[self.obj_to_datetime_cols].apply(lambda col: pd.to_datetime(col, format='%b-%Y', errors='coerce'))
        
        # special case for employment length
        self.data['employment_length'] = self.data['employment_length'].replace({"<1 year": "0", "10+ years": "15"})
        # Extract numeric values and convert to integers
        self.data['employment_length'] = self.data['employment_length'].str.extract('(\d+)').astype(float).fillna(0).astype(int)

        return
    
    # functions to handle correcting data types of various columns - purpose should be self explanatory
    def __set__int_to_obj_cols(self):
        self.int_to_obj_cols = ["loan_id","member_id",
                                "policy_code"]
        self._check_cols_exist(self.int_to_obj_cols)    
    
    
    def __set__float_to_int_cols(self):
        self.float_to_int_cols = ["mths_since_last_record",
                                  "collections_12_mths_ex_med",
                                  "mths_since_last_major_derog",
                                  ]
        self._check_cols_exist(self.float_to_int_cols)    
    
    def __set__obj_to_bool_cols(self):
        self.obj_to_bool_cols = ["payment_plan"]
        self._check_cols_exist(self.obj_to_bool_cols)
        
    def __set__obj_to_datetime_cols(self):
        self.obj_to_datetime_cols = ["issue_date",
                                     "earliest_credit_line",
                                     "last_payment_date",
                                     "next_payment_date",
                                     "last_credit_pull_date",
                                     ]
        self._check_cols_exist(self.obj_to_datetime_cols)


class DataPreprocessor(DataTransform):
    """Imputation class. Handles several aspects of data cleaning:
    - Drops columns containing too many null values (>50%)
    - Imputes values for columns containing moderate number of nulls (1 - 15%)
    - Drops other rows containing nulls
    - Performs Yeo-Johnson transform on skewed numeric columns
    - Drops outliers based on threshold of 5 standard deviations from the mean value
    - Drops columns with a correlation score of >0.9 with other columns
    """
    def __init__(self, raw_data):
        super().__init__(raw_data)
        
        # don't necessarily want to do this yet
        # self.one_hot_encode_cols()
        
        self.__handle_nulls()
        
        # self.__transform_skewed_cols()
        self.__drop_outliers()
        
        print("Before vs after")
        print(self.data.shape)
        self.__drop_correlated_cols()
        print(self.data.shape)
        
        print("Data cleaning complete")
        return
        
    
    def __handle_nulls(self):
        """Function to drop columns, rows, or impute based on nulls
        """
        # Handle columns with very large number of nulls
        self.__set_cols_to_drop()
        # self.data = self.data.drop(columns=self.cols_to_drop)
        
        # Handle columns with low number of nulls
        self.__set_rows_to_drop()
        self.data = self.data.dropna(subset=self.rows_to_drop, how='any')

        # Handle the rest with imputation, uses random forest classifier or regressor
        self.__set_cols_to_impute()
        for impute_col in self.cols_to_impute:
            self.__impute_nulls_predict(impute_col)
            
        return
    
    def __impute_nulls_predict(self, column_name):
        # idea taken from a kaggle challenge from a few months ago
        set_classifier_flag = 0
        # split data into train/test and null sets - based on target column containing null values
        null_data = self.data[self.data[column_name].isnull()]
        non_null_data = self.data.dropna(subset=[column_name])

        # Feature columns - just numerics works fine for now
        # feature_cols = [col for col in self.data.columns if col != column_name]
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Train model on non-null data
        X_features = non_null_data[numeric_columns]
        y_target = non_null_data[column_name]

        # Model training

        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_columns)])
        
        if y_target.dtype == object:
            model = RandomForestClassifier()
            print(f"Using classifier to impute for column: {column_name}")
            y_target = pd.get_dummies(y_target)
            new_columns = y_target.columns.tolist()
            set_classifier_flag = 1
        elif y_target.dtype == int or y_target.dtype == float:
            model = RandomForestRegressor()
            print(f"Using regressor to impute for column: {column_name}")
        else:
            print(f"Incorrect data type encountered in column to be imputed: {column_name} ; {y_target.dtype}")

        X_features = preprocessor.fit_transform(X_features)
        # Train the model
        model.fit(X_features, y_target)

        # Impute missing values
        X_null = null_data[numeric_columns]
        X_null = preprocessor.transform(X_null)
        imputed_values = model.predict(X_null)

        # Replace missing values with imputed values in the original dataset
        if set_classifier_flag:
            self.data = pd.get_dummies(self.data, columns=[column_name])
            self.data.loc[self.data.index.isin(null_data.index), new_columns] = imputed_values
        else:
            self.data.loc[self.data.index.isin(null_data.index), column_name] = imputed_values
        print(f"Imputation for {column_name} complete!")

    def __transform_skewed_cols(self):
        """Apply Yeo-Johnson transform to set columns
        """
        self.__set_skewed_cols()
        print("Transforming skewed columns...")
        for column in self.skewed_cols:
            if self.data.dtypes[column] == float or self.data.dtypes[column] == int:
                try:
                    transformed_data, _ = yeojohnson(self.data[column].dropna())
                    self.data[column] = transformed_data
                except ValueError:
                    print(f"Transformation failed for {column}")
                    pass
        
    def __drop_outliers(self, threshold=5):
        """Drop rows if a value contains a value over 5 standard deviations from the mean
        """
        for col in self.data:
            if self.data.dtypes[col] == int or self.data.dtypes[col] == float:
                mean = self.data[col].mean()
                std = self.data[col].std()
                outlier_indices = (self.data[col] - mean).abs() > (threshold * std)
                self.data = self.data[~outlier_indices]
    
    def __drop_correlated_cols(self):
        """Drops overly correlated columns: threshold = 0.9 or greater correlation
        """
        print("Dropping overly correlated columns")
        self.__set_correlated_cols()
        self.data = self.data.drop(self.correlated_cols, axis=1)
        
    def one_hot_encode_cols(self):
        """Convert object columns to encoded data.
        """
        self.OHE_cols = ['loan_grade',
                         'loan_sub_grade',
                         'home_ownership',
                         'verification_status',
                         'loan_status',
                         'purpose',
                         'application_type']
        super()._check_cols_exist(self.OHE_cols)
        if self.missing_columns:
            print("WARNING: expected columns are missing in the loaded data! Missing columns:")
            print(self.missing_columns)
            self.OHE_cols = [col for col in self.int_to_obj_cols if col not in self.missing_columns]
        self.data = pd.get_dummies(self.data, columns=self.OHE_cols)
    
    def __set_rows_to_drop(self):
        """Set columns with low number of nulls to drop those rows
        """
        self.rows_to_drop = ["collection_recovery_fee",
                             "last_payment_date",
                             "last_credit_pull_date",
                             "collections_12_mths_ex_med"]
        super()._check_cols_exist(self.rows_to_drop)
        
    def __set_cols_to_drop(self):
        """Set columns with large numbers of nulls
        """
        self.cols_to_drop = ["mths_since_last_delinq",
                             "mths_since_last_record",
                             "next_payment_date",
                             "mths_since_last_major_derog"]
        super()._check_cols_exist(self.cols_to_drop)
        
    def __set_cols_to_impute(self):
        """Set columns for data imputation
        """
        self.cols_to_impute = ["funded_amount",
                               "term",
                               "int_rate"]
        super()._check_cols_exist(self.cols_to_impute)
        
    # def set_cols_to_gate(self):
    #     self.cols_to_gate =['delinq_2yrs',
    #                         'inq_last_6mths',
    #                         'out_prncp',
    #                         'out_prncp_inv',
    #                         'total_rec_late_fee',
    #                         'recoveries',
    #                         'collection_recovery_fee',
    #                         'collections_12_mths_ex_med']
    #     super()._check_cols_exist(self.cols_to_impute)
            
    def __set_skewed_cols(self):
        """Set columns for Yeo-Johnson transform
        """
        self.skewed_cols = ['loan_amount',
                            'funded_amount',
                            'funded_amount_inv',
                            'int_rate',
                            'installment',
                            'annual_inc',
                            'dti',
                            'open_accounts',
                            'total_accounts',
                            'total_payment',
                            'total_payment_inv',
                            'total_rec_prncp',
                            'total_rec_int',
                            'last_payment_amount',
                            ]
        super()._check_cols_exist(self.skewed_cols)
        
    def __set_correlated_cols(self):
        """Set columns to drop, based on investigation in notebook to find most correlated columns
        """
        self.correlated_cols = ['60 months']
            # 'funded_amount', 
            #                     'funded_amount_inv',
            #                     'out_prncp_inv',
            #                     'total_payment_inv',
            #                     'installment',
            #                     'total_rec_prncp',
        super()._check_cols_exist(self.correlated_cols)
        
        
        
        
        
        
if __name__ == "__main__":
    data = pd.read_csv("RDS_data.csv", index_col=0)
    data_transformer = DataPreprocessor(data)
    # data_transformer = DataTransform(data)
    print(data_transformer.data.dtypes)
    # print(data_transformer.data['last_payment_date'].head())
    data_transformer.data.to_csv("datetime_test_not.csv")
    # print(data_transformer.data.shape)
    # print(data_transformer.data.isnull().sum().sum())