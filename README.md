# Exploratory Data Analysis Project:  Customer Loans in Finance

AI Core project in exploratory data analysis of financial data. Downloads online data table (provided user has credentials) and performs data cleaning and preprocessing in several stages. Contains custom feature extraction class and Random Forest Classifier model to make predictions on loans being paid back or charged off, and returns information on the strongest predictors of this being the case.

## Installation instructions 
Developed on Python 3.8.8. Required libraries: 
- numpy
- pandas
- yaml
- sys
- sqlalchemy
- scipy
- sklearn

These libraries will in future be exported to an env.yaml file to allow easy compatibility with required libraries.


## Usage instructions
Each python (.py) file can be run separately, or run through data_queries.ipynb to see the complete data cleaning, preprocessing and analysis pipeline.

## Contents
### Milestone files
- db_utils.py - contains class to access, download and save financial data table from SQL database stored on AWS
- preprocessing.py - handles data preprocessing and cleaning via classes.
- data_stats.py - contains generalised functions to get statistics and visualise dataset
- data_queries.ipynb - notebook to work through analysis objectives of milestone 4. Can either be run start to finish or observe saved outputs.

### Other files
- credentials.yaml - access credentials to RDS database - not stored in shared project. You will need your own credentials in order to use this project and download the correct data.
- data_examining.ipynb - experimental notebook containing iterative cells working towards changing column data types. Finished implementation found in preprocessing.py - this file is left for legacy purposes for now.
- data_imputation.ipynb - experimental notebook containing iterative cells working towards developing a data imputation methodology. Finished implementation found in preprocessing.py - this file is left for legacy purposes for now.
- random_forest_class.py - creates a Random Forest Classifier class to make predictions on loan_status. Uses past data as a training set and is able to make predictions on current loans. Accessable model parameters include predictions on open loans and more importantly, feature importances. This is to fulfil the milestone requirement of determining which features are better or worse predictors for whether a loan will be paid back or charged off.
- date_time_feature_extractor.py - contains a custom extractor to extract month and year features from datetime columns and convert to columns usable by the classifier


### License information
Michael Macleod 2023

**<--CODE REVIEW NOTES -->**
**<--Good Readme, could add your findings from milestone 4 into here-->**
**<--Could add the data dictionary in here or as a seperate file into the repo but up to you, not a necessity-->**
**<--Small point, you have "access credentials to RDS database - not stored in shared project" but you have the credentials in your task.txt file, unless you are planning on removing that file -->**