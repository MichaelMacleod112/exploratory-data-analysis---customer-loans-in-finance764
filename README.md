# Exploratory Data Analysis Project:  Customer Loans in Finance

## Contents
- [Introduction](##introduction)
- [Findings](##findings)
- [File Contents](##filecontents)
- [File Structure Tree](##file-structure-tree)
- [Usage](##usage)
- [Data Key](##datakey)
- [License](##license)

## Introduction
AI Core project in exploratory data analysis of financial data. Downloads online data table (provided user has credentials) and performs data cleaning and preprocessing in several stages. Contains custom feature extraction class and Random Forest Classifier model to make predictions on loans being paid back or charged off, and returns information on the strongest predictors of this being the case.

## Findings

Random forest classifier used to determine feature importances in predicting future loan status, i.e. full paid or charged off. Results below show Annual Income as the most important feature, an easy to understand result.

   ![Feature Importances](/screenshots/feature_importances.png)

## File Contents

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

## File Structure Tree

```
.
├── .gitignore
├── __pycache__
├── screenshots
│   ├── feature_importances.png
├── data_examining.ipynb
├── data_imputation.ipynb
├── data_queries.ipynb
├── data_stats.py
├── date_time_feature_extractor.py
├── db_utils.py
├── environment.yaml
├── preprocessing.py
├── random_forest_class.py
├── README.md
├── screenshots
└── task.txt
```

## Usage

#### 1. Clone the Repository

Navigate to the [GitHub repository](https://github.com/MichaelMacleod112/exploratory-data-analysis---customer-loans-in-finance764) and click on the "Code" button. Copy the repository URL and use the git clone command to clone the repository to your local machine.

#### 2. Navigate to the Project Directory

Use your terminal or command prompt to navigate to the directory where you cloned the repository.

#### 3. Create Conda Environment

- **Install Conda:**
  Ensure that you have Conda installed on your machine. If not, download and install Miniconda or Anaconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

- **Create Environment:**
  Open your terminal and run the following command to create a Conda environment using the provided YAML file:
```bash
  - conda env create -f environment.yaml
```

#### 4. Activate Conda Environment

Once the environment is created, activate it using the following command:

```bash
- conda activate EDA_project_env
```

#### 5. Verify Environment

Check that your command prompt or terminal displays the activated environment name.

#### 6. Run the Project

Now, you can run the project or execute any scripts within the activated Conda environment.

#### 7. Deactivate Environment

After finishing your work, deactivate the Conda environment using the following command:

```bash
- conda deactivate
```

Feel free to reach out if you encounter any issues or have questions.

## Loan Data Key Mapping and Data Type Adjustments

Below is a guide to the loan data fields, including key mapping and suggested data type adjustments.

- `id`: Unique id of the loan (mapped to `loan_id`)
- `member_id`: Id of the member who took out the loan
- `loan_amount`: Amount of loan the applicant received (data type adjusted: int to float)
- `funded_amount`: The total amount committed to the loan at that point in time
- `funded_amount_inv`: The total amount committed by investors for that loan at that point in time
- `term`: The number of monthly payments for the loan (data type adjusted: obj to time period)
- `int_rate`: Interest rate on the loan
- `instalment`: The monthly payment owned by the borrower (typo corrected: `installment`)
- `grade`: Loan company (LC) assigned loan grade (mapped to `loan_grade`)
- `sub_grade`: LC assigned loan sub-grade
- `employment_length`: Employment length in years (data type adjusted: obj to time period)
- `home_ownership`: The home ownership status provided by the borrower
- `annual_inc`: The annual income of the borrower
- `verification_status`: Indicates whether the borrower's income was verified by the LC or the income source was verified
- `issue_date`: Issue date of the loan (data type adjusted: obj to datetime)
- `loan_status`: Current status of the loan
- `payment_plan`: Indicates if a payment plan is in place for the loan (data type adjusted: obj to bool)
- `purpose`: A category provided by the borrower for the loan request
- `dti`: A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income
- `delinq_2yr`: The number of 30+ days past-due payments in the borrower's credit file for the past 2 years
- `earliest_credit_line`: The month the borrower's earliest reported credit line was opened (data type adjusted: obj to datetime)
- `inq_last_6mths`: The number of inquiries in the past 6 months (excluding auto and mortgage inquiries)
- `mths_since_last_record`: The number of months since the last public record (data type adjusted: float to int or datetime period)
- `open_accounts`: The number of open credit lines in the borrower's credit file
- `total_accounts`: The total number of credit lines currently in the borrower's credit file
- `out_prncp`: Remaining outstanding principal for the total amount funded
- `out_prncp_inv`: Remaining outstanding principal for a portion of the total amount funded by investors (note: similarity)
- `total_payment`: Payments received to date for the total amount funded
- `total_rec_int`: Interest received to date
- `total_rec_late_fee`: Late fees received to date
- `recoveries`: Post charge-off gross recovery
- `collection_recovery_fee`: Post charge-off collection fee
- `last_payment_date`: Date on which the last monthly payment was received (data type adjusted: obj to datetime)
- `last_payment_amount`: Last total payment amount received
- `next_payment_date`: Next scheduled payment date (data type adjusted: obj to datetime)
- `last_credit_pull_date`: The most recent month LC pulled credit for this loan (data type adjusted: obj to datetime)
- `collections_12_mths_ex_med`: Number of collections in the last 12 months, excluding medical collections (data type adjusted: float to int)
- `mths_since_last_major_derog`: Months since the most recent 90-day or worse rating (data type adjusted: float to int)
- `policy_code`: Publicly available policy_code=1 new products not publicly available policy_code=2 (data type adjusted: int to object)
- `application_type`: Indicates whether the loan is an individual application or a joint application with two co-borrowers


### License information
[MIT](https://choosealicense.com/licenses/mit/)
