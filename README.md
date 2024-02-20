# Exploratory Data Analysis Project:  Customer Loans in Finance

## Contents
- [Introduction](#introduction)
- [Project Brief](#project-brief-business-intelligence-enhancement-with-power-bi)
- [Business Intelligence Report](#business-intelligence-report-dummy-sales-data)
- [Report Structure](#report-structure)
- [File Contents](#file-contents)
- [File Structure Tree](#file-structure-tree)
- [Usage](#usage)
- [Report Contents](#report-contents)
- [License](#license)

## Introduction
AI Core project in exploratory data analysis of financial data. Downloads online data table (provided user has credentials) and performs data cleaning and preprocessing in several stages. Contains custom feature extraction class and Random Forest Classifier model to make predictions on loans being paid back or charged off, and returns information on the strongest predictors of this being the case.


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
  - conda env create -f env.yml
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
