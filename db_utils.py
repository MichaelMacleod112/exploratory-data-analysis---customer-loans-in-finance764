import sys
import pandas as pd
import yaml
from sqlalchemy import create_engine

class RDSDatabaseConnector():
    """Holds functionality to access database server holding RDS data.
    Functions to create an SQL engine, download the tabular data, and save to a csv file.

    """
    
    def __init__(self):
        self.credentials = self._load_credentials()
        
        self.__post_init__()
    
    def __post_init__(self):
        self.database_url = f"postgresql://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}"
        self._initialise_engine()
    
    def _load_credentials(self)->dict:
        try:
            credentials_path = 'credentials.yaml'
            with open(credentials_path, 'r') as file:
                credentials = yaml.safe_load(file)
                return credentials
        except FileNotFoundError:
            print("Error: database credentials file not found")
            return None
        
    def _initialise_engine(self)->None:
        try:
            self.engine = create_engine(self.database_url)
        except Exception as e:
            print(f"Error initialising SQL engine: {e}")
            sys.exit(1)
        
    def _extract_RDS_data(self, table_name = "loan_payments")->pd.DataFrame:
        """_summary_

        Args:
            table_name (str, optional): _description_. Defaults to "loan_payments".

        Returns:
            pd.DataFrame: _description_
        """
        # Connect to database
        with self.engine.connect() as connection:
            # Execute SQL query to fetch data
            query = f"SELECT * FROM {table_name}"
            try:
                result_proxy = connection.execute(query)
                dataframe = pd.DataFrame(result_proxy.fetchall(), columns=result_proxy.keys())
            except Exception as e:
                print(e)
                sys.exit(1)

            return dataframe
        
    def save_data_as_csv(self)->None:
        self._extract_RDS_data().to_csv("RDS_data.csv")
        
# print(RDSDatabaseConnector.load_credentials())
if __name__ == "__main__":
    rds_inst = RDSDatabaseConnector()
    rds_inst.save_data_as_csv()