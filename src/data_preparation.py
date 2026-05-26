# Standard library imports
import logging
import re
from typing import Any, Dict

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class DataPreparation:
    """
    A class used to clean and preprocess final test score data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        # Drop student id duplicates, opting to keep the assumed latest entry by using keep="last"
        df_cleaned = df.sort_index()
        df_cleaned = df.drop_duplicates(subset="student_id", keep="last")
        logging.info(f"Shape after dropping duplicates: {df_cleaned.shape}")

        # Drop rows with age less than 15, as they are likely to be data entry errors
        df_cleaned = df_cleaned[df_cleaned["age"] >= 15]
        logging.info(f"Shape after filtering by age: {df_cleaned.shape}")

        # Fix NaN values for CCA
        df_cleaned["CCA"].fillna("None", inplace=True)

        # For attendance_rate NaN values, we can fill them with the average attendance rate for students with similar hours_per_week
        df_cleaned['attendance_rate'] = self._fill_missing_attendance_rate(df_cleaned, 'attendance_rate')

        # Fix inconsistent categorical/boolean values for CCA and tuition columns
        df_cleaned['CCA'] = df_cleaned['CCA'].replace({'ARTS': 'Arts', 'SPORTS': 'Sports', 'CLUBS': 'Clubs', 'NONE': 'None'})
        df_cleaned['tuition'] = df_cleaned['tuition'].replace({'Y': 'Yes', 'N': 'No'})

        # Drop final_test_score rows with NaN values
        df_cleaned.dropna(subset=["final_test"], inplace=True)
        logging.info(f"Shape after dropping NaN values: {df_cleaned.shape}")
        
        logging.info("Data cleaning completed.")
        return df_cleaned

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(drop="first",handle_unknown="ignore"))]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder="passthrough",
            n_jobs=-1,
        )
        return preprocessor

    @staticmethod
    def _fill_missing_attendance_rate(
        df: pd.DataFrame, attendance_column: str
    ) -> pd.DataFrame:
        """
        Fills missing values in the 'attendance_column' using the 'id_column'.

        Args:
        -----
        df (pd.DataFrame): The DataFrame containing the columns to be filled.
        attendance_column (str): The name of the column containing the attendance rates to be filled.

        Returns:
        --------
        pd.DataFrame: The DataFrame with missing values in 'attendance_column' filled.
        """
        avg_attendance_by_hours = df.groupby('hours_per_week')['attendance_rate'].transform('mean')
        df[attendance_column] = df[attendance_column].fillna(avg_attendance_by_hours)
        return df[attendance_column]