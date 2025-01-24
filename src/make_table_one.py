from library.table_one import MakeTableOne
import numpy as np
import pandas as pd
from configuration.config import config
from library.table_one import MakeTableOne
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict
from tabulate import tabulate
from library.compute_sparse_encoding import multi_response_col
from configuration.config import aliases, dtypes
from visualization import get_labels
from typing import Dict
import re

def apply_labels_to_table(df: pd.DataFrame,
                          labels: Dict[str, Dict[int, str]],
                          var_column:str='variable') -> pd.DataFrame:
    """
    Apply label mappings to categorical variables in a DataFrame based on a labels dictionary.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the 'variable' column and associated data.
    - labels (Dict[str, Dict[int, str]]): A dictionary mapping variable names to numeric-to-label mappings.
    - var_column (str): The name of the variable column. Here we have the names of the variables and the numeric
        scoring we need to change

    Returns:
    - pd.DataFrame: A modified DataFrame with numeric scores replaced by corresponding labels for categorical variables.
    """

    def replace_numeric_with_label(value: str, mapping: Dict[int, str]) -> str:
        """
        Replace numeric values in a string with corresponding labels based on a mapping.

        Parameters:
        - value (str): The input string containing numeric values.
        - mapping (Dict[int, Dict[int, str]]): A dictionary mapping numeric values to labels.

        Returns:
        - str: Modified string with numeric values replaced by labels.
        """
        match = re.match(r"(\d+)", str(value))  # Match numeric part
        if match:
            num = int(match.group(1))  # Extract numeric value
            label = mapping.get(num, None)  # Replace with label if found
            if label:
                return re.sub(r"^\d+", label, str(value))  # Replace numeric with label
        return value  # Return original value if no match

    def is_string(value):
        """
        Helper function to check if a string cannot be converted to an integer
        Usage:
        df_tab_one['question'] = df_tab_one['variable'].apply(is_string)
        :param value:
        :return:
        """
        try:
            int(value)
            return False  # If it can be converted to an integer, it's not a string
        except ValueError:
            return True  # If it raises a ValueError, it's a string

    # map to which rows have a column name and which ones are numerical scores to replace
    df['question'] = df['variable'].apply(is_string)

    # Iterate through the labels dictionary to apply mappings
    for variable, mapping in labels.items():
        # variable = [*labels.keys()][0]
        # mapping = labels.get(variable)
        if variable in df[var_column].values:
            match_idx = df[df[var_column] == variable].index[0]
            # Find the next row where 'question' is True
            next_true_idx = df.loc[match_idx + 1:][df['question'] == True].index
            if not next_true_idx.empty:
                next_idx = next_true_idx[0]  # Get the first row where question is True
                # Modify rows between match_idx and next_idx (inclusive)
                df.loc[match_idx:next_idx, var_column] = df.loc[match_idx:next_idx, var_column].apply(
                    lambda x: replace_numeric_with_label(x, mapping)
                )
            # Replace numeric values with labels for the corresponding variable
            df.loc[df[var_column] == variable, var_column] = df.loc[df[var_column] == variable, var_column].replace(
                mapping)
    return df


if __name__ == "__main__":
    df_data = pd.read_csv(config.get('data_path').get('pp_dataset'))
    data_dict = pd.read_excel(config.get('data_path').get('asq_dictionary'))
    output_path = config.get('results_path').joinpath('table_one')

    df_dict = pd.read_excel(config.get('data_path')['base'].joinpath('asq_dictionary_v4.xlsx'))
    df_dict_scoring = df_dict.loc[df_dict['Column Name'].isin(aliases.keys()), ['Column Name', 'Numeric Scoring Code ']]

    #%% Dictionary with the formal numeric scoring of each ordinal/categorical variable
    labels = {}
    for alias, name_ in aliases.items():
        if dtypes.get(alias) != 'continuous' and not alias in labels.keys():
            label = get_labels(df=data_dict, col_name=alias)
            labels.update(label)

    labels = {key: val for key, val in labels.items() if isinstance(val, dict)}

    df_data.loc[df_data['id_subject'] == 964, 'dem_0500'] = 1

    # %% Make the table one
    continuous_var = [var for var, dtype_ in dtypes.items() if dtype_ == 'continuous']
    categorical_var = [var for var, dtype_ in dtypes.items() if dtype_ != 'continuous']
    column_groups = 'time'

    table_one_race = MakeTableOne(df=df_data,
                                  continuous_var=continuous_var,
                                  categorical_var=categorical_var,
                                  strata=column_groups)

    df_table_one = table_one_race.create_table()
    df_tab_one = table_one_race.group_variables_table(df_table_one)
    df_tab_one_modified = apply_labels_to_table(df=df_tab_one, labels=labels)

    df_tab_one_modified['variable_code'] = ''

    df_tab_one_modified.loc[df_tab_one_modified.question, 'variable_code'] = df_tab_one_modified.loc[df_tab_one_modified.question, 'variable']

    df_tab_one_modified['variable'] = df_tab_one_modified['variable'].replace(aliases)

    df_tab_one_modified.drop(columns=['question'], inplace=True)

    df_tab_one.to_csv(output_path.joinpath('table_one.csv'), index=False)





