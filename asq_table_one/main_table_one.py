import pandas as pd
import pathlib
from asq_table_one.table_one import MakeTableOne
import json
from typing import Dict, List
from asq_table_one.asq_data_dictionary import AsqDataDictionary
import re
import bisect

def map_asq_in_table_one(asq_dict: Dict, df_asq_table_one: pd.DataFrame) -> pd.DataFrame:
    def first_occurrence_greater_than_index(lst: List, index: int):
        # Use bisect_right to find the insertion point for the index in a sorted list
        pos = bisect.bisect_right(lst, index)

        # Check if the position is within bounds and return the value, else None
        return lst[pos] if pos < len(lst) else None

    division_value = '--'
    # We want to replace variable name with Question Name (Abbreviated) + description multi-response
    df_tab_one = df_asq_table_one.copy()
    col_dist = [*df_tab_one.columns][1]
    div_indexes = df_tab_one.loc[df_tab_one[col_dist] == division_value, :].index.to_list()

    for idx_, var_ in df_tab_one.iterrows():
        var_value = var_['variable']  # Access the 'variable' column in the row
        # Assign the formal name to the variable using the match from the dictionary
        formal_name_dict = asq_dict.get(var_value)  # Retrieve the formal name from asq_dict
        if isinstance(formal_name_dict, dict):
            numeric_scoring = formal_name_dict.get('numeric_scoring')
            formal_name = formal_name_dict.get('description')
            df_tab_one.loc[idx_, 'variable'] = formal_name

            if var_[col_dist] == division_value:
                # We have a categorical/ordinal response
                # Label numbers of the categories to their true labels
                idx_start_question = idx_ + 1
                idx_end_question = first_occurrence_greater_than_index(lst=div_indexes, index=idx_)

                # Check if idx_end_question is None
                if idx_end_question is not None:
                    idx_end_question -= 1  # Adjust to the correct range
                    df_tab_one.loc[idx_start_question:idx_end_question, 'variable'] = df_tab_one.loc[idx_start_question:idx_end_question, 'variable'].map(numeric_scoring)
                else:
                    print(f"Warning: No valid end index found for idx_={idx_}. Skipping this range.\n{df_tab_one.loc[idx_, :]}")
    return df_tab_one

if __name__ == '__main__':
    data_dict_path = pathlib.Path(r'asq_dictionary_v4.xlsx')
    asq_dictionary = pathlib.Path(r'asq_dictionary_description.json')
    asq_data_path = pathlib.Path(r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\MiglabPsyc235\data\pproc\asq_for_tab_one.csv')

    # Step 1: Load the dataset
    df_data = pd.read_csv(asq_data_path)
    # Step 2: Load the data dictionary
    asq_dict_constructor = AsqDataDictionary(output_path=asq_dictionary,
                                    data_dict_path=data_dict_path,
                                    overwrite=False)

    asq_dict = asq_dict_constructor.run()
    asq_dict_constructor.print_column_dict(asq_dict)
    dtypes = asq_dict_constructor.get_columns_dtypes(asq_dict=asq_dict, df_asq=df_data)
    # %% Step 3: Create table one from the data
    # dtypes = classify_columns_dtypes(df=df_data)
    # Problems: narc_2100 is continuus but label as categorical, make a function that uses the df_data
    tab_one_clusters = MakeTableOne(df=df_data,
                                    strata='time',
                                    continuous_var=dtypes.get('continuous'),
                                    categorical_var=dtypes.get('categorical'),
                                    )

    df_tab_one_clusters = tab_one_clusters.create_table()

    df_tab_one_clusters_edited = tab_one_clusters.group_variables_table(df=df_tab_one_clusters)

    # %% Step 4: Edit the labels
    df_tab_one_asq = map_asq_in_table_one(asq_dict=asq_dict, df_asq_table_one=df_tab_one_clusters_edited)







