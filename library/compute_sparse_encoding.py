"""
ASQ questions that are stored as a string list format, we will compute sparse encoding to this column
The dictionary value is the name in the asq_dictionary.xlsx file.
"""
import pandas as pd
from typing import Optional
import ast

multi_response_col = {
 'mdhx_sleep_problem': 'problem',
 'mdhx_sleep_diagnosis': 'mdhx_0120',
 'mdhx_sleep_treatment': 'treatment',
 'mdhx_pap_problem': 'problem',
 'mdhx_pap_improvement': 'improvement',
 'mdhx_cardio_problem': 'problem',
 'mdhx_cardio_surgery': 'surgery',
 'mdhx_pulmonary_problem': 'problem',
 'mdhx_ent_surgery': 'surgery',
 'mdhx_ent_problem': 'problem',
 'mdhx_dental_work': 'procedure',
 'mdhx_orthodontics': 'procedure',
 'mdhx_gi_problem': 'problem',
 'mdhx_neurology_problem': 'problem',
 'mdhx_metabolic_endocrine_problem': 'problem',
 'mdhx_urinary_kidney_problem': 'problem',
 'mdhx_pain_fatigue_problem': 'problem',
 'mdhx_headache_problem': 'problem',
 'mdhx_psych_problem': 'problem',
 'mdhx_anxiety_problem': 'problem',
 'mdhx_eating_disorder': 'disorder',
 'mdhx_other_problem': 'problem',
 'mdhx_cancer': 'cancer',
 'mdhx_autoimmune_disease': 'disease',
 'mdhx_hematological_disease': 'disease',
 'famhx_insomnia': 'relation',
 'famhx_sleep_apnea': 'relation',
 'famhx_narcolepsy': 'relation',
 'famhx_rls': 'relation',
 'famhx_other_sleep_disorder': 'relation',
 'famhx_sleepwalk': 'relation',
 'famhx_fibromyalgia': 'relation',
 'famhx_depression': 'relation',
 'famhx_anxiety': 'relation',
 'famhx_psych_illness': 'relation',
 'famhx_psych_treatment': 'relation',
 'famhx_sleep_death': 'relation',
 'bthbts_sleep_disruption': 'disruption',
 'bthbts_employment': 'employment',
 'sched_rotating_shift': 'shift',
}

def compute_sparse_encoding(multi_response_col: list[str],
                            df: pd.DataFrame,
                            nan_int: Optional[int] = -200) -> pd.DataFrame:
    """
    Compute sparse encoding to the multiple response columns
    :para, df: pd.Dataframe, dataset
    :param multi_response_col: list[str], columns that are of multiple response type in the dataset
    :para nan_int: Optional[int], integer to mark the nan when doing the exploded dataframe
    :return:
    """

    def make_list(x, nan_replace: Optional[str] = '[-200]'):
        """
        To do the explode all values most be list format. Because they are saved as astrings in the .csv we must
        make all the cells as strings and nans are not recognized by the ast.literal_eval method.
        :param x:
        :param nan_replace: str, optional, string to replace the nan
        :return:
        """
        if pd.isna(x):
            return nan_replace
        elif isinstance(x, list) or isinstance(x, int):
            return str(x)
        else:
            return x

    def sparse_encoding(df_exploded: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Multiple responses can have more than one category, this the index in the df_exploded will have duplicates
        for the asq responses that have more than one response in a cell. Therefore, the traditional
        one-hot-encoding algorithm will result in more rows than it should.

        This function takes this duplicates into consideration and applies the categorical encoding. Be aware, that
        this is not one-hot because in a row we can have more than one category.

        :param df_exploded: pd.Dataframe, exploded dataframe of the column we will one hot encode
        :param column_name: str, column name of the exposed dataframe we are using
        :return:
            pd.Dataframe with the categorical encoding
        """
        # pre-allocate the dummy frame, get the column names based on the unique elements
        values = df_exploded[column_name].unique()
        values.sort()
        columns = [f'{column_name}_{value}' for value in values]
        df_dummy = pd.DataFrame(data=0,
                                columns=columns,
                                index=range(0, df_exploded.index.nunique()))
        # make an index column for us to use as indexes since we need to use uniques when allocating
        df_exploded.reset_index(inplace=True, drop=False, names=['asq_index'])
        for val_, col_ in zip(values, columns):
            # Get the indices where the condition is true
            indices = df_exploded.loc[df_exploded[column_name] == val_, 'asq_index'].values
            df_dummy.loc[indices, col_] = 1

        return df_dummy

    dataset = df.copy()
    for multi_resp_ in multi_response_col:
        print(f'\n{multi_resp_}')
        if not multi_resp_ in dataset.columns:
            print('not in columns')
            continue
        print('processing')
        # multi_resp_ = multi_response_col[0]
        # Make a copy of the column containing lists
        df_multi_resp = pd.DataFrame(
            data=dataset[multi_resp_].copy(),
            columns=[multi_resp_]
        )
        # make all cell with sme str(list) format
        df_multi_resp[multi_resp_] = df_multi_resp[multi_resp_].apply(make_list,
                                                                      nan_replace=f'[{nan_int}]')
        # convert as a list
        df_multi_resp[multi_resp_] = df_multi_resp[multi_resp_].apply(lambda x: ast.literal_eval(x))
        # explode all the lists
        df_exploded = df_multi_resp.explode(column=multi_resp_)
        df_sparse = sparse_encoding(df_exploded=df_exploded,
                                    column_name=multi_resp_)

        df_sparse.rename(columns={f'{multi_resp_}_{nan_int}': f'{multi_resp_}_nan'},
                         inplace=True)
        if not dataset.shape[0] == df_sparse.shape[0]:
            raise ValueError(f'Unmatch dimensions in the rows for columns: '
                             f'{multi_resp_} - ({dataset.shape[0]} vs {df_sparse.shape[0]} )')

        # remove the original column
        dataset.drop(columns=multi_resp_, inplace=True)

        # append the one-hot-encoded version
        dataset = pd.concat([dataset, df_sparse], axis=1)
    return dataset
