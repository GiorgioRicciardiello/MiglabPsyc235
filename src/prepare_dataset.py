"""
Once the subject pairs have been matches and manually filter to maximize the number of matches, we will proceed with
compiling the final version of the dataset and pre-processing the columns.


Operations:
1. Assign subject id to each record with duplicates
2. For duplicates with more than a pair, select the most distant one
3. Select the columns of interest
4. Process the time column into valid units for analysis

"""
import pathlib

import pandas as pd
import numpy as np
from configuration.config import config, aliases, dtypes
from library.fuzzy_search import FuzzySearch, NameDateProcessor
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle
from typing import Tuple, Optional
from library.compute_sparse_encoding import compute_sparse_encoding, multi_response_col

def calculate_time_delta(df):
    """
    Creates a time delta column
    :param df:
    :return:
    """
    if len(df) > 1:
        # Sort by 'asq_created_at'
        df = df.sort_values(by='asq_created_at')
        # Calculate the time difference (in days) between the first and last record
        df['time_delta'] = df['asq_created_at'].diff().dt.days
    else:
        # If only one record, no time difference
        df['time_delta'] = None
    return df

def select_most_distant_in_quarter(df):
    """
    Function to select the most distant record in time within the same quarter.
    :param df: DataFrame for a single subject
    :return: DataFrame with the most distant record in the same quarter
    """
    if len(df) > 1:
        # Sort by 'quarter' and 'asq_created_at'
        df = df.sort_values(by=['quarter', 'asq_created_at'])

        # Group by quarter and find the most distant records within each quarter
        result = df.groupby('quarter').apply(lambda x: x.iloc[[0, -1]] if len(x) > 1 else x)

        # Remove the additional group keys introduced by `groupby`
        result = result.reset_index(drop=True)

        return result
    else:
        # If there is only one record, return it as is
        return df


def check_same_quarter(df):
    # If there are more than one record, check if all quarters are the same
    if len(df) > 1:
        df['same_quarter'] = df['quarter'].nunique() == 1
    else:
        # For subjects with only one record, mark as False (not a pair)
        df['same_quarter'] = False
    return df

def compute_cyclic_time_diff(group):
    """Function to calculate cyclic time difference"""
    # Convert times to minutes from midnight
    times = group.dt.hour * 60 + group.dt.minute
    # Sort the times within the group
    sorted_times = times.sort_values()
    # Calculate the forward difference (end_time - start_time)
    diff = sorted_times.iloc[-1] - sorted_times.iloc[0]
    # Check for cyclic adjustment (if crossing midnight)
    cyclic_diff = min(diff, 1440 - diff)  # 1440 minutes in a day
    return cyclic_diff

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    df_asq = pd.read_csv(config.get("data_path")['asq'].joinpath("pp_3_asq_clinical_research_merge.csv"),
                         low_memory=False)
    df_asq = compute_sparse_encoding(multi_response_col=multi_response_col,
                                     df=df_asq)
    df_asq['survey_id'] = df_asq['origin'].apply(lambda x: x[0]) + ['_'] * df_asq['origin'].shape[0] + df_asq[
        'survey_id'].astype(str)
    df_pairs = pd.read_excel(config.get("data_path")['base'].joinpath("matches_scorer_pairs_manually_verified.xlsx"))
    # get only the verified pairs
    df_pairs = df_pairs.loc[df_pairs['1 is match 0 is no match'] == 1, :]
    df_pairs.drop(columns='1 is match 0 is no match', inplace=True)
    df_pairs['asq_created_at'] = pd.to_datetime(df_pairs['asq_created_at'])
    # get the full ASQ for the matches patients
    subject_counts = df_pairs['subject_name'].value_counts()
    print(f'Total count of pairs: \n{subject_counts.value_counts()}')

    # %% compute sparse encoding

    # %% Assign subject id
    df_pairs['id_subject'] = df_pairs.groupby(['subject_name', 'asq_dob']).ngroup() + 1

    # %% Column to show how many duplicates are per subject
    subject_counts = df_pairs['subject_name'].value_counts()
    df_pairs['subject_count'] = df_pairs['subject_name'].map(subject_counts)

    df_pairs = df_pairs.loc[df_pairs['subject_count'] > 1, :]

    #%% select the two most distant records in terms of time
    # Apply the function to each group of subjects
    df_pairs_most_distant = df_pairs.groupby('subject_name').apply(select_most_distant_in_quarter).reset_index(drop=True)


    # %% Check if pairs are in the same quarter
    df_pairs_most_distant = df_pairs_most_distant.groupby('subject_name').apply(check_same_quarter).reset_index(drop=True)
    df_diff_quarter = df_pairs_most_distant.loc[df_pairs_most_distant['same_quarter'] == False, :]

    # we have some subjects that took the course more than once!
    # Let's take the pair of the first class session they took to avoid attrition bias

    df_first_class = df_diff_quarter.sort_values(by='asq_created_at').groupby('subject_name').head(2).reset_index()

    df_pairs_most_distant = df_pairs_most_distant.loc[df_pairs_most_distant['same_quarter'] == True, :]

    df_pairs_most_distant = pd.concat([df_pairs_most_distant, df_first_class], ignore_index=True)
    df_pairs_most_distant.drop(columns=['same_quarter', 'index'], inplace=True)

    # %% Check if all id_subjects have a pair
    subject_counts = df_pairs_most_distant.groupby('id_subject').size()

    # Check if all subjects have exactly two records
    subjects_with_pairs = subject_counts[subject_counts == 2]

    # Find subjects that do not have exactly two records
    subjects_without_pairs = subject_counts[subject_counts != 2]

    # Display results
    print(f"Number of subjects with a pair: {len(subjects_with_pairs)}")
    print(f"Number of subjects without a pair: {len(subjects_without_pairs)}")

    if len(subjects_without_pairs) > 0:
        print(f"Subjects without a pair: \n{subjects_without_pairs}")

    # %% Calculate time difference in days for each pair
    # Apply the function to each subject and calculate 'time_delta'
    df_pairs_most_distant = df_pairs_most_distant.groupby('subject_name').apply(calculate_time_delta).reset_index(drop=True)
    # plot the time delta
    time_delta_counts = df_pairs_most_distant['time_delta'].value_counts().sort_index(ascending=False)
    plt.figure(figsize=(20, 6))
    sns.barplot(x=time_delta_counts.index, y=time_delta_counts.values, palette='viridis')
    plt.title('Frequency of Time Delta Between Records')
    plt.xlabel('Time Delta (Days)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.7)
    plt.tight_layout()
    plt.show()
    # %% Quantify the dataset
    df_pairs = df_pairs_most_distant.groupby('id_subject').filter(lambda x: (x['time_delta'] >= 20).any())
    subject_counts = df_pairs.groupby('id_subject').size()
    # Check if all subjects have exactly two records
    subjects_with_pairs = subject_counts[subject_counts == 2]
    # Find subjects that do not have exactly two records
    subjects_without_pairs = subject_counts[subject_counts != 2]
    # Display results
    print(f'Number of unique subjects: {df_pairs.id_subject.nunique()}')
    print(f"Number of subjects with a pair: {len(subjects_with_pairs)}")
    print(f"Total number of ASQs: {df_pairs.shape[0]}")



    # %% Couple the subjects with the ASQ
    df_pairs.rename(columns={'asq_survey_id': 'survey_id'}, inplace=True)
    df_pairs['survey_id'] = df_pairs['survey_id'].astype(str)
    df_asq_pairs = pd.merge(left=df_pairs[['id_subject', 'quarter', 'subject_name', 'survey_id']],
             right=df_asq,
             on=['survey_id'])

    col_first = [
        'id_subject',
        'quarter',
        'subject_name',
        'name',
        'completed',
        'created_at',
    ]
    columns = [col for col in df_asq_pairs.columns if col not in col_first]
    df_asq_pairs = df_asq_pairs[col_first+columns]

    col_id = [col for col in columns if col.endswith('_id') and col != 'survey_id']
    df_asq_pairs.drop(columns=col_id, inplace=True)
    df_asq_pairs.drop(columns=['mrn', 'next_module', 'origin'], inplace=True)

    df_asq_pairs['completed'] = pd.to_datetime(df_asq_pairs['completed'])

    # Group by 'id_subject' and calculate the number of days between each completed record
    df_asq_pairs = df_asq_pairs.sort_values(by=['id_subject', 'completed'])
    df_asq_pairs['days_between'] = df_asq_pairs.groupby('id_subject')['completed'].diff().dt.days

    # %% Feature engineering
    # Sleep Time (minutes + hours*60)
    df_asq_pairs[['sched_2310', 'sched_2300', 'sched_3810', 'sched_3800', 'sched_3710', 'sched_3700' ]]

    # Check positivity
    # df_asq_pairs.loc[df_asq_pairs['sched_total_sleep_time_week'] < 0, ['sched_2310', 'sched_2300', 'sched_total_sleep_time_week']]

    df_asq_pairs[['sched_2310', 'sched_2300', 'sched_3810', 'sched_3800', 'sched_3710', 'sched_3700']] = df_asq_pairs[
        ['sched_2310', 'sched_2300', 'sched_3810', 'sched_3800', 'sched_3710', 'sched_3700']].applymap(
        lambda x: x if x >= 0 else np.nan)

    df_asq_pairs[['sched_2310', 'sched_2300', 'sched_3810', 'sched_3800', 'sched_3710', 'sched_3700']].describe()

    # Sleep Latency
    df_asq_pairs['sched_total_sleep_time_week'] = df_asq_pairs['sched_2310'] + (df_asq_pairs['sched_2300'] * 60)
    df_asq_pairs['sched_total_sleep_time_weekend'] = df_asq_pairs['sched_3810'] + (df_asq_pairs['sched_3800'] * 60)
    df_asq_pairs['sched_total_sleep_latency_weekend'] = df_asq_pairs['sched_3710'] + (df_asq_pairs['sched_3700'] * 60)
    df_asq_pairs['sched_total_sleep_latency_week'] = df_asq_pairs['sched_2210'] + (df_asq_pairs['sched_2200'] * 60)

    df_asq_pairs['sched_total_sleep_latency'] = df_asq_pairs['sched_2210'] + (df_asq_pairs['sched_2200'] * 60)
    # limit the user input on sleep latency and time
    col_latency = [col for col in df_asq_pairs if 'sched_total_sleep_latency' in col]
    for col_latency in col_latency:
        # limit_max = df_asq_pairs[col_latency].mean() + 3 * df_asq_pairs[col_latency].std()
        limit_max = 200  # upper clip at 200 minutes ~ 3.3 hours of sleep latency
        df_asq_pairs[col_latency] = df_asq_pairs[col_latency].clip(lower=0, upper=limit_max)

    col_sleep_time = [col for col in df_asq_pairs if 'sched_total_sleep_time' in col]
    for col_latency in col_sleep_time:
        limit_max = 800  # upper clip at 800 minutes ~ 13 hours of total sleep
        df_asq_pairs[col_latency] = df_asq_pairs[col_latency].clip(lower=0, upper=limit_max)

    # Bed Time
    # Variables contain NaNs sow we cannot use the full datase
    # use the features: sched_0900, sched_1000, sched_1900, sched_2000
    df_asq_pairs['sched_0900'] = pd.to_datetime(df_asq_pairs['sched_0900'])
    df_asq_pairs['sched_1900'] = pd.to_datetime(df_asq_pairs['sched_1900'])

    df_asq_pairs['sched_bed_time_week_diff'] = df_asq_pairs.groupby('id_subject')['sched_0900'].transform(
        compute_cyclic_time_diff
    )

    df_asq_pairs['sched_bed_time_weekend_diff'] = df_asq_pairs.groupby('id_subject')['sched_1900'].transform(
        compute_cyclic_time_diff
    )

    # df_asq_pairs[['id_subject','sched_0900','sched_bed_time_week_diff']]

    df_asq_pairs['sched_bed_time_week_diff'].describe()

    df_asq_pairs[['id_subject', 'sched_0900', 'sched_bed_time_week_diff']]

    df_asq_pairs['sched_0900_total_minutes'] = df_asq_pairs['sched_0900'].apply(lambda x: x.hour * 60 + x.minute)
    df_asq_pairs['sched_1900_total_minutes'] = df_asq_pairs['sched_1900'].apply(lambda x: x.hour * 60 + x.minute)
    # mean_minutes = df_asq_pairs['sched_1900_total_minutes'].mean()
    # mean_time = f"{int(mean_minutes // 60):02}:{int(mean_minutes % 60):02}"
    # std_dev_minutes = df_asq_pairs['sched_1900_total_minutes'].std()
    # std_hours = int(std_dev_minutes // 60)
    # std_minutes = int(std_dev_minutes % 60)
    # std_time = f"{std_hours:02}:{std_minutes:02}:00"
    #

     #%% Data Cleaning
    col_scores = ['gad_0900', 'phq_1100', 'fosq_1100']

    df_asq_pairs[col_scores] = df_asq_pairs[col_scores].applymap(lambda x: x if x >= 0 else np.nan)
    # df_asq_pairs['gad_0900'].describe()
    # df_asq_pairs['phq_1100'].describe()
    # df_asq_pairs['fosq_1100'].describe()
    # df_asq_pairs['isi_score'].describe()

    # negative responses must be removed
    # soclhx_0500 considers the Events per week, people can max train 2 times per week = 14 events
    df_asq_pairs['soclhx_0500'] = df_asq_pairs['soclhx_0500'].clip(lower=0, upper=14)
    # if wrote decimals between 0.1 and 1, set it to one event per week
    df_asq_pairs['soclhx_0500'] = df_asq_pairs['soclhx_0500'].apply(
        lambda x: 1 if 0.1<= x <= 1 else int(x) if not pd.isna(x) else x
    )
    # round the rest to the first decimal points
    df_asq_pairs['soclhx_0500'] = df_asq_pairs['soclhx_0500'].round(1)

    # here we set the do not know to zero, we could remove them or map them to -1. Never and do not know might be
    # similar a never user might not know actually that he\she does that.
    # Snorting gasping ordinal
    df_asq_pairs['map_0300'] = df_asq_pairs['map_0300'].clip(lower=0)
    df_asq_pairs['map_0300'] = df_asq_pairs['map_0300'].fillna(0)
    df_asq_pairs['map_0300'] = df_asq_pairs['map_0300'].astype(int)

    # Perspire heavily during the night - ordinal
    df_asq_pairs['osa_0200'] = df_asq_pairs['osa_0200'].clip(lower=0)
    df_asq_pairs['osa_0200'] = df_asq_pairs['osa_0200'].fillna(0)
    df_asq_pairs['osa_0200'] = df_asq_pairs['osa_0200'].astype(int)


    # %% Additional
    # df_asq_pairs = make_ess_score(df_asq_pairs)
    columns = ['id_subject',
               'quarter',
               # 'subject_name',
               # 'name',
               'completed',
               'created_at',
               'survey_id',
               'dob',
               'start_time',]
    columns.extend([*aliases.keys()])
    df_asq_pairs = df_asq_pairs[columns]

    # Columns that should be constant for both sumbissions
    #  Group by 'id_subject' and check if the values in 'dem_0500' are different
    df_diff_dem_0500 = df_asq_pairs.groupby('id_subject').filter(lambda x: x['dem_0500'].nunique() > 1)
    df_asq_pairs.loc[df_asq_pairs['id_subject'] == 964, 'dem_0500'] = 1

    # %% divide in two groups
    subject_counts = df_asq_pairs['id_subject'].value_counts()
    subjects_with_two_records = subject_counts[subject_counts == 2].index
    df_asq_pairs = df_asq_pairs[df_asq_pairs['id_subject'].isin(subjects_with_two_records)]

    df_asq_pairs['time'] = np.nan
    df_asq_pairs.loc[df_asq_pairs.groupby('id_subject').nth(0).index, 'time'] = 'first'

    df_asq_pairs.loc[df_asq_pairs.groupby('id_subject').nth(1).index, 'time'] = 'second'

    # %% save the dataset
    df_asq_pairs.to_csv(config.get('data_path').get('pp_dataset'), index=False)



















