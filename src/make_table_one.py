from library.table_one import MakeTableOne
import numpy as np
import pandas as pd
from configuration.config import config
from library.table_one import MakeTableOne
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict
from tabulate import tabulate
from graphviz import Digraph
from library.compute_sparse_encoding import multi_response_col


if __name__ == "__main__":
    df_data = pd.read_csv(config.get('data_path').get('pp_dataset'))
    output_path = config.get('results_path').joinpath('table_one')
    aliases = {
        'dem_0110': 'Age',
        'dem_0500': 'Sex At Birth',
        'dem_0800': 'BMI',
        'dem_1000': 'Racial Category',
        'map_0300': 'Snorting Gasping',
        'osa_0200': 'Perspire heavily during the night',
        'ess_0900': 'ESS Score',
        'isi_0100': 'Difficulty falling asleep',
        'isi_0200': 'Difficulty staying asleep',
        'isi_0300': 'Problem waking up too early',
        'cir_0200': 'What time of day would you get up',
        'fosq_0100': 'Difficulty concentrating',
        'fosq_0200': 'Difficulty remembering',
        'gad_0900': 'GAD-2 Score',
        'phq_1100': 'PHQ-2 Score',
        'fosq_1100': 'FOSQ Score',
        'isi_score': 'ISI Score',
        'sched_total_sleep_time_week': 'Total Sleep Time Week',
        'sched_total_sleep_time_weekend': 'Total Sleep Time Weekend',
    }
    df_dict = pd.read_excel(config.get('data_path')['base'].joinpath('asq_dictionary_v4.xlsx'))
    df_dict_scoring = df_dict.loc[df_dict['Column Name'].isin(aliases.keys()), ['Column Name', 'Numeric Scoring Code ']]

    # Group by 'id_subject' and check if the values in 'dem_0500' are different
    df_diff_dem_0500 = df_data.groupby('id_subject').filter(lambda x: x['dem_0500'].nunique() > 1)
    df_data.loc[df_data['id_subject'] == 964, 'dem_0500'] = 1
    # %% divide in two groups
    subject_counts = df_data['id_subject'].value_counts()
    subjects_with_two_records = subject_counts[subject_counts == 2].index
    df_data = df_data[df_data['id_subject'].isin(subjects_with_two_records)]
    df_data['time'] = np.nan
    df_data.loc[df_data.groupby('id_subject').nth(0).index, 'time'] = 'first'
    df_data.loc[df_data.groupby('id_subject').nth(1).index, 'time'] = 'second'
    df_second_replicate = df_data.groupby('id_subject').nth(1)
    # %% define the columns to include in the table one
    columns = ['fosq_0200',
     'mdhx_6310',
     'bthbts_0360',
     'sched_4150',
     'sched_total_sleep_time_weekend',
     'sched_3800',
     'isi_0100',
     'mdhx_6100',
     'dem_1000',
     'mdhx_6320',
     'bthbts_0350',
     'isi_0300',
     'dem_0500',
     'sched_2310',
     'map_0300',
     'mdhx_5920',
     'soclhx_1000',
     'sched_1000',
     'dem_0800',
     'mdhx_0200',
     'mdhx_5700',
     'sched_1900',
     'mdhx_5600',
     'sched_2000',
     'gad_0900',
     'mdhx_6600',
     'sched_3700',
     'soclhx_0900',
     'ess_0900',
     'isi_score',
     'mdhx_6400',
     'mdhx_5710',
     'sched_0900',
     'cir_0200',
     'fosq_0100',
     'mdhx_5720',
     'mdhx_0900',
     'soclhx_0100',
     'bthbts_0320',
     'score',
     'mdhx_5900',
     'mdhx_6500',
     'phq_1100',
     'sched_2210',
     'soclhx_0510',
     'fosq_1100',
     'mdhx_5910',
     'bthbts_0310',
     'dem_0110',
     'sched_total_sleep_time_week',
     'mdhx_6300',
     'osa_0200',
     'isi_0200']
    for mcol_ in multi_response_col.keys():
        columns.extend(
            [col for col in df_data if mcol_ in col and not col.endswith('nan')]
        )

    # multi_response_col_selected = [
    #     'mdhx_sleep_problem',
    #     'mdhx_cardio_problem',
    #     'mdhx_pulmonary_problem',
    #     'mdhx_ent_surgery',
    #     'mdhx_sleep_problem',
    #     'mdhx_ent_problem',
    #     'mdhx_psych_problem',
    #     'bthbts_sleep_disruption',
    #     'bthbts_employment',
    # ]
    # [col for col in multi_response_col_selected if not col in multi_response_col]


    for col in df_data[columns]:
        print(f'-{col}: \n\t\t{df_data[col].unique()}\n')

    # %% Time column format
    col_time_format = [
        'sched_0900',
     'sched_1000',
     'sched_1900',
     'sched_2000',
    'soclhx_1000']
    for col_time_ in col_time_format:
        # Step 1: Convert time strings to total seconds
        df_data[col_time_+'_seconds'] = pd.to_timedelta(df_data[col_time_]).dt.total_seconds()

        # Step 2: Calculate mean and standard deviation in seconds
        mean_seconds = df_data[col_time_+'_seconds'].mean()
        std_seconds = df_data[col_time_+'_seconds'].std()

        # Step 3: Convert results back to time format (optional)
        mean_time = pd.to_timedelta(mean_seconds, unit='s')
        std_time = pd.to_timedelta(std_seconds, unit='s')

        mean_time_only = str(mean_time).split(' ')[-1].split('.')[0]  # Splits "0 days HH:MM:SS" and takes the time part
        std_time_only = str(std_time).split(' ')[-1].split('.')[0]

        # Convert seconds back to hours for plotting and ticks
        df_data[col_time_+'_hours'] = df_data[col_time_+'_seconds'] / 3600
        mean_hours = mean_seconds / 3600
        std_hours = std_seconds / 3600

        # Plot histogram
        plt.figure(figsize=(10, 6))
        # plt.hist(df_data[col_time_+'_hours'],
        #          bins=10,
        #          color='skyblue',
        #          edgecolor='black',
        #          hue='time',
        #          alpha=0.7)
        sns.histplot(
            data=df_data,
            x=col_time_+'_hours',
            hue='time',
            bins=10,
            kde=False,
            palette='pastel',
            edgecolor='black',
            alpha=0.7
        )

        plt.axvline(mean_hours,
                    color='red',
                    linestyle='-',
                    label=f'Mean ({mean_hours:.2f} hrs)')
        plt.axvline(mean_hours + std_hours,
                    color='orange',
                    linestyle='--',
                    label=f'+1 Std ({mean_hours + std_hours:.2f} hrs)')
        plt.axvline(mean_hours - std_hours,
                    color='orange',
                    linestyle='--',
                    label=f'-1 Std ({mean_hours - std_hours:.2f} hrs)')
        plt.xticks(
            np.arange(0, 25, 2),  # Generate ticks from 0 to 24 hours
            [f'{int(h)}:00' for h in np.arange(0, 25, 2)]  # Convert to "HH:MM" format
        )
        plt.xlabel('Time (Hours)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Histogram of Scheduled Times \n{col_time_}', fontsize=14)
        plt.legend()
        plt.xlim(0, 24)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Print results
        print("Mean Time:", mean_time)
        print("Standard Deviation of Time:", std_time)

    df_data[columns + ['time']].to_csv('asq_for_tab_one.csv', index=False)
    # %% Make the table one
    # Table 1 - Demographic characteristics of the sample across ethnic groups
    continuous_var = [col for col in df_data[columns] if df_data[col].nunique() > 7 and not col in col_time_format]
    categorical_var = [col for col in df_data[columns] if not col in continuous_var and not col in col_time_format]
    column_groups = 'time'

    table_one_race = MakeTableOne(df=df_data,
                                  continuous_var=continuous_var,
                                  categorical_var=categorical_var,
                                  strata=column_groups)

    df_table_one = table_one_race.create_table()
    df_tab_one = table_one_race.group_variables_table(df_table_one)

    df_tab_one.to_csv(output_path.joinpath('table_one_race_edited.csv'), index=False)


    continuous_var = ['gad_0900', 'phq_1100', 'fosq_1100', 'isi_score', 'sched_total_sleep_time_week', 'sched_total_sleep_time_weekend']



