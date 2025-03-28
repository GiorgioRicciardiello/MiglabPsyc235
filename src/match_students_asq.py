"""
Use the name of the roster to match the name of the students in the ASQ database.
"""
import pathlib

import pandas as pd
import numpy as np
from configuration.config import config
from library.fuzzy_search import FuzzySearch, NameDateProcessor
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle
from typing import Tuple, Optional


def convert_date(date_str, academic_year):
    """
    Define the function to convert date strings to datetime
    :param date_str:
    :param academic_year:
    :return:
    """
    # Remove ordinal suffix

    # Determine if the date belongs to the start or end year
    if 'October' in date_str or 'February' in date_str:
        year = academic_year.split('-')[0]
    else:
        year = academic_year.split('-')[1]

    # Concatenate the date string with the correct year
    return pd.to_datetime(f"{date_str} {year}", format='%d %B %Y')


def fuzzy_search_wrapper(df_students: pd.DataFrame,
                         df_academic_calendar: pd.DataFrame,
                         df_asq: pd.DataFrame,
                         scorer: str) -> pd.DataFrame:
    """
    Fuzzy search wrapper tailor for matching the ASQs with the students. For reducing the possible of matches within
    the fuzzy filter threshold we use the academic calendar to limit the ASQ for the period of the course the students
    were enrolled. This also optimizes the search by reducing the search of possible matches at each iteration.
    :param df_students:
    :param df_academic_calendar:
    :param df_asq:
    :param scorer:
    :return:
        dataframe with the results of the matches and the score
    """
    df_matches = pd.DataFrame()
    for idx, calendar in df_academic_calendar.iterrows():
        # calendar = df_academic_calendar.loc[4, :]
        year = int(calendar['Academic Year'].split('-')[0])
        df_enrolled = df_students.loc[(df_students['year'] == year) & (df_students['quarter'] == calendar.Term), :]
        if df_enrolled.shape[0] == 0:
            logging.info(f'No students in the {calendar.Term} - {year} quarter term')
            continue
        # we could use the start_time or completed or created_at
        df_asq_quarter = df_asq.loc[(df_asq.created_at >= calendar['Start Date']) &
                                    (df_asq.created_at <= calendar['End Date']), :]
        # we use this to later manually check per quarter which ones were missing
        df_asq_quarter = df_asq_quarter[['name', 'date_of_birth', 'survey_id', 'completed', 'created_at']]
        df_asq_quarter.to_csv(config.get('results_path').joinpath(f'asq_{calendar.Term}_{year}.csv'), index=False)
        logging.info(
            f'Searching for {df_enrolled.shape[0]} students in {df_asq_quarter.shape[0]} ASQs for '
            f'the {calendar.Term} - {year} quarter term')

        fuzzy_search = FuzzySearch(asq_df=df_asq_quarter,
                                   subjects_df=df_enrolled,
                                   scorer=scorer)

        matches = fuzzy_search.search_by_name_matches(fuzzy_filter=60)
        matches['quarter'] = f'{calendar.Term}-{year}'
        df_matches = pd.concat([df_matches, matches])
    df_matches.rename(columns={'subject_idx': 'idx_student_db',
                               'subject_id': 'idx_asq_db'
                               }, inplace=True)
    df_matches.sort_values(by='score', inplace=True, ascending=False)
    return df_matches


def compare_students_matched(df_students: pd.DataFrame,
                             df_match_res: pd.DataFrame,
                             figsize: Optional[Tuple[int, int]] = (12, 8),
                             output_path:Optional[pathlib.Path]=None) -> None:
    """
    The students dataset contains all the students for each quarter. This can be used as the best number of matches
    we can get. Therefore, by plotting the expected against the obtained (matched with the ASQ) we can visualize
    hoe effective was the algorithm
    :param df_students:
    :param df_match_res:
    :param figsize:
    :return:
    """
    # Step 1: Identify unique subject_name and idx_student_db pairs
    unique_subjects = df_match_res[['subject_name', 'idx_student_db']].drop_duplicates(keep='first')

    # Step 2: Filter the dataframe to keep only the rows where subject_name is unique
    filtered_df = df_match_res.loc[unique_subjects.index, :]

    # Step 3: Select the quarter column
    unique_subjects_quarter = filtered_df['quarter']

    # Combine df_students and unique_subjects_quarter
    df_combined = {
        'Academic year': df_students['Academic year'].to_list() + unique_subjects_quarter.to_list(),
        'Dataset': ['Student'] * df_students['Academic year'].shape[0] + ['Matches'] *
                   unique_subjects_quarter.shape[0]
    }
    df_combined = pd.DataFrame(df_combined)
    df_combined['Academic year'] = df_combined['Academic year'].str.replace('-', ' ')

    # Bar plot of the number of students per academic year comparing df_results and df_students
    plt.figure(figsize=figsize)
    sns.countplot(data=df_combined, x='Academic year', hue='Dataset', palette='Set2', alpha=0.9)
    plt.title(f'Number of Students per Academic Year (Comparison)\n'
              f'Students Unique Names: {df_students.name.nunique()}\n'
              f'Matches Unique Names: {unique_subjects.shape[0]}')
    plt.xlabel('Academic Year')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.grid(alpha=0.7)
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def make_datetime_timezone_unaware(df: pd.DataFrame) -> pd.DataFrame:
    """Function to convert datetime columns to timezone-unaware"""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    return df


warnings.filterwarnings('ignore')
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(config.get('results_path').joinpath('student_search.log')),
                                  logging.StreamHandler()],
                        )
    #%% ASQ
    df_asq = pd.read_csv(config.get("data_path")['asq'].joinpath("pp_3_asq_clinical_research_merge.csv"),
                         low_memory=False)
    df_asq['survey_id'] = df_asq['origin'].apply(lambda x: x[0]) + ['_'] * df_asq['origin'].shape[0] + df_asq[
        'survey_id'].astype(str)
    df_asq = df_asq[~df_asq['name'].isna()]
    df_asq['name'] = df_asq['name'].astype(str)
    df_asq = df_asq[df_asq['name'] != 'O-l']
    df_asq['completed'] = pd.to_datetime(df_asq.completed)
    df_asq['created_at'] = pd.to_datetime(df_asq['created_at'], format='mixed', utc=True)
    df_asq.rename(columns={'dob': 'date_of_birth'}, inplace=True)
    # filter by age
    df_asq = df_asq.loc[(df_asq['dem_0110'] >= 18) & (df_asq['dem_0110'] <= 25), :]
    df_asq.to_csv(config.get("data_path")['asq'].joinpath("pp_3_asq_clinical_research_merge.csv"), index=False)
    #%% Students
    df_students = pd.read_csv(config.get('data_path').get('dataset_students'))
    df_students['ID'] = df_students['ID'].astype(int)
    df_students['Academic year'] = df_students['quarter'] + ' ' + df_students['year'].astype(str)
    df_students.sort_values(by='year', inplace=True)
    # all the students from the roster dataset

    #%% academic calendar dates
    df_academic_calendar = pd.read_excel(config.get('data_path')['calendar'])
    df_academic_calendar.columns = df_academic_calendar.columns.str.strip()
    df_academic_calendar['Start Date'] = df_academic_calendar['Start Date'].str.strip()
    df_academic_calendar['End Date'] = df_academic_calendar['End Date'].str.strip()
    # Convert 'Start Date' and 'End Date' to datetime format
    df_academic_calendar['Start Date'] = pd.to_datetime(
        df_academic_calendar['Start Date'] + ' ' + df_academic_calendar['Academic Year'].str.split('-').str[0],
        format='%B %d %Y', utc=True)
    df_academic_calendar['End Date'] = pd.to_datetime(
        df_academic_calendar['End Date'] + ' ' + df_academic_calendar['Academic Year'].str.split('-').str[0],
        format='%B %d %Y', utc=True)

    assert all(df_academic_calendar['Start Date'] < df_academic_calendar['End Date'])

    # %% do the fuzz search
    scorers = ['R',
               'PR',
               'TSeR',
               'TSoR',
               'PTSeR',
               'PTSoR',
               'WR',
               'QR',
               'UWR',
               'UQR']

    if not config.get('results_path').joinpath('matches_scorer.pkl').exists():
        # generate the matches using all the possible fuzzy methods to increase the chances of capturing all the
        # matches between the datasets
        matches = {}
        for scorer in scorers:
            df_matches = fuzzy_search_wrapper(df_students=df_students,
                                              df_academic_calendar=df_academic_calendar,
                                              df_asq=df_asq,
                                              scorer=scorer)

            df_matches = make_datetime_timezone_unaware(df_matches)
            df_matches.to_excel(config.get('results_path').joinpath('scorer_frames', f'Matches_{scorer}.xlsx'),
                                index=False)
            # bar plot for each method to evaluate the matches
            score_counts = df_matches['score'].value_counts().sort_index()
            plt.figure(figsize=(12, 6))
            sns.barplot(x=score_counts.index, y=score_counts.values, palette="viridis")
            bars = plt.gca().patches
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')
            plt.xlabel('Score')
            plt.ylabel('Count')
            plt.title(f'Scorer {scorer} - Count of Scores in df_matches Dim {df_matches.shape[0]}')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout(pad=3.0)  # Increase space between the bars and the plot borders
            plt.savefig(config.get('results_path').joinpath(f'ScorerBar_{scorer}.png'), dpi=300)
            plt.show()

            matches[scorer] = {
                'df_matches': df_matches,
                'scorer_count': score_counts
            }

        # save the matches as separate frames
        for scorer, frame_score in matches.items():
            # scorer = 'UQR'
            # frame_score = matches.get(scorer)
            df_frame = frame_score.get('df_matches')
            scorer_count = frame_score.get('scorer_count')
            df_frame.to_csv(config.get('results_path').joinpath(f'matches_method_{scorer}.csv'), index=False)

        with open(config.get('results_path').joinpath('matches_scorer.pkl'), 'wb') as file:
            pickle.dump(matches, file)

    # %% Check the best scorer
    with open(config.get('results_path').joinpath('matches_scorer.pkl'), 'rb') as file:
        matches_dict = pickle.load(file)

    # make a single frame with all
    df_matches = pd.DataFrame()
    for method, value in matches_dict.items():
        # method = [*matches_dict.keys()][0]
        df_match = value.get('df_matches')
        df_match['method'] = method
        df_matches = pd.concat([df_matches, df_match])
    # The PTSeR method is not good, threfore we remove it
    df_matches = df_matches.loc[df_matches['method'] != 'PTSeR', :]
    df_matches.reset_index(drop=True, inplace=True)

    df_matches = df_matches.sort_values(by=['score', 'subject_name'],
                                        ascending=[False, True], inplace=False)


    # %% Filtering as many repetitions as possible
    # The best method that matches was the R method. No matter the score, if asq_name, subject_name, score, and
    # asq_survey_id are the same and only the method R is different. Preserve only the R match
    df_r = df_matches[df_matches['method'] == 'R']
    # Identify duplicates based on the key columns excluding the 'method'
    key_columns = ['asq_name', 'subject_name', 'score', 'asq_survey_id']

    # Identify rows in the original DataFrame that match the rows in df_r based on the key columns
    df_duplicates = df_matches[df_matches.duplicated(subset=key_columns, keep=False) &
                               (df_matches['method'] != 'R')]
    # Combine the filtered DataFrame with non-duplicate rows from the original DataFrame
    df_results = pd.concat([df_r, df_matches[~df_matches.index.isin(df_duplicates.index)]])
    df_results = df_results.drop_duplicates(subset=key_columns, keep='first').sort_index()
    df_results = df_results.sort_values(by=['score', 'subject_name'],
                                        ascending=[False, True], inplace=False)

    compare_students_matched(df_students=df_students,
                             df_match_res=df_results,
                             figsize=(12, 6),
                             output_path=config.get('results_path').joinpath('BarPlotMatchAllScores.png'))

    df_results.to_excel(config.get('results_path').joinpath('matches_scorer_result.xlsx'), index=False)

    # %% Count of uniques
    # use the count of unique subjects to determine how false positive should we remove per subject
    unique_subject_dict = df_results['subject_name'].value_counts().to_dict()
    unique_subject = pd.Series(unique_subject_dict)
    unique_subject = unique_subject.reset_index()
    unique_subject.columns = ['subject_name', 'count']

    idx_to_remove = []
    for idx, row in unique_subject.iterrows():
        # row = unique_subject.loc[0, :]
        name_ = row['subject_name']
        count_ = row['count']
        if count_ < 5:
            continue

        df_duplications = df_results[df_results['subject_name'] == name_]
        duplicate_indices = [index for index, value in df_duplications['score'].isin([100]).items() if not value]
        if len(duplicate_indices) > 0:
            idx_to_remove.extend(duplicate_indices)
            continue

    result_df_cleaned = df_results.drop(index=idx_to_remove)
    # subjects that appear only once are removed since we are searching for a change in responses from when the
    # quarter started and the quarter ended
    subject_counts = result_df_cleaned['subject_name'].value_counts()
    subjects_to_remove = subject_counts[subject_counts == 1].index
    df_results_pairs = result_df_cleaned[~result_df_cleaned['subject_name'].isin(subjects_to_remove)]

    compare_students_matched(df_students=df_students,
                             df_match_res=df_results_pairs,
                             figsize=(12, 6),
                             output_path=config.get('results_path').joinpath('BarPlotMatchPairUniques.png'))

    df_results_pairs.to_excel(config.get('results_path').joinpath('matches_scorer_pairs_manually_verified.xlsx'), index=False)

    # mark the asq partitions per quarter which users had been found
    asq_files = list(config.get('results_path').glob('asq_*.csv'))
    df_results_pairs['quarter'] = df_results_pairs['quarter'].apply(lambda x: x.replace('-', ' '))
    for asq_file in asq_files:
        asq_quarter = pd.read_csv(asq_file)
        asq_file_name = asq_file.name
        parts = asq_file_name.split('_')  # ['asq', 'Spring', '2014.csv']
        quarter = parts[1] + ' ' + parts[2].replace('.csv', '')  # 'Spring 2014'
        df_results_pairs_quarter = df_results_pairs.loc[df_results_pairs['quarter'] == quarter]['asq_survey_id'].to_list()
        asq_quarter['match'] = asq_quarter['survey_id'].isin(df_results_pairs_quarter).astype(int)
        # Reorder columns to put 'match' first
        cols = ['match'] + [col for col in asq_quarter.columns if col != 'match']
        asq_quarter = asq_quarter[cols]
        asq_quarter = asq_quarter.sort_values(by=['match', 'name', 'date_of_birth'], ascending=[False, True, False])
        asq_quarter.to_csv(asq_file, index=False)

    # %%
    # unique_subject_dict = df_matches['subject_name'].value_counts().to_dict()
    # unique_subject = pd.Series(unique_subject_dict)
    # unique_subject = unique_subject.reset_index()
    # unique_subject.columns = ['subject_name', 'count']
    # idx_to_remove = []
    # for idx, row in unique_subject.iterrows():
    #     name_ = row['subject_name']
    #     count_ = row['count']
    #     if count_ < 5:
    #         continue
    #     df_duplications = df_matches[df_matches['subject_name'] == name_]
    #
    #     if df_duplications['score'].max() == 100:
    #         indices_to_remove = df_duplications[df_duplications['score'] < 100,:]
    #         if
    #         idx_to_remove.extend(indices_to_remove)
    #
    # df_matches_cleaned = df_matches.drop(index=idx_to_remove)
    #
    # # Further remove subject_names that appear only once
    # subject_counts = df_matches_cleaned['subject_name'].value_counts()
    # subjects_to_remove = subject_counts[subject_counts == 1].index
    # df_matches_cleaned = df_matches_cleaned[~df_matches_cleaned['subject_name'].isin(subjects_to_remove)]
    #
    #
    #
    # # Example usage
    # compare_students_matched(df_students=df_students,
    #                          df_match_res=df_matches_cleaned,
    #                          figsize=(8, 6))
    #
    #
    # df_matches

    # %% Visualize the collection
    # 1. Bar plot on how many unique subject_names we have in each quarter period
    df_results = pd.read_excel(config.get('data_path').get('base').joinpath('matches_scorer_pairs_manually_verified.xlsx'))
    unique_subjects_per_quarter = df_results.groupby('quarter')['subject_name'].nunique()
    df_students_avail = df_students.groupby('Academic year').size()
    df_students_avail = df_students_avail.reset_index(name='Count')
    # Plot the 'unique_subjects_per_quarter' data
    unique_subjects_per_quarter.sort_index(inplace=True)

    # Plot the 'df_students_avail' data
    # df_students_avail.set_index('Academic year', inplace=True)
    # df_students_avail = df_students_avail.reindex(index=unique_subjects_per_quarter.index)  # Align the indexes

    df_students_avail['Dataset'] = 'Roster'
    unique_subjects_df = pd.DataFrame({
        'Quarter': unique_subjects_per_quarter.index,
        'Count': unique_subjects_per_quarter.values,
        'Dataset': 'ASQ'
    })

    df_students_avail['Quarter'] = df_students_avail['Academic year'].str.replace(' ', '-')
    df_combined = pd.concat([
        df_students_avail[['Quarter', 'Count', 'Dataset']],
        unique_subjects_df
    ])
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df_combined, x='Quarter', y='Count', hue='Dataset', palette='Set2', alpha=0.9)
    plt.title('Unique Subjects per Quarter (Roster vs Matched ASQ)', fontsize=16)
    plt.xlabel('Quarter', fontsize=14)
    plt.ylabel('Number of Unique Subjects', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(fontsize=12, title='Dataset')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.7)
    plt.show()





    # 2. How many unique subjects we have per score
    unique_subjects_per_score = df_results.groupby('score')['subject_name'].nunique()

    plt.figure(figsize=(10, 6))
    unique_subjects_per_score.plot(kind='bar', color='skyblue')
    plt.title('Unique Subjects per Score')
    plt.xlabel('Score')
    plt.ylabel('Number of Unique Subjects')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(alpha=.7)
    plt.show()

    # Additional plots for a good overview of the results
    # 3. Distribution of scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df_results['score'], bins=10, kde=True, color='salmon')
    plt.title('Distribution of Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.grid(alpha=.7)
    plt.show()

    # 4. Count plot of methods used
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_results, x='method', palette='pastel')
    plt.title('Count of Methods Used')
    plt.xlabel('Method')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.grid(alpha=.7)
    plt.show()

    # %% Visualize the students dataset
    df_students['Academic year'] = df_students['quarter'] + ' ' + df_students['year'].astype(str)
    # 1. Bar plot of the number of students per academic year
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_students, x='Academic year', color='skyblue')
    plt.title(f'Students Dataset \n Number of Students per Academic Year\nTotal: {df_students.shape[0]}')
    plt.xlabel('Academic Year')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=45)
    plt.grid(0.7)
    plt.tight_layout()
    plt.show()

    unique_subject_dict = df_results['subject_name'].value_counts().to_dict()
    unique_subject = pd.Series(unique_subject_dict)
    unique_subject = unique_subject.reset_index()
    unique_subject.columns = ['subject_name', 'count']

    unique_subjects = df_results[['subject_name', 'idx_student_db']].drop_duplicates(keep=False)
    # Step 2: Filter the dataframe to keep only the rows where subject_name is unique
    filtered_df = df_results[df_results['subject_name'].isin(unique_subjects)]
    # Step 3: Select the quarter column
    unique_subjects_quarter = filtered_df['quarter']

    # Combine df_students and df_results
    df_combined = {
        'Academic year': df_students['Academic year'].to_list() + unique_subjects_quarter.to_list(),
        'Dataset': ['Student Datase'] * df_students['Academic year'].shape[0] + ['Results'] *
                   unique_subjects_quarter.shape[0]
    }
    df_combined = pd.DataFrame(df_combined)
    df_combined['Academic year'] = df_combined['Academic year'].str.replace('-', ' ')

    # Bar plot of the number of students per academic year comparing df_results and df_students
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df_combined, x='Academic year', hue='Dataset', palette='viridis')
    plt.title('Number of Students per Academic Year (Comparison)')
    plt.xlabel('Academic Year')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=45)
    plt.legend(title='Dataset')
    plt.show()

    # If a subject_name has a score greater than 87 and method == 'R'
    # remove the subject name from the rest of the rows
    # subjects_good_match = df_matches[(df_matches['score'] >= 87) &
    #                                 (df_matches['method'] == 'R')
    #                                           ]['subject_name'].unique()
    #
    # df_filtered = df_matches[~((df_matches['subject_name'].isin(subjects_good_match)) &
    #                            (df_matches['score'] < 87) &
    #                            (df_matches['method'] != 'R'))]
    # # in the score of 100 we have repetitions of the different methods, preserve only the R method
    #
    # df_filtered_r_best = df_filtered[((df_filtered['subject_name'].isin(subjects_good_match)) &
    #                                           (df_matches['score'] >= 87) &
    #                                           (df_matches['method'] == 'R'))]
    #
    # df_filtered = df_filtered[df_filtered['score'] < 87]
    # df_filtered_r_best = pd.concat([df_filtered_r_best, df_filtered])
    # df_filtered_r_best = df_filtered_r_best.sort_values(by=['score', 'subject_name'],
    #                                     ascending=[False, True], inplace=False)

    # df_sorted = df_matches.sort_values(by=['subject_name', 'score'],
    #                                    inplace=False,
    #                                    ascending=[True, False])
    # df_grouped_sorted = df_sorted.groupby('score').apply(
    #     lambda x: x.sort_values(by=['asq_name', 'subject_name'], ascending=[True, True])).reset_index(drop=True)
    # df_grouped_sorted.sort_values(by=['score'], inplace=True, ascending=False)

    # if a subject is aleady matched at 100% remove it from the other ones
    subjects_with_100 = df_matches[df_matches['score'] == 100]['subject_name'].unique()
    df_matches_unique_100 = df_matches[
        ~((df_matches['subject_name'].isin(subjects_with_100)) & (df_matches['score'] != 100))]
    df_matches_unique_100.sort_values(by='score', inplace=True, ascending=False)
    df_matches_unique_100.loc[df_matches_unique_100['method'] == 'R', :]

    df_filtered = df_matches.groupby('subject_name').filter(lambda x: (x['score'] >= 90).any())

    df_filtered.sort_values(by='score', inplace=True)

    df_matches.method.unique()

    print('sdf')

    data = {
        'asq_name': ['A', 'A', 'B', 'B', 'C'],
        'subject_name': ['sub1', 'sub1', 'sub2', 'sub2', 'sub3'],
        'score': [1, 1, 2, 2, 3],
        'asq_survey_id': [101, 101, 102, 102, 103],
        'method': ['R', 'X', 'R', 'Y', 'R']
    }

    df_matches = pd.DataFrame(data)

    # Filter to preserve only the rows with method 'R' where other columns match
    df_filtered = df_matches[df_matches['method'] == 'R']

    # Merge the filtered DataFrame back with the original DataFrame on the key columns,
    # but only keeping the rows from the original that are not in the filtered DataFrame
    merged_df = df_matches.merge(df_filtered, on=['asq_name', 'subject_name', 'score', 'asq_survey_id'], how='outer',
                                 indicator=True)

    # Keep only the rows from the filtered DataFrame and the rows from the original DataFrame that were not filtered out
    df_results = merged_df[(merged_df['_merge'] == 'both') | (merged_df['_merge'] == 'left_only')]

    # Drop the '_merge' column as it is no longer needed
    df_results = df_results.drop(columns=['_merge'])
