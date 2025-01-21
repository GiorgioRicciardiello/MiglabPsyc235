"""
Create the students datasets.
"""
import pandas as pd
from pathlib import Path
from configuration.config import config

def main():
    path_students = Path(config.get('data_path').get('students'))
    files_roster = path_students.glob('*.xlsx')
    df_students = pd.DataFrame()
    for file in files_roster:
        name = file.stem.replace('  ', ' ')
        if 'psyc' not in name.lower():
            continue
        df = pd.read_excel(file)
        df['quarter'] = name.split(' ')[2]
        df['year'] = int(name.split(' ')[3])
        df_students = pd.concat([df_students, df])
    df_students.reset_index(drop=True, inplace=True)
    # Filter out rows where 'ID' is NaN
    df_students = df_students[~df_students['ID'].isna()]
    # Drop columns that are all NaNs
    df_students = df_students.dropna(axis=1, how='all')
    # remove the withdrawn students
    df_students = df_students[df_students['Status Note'].isna()]
    # take the columns of interest
    df_students = df_students[['ID', 'Name', 'Level', 'quarter', 'year']]
    # Clean Name
    df_students['Name'] = df_students['Name'].apply(lambda x: ' '.join(x.split(',')[::-1]) if ',' in x else x)
    df_students.rename(columns={'Name':'name'}, inplace=True)
    df_students.to_csv(config.get('data_path').get('dataset_students'), index=False)

if __name__ == '__main__':
    main()




