import pathlib

#%% define the directories
# Define root path
root_path = pathlib.Path(__file__).resolve().parents[1]
# Define raw data path
data_raw_path = root_path.joinpath('data')
results_path = root_path.joinpath('results')
logging_path = root_path.joinpath('logging')

# Construct the config dictionary
config = {
    'root_path': root_path,
    'data_path': {
        'base': data_raw_path,
        'asq': data_raw_path.joinpath('asq'),
        'students': data_raw_path.joinpath('class_students'),
        'dataset_students': data_raw_path.joinpath('dataset_students.csv'),
        'calendar': data_raw_path.joinpath('calendar.xlsx'),
        'pp_dataset': data_raw_path.joinpath('pproc').joinpath('asq_pair.csv'),
        'asq_dictionary': data_raw_path.joinpath('asq_dictionary_v4.xlsx'),
    },
    'results_path': results_path,
    'logging_path': logging_path,
}

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
    'soclhx_0500': 'Exercise per week',
    # columns that will not be used for the statistical significance
    'sched_total_sleep_time_week': 'Total Sleep Time Week',
    'sched_total_sleep_time_weekend': 'Total Sleep Time Weekend',
    'sched_total_sleep_latency_week': 'Total Sleep Latency Week',
    'sched_total_sleep_latency_weekend': 'Total Sleep Time Weekend',
    # 'sched_bed_time_week_diff': 'Bed Time Week Difference',
    # 'sched_bed_time_weekend_diff': 'Bed Time Weekend Difference',
}
dtypes = {
    'dem_0110': 'continuous',
    'dem_0500': 'binary',
    'dem_0800': 'continuous',
    'dem_1000': 'categorical',
    'map_0300': 'ordinal',
    'osa_0200': 'ordinal',
    'ess_0900': 'continuous',
    'isi_0100': 'ordinal',
    'isi_0200': 'ordinal',
    'isi_0300': 'ordinal',
    'cir_0200': 'categorical',
    'fosq_0100': 'ordinal',
    'fosq_0200': 'ordinal',
    'gad_0900': 'continuous',
    'phq_1100': 'continuous',
    'fosq_1100': 'continuous',
    'isi_score': 'continuous',
    'soclhx_0500': 'continuous',
    'sched_total_sleep_time_week': 'continuous',
    'sched_total_sleep_time_weekend': 'continuous',
    'sched_total_sleep_latency_week': 'continuous',
    'sched_total_sleep_latency_weekend': 'continuous',
    # 'sched_bed_time_week_diff': 'continuous',
    # 'sched_bed_time_weekend_diff': 'continuous',
}
