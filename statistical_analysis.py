"""
Statistical Analysis Script for Within-Subject Comparisons

Author: giocrm@stanford.edu

Description:
This script performs statistical analysis for within-subject comparisons of sleep-related variables.
Key features include:
- Implementation of paired statistical tests (e.g., Wilcoxon Signed-Rank, Paired t-test, McNemar).
- Analysis of continuous, binary, ordinal, and categorical data types.
- Calculation of effect sizes and p-value adjustments using the Benjamini-Hochberg procedure.
- Visualization of data distributions and test results.
- Structured to handle large datasets with customizable configurations.

Dependencies:
- NumPy, pandas, SciPy, StatsModels, Seaborn, Matplotlib, Tabulate, Graphviz, Pingouin.

Usage:
Configure paths and data types in `configuration.config` before running the script.

"""

from library.table_one import MakeTableOne
import numpy as np
import pandas as pd
from configuration.config import config, aliases, dtypes
from library.table_one import MakeTableOne
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from scipy.stats import chi2
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from sklearn.utils import resample
from tqdm import tqdm
from library.within_tests import format_p_value, apply_statistical_tests_with_bootstrap, apply_statistical_tests, calculate_wilcoxon_sample_size


if __name__ == "__main__":
    df_data = pd.read_csv(config.get('data_path').get('pp_dataset'))
    # %% define the output folders
    output_path_stats = config.get('results_path').joinpath('statistics')
    output_path_plots = config.get('results_path').joinpath('plots')

    if not output_path_stats.exists():
        output_path_stats.mkdir(exist_ok=True)

    if not output_path_plots.exists():
        output_path_plots.mkdir(exist_ok=True)

    # df_dict = pd.read_excel(config.get('data_path')['base'].joinpath('asq_dictionary_v4.xlsx'))
    # df_dict_scoring = df_dict.loc[df_dict['Column Name'].isin(aliases.keys()), ['Column Name', 'Numeric Scoring Code ']]

    # remove the variables that are constant within subjects (Age [dem_0110], Gender [dem_0500], Race [dem_1000])
    df_data.drop(columns=['dem_0110', 'dem_0500', 'dem_1000'], inplace=True)

    # %% Evaluate time features
    cols = ['sched_total_sleep_time_week',
         'sched_total_sleep_time_weekend',
         'sched_total_sleep_latency_week',
         'sched_total_sleep_latency_weekend',]
    df_data.groupby(by='time')['sched_total_sleep_time_week'].mean()/60
    df_data.groupby(by='time')['sched_total_sleep_time_week'].std()/60


    # %% Within subject comparison.
    # We want to compare how the results of the same subject changed after taking the class. We will not compare
    # between subjects e.g., student 1 with student 20.
    results = apply_statistical_tests(df=df_data,
                                      dtypes=dtypes,
                                      pair_id='id_subject')
    df_results = pd.DataFrame(results).T
    df_results.reset_index(inplace=True, drop=False, names='variable')
    # df_results = df_results.loc[~df_results['variable'].isin(['dem_0110', 'dem_0500']), :]
    df_results['variable'] = df_results.variable.map(aliases)

    # Apply the Benjamini-Hochberg procedure to adjust the p-values
    df_results['p_value_adjusted'] = multipletests(df_results['p_value'], method='fdr_bh')[1]
    df_results['effect_size'] = df_results['effect_size'].round(4)

    # format the p values
    df_results['p_value_formatted'] = df_results.p_value.apply(format_p_value)
    df_results['p_value_adjusted_formatted'] = df_results.p_value_adjusted.apply(format_p_value)

    df_results.sort_values(by='p_value_adjusted_formatted', ascending=True, inplace=True)

    df_results.to_excel(output_path_stats.joinpath('hypothesis_testing.xlsx'), index=False)

    # %% Bootstrap to compute better effect size computation
    boot_res = apply_statistical_tests_with_bootstrap(df=df_data,
                                                      dtypes=dtypes,
                                                      pair_id='id_subject',
                                                      n_bootstraps=2000
                                                      )
    df_boot_res = pd.DataFrame(boot_res).T
    df_boot_res.reset_index(inplace=True, drop=False, names='variable')
    df_boot_res['variable'] = df_boot_res.variable.map(aliases)
    df_boot_res['p_value_adjusted'] = multipletests(df_boot_res['p_value'], method='fdr_bh')[1]
    for col_round in ['boot_effect_size', 'wilcoxon_effect_size', 'ci_upper', 'ci_lower']:
        df_boot_res[col_round] = df_boot_res[col_round].astype(float)
        df_boot_res[col_round] = df_boot_res[col_round].round(3)

    for dfloat in ['p_value', 'p_value_adjusted']:
        df_boot_res[dfloat] = df_boot_res[dfloat].astype(float)

    df_boot_res.sort_values(by='p_value_adjusted', ascending=True, inplace=True)

    df_boot_res['p_value_adjusted_formatted'] = df_boot_res.p_value_adjusted.apply(format_p_value)
    df_boot_res['p_value_formatted'] = df_boot_res.p_value.apply(format_p_value)

    df_boot_res['CI'] = df_boot_res[['ci_lower', 'ci_upper']].apply(lambda row: [row['ci_lower'], row['ci_upper']],
                                                                    axis=1)

    df_boot_res.to_excel(output_path_stats.joinpath('hypothesis_testing_bootstrap.xlsx'), index=False)


    # %% Power calculation
    sample_size = len(df_data)
    # Assuming alpha = 0.05, effect size = 0.5 (medium), and a power of 0.8
    effect_size = 0.5
    alpha = 0.05
    power = 0.8

    # Initialize the power analysis
    analysis = TTestIndPower()

    # Calculate the required sample size
    required_sample_size = analysis.solve_power(effect_size=effect_size,
                                                alpha=alpha,
                                                power=power,
                                                alternative='two-sided')

    # Compare the required sample size with the actual sample size
    if sample_size >= required_sample_size:
        print(f"Sample size is sufficient: {sample_size} participants (required: {int(required_sample_size)}).")
    else:
        print(f"Sample size is insufficient: {sample_size} participants (required: {int(required_sample_size)}).")

    # Output justification
    print(f"Sample size justification: With an assumed effect size of {effect_size}, alpha of {alpha}, "
          f"and a power of {power}, the minimum required sample size is {int(required_sample_size)}.")

    equired_sample_size = calculate_wilcoxon_sample_size(effect_size=0.5, alpha=0.05, power=0.8)
    print(f"Required sample size for Wilcoxon signed-rank test: {required_sample_size} pairs")

    # %% manually do the statistics
    # from scipy.stats import skew
    # def wilcoxon_signed_rank_manual(x, y):
    #     # Step 1: Calculate Differences
    #     differences = np.array(x) - np.array(y)
    #     print("Step 1: Differences (X - Y):", differences)
    #
    #     # Step 2: Exclude Zero Differences
    #     non_zero_diff = differences[differences != 0]
    #     print("Step 2: Non-zero Differences:", non_zero_diff)
    #
    #     # Step 3: Rank Absolute Differences
    #     abs_diff = np.abs(non_zero_diff)
    #     ranks = pd.Series(abs_diff).rank(method="average").values
    #     print("Step 3: Ranks of Absolute Differences:", ranks)
    #
    #     # Step 4: Assign Signed Ranks
    #     signed_ranks = np.sign(non_zero_diff) * ranks
    #     print("Step 4: Signed Ranks:", signed_ranks)
    #
    #     # Step 5: Sum Positive and Negative Ranks
    #     w_plus = signed_ranks[signed_ranks > 0].sum()
    #     w_minus = signed_ranks[signed_ranks < 0].sum()
    #     print("Step 5: W+ (Positive Rank Sum):", w_plus)
    #     print("Step 5: W- (Negative Rank Sum):", w_minus)
    #
    #     # Step 6: Compute Test Statistic
    #     w_statistic = min(w_plus, abs(w_minus))
    #     print("Step 6: Test Statistic (W):", w_statistic)
    #
    #     # Visualize Each Step
    #     plt.bar(range(len(non_zero_diff)), non_zero_diff, color="skyblue", label="Differences")
    #     plt.axhline(0, color="gray", linestyle="--", label="Zero Line")
    #     plt.xticks(range(len(non_zero_diff)), labels=[f"Pair {i + 1}" for i in range(len(non_zero_diff))])
    #     plt.ylabel("Differences (X - Y)")
    #     plt.title("Step 1: Differences Between Paired Values")
    #     plt.legend()
    #     plt.show()
    #
    #     return {
    #         "differences": differences,
    #         "non_zero_diff": non_zero_diff,
    #         "ranks": ranks,
    #         "signed_ranks": signed_ranks,
    #         "w_plus": w_plus,
    #         "w_minus": w_minus,
    #         "test_statistic": w_statistic
    #     }
    #
    #
    # col = 'isi_0300'
    # pair_id = 'id_subject'
    # paired_data = df_data[[col, pair_id]].dropna()
    # paired_data = paired_data.groupby(pair_id).agg(list)
    # # Ensure we have exactly two observations per subject
    # paired_data = paired_data[paired_data[col].apply(len) == 2]
    # # Extract paired values
    # values1 = [pair[0] for pair in paired_data[col]]
    # values2 = [pair[1] for pair in paired_data[col]]
    #
    # # Calculate skewness
    # skewness = skew(values1)
    # print(f"Skewness: {skewness:.2f}")
    #
    # np.mean(values1), np.median(values1)
    #
    # np.mean(values2), np.median(values2)
    #
    # wilcoxon_signed_rank_manual(x=values1,
    #                             y=values2)
    #
    # import pingouin as pg
    # diff = np.array(values1) - np.array(values2)
    # np.mean(diff), np.median(diff)  # close to zero so wilcoxon test is okay
    # pg.wilcoxon(diff, alternative='two-sided', correction=True)
    #
    # stats_wilcoxon = stats.wilcoxon(x=diff,
    #                                 # y=values2,
    #                                 correction=True,
    #                                 zero_method='wilcox',
    #                                 alternative='two-sided',
    #                                 method='approx')
    # stat = stats_wilcoxon.statistic
    # n = len(diff[diff != 0])  # Exclude zero differences (ties)
    # # Calculate expected value and variance under H0
    # E_T_plus = n * (n + 1) / 4
    # Var_T_plus = n * (n + 1) * (2 * n + 1) / 24
    # # Calculate Z-score
    # Z = (stat - E_T_plus) / np.sqrt(Var_T_plus)
    # # Calculate effect size r
    # r = Z / np.sqrt(n)
    #

    # %% Time variables
    # Varibales contain NaNs sow we cannot use the full datase
    # use the fetures: sched_0900, sched_1000, sched_1900, sched_2000


