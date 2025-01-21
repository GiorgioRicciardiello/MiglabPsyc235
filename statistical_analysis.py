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
from tabulate import tabulate
from graphviz import Digraph
from statsmodels.stats.multitest import multipletests
from collections import Counter


# def apply_statistical_tests(df:pd.DataFrame,
#                             dtypes:dict[str, str],
#                             col_strata:str):
#     """
#     Perform within-subject comparison statistics
#     :param df:
#     :param dtypes:
#     :return:
#     """
#     results = {}
#     group_names = df[col_strata].unique()
#     for col, dtype in dtypes.items():
#         # col = 'ess_score'
#         # dtype = dtypes.get(col)
#         # Extract the subject pairs for the given column
#         paired_data = df[[col, 'id_subject']].dropna()
#         paired_data = paired_data.groupby('id_subject').agg(list)
#
#         # Ensure we have exactly two observations per subject
#         paired_data = paired_data[paired_data[col].apply(len) == 2]
#         # Remove rows where any value in the pair is negative
#         paired_data = paired_data[paired_data[col].apply(lambda x: all(val >= 0 for val in x))]
#
#         # values1 = paired_data.loc[paired_data[col_strata] == group_names[0]]
#         # values2 = paired_data.loc[paired_data[col_strata] == group_names[1]]
#         values1 = [pair[0] for pair in paired_data[col]]
#         values2 = [pair[1] for pair in paired_data[col]]
#
#         # df_values1 = pd.DataFrame(values1)
#         # sns.histplot(df_values1, bins=30, kde=True)
#         # plt.show()
#         # unique_counts = {item: values1.count(item) for item in set(values1)}
#
#         n = len(values1)
#         if dtype == 'continuous':
#             stat, normality_p = stats.shapiro(np.array(values1) - np.array(values2))
#             # if normality_p > 0.05 or stat < 0.5:
#             if stat < 0.6:
#                 stats_wilcoxon = stats.wilcoxon(x=values1,
#                                            y=values2,
#                                            zero_method="pratt",
#                                            alternative="two-sided")
#                 t_stat = stats_wilcoxon.statistic
#                 p_val = stats_wilcoxon.pvalue
#                 method = 'Wilcoxon'
#             else:
#                 # Paired t-test
#                 t_stat, p_val = stats.ttest_rel(a=values1,
#                                                 b=values2,
#                                                 alternative='two-sided')
#                 method = 'Paired t-test'
#
#         elif dtype == 'binary':
#             table = pd.crosstab(np.array(values1), np.array(values2))
#             b = table.iloc[0, 1]  # Discordant pair (Yes, No)
#             c = table.iloc[1, 0]  # Discordant pair (No, Yes)
#
#             if (b + c) < 25:
#                 result = mcnemar(table, exact=True)
#             else:
#                 result = mcnemar(table, exact=False)
#             p_val = result.pvalue
#             t_stat = None
#             method = 'McNemar Test'
#             effect_size = b / (b + c) if (b + c) > 0 else None  # Proportion of discordance
#
#         elif dtype == 'ordinal':
#             # Wilcoxon signed-rank test for ordinal data, before-after or time_0 - time_1
#             diff = list(set(values1) - set(values2))  # warning: compute the difference before to avoid errors in approx
#             stats_wilcoxon = stats.wilcoxon(x=diff,
#                                             # y=values2,
#                                             zero_method='pratt',
#                                             alternative='two-sided')
#             t_stat = stats_wilcoxon.statistic
#             p_val = stats_wilcoxon.pvalue
#             method = 'Wilcoxon Signed-Rank Test'
#
#         elif dtype == 'categorical':
#             # McNemar's test or Cochran’s Q test could be considered for categorical, but let's stick to McNemar for binary categories
#             table = pd.crosstab(np.array(values1), np.array(values2))
#
#             if table.shape == (2, 2):  # Binary categorical data
#                 result = mcnemar(table, exact=True)
#                 p_val = result.pvalue
#                 t_stat = None
#                 method = 'McNemar Test'
#             elif table.shape[1] > 2:
#                 result = cochrans_q(table)
#                 t_stat = result.statistic
#                 p_val = result.pvalue
#                 method = "Cochran's Q Test"
#             else:
#                 # If more than two categories, no direct method, we could expand this to include Cochran's Q test
#                 p_val = None
#                 t_stat = None
#                 method = 'Unsupported Categorical Test'
#         else:
#             p_val = None
#             t_stat = None
#             method = 'Unknown'
#
#         results[col] = {
#             'sample_size': n,
#             'p_value': p_val,
#             'effect_size': t_stat,
#             'method': method,
#             'dtype': dtype,
#         }
#
#     return results
#


def format_p_value(p,
                   decimal_places:Optional[int]=4,
                   sci_decimal_places:Optional[int]=2):
    """
    Format p-values to display in scientific notation if very small,
    with customizable decimal places for both formats.

    Args:
        p (float): The p-value to format.
        decimal_places (int): Number of decimal places for regular rounding.
        sci_decimal_places (int): Number of decimal places for scientific notation.

    Returns:
        str: Formatted p-value as a string.

    Example Usage:
    # Apply formatting with dynamic decimal places
    df_results['p_value_formatted'] = df_results.p_value.apply(lambda p: format_p_value(p, decimal_places=5, sci_decimal_places=3))
        p_value	        p_value_formatted
        2.985507e-07	    2.986e-07
        3.904867e-07	    3.905e-07
        1.077162e-06	    1.077e-06
        2.367503e-05	    0.00002
        4.527726e-05	    0.00005

    """
    if p < 10 ** (-decimal_places):
        return f"{p:.{sci_decimal_places}e}"  # Scientific notation
    return round(p, decimal_places)  # Regular rounding

def apply_statistical_tests(df: pd.DataFrame,
                            dtypes: dict[str, str],
                            pair_id:str='id_subject'):
    """
    Perform within-subject comparison statistics with rigorous measures.
    :param df: DataFrame containing the data.
    :param dtypes: Dictionary mapping column names to their data types.
    :param col_strata: Column used for grouping.
    :return: Dictionary with statistical results for each variable.
    """

    def compute_wilcoxon(diff_:np.ndarray) -> dict:
        """
        Compute the wilcoxon test and compute the effect size
        :param diff_:
        :return:
        """
        stats_wilcoxon = stats.wilcoxon(x=diff_,
                                        # y=values2,
                                        correction=True,
                                        zero_method='wilcox',
                                        alternative='two-sided',
                                        method='approx')
        stat = stats_wilcoxon.statistic
        p_val = stats_wilcoxon.pvalue
        n = len(diff_[diff_ != 0])  # Exclude zero differences (ties)
        # Calculate expected value and variance under H0
        E_T_plus = n * (n + 1) / 4
        Var_T_plus = n * (n + 1) * (2 * n + 1) / 24
        # Calculate Z-score
        Z = (stat - E_T_plus) / np.sqrt(Var_T_plus)
        # Calculate effect size r
        effect_size = Z / np.sqrt(n)
        method = 'Wilcoxon Signed-Rank Test'
        return {'effect_size':effect_size,
                'statistic':stat,
                'p_value':p_val,
                'method':method}

    # def compute_effect_size(t_stat, n):
    #     """
    #     Compute the effect size for Wilcoxon Signed-Rank Test using standardized test statistic.
    #     https://www.youtube.com/watch?v=74vSNwyd2ys&ab_channel=Statorials
    #
    #     Using the Expected Sum of Ranks: n*(n+1)/4
    #     Standard deviation of ranks: sqrt((n(n+1)(2n+1))/6)
    #     Standardize the test statistic: Z = (W-Expected Sum of Ranks)/ Standard Deviation of Ranks
    #     Parameters:
    #     - t_stat (float): Wilcoxon test statistic.
    #     - n (int): Total sample size (non-zero differences).
    #
    #     Returns:
    #     - float: Standardized effect size (Cohen's d equivalent for Wilcoxon).
    #     """
    #     # Compute the standard deviation of the ranks
    #     sd_ranks = np.sqrt(n * (n + 1) * (2 * n + 1) / 6)
    #
    #     # Standardize the test statistic
    #     z_stat = (t_stat - (n * (n + 1) / 4)) / sd_ranks
    #
    #     # Return the absolute value of the effect size
    #     return np.abs(z_stat)

    results = {}
    for col, dtype in dtypes.items():
        print(f'Hypothesis testing: {col} - dtype: {dtype}\n')
        # Extract the subject pairs for the given column
        paired_data = df[[col, pair_id]].copy().dropna()
        paired_data = paired_data.groupby(pair_id).agg(list)

        # Ensure we have exactly two observations per subject
        paired_data = paired_data[paired_data[col].apply(len) == 2]

        # Remove rows where any value in the pair is negative
        paired_data = paired_data[paired_data[col].apply(lambda x: all(val >= 0 for val in x))]

        # Convert non-continuous values to integers
        if dtype in ['binary', 'ordinal', 'categorical']:
            paired_data[col] = paired_data[col].apply(lambda x: [int(val) for val in x])

        # Extract paired values
        values1 = [pair[0] for pair in paired_data[col]]
        values2 = [pair[1] for pair in paired_data[col]]

        n = len(values1)

        if n == 0:
            results[col] = {
                'sample_size': n,
                'statistic': None,
                'p_value': None,
                'effect_size': None,
                'method': 'No valid pairs',
                'dtype': dtype,
            }
            continue

        diff = np.round(np.array(values1) - np.array(values2), 3)
        # symmetry: np.mean(diff), np.median(diff) -> must be similar
        if dtype == 'continuous':
            stat, normality_p = stats.shapiro(diff)
            print(f'\tShapiro test: {col}\t {stat}, {normality_p}\n')
            if normality_p >= 0.05:  # Data is normally distributed
                t_stat, p_val = stats.ttest_rel(values1, values2)
                method = 'Paired t-test'
                # effect_size = t_stat
                # Compute Cohen's d for paired samples
                mean_diff = np.mean(diff)
                sd_diff = np.std(diff, ddof=1)
                effect_size = mean_diff / sd_diff  # Cohen's d

            else:  # Data is not normally distributed
                res = compute_wilcoxon(diff_=diff)
                effect_size = res.get('effect_size')
                t_stat = res.get('statistic')
                p_val = res.get('p_value')
                method = res.get('method')

        elif dtype == 'binary':
            table = pd.crosstab(np.array(values1), np.array(values2))
            b = table.iloc[0, 1]  # Discordant pair (Yes, No)
            c = table.iloc[1, 0]  # Discordant pair (No, Yes)

            if (b + c) < 25:
                result = mcnemar(table, exact=True)
            else:
                result = mcnemar(table, exact=False)
            p_val = result.pvalue
            t_stat = None
            method = 'McNemar Test'
            effect_size = b / (b + c) if (b + c) > 0 else None  # Proportion of discordance

        elif dtype == 'ordinal':
            # Wilcoxon signed-rank test for ordinal data, before-after or time_0 - time_1
            res = compute_wilcoxon(diff_=diff)
            effect_size = res.get('effect_size')
            t_stat = res.get('statistic')
            p_val = res.get('p_value')
            method = res.get('method')

        elif dtype == 'categorical':
            table = pd.crosstab(np.array(values1), np.array(values2))

            if table.shape == (2, 2):  # Binary categorical data
                result = mcnemar(table, exact=True)
                p_val = result.pvalue
                t_stat = result.statistic
                b = table.iloc[0, 1]  # Off-diagonal counts
                c = table.iloc[1, 0]
                odds_ratio = b / c if c != 0 else np.inf
                effect_size = np.log(odds_ratio)  # Log odds ratio as effect size
                method = 'McNemar Test'

            else:
                # Stuart-Maxwell Test for multi-categorical data
                marginals = table.sum(axis=1) - table.sum(axis=0)
                observed = marginals ** 2
                expected = table.sum().sum() / (2 * table.shape[0])  # Expected under null
                test_stat = (observed / expected).sum()

                # Degrees of freedom
                dof = table.shape[0] - 1
                p_val = chi2.sf(test_stat, dof)

                # Effect size: Cramér's V
                total = table.values.sum()
                effect_size = np.sqrt(test_stat / (total * (table.shape[0] - 1)))

                t_stat = np.sqrt(test_stat)  # Pseudo t-statistic
                method = 'Stuart-Maxwell Test'


        else:
            p_val = None
            t_stat = None
            method = 'Unknown'
            effect_size = None

        results[col] = {
            'sample_size': n,
            'p_value': p_val,
            'effect_size': effect_size,
            'statistic': t_stat,
            'method': method,
            'dtype': dtype,
        }

    return results



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
    # remove the variables that are constant within subjects (Age, Gender)
    df_results = df_results.loc[~df_results['variable'].isin(['dem_0110', 'dem_0500']), :]
    df_results['variable'] = df_results.variable.map(aliases)
    df_results.sort_values(by='p_value', ascending=True, inplace=True)

    # Apply the Benjamini-Hochberg procedure to adjust the p-values
    df_results['p_value_adjusted'] = multipletests(df_results['p_value'], method='fdr_bh')[1]
    df_results['effect_size'] = df_results['effect_size'].round(4)

    # format the p values
    df_results['p_value_formatted'] = df_results.p_value.apply(format_p_value)
    df_results['p_value_adjusted_formatted'] = df_results.p_value_adjusted.apply(format_p_value)

    df_results.to_excel(output_path_stats.joinpath('hypothesis_testing.xlsx'), index=False)

    # %% manually do the statistics
    from scipy.stats import skew
    def wilcoxon_signed_rank_manual(x, y):
        # Step 1: Calculate Differences
        differences = np.array(x) - np.array(y)
        print("Step 1: Differences (X - Y):", differences)

        # Step 2: Exclude Zero Differences
        non_zero_diff = differences[differences != 0]
        print("Step 2: Non-zero Differences:", non_zero_diff)

        # Step 3: Rank Absolute Differences
        abs_diff = np.abs(non_zero_diff)
        ranks = pd.Series(abs_diff).rank(method="average").values
        print("Step 3: Ranks of Absolute Differences:", ranks)

        # Step 4: Assign Signed Ranks
        signed_ranks = np.sign(non_zero_diff) * ranks
        print("Step 4: Signed Ranks:", signed_ranks)

        # Step 5: Sum Positive and Negative Ranks
        w_plus = signed_ranks[signed_ranks > 0].sum()
        w_minus = signed_ranks[signed_ranks < 0].sum()
        print("Step 5: W+ (Positive Rank Sum):", w_plus)
        print("Step 5: W- (Negative Rank Sum):", w_minus)

        # Step 6: Compute Test Statistic
        w_statistic = min(w_plus, abs(w_minus))
        print("Step 6: Test Statistic (W):", w_statistic)

        # Visualize Each Step
        plt.bar(range(len(non_zero_diff)), non_zero_diff, color="skyblue", label="Differences")
        plt.axhline(0, color="gray", linestyle="--", label="Zero Line")
        plt.xticks(range(len(non_zero_diff)), labels=[f"Pair {i + 1}" for i in range(len(non_zero_diff))])
        plt.ylabel("Differences (X - Y)")
        plt.title("Step 1: Differences Between Paired Values")
        plt.legend()
        plt.show()

        return {
            "differences": differences,
            "non_zero_diff": non_zero_diff,
            "ranks": ranks,
            "signed_ranks": signed_ranks,
            "w_plus": w_plus,
            "w_minus": w_minus,
            "test_statistic": w_statistic
        }


    col = 'isi_0300'
    pair_id = 'id_subject'
    paired_data = df_data[[col, pair_id]].dropna()
    paired_data = paired_data.groupby(pair_id).agg(list)
    # Ensure we have exactly two observations per subject
    paired_data = paired_data[paired_data[col].apply(len) == 2]
    # Extract paired values
    values1 = [pair[0] for pair in paired_data[col]]
    values2 = [pair[1] for pair in paired_data[col]]

    # Calculate skewness
    skewness = skew(values1)
    print(f"Skewness: {skewness:.2f}")

    np.mean(values1), np.median(values1)

    np.mean(values2), np.median(values2)

    wilcoxon_signed_rank_manual(x=values1,
                                y=values2)

    import pingouin as pg
    diff = np.array(values1) - np.array(values2)
    np.mean(diff), np.median(diff)  # close to zero so wilcoxon test is okay
    pg.wilcoxon(diff, alternative='two-sided', correction=True)

    stats_wilcoxon = stats.wilcoxon(x=diff,
                                    # y=values2,
                                    correction=True,
                                    zero_method='wilcox',
                                    alternative='two-sided',
                                    method='approx')
    stat = stats_wilcoxon.statistic
    n = len(diff[diff != 0])  # Exclude zero differences (ties)
    # Calculate expected value and variance under H0
    E_T_plus = n * (n + 1) / 4
    Var_T_plus = n * (n + 1) * (2 * n + 1) / 24
    # Calculate Z-score
    Z = (stat - E_T_plus) / np.sqrt(Var_T_plus)
    # Calculate effect size r
    r = Z / np.sqrt(n)


    # %% Time variables
    # Varibales contain NaNs sow we cannot use the full datase
    # use the fetures: sched_0900, sched_1000, sched_1900, sched_2000

    # %% plot
    data = {
        "Spring-2014": 250,
        "Spring-2015": 206,
        "Spring-2016": 219,
        "Spring-2017": 253,
        "Spring-2018": 19,
        "Spring-2020": 226,
        "Spring-2021": 254,
        "Spring-2022": 127,
        "Winter-2014": 50,
        "Winter-2015": 226,
        "Winter-2016": 246,
        "Winter-2017": 257,
        "Winter-2018": 244,
        "Winter-2019": 259,
        "Winter-2020": 247,
        "Winter-2021": 275,
    }
    # Extract data for plotting
    quarters = list(data.keys())
    counts = list(data.values())

    # Plot settings
    plt.figure(figsize=(10, 6))
    plt.bar(quarters, counts, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel("Quarter", fontsize=12)
    plt.ylabel("Student Participation Count", fontsize=12)
    plt.title("Student Participation by Quarter", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid lines for clarity
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout for a professional look
    plt.tight_layout()

    # Save the figure
    # plt.savefig("student_participation_by_quarter.png", dpi=300)

    # Display the plot
    plt.show()

