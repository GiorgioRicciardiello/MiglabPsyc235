import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from scipy.stats import chi2
from typing import Optional, Union, Dict
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from sklearn.utils import resample
from tqdm import tqdm

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
        if not col in df.columns:
            continue
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


def calculate_wilcoxon_sample_size(effect_size, alpha=0.05, power=0.8):
    """
    Calculate the required sample size for the Wilcoxon signed-rank test.

    Parameters:
    - effect_size (float): Expected effect size (r).
    - alpha (float): Significance level (default is 0.05).
    - power (float): Statistical power (default is 0.8).

    Returns:
    - sample_size (int): Required number of pairs.
    """
    # Initialize power analysis
    analysis = NormalIndPower()

    # Calculate the z-scores for alpha (two-sided) and power
    z_alpha = np.abs(np.percentile(np.random.normal(size=100000), alpha / 2 * 100))
    z_beta = np.abs(np.percentile(np.random.normal(size=100000), alpha * 100))

    # Compute the required sample size (approximation based on normality assumption)
    sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')

    return int(np.ceil(sample_size))


def apply_statistical_tests_with_bootstrap(df: pd.DataFrame,
                                           dtypes: dict[str, str],
                                           pair_id: str = 'id_subject',
                                           n_bootstraps: int = 2000,
                                           method:str='cohen'):
    """
    Perform within-subject comparison statistics with rigorous measures and bootstrap-based Cohen's d.
    :param df: DataFrame containing the data.
    :param dtypes: Dictionary mapping column names to their data types.
    :param pair_id: Column used for grouping.
    :param n_bootstraps: Number of bootstrap iterations.
    :param method: Method to use for bootstrapping. Cohen's d., else wilcoxon r=Z/N'
    :return: Dictionary with statistical results for each variable.
    """
    def compute_wilcoxon(diff_: np.ndarray) -> dict:
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
        return {'effect_size': effect_size,
                'statistic': stat,
                'p_value': p_val,
                'method': method}


    def bootstrap_wilcoxon(diff_: np.ndarray,
                           n_bootstraps:int=2000) -> dict:
        """
        Compute the Wilcoxon test and compute the effect size
        :param diff_: Differences between paired values
        :return: Dictionary with Wilcoxon results and effect size
        """
        original_effect_size = compute_wilcoxon(diff_).get('statistic')
        boot_effect_sizes = []
        for _ in range(n_bootstraps):
            boot_diff = resample(diff_)
            boot_stat = compute_wilcoxon(boot_diff).get('statistic')
            boot_E_T_plus = len(boot_diff) * (len(boot_diff) + 1) / 4
            boot_Var_T_plus = len(boot_diff) * (len(boot_diff) + 1) * (2 * len(boot_diff) + 1) / 24
            boot_Z = (boot_stat - boot_E_T_plus) / np.sqrt(boot_Var_T_plus)
            boot_effect_sizes.append(boot_Z / np.sqrt(len(boot_diff)))
        # Calculate bootstrap confidence intervals for effect size
        ci_lower = np.percentile(boot_effect_sizes, 2.5)
        ci_upper = np.percentile(boot_effect_sizes, 97.5)
        return {
            'effect_size': original_effect_size,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': 'Wilcoxon Signed-Rank Test'
        }


    def bootstrap_cohen_d(diff:np.ndarray,
                          n_bootstraps=2000):
        """
        Calculate Cohen's d using bootstrapping.
        :param values1: First set of paired values.
        :param values2: Second set of paired values.
        :param n_bootstraps: Number of bootstrap iterations.
        :return: Mean Cohen's d with bootstrap confidence intervals.
        """
        original_mean_diff = np.mean(diff)
        original_std_diff = np.std(diff, ddof=1)
        original_cohen_d = original_mean_diff / original_std_diff

        boot_cohen_ds = []
        for _ in range(n_bootstraps):
            boot_diff = resample(diff)
            mean_diff = np.mean(boot_diff)
            std_diff = np.std(boot_diff, ddof=1)
            boot_cohen_ds.append(mean_diff / std_diff if std_diff > 0 else 0)

        return {
            'effect_size': original_cohen_d,
            'ci_lower': np.percentile(boot_cohen_ds, 2.5),
            'ci_upper': np.percentile(boot_cohen_ds, 97.5),
            'method': 'Cohen d'

        }

    results = {}
    for col, dtype in tqdm(dtypes.items(), desc="Processing Columns With Bootstrap"):
        if not col in df.columns:
            continue
        print(f'Hypothesis testing: {col} - dtype: {dtype}: Bootstrap {n_bootstraps}\n')
        # Extract the subject pairs for the given column
        paired_data = df[[col, pair_id]].copy().dropna()
        paired_data = paired_data.groupby(pair_id).agg(list)
        # Ensure we have exactly two observations per subject
        paired_data = paired_data[paired_data[col].apply(len) == 2]
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
                'cohen_d': None,
                'ci_lower': None,
                'ci_upper': None,
                'method': 'No valid pairs',
                'dtype': dtype,
            }
            continue

        diff = np.round(np.array(values1) - np.array(values2), 3)

        # compte the wilcoxon rank test
        res = compute_wilcoxon(diff)
        wilcoxon_effect_size = res.get('effect_size')
        t_stat = res.get('statistic')
        p_val = res.get('p_value')

        # Compute bootstrap Cohen's d
        if method == 'cohen':
            bootstrap_results = bootstrap_cohen_d(diff=diff,
                                                  n_bootstraps=n_bootstraps)
        else:
            bootstrap_results = bootstrap_wilcoxon(diff_=diff,
                                                   n_bootstraps=n_bootstraps)
        effect_size = bootstrap_results.get('effect_size')
        ci_lower = bootstrap_results.get('ci_lower')
        ci_upper = bootstrap_results.get('ci_upper')
        boot_method =  bootstrap_results.get('method')

        results[col] = {
            'sample_size': n,
            'p_value': p_val,
            'wilcoxon_effect_size': wilcoxon_effect_size,
            'boot_effect_size': effect_size,
            'boot_method': boot_method,
            'ci_lower': np.round(ci_lower, 4),
            'ci_upper': np.round(ci_upper, 4),
            'statistic': t_stat,
            'dtype': dtype,
        }

    return results
