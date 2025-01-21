"""
Visualization of the feature distribution
"""
import pathlib
import shutil

import pandas as pd
from configuration.config import config, aliases, dtypes
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, Tuple, List
import numpy as np
import warnings
warnings.simplefilter('ignore')

def plot_distribution(df:pd.DataFrame,
                      column:str,
                      alias:str,
                      dtype:str,
                      output_path:Optional[pathlib.Path] = None,
                      figsize:Tuple[int,int] = (8,6),
                      visualize:bool = False,
                      ):
    """
    Plots a distribution based on the column's dtype: histogram for continuous,
    bar plot for binary, categorical, or ordinal data. Adds relevant statistics
    for hypothesis testing.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name in the DataFrame.
        alias (str): Alias of the column name.
        dtype (str): dtype of the column (continuous, binary, ordinal, or categorical).
        output_path (pathlib.Path, optional): Path where to save the figure. Defaults to None.
    """
    sample_size = df.loc[~df[column].isna(), column].shape[0]
    total_sample_size = df.shape[0]
    proportion_sample_size = np.around(sample_size/total_sample_size, 3) * 100
    # Set theme for the plot
    sns.set_theme(style="whitegrid", palette="muted")
    # Initialize the plot
    plt.figure(figsize=figsize)
    if dtype == 'continuous':
        # Histogram for continuous data
        sns.histplot(df[column], kde=True, color="skyblue")
        plt.title(f"{alias} (Continuous) - Samples {proportion_sample_size}%", fontsize=14)

        # Add statistics
        mean = df[column].mean()
        median = df[column].median()
        std_dev = df[column].std()
        plt.axvline(mean,
                    color='red',
                    linestyle='--',
                    label=f"Mean: {mean:.2f}")
        plt.axvline(median,
                    color='green',
                    linestyle='--',
                    label=f"Median: {median:.2f}")
        plt.legend()

    elif dtype in ['binary', 'categorical', 'ordinal']:
        # Bar plot for binary, categorical, or ordinal data
        value_counts = df[column].value_counts(normalize=True).reset_index()
        sns.barplot(data=value_counts,
                    x=column,
                    y="proportion",
                    palette="pastel")
        plt.title(f"{alias} ({dtype.capitalize()}) - Samples {proportion_sample_size}%", fontsize=14)
        plt.xlabel(alias)
        plt.ylabel("Proportion")

        # # Add percentages on bars
        # for index, row in value_counts.iterrows():
        #     plt.text(index, row[column], f"{row[column]*100:.1f}%", ha='center', va='bottom')

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Finalize plot
    plt.tight_layout()
    if output_path is not None:
        file_name = f'{column} - {alias}.png'
        plt.savefig(output_path.joinpath(file_name), bbox_inches='tight', dpi=300)
    if visualize:
        plt.show()
    plt.close()



def plot_distribution_with_hue(df: pd.DataFrame,
                               column: str,
                               alias: str,
                               dtype: str,
                               hue: str = 'time',
                               figsize: Tuple[int, int] = (8, 6),
                               output_path: Optional[pathlib.Path] = None,
                               visualize:bool=False):
    """
    Plots a distribution based on the column's dtype: histogram for continuous,
    bar plot for binary, categorical, or ordinal data, with a hue to compare time points.
    Adds relevant statistics for hypothesis testing.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name in the DataFrame.
        alias (str): Alias of the column name.
        dtype (str): dtype of the column (continuous, binary, ordinal, or categorical).
        hue (str): Column to use as hue (e.g., 'time').
        figsize (Tuple[int, int]): Size of the figure. Defaults to (8, 6).
        output_path (pathlib.Path, optional): Path where to save the figure. Defaults to None.
    """
    # Filter out missing values for the column
    df_filtered = df.loc[~df[column].isna(), :]
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=figsize)

    if dtype == 'continuous':
        # Histogram with hue for continuous data
        sns.histplot(data=df_filtered, x=column, hue=hue, kde=True, palette="pastel", element="step")
        plt.title(f"{alias} (Continuous) - Comparison by {hue.capitalize()}", fontsize=14)
        plt.xlabel(alias)
        plt.ylabel("Density")
        plt.legend(title=hue.capitalize())

    elif dtype in ['binary', 'categorical', 'ordinal']:
        # Bar plot with hue for categorical data
        sns.countplot(data=df_filtered, x=column, hue=hue, palette="pastel")
        plt.title(f"{alias} ({dtype.capitalize()}) - Comparison by {hue.capitalize()}", fontsize=14)
        plt.xlabel(alias)
        plt.ylabel("Count")
        plt.legend(title=hue.capitalize())

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Finalize plot
    plt.tight_layout()
    if output_path is not None:
        file_name = f'{column} - {alias}.png'
        plt.savefig(output_path.joinpath(file_name), bbox_inches='tight', dpi=300)
    if visualize:
        plt.show()
    plt.close()


def plot_within_subject_comparison(df: pd.DataFrame,
                                   column: str,
                                   alias: str,
                                   subject_id: str = 'subject_id',
                                   time_col: str = 'time',
                                   figsize: Tuple[int, int] = (10, 6),
                                   output_path: Optional[pathlib.Path] = None,
                                   visualize:bool=False):
    """
    Creates a plot to compare within-subject changes for a given column over time.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name for the variable to compare.
        alias (str): Alias for the column (for labeling).
        subject_id (str): Column representing subject identifiers.
        time_col (str): Column representing the time points.
        figsize (Tuple[int, int]): Figure size.
        output_path (pathlib.Path, optional): Path to save the plot. Defaults to None.
    """
    # Filter missing values
    df_filtered = df.loc[~df[column].isna(), :]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)

    # Create a line plot connecting within-subject values
    sns.lineplot(data=df_filtered,
                 x=time_col,
                 y=column,
                 hue=subject_id,
                 palette="cool",
                 alpha=0.5,
                 linewidth=0.7,
                 legend=False)

    # Add points for individual measurements
    sns.scatterplot(data=df_filtered,
                    x=time_col,
                    y=column,
                    hue=subject_id,
                    palette="cool",
                    legend=False)

    # Add overall mean trend for clarity
    sns.pointplot(data=df_filtered,
                  x=time_col,
                  y=column,
                  ci='sd',
                  color='black',
                  markers="D",
                  scale=1.5,
                  label='Mean ± SD')

    plt.title(f"Within-Subject Comparison of {alias}", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel(alias)
    plt.tight_layout()

    # Save or display the plot
    if output_path is not None:
        file_name = f'{column} - {alias}.png'
        plt.savefig(output_path.joinpath(file_name), bbox_inches='tight', dpi=300)
    if visualize:
        plt.show()
    plt.close()


def plot_within_subject_and_distribution(df: pd.DataFrame,
                                         column: str,
                                         alias: str,
                                         dtype: str,
                                         subject_id: str = 'subject_id',
                                         time_col: str = 'time',
                                         hue: str = 'time',
                                         figsize: Tuple[int, int] = (16, 6),
                                         output_path: Optional[pathlib.Path] = None,
                                         visualize: bool = False):
    """
    Creates a subplot with within-subject comparison (left) and distribution with hue (right).

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name for the variable to compare.
        alias (str): Alias for the column (for labeling).
        dtype (str): dtype of the column (continuous, binary, ordinal, or categorical).
        subject_id (str): Column representing subject identifiers.
        time_col (str): Column representing the time points.
        hue (str): Column to use as hue (e.g., 'time').
        figsize (Tuple[int, int]): Figure size. Defaults to (16, 6).
        output_path (pathlib.Path, optional): Path to save the plot. Defaults to None.
        visualize (bool): Whether to display the plot. Defaults to False.
    """
    sample_size = df.loc[~df[column].isna(), column].shape[0]
    total_sample_size = df.shape[0]
    proportion_sample_size = np.around(sample_size/total_sample_size * 100, 2)

    # Filter missing values
    df_filtered = df.loc[~df[column].isna(), :]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Within-subject comparison
    sns.lineplot(data=df_filtered,
                 x=time_col,
                 y=column,
                 hue=subject_id,
                 palette="cool",
                 alpha=0.5,
                 linewidth=0.7,
                 legend=False,
                 ax=axes[0])
    sns.scatterplot(data=df_filtered,
                    x=time_col,
                    y=column,
                    hue=subject_id,
                    palette="cool",
                    legend=False,
                    ax=axes[0])
    sns.pointplot(data=df_filtered,
                  x=time_col,
                  y=column,
                  ci='sd',
                  color='black',
                  markers="D",
                  scale=1.5,
                  label='Mean ± SD',
                  ax=axes[0])
    axes[0].set_title(f"Within-Subject Comparison\n {alias}", fontsize=14)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(alias)

    # Plot 2: Distribution with hue
    if dtype == 'continuous':
        sns.histplot(data=df_filtered,
                     x=column,
                     hue=hue,
                     kde=True,
                     palette="pastel",
                     element="step", ax=axes[1])
        axes[1].set_title(f"Distribution Both Times - Sample Size {proportion_sample_size}% \n{alias} (Continuous)",
                          fontsize=14)
        axes[1].set_xlabel(alias)
        axes[1].set_ylabel("Density")

        # Calculate means and std deviations for each hue group
        stats = df_filtered.groupby(hue)[column].agg(['mean', 'std']).reset_index()
        stats['mean'] = stats['mean'].round(2)
        stats['std'] = stats['std'].round(2)

        # Get the unique hue categories and their corresponding colors from the palette
        hue_categories = stats[hue].unique()
        palette = sns.color_palette("pastel", len(hue_categories))
        color_map = dict(zip(hue_categories, palette))

        # Add vertical lines for means with distinct colors
        for _, row in stats.iterrows():
            axes[1].axvline(
                row['mean'],
                color=color_map[row[hue]],
                linestyle='--',
                linewidth=1.5,
                label=f"{row[hue]}: μ={row['mean']}, σ={row['std']}"
            )

        # Add legend with the statistics
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles=handles,
                       title=f"{hue.capitalize()} (Mean ± Std)",
                       fontsize=10,
                       title_fontsize=11)

    elif dtype in ['binary', 'categorical', 'ordinal']:
        sns.countplot(data=df_filtered,
                      x=column,
                      hue=hue,
                      palette="pastel",
                      ax=axes[1])
        axes[1].set_title(f"Distribution Both Times - Sample Size {proportion_sample_size}%\n{alias} ({dtype.capitalize()})", fontsize=14)
        axes[1].set_xlabel(alias)
        axes[1].set_ylabel("Count")
        axes[1].legend(title=hue.capitalize())

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Final adjustments
    plt.tight_layout()

    # Save or display the plot
    if output_path is not None:
        file_name = f'{column} - {alias}.png'
        plt.savefig(output_path.joinpath(file_name), bbox_inches='tight', dpi=300)
    if visualize:
        plt.show()
    plt.close()


if __name__ == "__main__":
    df_data = pd.read_csv(config.get('data_path').get('pp_dataset'))
    output_path_plots = config.get('results_path').joinpath('plots')
    # %% Select only the first time of sampling
    output_path_plots_baseline = output_path_plots.joinpath('baseline')
    if not output_path_plots_baseline.exists():
        output_path_plots_baseline.mkdir(exist_ok=True, parents=True)
    df_plot = df_data.loc[df_data['time'] == 'first', :]
    # Call the function
    for col_name, alias_ in aliases.items():
        # col_name = [*aliases.keys()][1]
        # alias_ = aliases.get(col_name)
        dtype = dtypes.get(col_name)
        print(f'Visualization: {col_name} ({alias_}) - {dtype}')
        plot_distribution(df=df_plot,
                          column=col_name,
                          alias=alias_,
                          dtype=dtype,
                          output_path=output_path_plots_baseline,
                          visualize=False)

    # %% Visualize both time samplings
    # Call the function
    output_path_plots_time_contrast = output_path_plots.joinpath('time_contrast')
    if not output_path_plots_time_contrast.exists():
        output_path_plots_time_contrast.mkdir(exist_ok=True, parents=True)
    for col_name, alias_ in aliases.items():
        # col_name = [*aliases.keys()][1]
        # alias_ = aliases.get(col_name)
        dtype = dtypes.get(col_name)
        print(f'Visualization: {col_name} ({alias_}) - {dtype}')
        plot_distribution_with_hue(df=df_data,
                                   column=col_name,
                                   alias=alias_,
                                   dtype=dtype,
                                   hue='time',
                                   output_path=output_path_plots_time_contrast,
                                   visualize=False)

    # %% Within subject comparison
    output_path_plots_within_comparison = output_path_plots.joinpath('within_comparison')
    if not output_path_plots_within_comparison.exists():
        output_path_plots_within_comparison.mkdir(exist_ok=True, parents=True)
    for col_name, alias_ in aliases.items():
        dtype = dtypes.get(col_name)
        print(f"Visualization: {col_name} ({alias_}) - {dtype}")
        plot_within_subject_comparison(df_data,
                                       column=col_name,
                                       alias=alias_,
                                       subject_id='id_subject',
                                       time_col='time',
                                       output_path=output_path_plots_within_comparison,
                                       visualize=False)

    # %% Within and hue distribution all at one
    output_path_plots_within_and_hue = output_path_plots.joinpath('within_and_hue')
    if not output_path_plots_within_and_hue.exists():
        output_path_plots_within_and_hue.mkdir(exist_ok=True, parents=True)
    for col_name, alias_ in aliases.items():
        dtype = dtypes.get(col_name)
        print(f"Visualization: {col_name} ({alias_}) - {dtype}")
        plot_within_subject_and_distribution(df=df_data,
                                             column=col_name,
                                             alias=alias_,
                                             subject_id='id_subject',
                                             hue='time',
                                             dtype=dtype,
                                             output_path=output_path_plots_within_and_hue,
                                             visualize=True)



