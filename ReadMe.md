# Stanford PSYC 135

## Objective 
Study if students get an improvement sleep after following the Stanford lecture PSYC 135 

## Methods 
The Alliance Sleep Questionnaire (ASQ) is filled by each student at the beginning and en of the quarter.
The questionnaire contains subjective sleep symptoms and can capture how sleep is perceived by each student.

Parametric and non-parametric statistic analysis will be performed in the responses to compare within subject 
variability.

## Results 



Pproc folder contains the final dataset with the proper match of students 


# Stanford PSYC 135
## Main script used for the paper
Main script utilzied to compute the statistical analyzes from the matched and pre-process dataset between the ASQ and the class roster: `statistical_analysis.py`

## src
1. Start by processing the different roster files with the student information with the script `dataset_students.py`
2. Use the fuzzy search algorithm to find the students in the ASQ responses with the script `match_students.py`
3. Pre-processing, filter the dataset of candidates between the students and the ASQ using the script `prepare_dataset.py`
4. Once we have the final dataset, we can start the analysis with the scripts: 
   - 'make_table_one.py' 
   - 'visualization.py'

## results
1. The folder `results\plots` contains the mane plots utilized for the paper
2. The folder `results\statistics` contains the final dataset after the hypothesis testing of within groups comparison