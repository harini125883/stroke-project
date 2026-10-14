# EDA and Visualization of Stroke Risk Factors + Deep Learning (MLP)

**Course:** U21ADP05 Exploratory Data Analysis and Visualization - Assignment II
**Student:** [Your Name], Roll No. [Roll Number] | Dept. of AI & Data Science, KPR Institute of Engineering and Technology

## Overview
Case study on the Kaggle *Stroke Prediction Dataset* (5,110 patients). Includes data cleaning, 14 visualizations
(Fry's 7-stage visualization pipeline), statistical/distribution analysis, and a tuned Keras MLP classifier with evaluation plots.

## Key results (test set, 767 patients / 38 strokes)
| Model | ROC-AUC | Recall | Precision |
|---|---|---|---|
| MLP (128-64, dropout 0.4) | 0.813 | 0.763 | 0.108 |
| Logistic regression (baseline) | 0.837 | 0.816 | 0.144 |

Age is by far the strongest predictor; gender and residence are not significantly associated with stroke.

## Repository structure
```
data/healthcare-dataset-stroke-data.csv   dataset (source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
stroke_eda_dl.py / stroke_eda_dl.ipynb    full commented code (EDA, preprocessing, model, evaluation)
figures/                                  all generated figures (fig01-fig14)
report/Stroke_EDA_IEEE_Report.docx/.pdf   IEEE-format project report
results.json, hyperparameter_tuning_results.csv, run_log.txt
requirements.txt
```
## How to run
```
pip install -r requirements.txt
python stroke_eda_dl.py        # run from the repository root
```
Note: the source file has 10 predictors; five features were engineered (age_group, bmi_category, glucose_category,
comorbidity_count, is_senior) to reach the required 15.
Research/education use only - not a clinical tool.
