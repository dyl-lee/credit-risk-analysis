# credit-risk-analysis

## Overview
Credit card credit dataset was used in this classification problem, employing various resampling algorithms (RandomOverSampler, SMOTE, ClusterCentroids, SMOTEENN) and 2 different ensemble classifiers ( to address an imbalanced dataset and predict high credit risk. 

## Resources
### Data sets
* [LoanStats_2019Q1.csv](/resources/LoanStats_2019Q1.csv)

### Software 
* Python 3.7.10
  * sklearn.linear_model module
* Visual Studio Code 1.64.2
* Jupyter Notebook

## Results
Dataset was read in as a DataFrame and preprocessed:
1. Dropped null columns where all values are null
2. Dropped null rows
3. Removed 'Issued' loan status
4. Converted interest rate to numerical data type
5. Converted target column values to 'low_risk' and 'high_risk' based on their values ('Current' was considered low risk while 'Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period' were considered high risk.)
6. Converted other categorical variables into dummy/indicator variables using Pandas get_dummies() and identify features (everything but loan_status) and targets (loan_status)
7. Split the dataset into training and testing sets. The models were trained, made to predict with the testing dataset and their performance assessed by various classifier metrics. 

### Resampling Algorithms
#### Random OverSampler 
![ros](/resources/images/naive_ros_metrics.png) 
#### SMOTE
![smote](/resources/images/SMOTE_metrics.png)
#### Cluster Centroids 
![cc](/resources/images/clustered_centroid_metrics.png)
#### SMOTEENN
![smoteenn](/resources/images/SMOTEENN_metrics.png)

### Ensemble Classifiers
#### Balanced Random Forest
![BRF](/resources/images/balanced_random_forest_metrics.png)
#### AdaBoost
![adaboost](/resources/images/adaboost_metrics.png)

#### Table 1: Summary of Classification Metrics for High Risk Loan Status per Resampling Algorithm
| | Balanced Accuracy Score | Precision Score | Recall Score | F1 Score |
| --- | --- | --- | --- | ---|
| Random OverSampler | 0.65 | 0.01 | 0.71 | 0.02 |
| SMOTE | 0.66 | 0.01 | 0.63 | 0.02 |
| Cluster Centroids | 0.54 | 0.01 | 0.69 | 0.01 |
| SMOTEENN | 0.54 | 0.01 | 0.69 | 0.01 |
| Balanced Random Forest | 0.79 | 0.03 | 0.70 | 0.06 |
| AdaBoost | 0.93 | 0.09 | 0.92 | 0.16 |

## Summary

