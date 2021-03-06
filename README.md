# credit-risk-analysis

## Overview
Credit card credit dataset was used in this classification problem, employing various resampling algorithms (RandomOverSampler, SMOTE, ClusterCentroids, SMOTEENN) and 2 different ensemble classifiers (Balanced Random Forest, AdaBoost) to address an imbalanced dataset and predict high credit risk. 

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
7. Split the dataset into training and testing sets. The models were trained, made to predict values with the testing dataset and their performance assessed by various classifier metrics. 

### Resampling Algorithms
#### Random OverSampler 
![ros](/resources/images/naive_ros_metrics.png) 
* Balanced Accuracy: 0.65
* High Risk Class
  * Precision Score: 0.01
  * Recall Score: 0.71
  * F1 Score: 0.02
* Low Risk Class
  * Precision Score: 1.00
  * Recall Score: 0.58
  * F1 Score: 0.73
#### SMOTE
![smote](/resources/images/SMOTE_metrics.png)
* Balanced Accuracy: 0.66
* High Risk Class
  * Precision Score: 0.01
  * Recall Score: 0.63
  * F1 Score: 0.02
* Low Risk Class
  * Precision Score: 1.00
  * Recall Score: 0.68
  * F1 Score: 0.81
#### Cluster Centroids 
![cc](/resources/images/clustered_centroid_metrics.png)
* Balanced Accuracy: 0.54 
* High Risk Class
  * Precision Score: 0.01
  * Recall Score: 0.69
  * F1 Score: 0.01
* Low Risk Class
  * Precision Score: 1.00
  * Recall Score: 0.40
  * F1 Score: 0.57
#### SMOTEENN
![smoteenn](/resources/images/SMOTEENN_metrics.png)
* Balanced Accuracy: 0.54
* High Risk Class
  * Precision Score: 0.01
  * Recall Score: 0.69
  * F1 Score: 0.01
* Low Risk Class
  * Precision Score: 1.00
  * Recall Score: 0.40
  * F1 Score: 0.57

### Ensemble Classifiers
#### Balanced Random Forest
![BRF](/resources/images/balanced_random_forest_metrics.png)
* Balanced Accuracy: 0.79
* High Risk Class
  * Precision Score: 0.03
  * Recall Score: 0.70
  * F1 Score: 0.06
* Low Risk Class
  * Precision Score: 1.00
  * Recall Score: 0.87
  * F1 Score: 0.70
#### AdaBoost
![adaboost](/resources/images/adaboost_metrics.png)
* Balanced Accuracy: 0.93
* High Risk Class
  * Precision Score: 0.09
  * Recall Score: 0.92
  * F1 Score: 0.16
* Low Risk Class
  * Precision Score: 1.00
  * Recall Score: 0.94
  * F1 Score: 0.97

#### Table 1: Summary of Classification Metrics for High Risk Loan Status
| | Balanced Accuracy Score | Precision Score | Recall Score | F1 Score |
| --- | --- | --- | --- | ---|
| Random OverSampler | 0.65 | 0.01 | 0.71 | 0.02 |
| SMOTE | 0.66 | 0.01 | 0.63 | 0.02 |
| Cluster Centroids | 0.54 | 0.01 | 0.69 | 0.01 |
| SMOTEENN | 0.54 | 0.01 | 0.69 | 0.01 |
| Balanced Random Forest | 0.79 | 0.03 | 0.70 | 0.06 |
| AdaBoost | 0.93 | 0.09 | 0.92 | 0.16 |

## Summary
To give context to the confusion matrices, the following labels can be used:
| | Predicted High Risk | Predicted Low Risk |
| --- | --- | --- |
| **Actual High Risk** | TP | FN |
| **Actual Low Risk** | FP | TN |

Where:
* True Positive (TP): Number of high-risk candidates correctly labeled as high-risk
* False Negative (FN): Number of high-risk candidates mislabeled as low-risk
* False Positive (FP): Number of low-risk candidates mislabeled as high-risk
* True Negative (TN): Number of low-risk candidates correctly labeled as low-risk 

The credit risk dataset naturally lends to imbalanced classes, there are considerably more low-risk loans than high-risk. Here, the attempts to address this imbalance were resampling (oversampling, undersampling, combination) and ensemble learning to yield a model that could reliably predict credit risk. The definition of credit risk here is defined as the possibility of a borrower failing to repay, or defaulting, a loan. In the context of detecting credit risk, we can think of the FN as the group that is more likely to default than FP (although committing this error can still be costly as missed opportunities for good business). In selecting which model to use for the logistic regression of the credit dataset, evaluation metrics to consider should account for the negative classifications such as recall and F1 scores. Therefore, with a recall of 0.92 and F1 score of 0.16, the AdaBoost learning model is recommended.
