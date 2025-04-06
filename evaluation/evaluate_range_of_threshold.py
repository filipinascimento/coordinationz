'''
Author: Manita Pote

This script evaluates the classification based on the pvalue.
The dataframe passed for the evaluation should have following columns:
- userid
    Datatype : String
    Description : The  userid of the account.
- pvlaue 
    Datatype : Float
    Description : The pvalue associated with the account.
- quantile
    Datatype : Float
    Description : The quntile value associated with the account.
- y_true
    Datatype: Boolean
    Description: The value that determines weather it is IO or control. 1 means IO, 0 means control
'''

import pandas as pd
import numpy as np
import os
import argparse

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def auc_roc_pr_analysis(scores, labels, 
                        thresholds, 
                        threshold_type='pvalue',
                        fixed_threshold=False
                       ):
    """
    Analyze AUC-ROC and AUC-PR using either p-value or quantile as thresholds.
    
    Parameters:
    - scores: array-like, predicted scores or probabilities.
    - labels: array-like, true binary labels (0 or 1).
    - thresholds: list, thresholds for p-value or quantile.
    - threshold_type: str, 'pvalue' (lower means positive) or 'quantile' (higher means positive).
    
    Returns:
    - auc_roc_score: float, computed AUC-ROC.
    - auc_pr_score: float, computed AUC-PR.
    """
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    for threshold in thresholds:
        if threshold_type == 'pvalue':
            predictions = np.where(scores <= threshold, 1, 0)  # P-value: lower is positive
        # elif threshold_type == 'quantile':
        #     # quantile_value = np.quantile(scores, threshold)
        #     predictions = np.where(scores >= threshold, 1, 0)  # Quantile: higher is positive
        # else:
        #     raise ValueError("threshold_type must be 'pvalue' or 'quantile'")
        
        # True Positives, False Positives, etc.
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 1 - Specificity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr  # Recall is the same as TPR
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if fixed_threshold == True:
            return f1
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    results = list(sorted(
        zip(thresholds, f1_list, precision_list, recall_list),
        key=lambda x: x[0]  # sort by the first element (fpr_list)
    ))

    print(len(results[0]))
    
    # Sort FPR and TPR for ROC curve
    fpr_list, tpr_list = zip(*sorted(zip(fpr_list, tpr_list)))
    auc_roc_score = auc(fpr_list, tpr_list)

    # Sort Precision and Recall for PR curve
    recall_list, precision_list = zip(*sorted(zip(recall_list, precision_list)))
    auc_pr_score = auc(recall_list, precision_list)
    
    return auc_roc_score, auc_pr_score, results


def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(description='Runs the evaluation of the IO and control data')
    
    parser.add_argument('--pvalue_quantile_file',
                        dest='pvalue_quantile_file',
                        help='The file that has pvalue and qunatile for the users.'
                       )
    
    parser.add_argument('--evaluation_type',
                        dest='evaluation_type',
                        default='both',
                        help='Evaluation type: pvalue, quantile or both. Both will run the evaluation at the same time.'
                       )

    return parser.parse_args()

if __name__ == "__main__":
    df_cosine, df_0_01 = load_pvalues(threshold=0.05)
    visualize(df_cosine, df_0_01)