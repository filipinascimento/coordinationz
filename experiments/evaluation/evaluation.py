import pandas as pd
import numpy as np

from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import StratifiedKFold

def evaluation_metrics(labels_map,n_splits=10):
    """
    Args:
        label_map: dataframe with the classification label for each userid (columns = ['userid', 'label' (0 or 1),'eigen_centrality])
    Returns:
        Average Performance across 10 folds
    """

    skf = StratifiedKFold(n_splits=n_splits)
    X = labels_map[['user_id','eigen_centrality']]
    y = labels_map['label']
    skf.get_n_splits(X, y)

    metrics = {"f1_score":0,"recall":0,"precision":0}

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        thresh = [i for i in range(1,100,2)]
        centrality_ranges = [np.percentile(X_train['eigen_centrality'].values,t) for t in thresh]
        max_centrality_range = -1
        max_f1_score = -1

        for cent in centrality_ranges:
            new_y_train = X_test['eigen_centrality'].apply(lambda x: 1 if x>=cent else 0)
            if(max_f1_score<f1_score(new_y_train,y_train)):
                max_f1_score = f1_score(new_y_train,y_train)
                max_centrality_range = cent

        y_test_preds = X_test['eigen_centrality'].apply(lambda x: 1 if x>=max_centrality_range else 0)

        metrics['f1_score']+=f1_score(y_test,y_test_preds)
        metrics['recall']+=recall_score(y_test,y_test_preds)
        metrics['precision']+=precision_score(y_test,y_test_preds)

    metrics['f1_score']/=float(n_splits)
    metrics['recall']/=float(n_splits)
    metrics['precision']/=float(n_splits)

    return metrics