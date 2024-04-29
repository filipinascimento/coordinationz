import pandas as pd
import numpy as np
from scipy import stats

import pandas as pd
import numpy as np
import datetime
import warnings
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from scipy.stats import ks_2samp

def read_data(input_path, filename):
    '''
    Read the data file from the given location
    
    :param input_path: Location to input file
    :param filename: Name of the file
    :return pandas dataframe
    '''
                        
    parts = filename.split('.')
                        
    if parts[-1] == 'csv':
        df = pd.read_csv(f'{input_path}/{filename}')
        
        print(df.info())
        
        return df
                        
    if parts[-1] == 'gz' and parts[-2] == 'pkl':
        df = pd.read_pickle(f'{input_path}/{filename}')
        
        if len(df) <= 1:
            print('The dataframe has just one column')
            
            raise Exception('Data insufficient')
        else:
            print(df.info())
            
            return df
    else:
    
        raise Exception(
            '--filename : Invalid file format. \n Only pkl.gz and csv accepted')
        

def ratios(df):
    n_users = df['userid'].nunique()
    retweets = df.loc[df['is_retweet'] == True]
    n_retweets = len(retweets)
    n_tweets = len(df.loc[(df['is_retweet'] == False) & \
                (~df['quoted_tweet_tweetid'].notnull()) & \
                (df['in_reply_to_tweetid'].isnull())])
    n_total = len(df)
    n_replies = len(df_mexico.loc[~df_mexico['in_reply_to_tweetid'].isnull()])
    n_quoted = len(df_mexico.loc[~df_mexico['quoted_tweet_tweetid'].isnull()])

    
    print('\n ------- Retweet --------------\n')
    df_retweet = df.loc[df['is_retweet'] == True]
    df_retweet_grp = df_retweet.groupby(
        ['retweet_tweetid']).size().to_frame(
        'retweet_count').reset_index()
    
    print('\n Retweet to user ratio :', round(n_retweets/n_users, 2))
    print('\n Retweet to tweet ratio :', round(n_retweets/n_tweets, 2))
    
    
    print('\n ------- Replies -------------\n')
    print('\n Replies to user ratio :', round(n_replies/n_users))
    print('\n Replies to tweet ratio : ', round(n_replies/n_tweets))
    print('\n Maximum replies a tweet got :',
          max(df_mexico_tweet_grp['retweet_count']))
    

def statistics(df, 
               column_to_groupby='poster_tweetid',
               column_to_take='age'
              ):
    '''
    Get the statistics of column
    '''
    df_stat = (df
              .groupby([column_to_groupby])[column_to_take]
              .describe()
               # .to_frame()
              .reset_index()
             )
    
    df_stat = df_stat.rename(columns={
        'count': f'count_{column_to_take}',
        'min': f'min_{column_to_take}',
        'max': f'max_{column_to_take}',
        'mean': f'mean_{column_to_take}',
        'median': f'median_{column_to_take}',
        'std': f'std_{column_to_take}',
        '50%': f'50%_{column_to_take}',
        '25%': f'25%_{column_to_take}',
        '75%': f'75%_{column_to_take}'
    })
    
    df_skew = (df
              .groupby([column_to_groupby])[column_to_take]
              .skew(skipna=False)
              .to_frame(f'skew_{column_to_take}')
              .reset_index()
             )
    df_kurtosis = (df
          .groupby([column_to_groupby])[column_to_take]
          .apply(pd.DataFrame.kurt)
          .to_frame(f'kurtosis_{column_to_take}')
          .reset_index()
         )
    
    df_group = df_stat.merge(df_skew, 
                             on=column_to_groupby)
    df_group = df_group.merge(df_kurtosis,
                              on=column_to_groupby)
    
    return df_group

def entropy(data):
    '''
    Calculates the entropy of distribution
    '''
    a, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def all_stat(df, 
             column_to_groupby='poster_tweetid',
             column_to_take='age'):
    '''
    Calculates the summary statistics of dataframe
    '''
    
    list_column = f'list_{column_to_take}'
    df_stat = (df
               .groupby([column_to_groupby])[column_to_take]
               .apply(list)
               .to_frame(list_column)
               .reset_index()
              )
    
    df_des = (df
          .groupby([column_to_groupby])[column_to_take]
          .describe()
           # .to_frame()
          .reset_index()
         )
    
    df_des = df_des.rename(columns={
        'count': f'count_{column_to_take}',
        'min': f'min_{column_to_take}',
        'max': f'max_{column_to_take}',
        'mean': f'mean_{column_to_take}',
        'median': f'median_{column_to_take}',
        'std': f'std_{column_to_take}',
        '50%': f'50%_{column_to_take}',
        '25%': f'25%_{column_to_take}',
        '75%': f'75%_{column_to_take}'
    })
    
    df_des = df_des.drop([f'count_{column_to_take}'], 
                         axis=1)
    
    df_skew = (df
              .groupby([column_to_groupby])[column_to_take]
              .skew(skipna=False)
              .to_frame(f'skew_{column_to_take}')
              .reset_index()
             )
    df_kurtosis = (df
          .groupby([column_to_groupby])[column_to_take]
          .apply(pd.DataFrame.kurt)
          .to_frame(f'kurtosis_{column_to_take}')
          .reset_index()
         )
    
    df_group = df_des.merge(df_skew, 
                             on=column_to_groupby)
    df_group = df_group.merge(df_kurtosis,
                              on=column_to_groupby)
    
    df_stat['min'] = df_stat[list_column].apply(
        lambda x: min(x)
    )
    df_stat['max'] = df_stat[list_column].apply(
        lambda x: max(x)
    )
    
    #Range
    df_stat[f'range_{column_to_take}'] = df_stat['max'] - df_stat['min']
    
    df_stat = df_stat.drop(columns=['min', 'max'])
    
    #IQR
    df_stat[f'iqr_{column_to_take}'] = df_stat[list_column].apply(
        lambda x: np.quantile(np.array(x), [0.75])[0] - np.quantile(np.array(x), [0.25])[0]
    
    ) 
    #Variance
    df_stat[f'var_{column_to_take}'] = df_stat[list_column].apply(
        lambda x: np.var(np.array(x)))
    # Coefficient of variation
    df_stat[f'cof_{column_to_take}'] = df_stat[list_column].apply(
        lambda x: np.std(np.array(x)) / np.mean(np.array(x)) * 100)
    # Mean Absolute Deviation
    df_stat[f'mad_{column_to_take}'] = df_stat[list_column].apply(
        lambda x: np.mean(np.absolute(np.array(x) - np.mean(np.array(x))))
    )
    # Entropy
    df_stat[f'entropy_{column_to_take}'] = df_stat[list_column].apply(entropy)
    
    df_group = df_group.merge(df_stat,
                              on=column_to_groupby)
    
    grps = df.groupby([column_to_groupby, 
                       'tweet_label']).groups.keys()
    df_grps = pd.DataFrame(data=grps, columns=[column_to_groupby, 
                                               'tweet_label'])
    
    df_group = df_group.merge(df_grps[[column_to_groupby, 
                                       'tweet_label']],
                              on=column_to_groupby)
    
    return df_group



def run_model(df,
              columns_not_include=['list_age'],
              model_type='random', 
              pca=False):
    '''
    Trains the model and prints the result
    :param df: Dataframe
    :param model_type: Type of model
    :param pca: Whether to do PCA or not
    :param columns_not_include: columns to not include
    '''
    print(df.columns)
          
    columns_not_include.extend(
        ['poster_tweetid','tweet_label'])
    
    columns_to_keep = list(set(df.columns) - set(columns_not_include))
    
    X = df[columns_to_keep]
    y = df['tweet_label']

    #PCA 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = df.index
    
    if pca == True:
        print('here')
        pca = PCA()

        # Fit the PCA object to the data and transform the data
        X = pca.fit_transform(X)
        print('After PCA shape ', X.shape)

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X,
                                                                                     y,
                                                                                     indices,
                                                        # random_state=104, 
                                                        stratify=y,
                                                        test_size=0.20, 
                                                        shuffle=True)

    print('Xtrain: ', len(X_train))
    print('Xtrain shape: ', X_train.shape)
    print('Xtest: ', len(X_test))
    print('Ytrain: ', len(y_train))
    print('Ytest: ', len(y_test))

    if model_type == 'logistic':
        model = LogisticRegression(random_state=0)

    if model_type == 'random':
        model = RandomForestClassifier(n_estimators=100, 
                                       # random_state=42
                                      )

    model.fit(X_train, y_train)
    
    print(model.score(X_train, y_train))

    y_pred = model.predict(X_test)

    result = classification_report(y_test, y_pred, labels=[0,1])

    print(result)
    
    #Cross validation
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = round(scores.mean(), 2)
    std_score = round(scores.std(), 2)
    
    print(f'Cross validation: mean {mean_score} accuracy with a standard deviation of {std_score}')
    
    #feature importance
    if model_type == 'random' and pca == False:
        importances = model.feature_importances_
        df_imp = pd.DataFrame({'Feature': columns_to_keep, 
                                    'Importance': model.feature_importances_})
        df_imp = df_imp.sort_values('Importance', 
                                         ascending=False).set_index('Feature')
        print(df_imp)

    #ROC curve
    lr_probs = model.predict_proba(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, lr_probs[:, 1])
    
    # Compute the AUC score
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', 
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', 
             lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    from sklearn.metrics import precision_recall_curve

    # y_true and y_scores are the true labels and predicted scores, respectively
    precision, recall, thresholds = precision_recall_curve(y_test,
                                                        lr_probs[:, 1])

    # plot the precision-recall curve
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
    
    df_pred = df.loc[indices_test]
    df_pred['pred'] = y_pred
    
    return df_pred


def KS_test(data1, data2):
    '''
    Tes the KS test for data1 and data2
    '''
    statistic, pvalue = ks_2samp(data1, 
                                 data2)


    print('KS test statistic:', statistic)
    print('p-value:', pvalue)
