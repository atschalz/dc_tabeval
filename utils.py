import random
import numpy as np
import os
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error, log_loss, mean_absolute_error

def set_seed(seed=42):
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
    
def get_metric(eval_metric_name):
    if eval_metric_name=="r2":
        return r2_score, "maximize"
    elif eval_metric_name=="auc":
        return roc_auc_score, "maximize"
    elif eval_metric_name=="mse":
        return lambda y_true, y_pred, **kwargs: mean_squared_error(y_true, y_pred), "minimize"
    elif eval_metric_name=="rmse":
        return lambda y_true, y_pred, **kwargs: np.sqrt(mean_squared_error(y_true, y_pred)), "minimize"
    elif eval_metric_name=="rmsle":
        return lambda y_true, y_pred, **kwargs: np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), "minimize"
    elif eval_metric_name=="rmsse":
        return r2_score, "maximize"
    # elif eval_metric_name=="ams":
    #     return ams.ams_score, "maximize"
    elif eval_metric_name=="gini":
        return lambda y_true, y_pred: (2*roc_auc_score(y_true, y_pred))-1, "maximize"
    elif eval_metric_name=="logloss":
        return log_loss, "minimize"
    elif eval_metric_name=="mlogloss":
        return lambda y_true, y_pred: log_loss(y_true, y_pred, labels=list(range(y_pred.shape[1]))), "minimize"
    elif eval_metric_name=="mae":
        return mean_absolute_error, "minimize"
    elif eval_metric_name=="norm_gini":
        return normalized_gini, "maximize"
    elif eval_metric_name=="multilabel":
        return multilabel_log_loss, "minimize"
    else:
        raise ValueError(f"Metric '{eval_metric_name}' not implemented.")    


def multilabel_log_loss(y_true, y_pred):
    """
    Compute the column-wise log loss for multi-label classification and then average over all columns.

    Parameters:
    - y_true (np.array): A binary array of shape (n_samples, n_classes) indicating the true labels.
    - y_pred (np.array): An array of shape (n_samples, n_classes) containing the predicted probabilities for each class.

    Returns:
    - float: The average log loss across all classes for the multi-label classification.
    """
    # Number of classes
    n_classes = y_true.shape[1]
    # Initialize total log loss
    total_log_loss = 0
    
    # Compute log loss for each class and sum up
    for i in range(n_classes):
        class_log_loss = log_loss(y_true[:, i], y_pred[:, i])
        total_log_loss += class_log_loss
    
    # Average log loss across all classes
    average_log_loss = total_log_loss / n_classes
    return average_log_loss

def gini(actual, pred):
    assert len(actual) == len(pred), "Length of actual values and predictions must be equal."
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float64)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    total_losses = all[:, 0].sum()
    gini_sum = all[:, 0].cumsum().sum() / total_losses
    
    gini_sum -= (len(actual) + 1) / 2.
    return gini_sum / len(actual)

def normalized_gini(actual, pred):
    return gini(actual, pred) / gini(actual, actual)
        
#### Functions from https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/163684:
## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 

## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

## Merging by concat to not lose dtypes
def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1






