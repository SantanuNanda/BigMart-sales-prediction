"""
utils.py - helper utilities (encoding, saving)
"""
import pandas as pd
import numpy as np

def mean_encode_smooth(series_train, target_train, m=200):
    agg = pd.DataFrame({'count': series_train.groupby(series_train).size(),
                        'mean': target_train.groupby(series_train).mean()})
    prior = target_train.mean()
    agg['smooth'] = (agg['count'] * agg['mean'] + m * prior) / (agg['count'] + m)
    return agg['smooth'].to_dict(), prior
