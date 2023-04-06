import pandas as pd
import numpy as np

from scipy.special import lambertw


def plogp(series):
    if type(series) == pd.Series:
        if (series < 0).any(): print('Warning: computing entropy of sequence with negative probabilities')
        return series.mask(~series.eq(0.), lambda x: -x*np.log2(np.abs(x)))
    else: 
        return (lambda x: -x * np.log2(x) if x else 0)(series)


def bernoulli_entropy_rate(p, q):
    return plogp(p) + plogp(q) - plogp(p+q)


def input_entropy(min, mean):
    return (-plogp(mean) + plogp(mean-1))/(min+mean)

def conditional_entropy(x):
    return plogp(x['TP'])+plogp(x['FP'])+plogp(x['FN'])+plogp(x['TN']) - plogp(x['TP']+x['FP'])-plogp(x['FN']+x['TN'])


def optimal_tau (tmin):
    return tmin/lambertw(tmin)


