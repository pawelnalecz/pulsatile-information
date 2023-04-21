import pandas as pd
import numpy as np
from typing import TypeVar, Iterable

from scipy.special import lambertw


T = TypeVar('T')

def plogp(series: T) -> T:
    if isinstance(series, pd.Series):
        # series = cast(pd.Series, series)
        if (series < 0).any(): 
            print('Warning: computing entropy of sequence with negative probabilities')
            print(series)
            raise ValueError()
        return series.mask(~series.eq(0.), lambda x: -x*np.log2(np.abs(x)))
    elif isinstance(series, np.ndarray):
        return - series * np.log2(series, out=np.zeros_like(series, dtype='float64'), where=series != 0)
    else:
        return (lambda x: -x * np.log2(x) if x else type(x)(0))(series)


def bernoulli_entropy_rate(p, q):
    return plogp(p) + plogp(q) - plogp(p+q)


def input_entropy(min, exp_mean):
    return (-plogp(exp_mean) + plogp(exp_mean-1))/(min+exp_mean)

def conditional_entropy(x):
    return plogp(x['TP'])+plogp(x['FP'])+plogp(x['FN'])+plogp(x['TN']) - plogp(x['TP']+x['FP'])-plogp(x['FN']+x['TN'])


def optimal_tau (tmin):
    return tmin/lambertw(tmin)


def binary_sequence_to_int(binary_sequence: Iterable):
    return sum(v << i for i, v in enumerate(binary_sequence))

T = TypeVar('T', bound=pd.Series | pd.DataFrame)

def rolling_binary_sequence_to_int(series: T, win_len: int) -> T:
    if win_len == 1:
        return series
    return sum(series.shift(i) * 2**(win_len - i - 1) for i in range(win_len)).shift(-(win_len // 2))


def is_local_max(x):
    return (x.diff().shift(-1) <= 0) & (x.diff() > 0)

def is_local_min(x):
    return is_local_max(-x)


