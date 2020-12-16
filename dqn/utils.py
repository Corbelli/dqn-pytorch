import numpy as np

def calc_rollling_mean(v, k=100):
    return [np.mean(v[(i-k+1):(i+1)]) if i-k+1 >= 0 else np.mean(v[0:(i+1)]) for i in range(len(v))]

def last_rolling_mean(v, k=100):
    i = len(v) - 1
    return np.mean(v[(i-k+1):(i+1)]) if i-k+1 >= 0 else np.mean(v[0:(i+1)]) 