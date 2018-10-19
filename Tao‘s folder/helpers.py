# -*- coding: utf-8 -*-
"""helpers."""

import numpy as np

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.nanmean(x, axis=0)
    if std_x is None:
        std_x = np.nanstd(x, axis=0)
    x = x - mean_x
    x = x / std_x
    return x, mean_x, std_x


def delta_angle_norm(a, b):
    """Caluculate difference between two angles
    normalize the result to ]-pi, pi]."""
    delta = a - b
    delta[delta < -np.pi] += 2 * np.pi
    delta[delta >  np.pi] -= 2 * np.pi
    return delta


def add_phi(x):
    """Add new phi features."""
    # PRI_lep_phi - PRI_tau_phi
    r1 = delta_angle_norm(x[:,18], x[:,15]).reshape(-1, 1)
    # PRI_met_phi - PRI_tau_phi
    r2 = delta_angle_norm(x[:,20], x[:,15]).reshape(-1, 1)
    # PRI_jet_leading_phi - PRI_tau_phi
    r3 = delta_angle_norm(x[:,25], x[:,15]).reshape(-1, 1)
    # PRI_jet_subleading_phi - PRI_tau_phi
    r4 = delta_angle_norm(x[:,28], x[:,15]).reshape(-1, 1)

    x = np.concatenate([x, r1, r2, r3, r4], axis=1)
    
    return x

def change_eta(x):
    """Change eta."""
    neg_eta_index = x[:,14] < 0
    # PRI_tau_eta
    x[neg_eta_index, 14] *= -1
    # PRI_lep_eta
    x[neg_eta_index, 17] *= -1
    # PRI_jet_leading_eta
    x[neg_eta_index, 24] *= -1
    # PRI_jet_subleading_eta
    x[neg_eta_index, 27] *= -1
    
    return x


def apply_log1p(x):
    """Apply log normalization to features with long tail."""
    long_tail = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
    x[:, long_tail] = np.log1p(x[:, long_tail])
    return x


def drop_phi(x):
    """Drop raw phi."""
    return np.delete(x, [15, 18, 20, 25, 28], axis=1)


def drop_na(x):
    """Drop columns full of NaN."""
    to_drop = np.where(np.all(np.isnan(x), axis=0))[0]
    x = np.delete(x, to_drop, axis=1)
    return x

def preprocessing(x, y, mean=None, std=None, test=False):
    """Preprocessing data."""
    
    # get index of four sets
    jet0_index = np.where(x[:,22]==0)[0]
    jet1_index = np.where(x[:,22]==1)[0]
    jet2_index = np.where(x[:,22]==2)[0]
    jet3_index = np.where(x[:,22]==3)[0]
    
    # use nan as missing value
    x[x==-999] = np.nan
    
    # feature engineering
    x = add_phi(x)
    x = change_eta(x)
    # apply log normalization on features 
    # with long tail distribution
    x = apply_log1p(x)
    x = drop_phi(x)
    
    # standarization
    if test:
        if mean is None or std is None:
            raise Error("Require mean and std in the test round!")
        x, _, _ = standardize(x, mean, std)
    else:
        x, mean, std = standardize(x)
    
    # select four groups and drop nan columns
    x_jet0 = drop_na(x[jet0_index, :])
    x_jet1 = drop_na(x[jet1_index, :])
    x_jet2 = drop_na(x[jet2_index, :])
    x_jet3 = drop_na(x[jet3_index, :])
    
    # fill remained NaN with 0
    x_jet0 = np.nan_to_num(x_jet0)
    x_jet1 = np.nan_to_num(x_jet1)
    x_jet2 = np.nan_to_num(x_jet2)
    x_jet3 = np.nan_to_num(x_jet3)
    
    
    if test:
        return [x_jet0, x_jet1, x_jet2, x_jet3], [jet0_index, jet1_index, jet2_index, jet3_index]
    else:
        # build y of four groups
        y_jet0 = y[jet0_index]
        y_jet1 = y[jet1_index]
        y_jet2 = y[jet2_index]
        y_jet3 = y[jet3_index]
        return [(x_jet0, y_jet0), 
                (x_jet1, y_jet1), 
                (x_jet2, y_jet2), 
                (x_jet3, y_jet3)], [mean, std]
    
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly