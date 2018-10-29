# -*- coding: utf-8 -*-
"""helper functions."""

import numpy as np
from implementations import *
from proj1_helpers import *

def standardize(x, mean=None, std=None):
    """Standardize data set."""
    if mean is None:
        mean = np.nanmean(x, axis=0)
    x = x - mean

    if std is None:
        std = np.nanstd(x, axis=0)
    x = x / std
    return x, mean, std


def get_jet_index(x):
    """Get index of three groups."""
    jet0_index = np.where(x[:,22]==0)[0]
    jet1_index = np.where(x[:,22]==1)[0]
    jet2_index = np.where(x[:,22]>=2)[0]
    return [jet0_index, jet1_index, jet2_index]


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


def apply_log1p(x):
    """Apply log normalization to features with long tail."""
    long_tail = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29]
    x[:, long_tail] = np.log1p(x[:, long_tail])
    return x


def drop_useless(x):
    """Drop useless columns."""
    # raw angles
    # eta: 14, 17, 24, 27
    # phi: 15, 18, 20, 25, 28
    raw_angle = [15, 18, 20, 25, 28]
    # columns of the same value (std is 0)
    same_cols = list(np.where(np.nanstd(x, axis=0)==0)[0])
    # columns full of NaN
    nan_cols = list(np.where(np.all(np.isnan(x), axis=0))[0])

    to_drop = list(set(raw_angle+same_cols+nan_cols))
    x = np.delete(x, to_drop, axis=1)
    return x


def fill_missing(x):
    """Fill missing values."""
    # use nan as missing value
    x[x==-999] = np.nan
    return x


def fill_nan(x):
    """Fill nan values."""
    # fill nan with 0
    x = np.nan_to_num(x)

    # # fill nan with the most frequently elements
    # for i in range(x.shape[1]):
    #     xi = x[:, i]
    #     value, count = np.unique(xi, return_counts=True)
    #     mode = value[np.argmax(count)]
    #     xi[np.isnan(xi)] = mode
    return x


def preprocessing(x_train, x_test):
    """Preprocess data."""
    # fill missing values with nan
    x_train = fill_missing(x_train)
    x_test = fill_missing(x_test)

    # add new phi features
    x_train = add_phi(x_train)
    x_test = add_phi(x_test)

    # apply log normalization
    x_train = apply_log1p(x_train)
    x_test = apply_log1p(x_test)

    # drop useless columns
    x_train = drop_useless(x_train)
    x_test = drop_useless(x_test)

    # standardization
    x_train, mean, std = standardize(x_train)
    x_test, _, _ = standardize(x_test, mean, std)

    # fill nan
    x_train = fill_nan(x_train)
    x_test = fill_nan(x_test)

    return x_train, x_test


def build_poly_feature(x, degree):
    """Build polynomial features for input data x."""
    # 0 to degree
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    # square root of absolute value
    poly = np.c_[poly, np.sqrt(np.abs(x))]
    # cross terms
    # vectorize the calculation
    # ref: https://stackoverflow.com/questions/22041561/python-all-possible-products-between-columns
    i, j = np.triu_indices(x.shape[1], 1)
    poly = np.c_[poly, x[:, i] * x[:, j]]
    return poly


def train_predict(x_train, y_train, x_test, y_test, degree, lambda_):
    """Train and predict."""
    # build polynomial features for
    tx_train = build_poly_feature(x_train, degree)
    tx_test = build_poly_feature(x_test, degree)
    # # normalization without the first column (full of 1)
#     tx_train[:, 1:], mean, std = standardize(tx_train[:, 1:])
#     tx_test[:, 1:], _, _ = standardize(tx_test[:, 1:], mean, std)

    w, _ = ridge_regression(y_train, tx_train, lambda_)
    y_pred_tr = predict_labels(w, tx_train)
    y_pred_te = predict_labels(w, tx_test)

    return y_pred_tr, y_pred_te
