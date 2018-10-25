# -*- coding: utf-8 -*-
"""run model and get prediction results."""

import numpy as np
from proj1_helpers import *
from implementations import *

# data path
train_data_path = "../data/train.csv"
test_data_path = "../data/test.csv"

# load training and test set
y_tr, x_tr, id_tr = load_csv_data(train_data_path)
_, x_te, id_te = load_csv_data(test_data_path)

# number of samples
N = x_tr.shape[0]
# number of features
m = x_tr.shape[1]

# TODO: more sophisticated data preprocessing
# replace missing data with nan
x_tr[x_tr == -999] = np.nan
x_te[x_tr == -999] = np.nan
# normalize
mean = x_tr.mean(0)
std = x_tr.std(0)
x_tr -= mean
x_tr /= std
x_te -= mean
x_te /= std
# fill nan
x_tr = np.nan_to_num(x_tr)
x_te = np.nan_to_num(x_te)

# initialize weights
initial_w = np.random.rand(m)

# TODO: cross validataion
# hyperparameters
max_iters = 100
gamma = 0.1
lambda_ = 0.1

# TODO: try with different models
# run ridge regression
weights, loss = ridge_regression(y_tr, x_tr, lambda_)

# predict
y_pred = predict_labels(weights, x_te)

# create submission
name = "submission.csv"
create_csv_submission(id_te, y_pred, name)
