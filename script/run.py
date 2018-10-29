# -*- coding: utf-8 -*-
"""generate submission file for Kaggle."""

# load necessary packages
import numpy as np
np.random.seed(12)
np.warnings.filterwarnings('ignore')
from proj1_helpers import *
from implementations import *
from helpers import *


print("Step1: Loading DataSet...")
# data path
train_data_path = "../data/train.csv"
test_data_path = "../data/test.csv"
# load training and test set
y_train, x_train, id_train = load_csv_data(train_data_path)
y_test, x_test, id_test = load_csv_data(test_data_path)
# get jet index of three groups
jet_index_train = get_jet_index(x_train)
jet_index_test = get_jet_index(x_test)


print("Step2: Training and Predicting...")
# hyper-parameter for each model
lambdas = [1e-6, 1e-4, 1e-4]
degrees = [15, 19, 17]

y_pred_test = np.zeros_like(y_test)
# iterate through three groups
for i in range(3):
    print("...For group {}...".format(i+1))
    lambda_ = lambdas[i]
    degree = degrees[i]
    train_index = jet_index_train[i]
    test_index = jet_index_test[i]
    # get train/test data for each group
    x_tr, y_tr = x_train[train_index], y_train[train_index]
    x_te, y_te = x_test[test_index], y_test[test_index]
    # data preprocess
    x_tr, x_te = preprocessing(x_tr, x_te)
    # get predict result
    _, y_pred_test[test_index] = train_predict(x_tr, y_tr, x_te, y_te, degree, lambda_)


print("Step3: Creating Submission File...")
create_csv_submission(id_test, y_pred_test, "../submission/submision.csv")


print("Done!")
