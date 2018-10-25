# -*- coding: utf-8 -*-
"""implementation of ML methods."""

import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_mse_loss(y, tx, w):
    """Calculate mse loss."""
    e = y - tx.dot(w)
    return 1/2 * np.mean(e**2)


def compute_gradient_linear(y, tx, w):
    """Compute the gradient of linear regression."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    # initialize weight
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient
        grad = compute_gradient_linear(y, tx, w)
        # update w by gradient descent update
        w = w - gamma * grad

    # calculate loss
    loss = compute_mse_loss(y, tx, w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    # initialize weight
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            # compute a stochastic gradient
            grad = compute_gradient_linear(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad

    # calculate loss
    loss = compute_mse_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def sigmoid(x):
    """Apply sigmoid function on x."""
    return 1.0 / (1 + np.exp(-x))


def compute_nl_loss(y, tx, w):
    """Compute the loss by negative log likelihood."""
    # Log loss is undefined for pred=0 or pred=1
    # so probabilities are clipped to max(epsilon, min(1 - epsilon, pred)).
    epsilon = 1e-15 
    pred = sigmoid(tx.dot(w))
    pred = np.clip(pred, epsilon, 1-epsilon)
    
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    
    return np.squeeze(- loss)


def compute_gradient_logistic(y, tx, w):
    """Compute the gradient of logistic regression."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using SGD."""
    # initialize weight
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            # compute a stochastic gradient
            grad = compute_gradient_logistic(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad

    # calculate loss
    loss = compute_nl_loss(y, tx, w)
    
    return w, loss


def compute_nl_loss_regularization(y, tx, w, lambda_):
    """Compute the loss by negative log likelihood and l2 regularization."""
    return compute_nl_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))


def compute_gradient_reg_logistic(y, tx, w, lambda_):
    """Compute the gradient of regularized logistic regression."""
    w_reg = w
    w_reg[0] = 0
    return compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w_reg


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using SGD."""
    # initialize weight
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            # compute a stochastic gradient with regularization
            grad = compute_gradient_reg_logistic(y_batch, tx_batch, w, lambda_)
            # update w through the stochastic gradient update
            w = w - gamma * grad

    # calculate loss with l2 regularization
    loss = compute_nl_loss_regularization(y, tx, w, lambda_)

    return w, loss
