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


def _check_dimension(y_true, y_pred):
    """Check if the dimension of true target and predicted target match."""
    # if 1D, reshape as 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    # check the number of samples
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Number of samples does not correspond: ",
                         "y_true has {}, y_pred has {}.".format(y_true.shape[0], y_pred.shape[0]))

    # check the number of output
    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("Number of output does not correspond: ",
                         "y_true has {}, y_pred has {}.".format(y_true.shape[1], y_pred.shape[1]))

    return y_true, y_pred


def compute_mse_loss(y, tx, w):
    """Calculate mse loss."""
    pred = tx.dot(w)
    y, pred = _check_dimension(y, pred)
    err = y - pred
    return np.mean(0.5 * err ** 2)


def compute_gradient_linear(y, tx, w):
    """Compute the gradient of linear regression."""
    pred = tx.dot(w)
    y, pred = _check_dimension(y, pred)
    err = y - pred
    grad = -tx.T.dot(err) / len(err)
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Parameters
    ----------
    y : Target
    tx : Data
    initial_w : Initial weight vector
    max_iters : Number of iterations
    gamma : Learning rate or step size in each iteration

    Returns
    -------
    w : Weight vector after final iteration
    loss : Corresponding loss calculated by mean square error
    """
    n_samples, n_features = tx.shape

    # check target
    if y.ndim > 2:
        raise ValueError("Target y has wrong shape: {}.".format(y.shape))
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if y.shape[0] != n_samples:
        raise ValueError("Number of samples in tx and y does not correspond: ",
                         "tx has {}, y has {}.".format(n_samples, y_shape[0]))

    n_output = y.shape[1]

    # check initial weight
    if initial_w.ndim == 1:
        initial_w = initial_w.reshape((-1, 1))
    if initial_w.shape != (n_features, n_output):
        raise ValueError("Initial weight has wrong shape: ",
                         "{} != {}.".format(initial_w.shape, (n_features, n_output)))

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
    """Linear regression using stochastic gradient descent.

    Parameters
    ----------
    y : Target
    tx : Data
    initial_w : Initial weight vector
    max_iters : Number of iterations
    gamma : Learning rate or step size in each iteration

    Returns
    -------
    w : Weight vector after final iteration
    loss : Corresponding loss calculated by mean square error
    """
    n_samples, n_features = tx.shape

    # check target
    if y.ndim > 2:
        raise ValueError("Target y has wrong shape: {}.".format(y.shape))
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if y.shape[0] != n_samples:
        raise ValueError("Number of samples in tx and y does not correspond: ",
                         "tx has {}, y has {}.".format(n_samples, y_shape[0]))

    n_output = y.shape[1]

    # check initial weight
    if initial_w.ndim == 1:
        initial_w = initial_w.reshape((-1, 1))
    if initial_w.shape != (n_features, n_output):
        raise ValueError("Initial weight has wrong shape: ",
                         "{} != {}.".format(initial_w.shape, (n_features, n_output)))

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
    """Least squares regression using normal equations.

    Parameters
    ----------
    y : Target
    tx : Data

    Returns
    -------
    w : Weight vector
    loss : Corresponding loss calculated by mean square error
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Parameters
    ----------
    y : Target
    tx : Data
    lambda_ : Regularization parameter

    Returns
    -------
    w : Weight vector
    loss : Corresponding loss calculated by mean square error
    """
    aI = 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
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
    y, pred = _check_dimension(y, pred)

    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))

    return np.squeeze(- loss).item()


def compute_gradient_logistic(y, tx, w):
    """Compute the gradient of logistic regression."""
    pred = sigmoid(tx.dot(w))
    y, pred = _check_dimension(y, pred)
    grad = tx.T.dot(pred - y)
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using GD.

    Parameters
    ----------
    y : Target
    tx : Data
    initial_w : Initial weight vector
    max_iters : Number of iterations
    gamma : Learning rate or step size in each iteration

    Returns
    -------
    w : Weight vector after final iteration
    loss : Corresponding loss calculated by negative log likelihood
    """
    n_samples, n_features = tx.shape

    # check target
    if y.ndim > 2:
        raise ValueError("Target y has wrong shape: {}.".format(y.shape))
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if y.shape[0] != n_samples:
        raise ValueError("Number of samples in tx and y does not correspond: ",
                         "tx has {}, y has {}.".format(n_samples, y_shape[0]))

    n_output = y.shape[1]

    # check initial weight
    if initial_w.ndim == 1:
        initial_w = initial_w.reshape((-1, 1))
    if initial_w.shape != (n_features, n_output):
        raise ValueError("Initial weight has wrong shape: ",
                         "{} != {}.".format(initial_w.shape, (n_features, n_output)))
    
    # initialize weight
    w = initial_w
    
    # losses
    losses = []
    # threshold
    threshold = 1e-8
    
    for n_iter in range(max_iters):
        # compute gradient
        grad = compute_gradient_logistic(y, tx, w)
        # update w through the gradient update
        w = w - gamma * grad
        # calculate loss
        loss = compute_nl_loss(y, tx, w)
        losses.append(loss)
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]


def compute_gradient_reg_logistic(y, tx, w, lambda_):
    """Compute the gradient of regularized logistic regression."""
    w_reg = w
    w_reg[0] = 0
    return compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w_reg


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using GD.

    Parameters
    ----------
    y : Target
    tx : Data
    lambda_ : Regularization parameter
    initial_w : Initial weight vector
    max_iters : Number of iterations
    gamma : Learning rate or step size in each iteration

    Returns
    -------
    w : Weight vector after final iteration
    loss : Corresponding loss calculated by negative log likelihood
    """
    n_samples, n_features = tx.shape

    # check target
    if y.ndim > 2:
        raise ValueError("Target y has wrong shape: {}.".format(y.shape))
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if y.shape[0] != n_samples:
        raise ValueError("Number of samples in tx and y does not correspond: ",
                         "tx has {}, y has {}.".format(n_samples, y_shape[0]))

    n_output = y.shape[1]

    # check initial weight
    if initial_w.ndim == 1:
        initial_w = initial_w.reshape((-1, 1))
    if initial_w.shape != (n_features, n_output):
        raise ValueError("Initial weight has wrong shape: ",
                         "{} != {}.".format(initial_w.shape, (n_features, n_output)))

    # initialize weight
    w = initial_w

    # losses
    losses = []
    # threshold
    threshold = 1e-8
    
    for n_iter in range(max_iters):
        # compute gradient
        grad = compute_gradient_reg_logistic(y, tx, w, lambda_)
        # update w through the gradient update
        w = w - gamma * grad
        # calculate loss
        loss = compute_nl_loss(y, tx, w)
        losses.append(loss)
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]

