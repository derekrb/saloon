#!/usr/bin/env python

import numpy as np


def ridge_reg(X, y):
    '''
    Fit a ridge regression.

    Parameters:
        X (matrix): An m x n matrix of training data
        y (matrix): A 1 x m matrix of binary payoffs

    Returns:
        A_inv (array): The inverted covariance matrix of the regression
        theta (array): The parameter vector of the regression
    '''

    I = np.matrix(np.identity(X.shape[1]))
    A = np.transpose(X) * X * I
    A_inv = np.linalg.inv(A)
    theta = A_inv * np.transpose(X) * y

    return A_inv, theta


def set_values(values, n):
    '''
    Generate an array of values to be associated with binary payoffs.
    Handles the case where all binary payoffs are valued equally.

    Parameters:
        values (array): A 1 x n array of values, or None if valued equally
               n (int): The length of the values array (i.e. number of arms)

    Returns:
        values (array): A 1 x n array of values
    '''

    if values is None:
        values = np.ones(n)

    return values


def get_samples(trials, successes, n_samples, smoothing, values):
    '''
    Draw expected payoffs from a beta distribution for many arms.

    Parameters:
           trials (array): A 1 x n array with total trials for each arm
        successes (array): A 1 x n array with total successes for each arm
          n_samples (int): The number of samples to pull from each arm's distribution
                           for Thompson Sampling.
        smoothing (float): The constant factor by which to divide all trials and successes
           values (array): A 1 x n array with the reward value for each arm

    Returns:
        samples (array): An n x n_samples array of expected rewards for arm n on sample n_sample
    '''

    #create an array of n rows (# of arms) and n_samples columns (# of samples) and fills with 0s
    samples = np.zeros((len(trials), n_samples))

    for i, _ in enumerate(trials):
        #for each arm (row), populate a random x value from the arm's distribution for every sample (column)
        samples[i, :] = np.random.beta(successes[i] / smoothing, (trials[i] - successes[i]) / smoothing, n_samples) * values[i]

    return samples


def get_weights(samples):
    '''
    Determine raw (unnormalized) weights for a bandit, given reward samples.

    Parameters:
        samples (array): An n x n_samples array with expected rewards for arm n on sample n_sample

    Returns:
        weights (array): A 1 x n array of unnormalized weights for each arm
    '''

    #argmax will find the winning x value from each sample (column) and make a 1 x n_samples array containing the index of the arm that won each sample
    #bincount will count the number of times each arm (index) won and insert the count into its respective index in a 1 x n array
    #then divide number of successes by number of samples to determine each arm's weight
    weights = np.bincount(np.argmax(samples, axis=0)) / float(samples.shape[1])

    #if arms at end of array didn't win any samples, bins are getting dropped; append 0s to the end
    if len(weights) < samples.shape[0]:
        weights = np.hstack((weights, np.zeros(samples.shape[0] - len(weights))))

    return weights


def normalize_weights(weights, baseline):
    '''
    Determine normalized weights for a bandit, given raw weights and constant baseline.

    Parameters:
         weights (array): A 1 x n array of unnormalized weights for each arm
        baseline (float): The minimum weight to give each arm

    Returns:
        n_weights (array): A 1 x n array with normalized weights for each arm
    '''

    if baseline > 0:
        #print "sum of weights is %s before adding baseline" % np.sum(self.weights)
        weights = (weights + baseline)
        #print "sum of weights is %s after adding baseline" % np.sum(self.weights)

    return weights / np.sum(weights)
