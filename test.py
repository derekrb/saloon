#!/usr/bin/env python

import saloon
import saloon.utils as utils
import numpy as np
import nose

### UTILS ###
def approx_equals(v1, v2):
    ''' Check equality of floats within +/- 5% '''
    rel = v1 / v2
    if rel > 1.05 or rel < 0.95:
        return False
    return True


### TESTS ###
def test_normalize_weights_no_baseline():
    ''' Test that normalized weights sum to 1 '''
    w = utils.normalize_weights(np.random.rand(5), 0)
    assert approx_equals(np.sum(w), 1.0)


def test_normalize_weights_with_baseline():
    ''' Test that normalized weights sum to 1, all weights are nonzero '''
    w = utils.normalize_weights(np.random.rand(5), 0.3)
    assert approx_equals(np.sum(w), 1.0)
    assert min(w) > 0.0


def test_get_weights():
    ''' Test that weight vector is of the correct dimensions '''
    w = utils.get_weights(np.random.rand(5, 1000))
    assert w.shape == (5,)


def test_get_weights_relative():
    ''' Test that arms with better payoffs get more weight '''
    s = np.ones((5, 10)) * 0.5
    s[0, :] = 0.55
    s[4, :] = 0.45
    w = utils.get_weights(s)
    assert w[0] >= w[1]
    assert w[1] >= w[4]


def test_set_values():
    ''' Test that set_values handles both None and arrays '''
    v1 = utils.set_values(None, 5)
    v2 = np.random.rand(5)
    v3 = utils.set_values(v2, 5)
    assert v1.all() == 1
    assert sum(v2) == sum(v3)


def test_ridge_reg():
    ''' Test that the ridge regression returns well-formed parameters '''

    inv = False
    I = np.matrix(np.identity(20))
    while not inv:
        X = np.matrix(np.round(np.random.rand(1000, 20)))
        if np.linalg.det(np.transpose(X) * X * I) != 0:
            inv = True

    y = np.matrix(np.round(np.random.rand(1000, 1)))
    A_inv, theta = utils.ridge_reg(X, y)
    assert A_inv.shape == (20, 20)
    assert theta.shape == (20, 1)


class TestSampling():
    ''' Tests for the get_samples function '''

    def __init__(self):
        self.t = np.ones(5) * 1000
        self.s = np.ones(5) * 10
        self.v = np.ones(5)

    def test_get_samples(self):
        '''
        Test that get_samples returns payoff probabilities of the correct
        dimensions and range of values.
        '''
        samples = utils.get_samples(self.t, self.s, 1000, 1.0, self.v)
        assert samples.shape == (5, 1000)
        assert np.max(samples) <= 1.0
        assert np.min(samples) >= 0.0

    def test_values_increases_payoff(self):
        ''' Test that increasing the value vector increases payoffs '''
        v = self.v
        v[0] = 1.5
        samples = utils.get_samples(self.t, self.s, 1000, 1.0, v)
        for i, r in enumerate(samples):
            if i == 0:
                continue
            assert np.sum(samples[0]) > np.sum(r)

    def test_smoothing_decreases_payoff(self):
        ''' Test that increasing the smoothing broadens payoffs '''
        s = self.s
        s[0] = 30
        u_samples = utils.get_samples(self.t, s, 1000, 1.0, self.v)
        s_samples = utils.get_samples(self.t, s, 1000, 100.0, self.v)
        assert np.var(u_samples[0]) < np.var(s_samples[0])


class TestSimple():
    ''' Tests for the SimpleThompsonBandit '''

    def __init__(self):
        self.bandit = saloon.SimpleThompsonBandit()
        self.t = np.ones(5) * 1000
        self.s = np.ones(5) * 10
        self.bandit.fit(self.s, self.t)


    def test_choose_arm_proportion(self):
        ''' Test that arms are chosen in reasonable proportion, given uniform input '''
        arms = []
        for i in range(10000):
            arms.extend(self.bandit.choose_arm())
        chosen = np.bincount(np.array(arms, dtype=int))
        assert max(chosen) < 1.5 * min(chosen)


class TestContext():
    ''' Tests for the ContextThompsonBandit '''

    def __init__(self):
        self.bandit = saloon.ContextThompsonBandit()
        self.X = np.round(np.random.rand(5000, 10))
        self.y = np.round(np.random.rand(5000, 1))
        self.a = np.round(np.random.rand(5000, 1) * 9)
        self.bandit.fit(self.X, self.y, self.a)


    def test_arm_count(self):
        ''' Test that the correct number of arms have been identified '''
        assert len(self.bandit.arms) == 10


    def test_choose_arm(self):
        ''' Test that an arm can be chosen '''
        c = np.ones(10)
        a = self.bandit.choose_arm(c)
        assert a >= 0
        assert a <= 9

    def test_values_changes_arm(self):
        ''' Test that increasing payoff value increase change of choosing an arm '''
        c = np.ones(10)

        u_arms = []
        for i in range(1000):
            u_arms.append(self.bandit.choose_arm(c))
        u_chosen = np.bincount(u_arms)

        self.bandit.values[0] = 1.5
        v_arms = []
        for i in range(1000):
            v_arms.append(self.bandit.choose_arm(c))
        v_chosen = np.bincount(v_arms)

        assert v_chosen[0] > 2 * u_chosen[0]
