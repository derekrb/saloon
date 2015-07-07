![image](https://raw.githubusercontent.com/derekrb/saloon/master/doc/saloon.gif)

# What is Saloon?

A place to create and train multi-armed bandits. Syntax is modeled on that of scikit-learn. Currently, only Contextual Thompson Sampling and Simple Thompson Sampling bandits are supported.

# Using Saloon

The documentation here will cover the basics of initializing and training a bandit, as well as using a bandit to select arms. For detailed information on how the bandit code operates, checkout the docstrings in the code itself.

# The Simple Thompson Bandit

The Simple Thompson Bandit is an effective tool for making decisions between a wide variety of available arms (e.g. many subject lines or templates) in which user-specific features tend not to matter. That is, every user is expected to have the same preferences for the available arms.

## Training

To train a bandit, you'll need to initialize it, and then feed it training data. Training data here consists of a vector of successes (e.g. clicks) by arm, and a vector of trials (e.g. sends) by arm. Here, the arm is implicit - it's the index of the element in the vector. That is, if your success vector is [10, 30, 12, 15], you have 10 successes for arm 0, 12 for arm 2, etc. So we have:

```python
from saloon import SimpleThompsonBandit

my_bandit = SimpleThompsonBandit()
my_bandit.fit(successes_by_arm, trials_by_arm)
```

Additionally, you can specify a number of samples to use for training (more is better), a baseline value to add to the weights assigned to each arm (to prevent some arms from getting no weight at all), and the value associated with each success, by arm. This can be important if the value of a success (e.g. revenue per click) varies from arm to arm. Specify these when training your bandit:

```python
my_bandit.fit(successes_by_arm, trials_by_arm, n_samples=10000, baseline=0.1, values=value_by_arm)
```

## Saving and Loading

You can save your bandit data to a MySQL table, so that it can be accessed at another time. This is helpful so that you don't have to train your bandit online at send time. To save your data, use the save_mysql() method. This method only writes to the columns "arm" and "weight" in your table, so you can include other columns as necessary for tracking.

```python
my_bandit.save_mysql(some_host, some_db, some_table, my_user, my_password)
```

You can load your bandit data from that table at send time using the load_mysql() method. Note that loading from a table overwrites any data currently in the bandit object.

```python
my_bandit = SimpleThompsonBandit()
my_bandit.load_mysql(some_host, some_db, some_table, my_user, my_password)
```

## Arm Selection

To select an arm, call choose_arm() on a trained bandit as follows:

```python
my_bandit.choose_arm()
```

In some cases, such as selecting many options to insert into a template, you may want to choose multiple arms at once (without replacement). This is supported with the keyword argument n:

```python
my_bandit.choose_arm(n=10)
```

# The Contextual Thompson Bandit

The Contextual Thompson Bandit goes a step further, allowing you to personalize arm selection based on a user's features. This is achieved using a Ridge Regression.

## Training

Training this bandit requires a bit more data. You'll need to pass in data about every training example. Specifically, the features, or "context", of that example, on which to train the model, the arm chosen for that example, and the result (a binary payoff). You can also pass in payoff values, as with the Simple Thompson Bandit:

```python
from saloon import ContextThompsonBandit

my_bandit = ContextThompsonBandit()
my_bandit.fit(contexts, results, arms, values=value_by_arm)
```

## Saving and Loading

Saving and loading follows the same convention as the Simple Thompson Bandit, except the columns used are now "arm", "theta", "a_inv" (the inverted covariance matrix of the Ridge Regression), and "values" (the payoff value of a sucess on each arm).

## Arm Selection

To select an arm, you must provide choose_arm() with the current context (i.e. featureset) of the user for whom you're choosing an arm:

```python
my_bandit.choose_arm(user_context)
```

