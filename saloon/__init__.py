#!/usr/bin/env python

import numpy as np
import utils
import logging
import json
import sqlalchemy
import ast
import psycopg2


class ContextThompsonBandit:
    '''
    Contextual Multi-armed Bandit using Thompson sampling from Gaussian posterior
    reward distributions for each arm. Fits the context set to each arm
    using ridge regression.

    Reference: http://research.microsoft.com/pubs/178904/p661.pdf
    '''


    def __init__(self, loglevel=logging.WARNING):
        logging.basicConfig(level=loglevel)
        self.log = logging.getLogger()
        self.A = {}
        self.A_inv = {}
        self.theta = {}

    def _load_raw(self,raw):
        for (arm, theta, a_inv, value) in raw:
            self.arms = np.append(self.arms, arm)
            self.values = np.append(self.values, value)
            self.theta[arm] = np.asarray(ast.literal_eval(theta))
            self.A_inv[arm] = np.asarray(ast.literal_eval(a_inv))

    def fit(self, X, y, a, values=None):
        '''
        Fit the bandit on contextual data.

        Parameters:
            X (array): An m x n array of contexts
            y (array): A 1 x m array of binary payoffs
            a (array): A 1 x m array indicating the arm chosen for each context in X

        Updates:
            self.A_inv (dict): A dict of covariance matrices for each arm
            self.theta (dict): A dict of parameter vectors for each arm
            self.arms (array): An array of arm indices
        '''

        X = np.matrix(X)
        y = np.matrix(y)

        self.arms = np.unique(a)
        self.values = utils.set_values(values, len(self.arms))

        for arm in self.arms:
            ix = (a == arm).flatten()
            try:
                self.A_inv[arm], self.theta[arm] = utils.ridge_reg(X[ix, :], y[ix])
            except Exception, e:
                self.log.error('Arm %d is broken: %s.' % (arm, e))
                raise SystemExit(1)


    def save_json(self, filename):
        '''
        Dump the theta vector and covariate matrix to json

        Parameters:
            filename (string): The name of the file to which to dump
        '''

        out = {}

        for arm in self.arms:
            out[arm] = {}
            out[arm]['theta'] = self.theta[arm].tolist()
            out[arm]['A_inv'] = self.A_inv[arm].tolist()

        with open(filename, 'w') as outfile:
            json.dump(out, outfile)
            outfile.close()


    def save_mysql(self, host, database, table, user, password):
        '''
        Dump the theta vector and covariate matrix to a MySQL table

        Parameters:
             host (string): Database server
               table (str): Table in which to store data. Must include the columns
                            arm (int), theta (text), a_inv (text)
                user (str): Database username
            password (str): Database password
        '''

        engine = sqlalchemy.create_engine('mysql://%s:%s@%s/%s' \
                    % (user, password, host, database))

        conn = engine.connect()

        for arm in self.arms:
            conn.execute('''
                         INSERT INTO %s (arm, theta, a_inv, values)
                         VALUES(%s, '%s', '%s', %s)
                         ON DUPLICATE KEY UPDATE arm = VALUES(arm), theta = VALUES(theta), a_inv = VALUES(a_inv)
                         ''' % (table, arm, self.theta[arm].tolist(), self.A_inv[arm].tolist(), self.values[arm])
            )


    def load_mysql(self, host, database, table, user, password, arm_col='arm', theta_col='theta', a_inv_col='a_inv', value_col='values'):
        '''
        Load theta vector and covariate matrix from a MySQL table

        Parameters:
             host (string): Database server
               table (str): Table in which to store data. Must include the columns
                            arm (int), weight (float)
                user (str): Database username
            password (str): Database password
             arm_col (str): Name of arm column (default 'arm')
           theta_col (str): Name of theta column (default 'theta')
           a_inv_col (str): Name of covariate matrix column (default 'a_inv')

        Updates:
            self.A_inv (dict): A dict of covariance matrices for each arm
            self.theta (dict): A dict of parameter vectors for each arm
            self.arms (array): An array of arm indices
        '''

        engine = sqlalchemy.create_engine('mysql://%s:%s@%s/%s' \
                    % (user, password, host, database))

        conn = engine.connect()
        raw = conn.execute('SELECT %s, %s, %s, %s FROM %s ORDER BY 1' % (arm_col, theta_col, a_inv_col, value_col, table)).fetchall()

        self.arms = np.array(())
        self.values = np.array(())

        self._load_raw(raw)

    def save_postgres(self, host, database, table, user, password, port=5432):
        '''
        Dump the theta vector and covariate matrix to a postgres table

        Parameters:
             host (string): Database server
               table (str): Table in which to store data. Must include the columns
                            arm (int), theta (text), a_inv (text)
                user (str): Database username
            password (str): Database password
                port (int): Database port (default is 5432)
        '''

        con = psycopg2.connect(host=host,port=port,user=user,password=password,database=database)
        cur = con.cursor()

        for arm in self.arms:
            cur.execute("""WITH new_value (arm, theta, a_inv, values) as (
                VALUES (%(arm)s,%(theta)s,%(a_inv)s,%(values)s))
                , upsert AS
                (
                    UPDATE %(table)s old
                        SET (arm, theta, a_inv) = (nv.arm, nv.theta, nv.a_inv)
                    FROM new_value nv
                    WHERE old.arm = nv.arm
                    RETURNING old.*
                )
                INSERT INTO %(table)s (arm, theta)
                    SELECT %(arm)s, %(theta)s
                    WHERE NOT EXISTS
                    (SELECT * FROM upsert)
                    """ % {'table':table,
                            'arm':arm,
                            'theta':self.theta[arm].tolist(),
                            'a_inv':self.A_inv[arm].tolist(),
                            'values':self.values[arm]})
            
    def load_postgres(self, host, database, table, user, password, port=5432, arm_col='arm', theta_col='theta', a_inv_col='a_inv', value_col='values'):
        '''
        Load theta vector and covariate matrix from a postgres table

        Parameters:
             host (string): Database server
               table (str): Table in which to store data. Must include the columns
                            arm (int), weight (float)
                user (str): Database username
            password (str): Database password
                port (int): Database port (default is 5432)
             arm_col (str): Name of arm column (default 'arm')
           theta_col (str): Name of theta column (default 'theta')
           a_inv_col (str): Name of covariate matrix column (default 'a_inv')

        Updates:
            self.A_inv (dict): A dict of covariance matrices for each arm
            self.theta (dict): A dict of parameter vectors for each arm
            self.arms (array): An array of arm indices
        '''

        engine = sqlalchemy.create_engine('postgresql://%s:%s@%s:%s/%s' \
                    % (user, password, host, port, database))

        conn = engine.connect()
        raw = conn.execute('SELECT %s, %s, %s, %s FROM %s ORDER BY 1' % (arm_col, theta_col, a_inv_col, value_col, table)).fetchall()

        self.arms = np.array(())
        self.values = np.array(())

        self._load_raw(raw)

    def choose_arm(self, x):
        '''
        Choose an arm using a fitted bandit.

        Parameters:
            x (array): A 1 x n context array

        Returns:
            chosen_arm (int): The index of the arm selected
        '''

        try:
            self.samples = np.zeros(len(self.arms))
        except NameError:
            self.log.error('No arms defined. Use the fit() method and then try again.')
            raise SystemExit(1)

        xt = np.transpose(np.matrix(x))

        for arm in self.arms:
            payoff = np.transpose(xt) * self.theta[arm]
            stdev = np.sqrt(np.transpose(xt) * self.A_inv[arm] * xt)
            sample = np.random.normal(payoff, stdev) * self.values[arm]
            self.samples[arm] = sample

        return np.argmax(self.samples)


class SimpleThompsonBandit:
    '''
    Simple Multi-armed bandit using Thompson sampling from a beta reward
    distribution for each arm.
    '''


    def __init__(self, loglevel=logging.WARNING):
        logging.basicConfig(level=loglevel)
        self.log = logging.getLogger()

    def _load_raw(self,raw):
        for (arm, weight) in raw:
            try:
                self.weights[arm] = weight
            except IndexError:
                self.log.error('Weight table is misaligned; index error on arm %s' % arm)

    def fit(self, successes, trials, n_samples=1000, baseline=0.0, values=None, smoothing=1.0):
        '''
        Generate the weights for each arm based on bandit history.

        Parameters:
            successes (array): A 1 x n array with total successes for each arm
               trials (array): A 1 x n array with total trials for each arm
              n_samples (int): The number of samples to pull from each arm's distribution
                               for Thompson Sampling.
             baseline (float): The minimum weight to give each ar
               values (array): A 1 x n array with the reward value for each arm, or None
            smoothing (float): The constant factor by which to divide all trials and successes

        Updates
            self.weights (array): A 1 x n array with normalized weights for each arm
        '''

        self.values = utils.set_values(values, len(trials))
        self.samples = utils.get_samples(trials, successes, n_samples, smoothing, self.values)
        self._raw_weights = utils.get_weights(self.samples)
        self.weights = utils.normalize_weights(self._raw_weights, baseline)


    def save_mysql(self, host, database, table, user, password):
        '''
        Dump arm weights to a MySQL table

        Parameters:
             host (string): Database server
               table (str): Table in which to store data. Must include the columns
                            arm (int), weight (float)
                user (str): Database username
            password (str): Database password
        '''

        engine = sqlalchemy.create_engine('mysql://%s:%s@%s/%s' \
                    % (user, password, host, database))

        conn = engine.connect()

        for arm, weight in enumerate(self.weights):
            conn.execute('''
                         INSERT INTO %s (arm, weight)
                         VALUES(%s, %s)
                         ON DUPLICATE KEY UPDATE arm = VALUES(arm), weight = VALUES(weight)
                         ''' % (table, arm, weight)
            )

    def load_mysql(self, host, database, table, user, password, arm_col='arm', weight_col='weight'):
        '''
        Load arm weights from a MySQL table

        Parameters:
             host (string): Database server
               table (str): Table in which to store data. Must include the columns
                            arm (int), weight (float)
                user (str): Database username
            password (str): Database password
             arm_col (str): Name of arm column (default 'arm')
          weight_col (str): Name of weight column (default 'weight')

        Updates:
            self.weights (array): A 1 x n array with normalized weights for each arm
        '''

        engine = sqlalchemy.create_engine('mysql://%s:%s@%s/%s' \
                    % (user, password, host, database))

        conn = engine.connect()
        raw = conn.execute('SELECT %s, %s FROM %s ORDER BY 1' % (arm_col, weight_col, table)).fetchall()

        self.weights = np.zeros(len(raw))

        self._load_raw(raw)

    def save_postgres(self, host, database, table, user, password, port=5432):
        '''
        Dump arm weights to a postgres table

        Parameters:
             host (string): Database server
               table (str): Table in which to store data. Must include the columns
                            arm (int), weight (float)
                user (str): Database username
            password (str): Database password
                port (int): Database port (default is 5432)
        '''

        con = psycopg2.connect(host=host,port=port,user=user,password=password,database=database)
        cur = con.cursor()

        for arm, weight in enumerate(self.weights):
            # Postgres equivalent of INSERT IGNORE (an upsert)
            cur.execute("""
            WITH new_value (arm, weight) as (
                VALUES (%(arm)s,%(weight)s))
                , upsert AS
                (
                    UPDATE %(table)s old
                        SET (arm,weight) = (nv.arm,nv.weight)
                    FROM new_value nv
                    WHERE old.arm = nv.arm
                    RETURNING old.*
                )
                INSERT INTO %(table)s (arm, weight)
                    SELECT %(arm)s,%(weight)s
                    WHERE NOT EXISTS 
                    (SELECT * FROM upsert)
            """ % {'table':table,'arm':arm,'weight':weight})
            con.commit()
        
    def load_postgres(self, host, database, table, user, password, port=5432, arm_col='arm', weight_col='weight'):
        '''
        Load arm weights from a postgres table

        Parameters:
             host (string): Database server
               table (str): Table in which to store data. Must include the columns
                            arm (int), weight (float)
                user (str): Database username
            password (str): Database password
                port (int): Database port (default is 5432)
             arm_col (str): Name of arm column (default 'arm')
          weight_col (str): Name of weight column (default 'weight')

        Updates:
            self.weights (array): A 1 x n array with normalized weights for each arm
        '''

        engine = sqlalchemy.create_engine('postgresql://%s:%s@%s:%s/%s' \
                    % (user, password, host, port, database))

        conn = engine.connect()
        raw = conn.execute('SELECT %s, %s FROM %s ORDER BY 1' % (arm_col, weight_col, table)).fetchall()

        self.weights = np.zeros(len(raw))

        self._load_raw(raw)


    def choose_arm(self, n=1):
        '''
        Select n arms to play given the weight distribution in self.weights

        Parameters:
            n (int): The number of arms to play

        Returns:
            arms (array): A 1 x n array of arms selected from self.weights
        '''

        try:
            temp_weights = np.copy(self.weights)
        except NameError:
            self.log.error('No weights defined. Use the fit() method and try again.')
            raise SystemExit(1)

        arms = np.zeros(n)

        for i in range(n):

            if sum(temp_weights) == 0.0:
                self.log.error('n greater than available arms. Choose smaller n.')
                raise SystemExit(1)               

            sum_weights = np.cumsum(temp_weights / sum(temp_weights))
            arms[i] = np.digitize(np.random.random_sample(1), sum_weights)[0]
            temp_weights[int(arms[i])] = 0.0

        return arms
