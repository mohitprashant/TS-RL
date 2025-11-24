# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 22:23:37 2025

@author: 18moh
"""

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
import numpy as np


class SVSSM:
    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5, seed=None):
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.sigma = tf.constant(sigma, dtype=tf.float32)
        self.beta  = tf.constant(beta, dtype=tf.float32)
        if seed: tf.random.set_seed(seed); np.random.seed(seed)
        
        self.stationary_std = self.sigma / tf.sqrt(1.0 - self.alpha**2)
        self.x = tf.random.normal(shape=[], mean=0.0, stddev=self.stationary_std)
        w_noise = tf.random.normal(shape=[], mean=0.0, stddev=1.0)
        self.y = self.beta * tf.math.exp(self.x / 2.0) * w_noise
        self.state_history = [self.x]
        self.observation_history = [self.y]

    def generate_data(self, steps):
        for _ in range(steps):
            v_n = tf.random.normal(shape=[], mean=0.0, stddev=1.0)
            w_n = tf.random.normal(shape=[], mean=0.0, stddev=1.0)
            self.x = (self.alpha * self.x) + (self.sigma * v_n)
            self.y = self.beta * tf.math.exp(self.x / 2.0) * w_n
            self.state_history.append(self.x)
            self.observation_history.append(self.y)
        return tf.stack(self.observation_history), tf.stack(self.state_history)