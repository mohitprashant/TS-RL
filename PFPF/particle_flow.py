# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 22:26:19 2025

@author: 18moh
"""

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
import numpy as np

class ParticleFlowBase:
    def __init__(self, alpha, sigma, beta, num_particles=200, flow_steps=20, mu_0=0.0, P_0=1.0):
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.sigma = tf.constant(sigma, dtype=tf.float32)
        self.beta  = tf.constant(beta, dtype=tf.float32)
        self.N = num_particles
        self.steps = flow_steps
        self.d_lambda = 1.0 / float(self.steps)
        
        self.particles = tf.random.normal([self.N], mean=mu_0, stddev=tf.sqrt(P_0))
        self.mu_history = [mu_0]

    def _get_grads_and_hessian(self, y, x):
        with tf.GradientTape() as t2:
            t2.watch(x)
            with tf.GradientTape() as t1:
                t1.watch(x)
                term_exp = (y**2) / (2.0 * self.beta**2) * tf.math.exp(-x)
                log_lik = -0.5 * x - term_exp
            grad = t1.gradient(log_lik, x)
        hess = t2.gradient(grad, x)
        if grad is None: grad = tf.zeros_like(x)
        if hess is None: hess = tf.zeros_like(x)
            
        return grad, hess

    def propagate(self):
        noise = tf.random.normal([self.N], mean=0.0, stddev=1.0)
        self.particles = (self.alpha * self.particles) + (self.sigma * noise)

    def estimate(self):
        mu = tf.reduce_mean(self.particles)
        self.mu_history.append(mu)
        return mu