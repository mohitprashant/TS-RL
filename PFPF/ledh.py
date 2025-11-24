# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 22:29:14 2025

@author: 18moh
"""

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
import numpy as np
from particle_flow import ParticleFlowBase


class LEDHFilter(ParticleFlowBase):
    def filter(self, observations):
        for t in range(tf.shape(observations)[0]):
            y_obs = observations[t]
            self.propagate()
            
            for k in range(self.steps):
                lam = float(k) * self.d_lambda
                mu_curr = tf.reduce_mean(self.particles)
                P_curr = tf.reduce_mean(tf.square(self.particles - mu_curr))
                grad, hess = self._get_grads_and_hessian(y_obs, self.particles)
                numerator = P_curr * grad
                denominator = 1.0 + (lam * P_curr * hess)
                denominator = tf.where(tf.abs(denominator) < 1e-4, 1e-4, denominator)
                
                slope = - (numerator / denominator)
                self.particles = self.particles + (slope * self.d_lambda)
            self.estimate()
        return tf.stack(self.mu_history)