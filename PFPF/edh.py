# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 22:27:42 2025

@author: 18moh
"""



import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
import numpy as np
from particle_flow import ParticleFlowBase

class EDHFilter(ParticleFlowBase):
    def filter(self, observations):
        for t in range(tf.shape(observations)[0]):
            y_obs = observations[t]
            self.propagate()
            
            for _ in range(self.steps):
                mu_curr = tf.reduce_mean(self.particles)
                P_curr = tf.reduce_mean(tf.square(self.particles - mu_curr))
                grad, _ = self._get_grads_and_hessian(y_obs, self.particles)
                drift = -0.5 * P_curr * grad
                self.particles = self.particles + (drift * self.d_lambda)
            
            self.estimate()
        return tf.stack(self.mu_history)