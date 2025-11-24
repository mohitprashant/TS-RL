# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 22:30:36 2025

@author: 18moh
"""


import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
import numpy as np
from particle_flow import ParticleFlowBase


class IPFFilter(ParticleFlowBase):
    def filter(self, observations):
        for t in range(tf.shape(observations)[0]):
            y_obs = observations[t]
            self.propagate()
            
            for k in range(self.steps):
                mu_curr = tf.reduce_mean(self.particles)
                P_curr = tf.reduce_mean(tf.square(self.particles - mu_curr))
                grad, hess = self._get_grads_and_hessian(y_obs, self.particles)
                
                deterministic_drift = -0.5 * P_curr * grad
                diffusion_scale = tf.sqrt(tf.abs(P_curr)) * 0.2 
                brownian_noise = tf.random.normal([self.N])
                
                update = (deterministic_drift * self.d_lambda) + \
                         (diffusion_scale * brownian_noise * tf.sqrt(self.d_lambda))
                self.particles = self.particles + update
            self.estimate()
        return tf.stack(self.mu_history)
    
    