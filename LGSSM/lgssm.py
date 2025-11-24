# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 20:13:09 2025

@author: 18moh
"""

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
import numpy as np


class LGSSM():
    """
    Linear Gaussian State Space Model
    
    State equation: x[t+1] = A @ x[t] + B @ w[t], where w[t] ~ N(0, I)
    Observation equation: y[t] = C @ x[t] + D @ v[t], where v[t] ~ N(0, I)
    """
    def __init__(self, dim : int, **kwargs) -> None :
        self.dim = dim
        
        self.x = None
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.seed = None
        
        self.y = None
        self.state_history = []
        self.observation_history = []
        
        # Conduct assertion checks
        valid_keys = ["x", "A", "B", "C", "D", "seed"]
        
        for key in valid_keys:
            if(key not in kwargs.keys()):
                if(key == 'x'):
                    setattr(self, key, tf.zeros((dim)))
                elif(key == 'seed'):
                    setattr(self, key, None)
                else:
                    setattr(self, key, tf.eye(dim))
                continue
            
            value = kwargs.get(key)
            
            if(key == 'x'):
                if(not isinstance(value, tf.Tensor)):
                    raise TypeError("Input must be a TensorFlow tensor.")
                tf.debugging.assert_equal(tf.shape(value), (dim), message=f"Shape mismatch: Expected ({dim})")
                
            elif(key == 'seed'):
                if(not isinstance(value, int)):
                    raise TypeError("Seed must be an integer value.")
                tf.random.set_seed(value)
                np.random.seed(value)
                
            else:
                if(not isinstance(value, tf.Tensor)):
                    raise TypeError("Input must be a TensorFlow tensor.")
                tf.debugging.assert_equal(tf.shape(value), (dim, dim), message=f"Shape mismatch: Expected ({dim},{dim})")
                
            setattr(self, key, value)
            
        # Generate initial observation
        y_init = self.C @ tf.expand_dims(self.x, axis=1) + self.D @ tf.expand_dims(tf.random.normal(shape=[self.dim]), axis=1)
        self.y = tf.squeeze(y_init)
        self.state_history.append(self.x)
        self.observation_history.append(self.y)
        
        
            
    def generate_next(self) -> tf.Tensor :
        """
        Generates the next state x[t+1] and observation y[t+1] and ppend 
        the new state and observation to their respective histories.
        
        Returns: The new observation vector y[t+1] of shape (dim, 1).
        """
        
        # x[t+1] = A @ x[t] + B @ w[t]
        x_next = self.A @ tf.expand_dims(self.x, axis=1) + self.B @ tf.expand_dims(tf.random.normal(shape=[self.dim]), axis=1)
        
        # y[t+1] = C @ x[t+1] + D @ v[t+1]
        y_next = self.C @ x_next + self.D @ tf.expand_dims(tf.random.normal(shape=[self.dim], dtype=tf.float32), axis=1)
        
        # Update the observation
        self.x = tf.squeeze(x_next)
        self.y = tf.squeeze(y_next)
        self.state_history.append(self.x)
        self.observation_history.append(self.y)
        
        return self.y
    
    
    def generate_data(self, steps : int) -> tf.Tensor :
        """
        Generates data for a specified number of steps, starting from the current state/observation.
        
        Args:
            steps (int): The number of steps (time points) to generate after the initial one.
            
        Returns:
            tf.Tensor: A tensor of shape (steps+1, dim, 1) containing the entire observation history 
                       [y[0], y[1], ..., y[steps]].
        """
        
        for _ in range(steps):
            self.generate_next()
            
        # Stack the observation history into a single tensor
        return tf.stack(self.observation_history, axis=0)
    
    
    def get_lgssm_params(self):
        """Returns: the LGSSM matrices and initial state/covariance."""
        
        Q = self.B @ tf.transpose(self.B) # Process noise covariance
        R = self.D @ tf.transpose(self.D) # Observation noise covariance
        x_0 = self.state_history[0] 
        P_0 = tf.zeros_like(self.A) 

        return self.A, self.C, Q, R, x_0, P_0
    


if __name__ == "__main__":
    lg = LGSSM(10, seed=42)
    print(lg.generate_data(3))

