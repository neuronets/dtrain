#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:41:10 2020

@author: Emi Z Liu
"""
import tensorflow as tf
import tensorflow_probability as tfp

# set up lightweight bayesian neural network
def sample_net(input_shape,
    activation = tf.nn.relu,
    batch_size = None):
    def one_layer(x, dilation_rate=(1, 1, 1)):
        x = tfp.layers.Convolution3DReparameterization(
                kernel_size=3,
                padding="same",
                dilation_rate=dilation_rate,
                activation=activation,
                name="layer/vwnconv3d",
            )(x)
        x = tf.keras.layers.Activation(activation, name="layer/activation")(x)
        return x
    
    inputs = tf.keras.layers.input(shape = input_shape, batch_size=batch_size, name="inputs")
    x = one_layer(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=x)