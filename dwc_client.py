#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_probability as tfp
import os
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import websockets
import uuid
import glob

client_id = 'A'
model = None
inputs = None
outputs = None

# set up lightweight bayesian neural network
def sample_net(input_shape,
    activation = tf.nn.relu,
    batch_size = None):
    def one_layer(x, dilation_rate=(1, 1, 1)):
        x = tfpl.Convolution3DReparameterization(
                filters,
                kernel_size=3,
                padding="same",
                dilation_rate=dilation_rate,
                activation=activation,
                name="layer/vwnconv3d",
            )(x)
        x = tfkl.Activation(activation, name="layer/activation")(x)
        return x
    
    inputs = tf.keras.layers.input(shape = input_shape, batch_size=batch_size, name="inputs")
    x = one_layer(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

def train(inputs, outputs):
    if model == None:
        model = sample_net(np.shape(inputs))
    # input shape: 4 x 1
    _op = 'adam'
    _loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    _metrics = ['accuracy']
    model.compile(optimizer=_op, loss=_loss, metrics=_metrics)
    model.fit(inputs=inputs, outputs=outputs, epochs=1, verbose=2)
    

def most_recent_consolidated():
    list_of_files = glob.glob('/server/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# ping server side
@app.get('/')
async def send_weights():
    return FileResponse('model-'+client_id+'.h5')

@app.post("/")
async def load_consolidated():
    model.load_weights(most_recent_consolidated())
    train()
    return {'consolidated weights': model}
