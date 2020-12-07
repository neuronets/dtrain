#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow_federated.python.learning import federated_averaging
import os
import asyncio
from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import HTMLResponse
import websockets
import copy

app = FastAPI()

datasets = []
num_clients = 1
model = None
clients = [0] #this would be a list of all client ids

@app.get('/')
async def root:
    return 'root'

@app.get("/{model}/")
async def update_model(model: tf.keras.Model):
    return model.load_weights('consolidated.h5')

@app.post("/weights")
async def load_weights(image: UploadFile = File(...)):
    model.load_weights(image.filename)
    datasets.append(copy.deepcopy(model))
    load_data()
    return {'consolidated weights': model}

async def load_data():
    for client_id in clients:
        await load_weights(client_id)

async def fed_avg():
    await load_data()
    
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))
    
    model = iterative_process.initialize()
    
    self.assertIsInstance(
        iterative_process.get_model_weights(model), model_utils.ModelWeights)
    self.assertAllClose(model.model.trainable,
                        iterative_process.get_model_weights(model).trainable)
    
    for _ in range(num_clients):
        model, _ = iterative_process.next(model, datasets)
        # self.assertIsInstance(
        #     iterative_process.get_model_weights(model), model_utils.ModelWeights)
        # self.assertAllClose(model.model.trainable,
        #                     iterative_process.get_model_weights(model).trainable)
      
    
    model.save_weights('consolidated.h5', save_format = 'h5')
    update_model(model)
    