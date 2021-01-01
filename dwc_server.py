#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import asyncio
from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import websockets
import copy
import numpy as np
import uuid

app = FastAPI()

datasets = []
num_clients = 1
model = None

@app.get('/')
async def root:
    return 'root'

@app.get("/{model}/")
async def update_model(model: tf.keras.Model):
    return FileResponse('consolidated.h5')

@app.post("/weights")
async def load_weights(image: UploadFile = File(...)): #do we still need to load a file here
    model.load_weights(image.filename)
    datasets.append(copy.deepcopy(model))
    dwc_frame()
    return {'consolidated weights': model}

def load_data():
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename[:6] != 'model-':
                continue
            model.load_weights(filename)
            datasets.append(copy.deepcopy(model))
    

async def dwc_frame():
    #load weights
    load_data()
    
    # need to figure out how to only run this code after all (or most) client weights
    
    # dwc implementation
    # what are the priors
    model = distributed_weight_consolidation(datasets, priors)
    
    model.save_weights(
        'server/consolidated-'+uuid.uuid4().__str__()+'.h5',
        save_format = 'h5')
    update_model(model)


# LOOK AT THIS LATERRRRR

async def distributed_weight_consolidation(models_weights, model_priors):
    # models is a list of weights of client-models; models = [model1, model2, model3...]
    num_layers =  int(len(model_weights[0])/2.0)
    num_datasets  = np.shape(model_weights)[0]
    consolidated_model = model_weights[0]
    mean_idx = [i for i in range(0,len(model_weights[0])) if i % 2 == 0]
    std_idx = [i for i in range(0,len(model_weights[0])) if i % 2 != 0]
    ep = 1e-5
    for i in range(num_layers):
        num_1 = 0; num_2 = 0
        den_1 = 0; den_2 = 0
        for m in range(num_datasets):
            model = model_weights[m]
            prior = model_priors[m]
            mu_s = model[mean_idx[i]]
            mu_o = prior[mean_idx[i]]
            sig_s = model[std_idx[i]]
            sig_o = prior[std_idx[i]]
            d1 = np.power(sig_s,2) + ep; d2= np.power(sig_o,2) + ep
            num_1 = num_1 + (mu_s/d1)
            num_2 = num_2 + (mu_o/d2)
            den_1 = den_1 + (1.0/d1)
            den_2 = den_2 + (1.0/d2)
        consolidated_model[mean_idx[i]] =  (num_1 - num_2)/(den_1 -den_2)
        consolidated_model[std_idx[i]] =  1/(den_1 -den_2)
    return consolidated_model
    