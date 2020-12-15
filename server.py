#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
#import asyncio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
#import websockets
import copy
import uuid
from dwc_implementation import distributed_weight_consolidation as dwc
from make_model import sample_net

app = FastAPI()
input_shape = (3,)
updated_file = ''
consolidation_func = dwc

@app.get("/")
async def root():
    return "root"

@app.get("/{model}/")
async def update_model(model: tf.keras.Model):
    return FileResponse(updated_file)

@app.post("/weights")
async def load_weights(image: UploadFile = File(...)): #do we still need to load a file here
    model = sample_net(input_shape)
    model.load_weights(image.filename)
    consolidate(consolidation_func)
    return {'consolidated weights': model}

def load_data():
    model = sample_net(input_shape)
    datasets = []
    client_ids = []
    path = os.getcwd()
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename[:6] != 'model-':
                continue
            model.load_weights(filename)
            datasets.append(copy.deepcopy(model))
            client_ids.append(filename[6:-3])
    return (datasets, client_ids)

def load_priors(client_ids):
    priors = []
    if not priors:
        prior = None
        for client in client_ids:
            filename = 'prior-'+client+'.h5'
            prior.load_weights(filename)
            priors.append(copy.deepcopy(prior))
    return priors

def load_data_and_priors():
    datasets, client_ids = load_data()
    priors = load_priors(client_ids)
    return (datasets, priors)

async def consolidate(consolidation_func, datasets, priors):
    datasets, priors = load_data_and_priors()
    
    model = consolidation_func(datasets, priors)
    
    updated_file = 'server/consolidated-'+uuid.uuid4().__str__()+'.h5'
    
    model.save_weights(
        updated_file,
        save_format = 'h5')
