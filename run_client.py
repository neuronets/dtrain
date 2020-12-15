#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:15:13 2020

@author: Emi Z Liu
"""
from client import Client
import time
import os

c = Client('A')
# stores most recent model file from server
model_file = ''

def read_data():
    datafiles = []
    path = 'data'
    for root, dirs, files in os.walk(path):
        for filename in files:
            if '.nii' not in filename:
                continue
            datafiles.append(os.path.join(path, filename))
    return datafiles

data = read_data()

def has_new_model(model_file):
    if model_file == '':
        return False
    try:
        model_file = c.load_consolidated()
        return True
    except:
        return False

def new_data():
    # need to find a way to speed this up, right now it's traversing
    # the entire directory which is very inefficient
    updated = read_data()
    if set(data) == set(updated):
        return False
    return True

while True:
    if has_new_model():
        c.train()
        c.save_weights()
    else:
        if new_data():
            c.train()
            c.save_weights()
        time.sleep(1)