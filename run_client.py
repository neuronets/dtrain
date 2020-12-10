#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:15:13 2020

@author: Emi Z Liu
"""
from client import Client
import time

c = Client('A')

def has_new_model():
    try:
        c.load_consolidated()
        return True
    except:
        return False

def new_data():
    pass

while True:
    if has_new_model():
        c.train()
        c.save_weights()
    else:
        if new_data():
            c.train()
            c.save_weights()
        time.sleep(1)