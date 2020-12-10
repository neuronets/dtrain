#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Aakanksha Rana, Emi Z Liu
"""
import numpy as np

def distributed_weight_consolidation(model_weights, model_priors):
    # models is a list of weights of client-models; models = [model1, model2, model3...]
    num_layers =  int(len(model_weights[0])/2.0)
    num_datasets  = np.shape(model_weights)[0]
    consolidated_model = model_weights[0]
    mean_idx = [i for i in range(len(model_weights[0])) if i % 2 == 0]
    std_idx = [i for i in range(len(model_weights[0])) if i % 2 != 0]
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
            num_1 += (mu_s/d1)
            num_2 += (mu_o/d2)
            den_1 += (1.0/d1)
            if m != num_datasets-1:
                den_2 +=(1.0/d2)
        consolidated_model[mean_idx[i]] =  (num_1 - num_2)/(den_1 -den_2)
        consolidated_model[std_idx[i]] =  1/(den_1 -den_2)
    return consolidated_model
    
