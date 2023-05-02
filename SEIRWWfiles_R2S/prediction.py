#!/usr/bin/env python
# coding: utf-8

import numpy as np
from SEIRWWfiles_R2S.SEIR_WW import SEIR_WW
from SEIRWWfiles_R2S.SEIR_WW_FWD import SEIR_WW_FWD
from SEIRWWfiles_R2S.WWinterpol import WWinterpol

def prediction(params,WWinds,YC,YW,C,winLength,S_init):
    
    # Predictions using only case data
    predsCase = np.zeros((1,len(WWinds)))
    params['RW'] = params['RW0'] / 10
    result = SEIR_WW(params,YC,YW,C,[True, False],WWinds,S_init)
    Xend = result[1]
    P = result[2]
    for jd in range(len(WWinds)):
        result = SEIR_WW_FWD(Xend[:,jd],C,P,WWinds[jd]+1,params,winLength)
        Yest = result[0]
        predsCase[0][jd] = np.sum(Yest[0])
    
    # Predictions using only WW data
    params['RW'] = params['RW0'] / 10
    predsWW = np.zeros((1,len(WWinds)))
    result = SEIR_WW(params,YC,YW,C,[False, True],WWinds,S_init)
    Xend = result[1]
    P = result[2]
    for jd in range(len(WWinds)):
        result = SEIR_WW_FWD(Xend[:,jd],C,P,WWinds[jd]+1,params,winLength)
        Yest = result[0]
        predsWW[0][jd] = np.sum(Yest[0])

    # Predictions using only WW data interpolated
    YWip = WWinterpol(YW)
    params['RW'] = params['RW0'] / 10
    predsWWip =  np.zeros((1,len(WWinds)))
    result = SEIR_WW(params,YC,YWip,C,[False, True],WWinds,S_init)
    Xend = result[1]
    P = result[2]
    for jd in range(len(WWinds)):
        result = SEIR_WW_FWD(Xend[:,jd],C,P,WWinds[jd]+1,params,winLength)
        Yest = result[0]
        predsWWip[0][jd] = np.sum(Yest[0])

    # Predictions using case & WW data
    params['RW'] = params['RW0']
    predsBoth = np.zeros((1,len(WWinds)))
    result = SEIR_WW(params,YC,YW,C,[True, True],WWinds,S_init)
    Xend = result[1]
    P = result[2]
    for jd in range(len(WWinds)):
        result = SEIR_WW_FWD(Xend[:,jd],C,P,WWinds[jd]+1,params,winLength)
        Yest = result[0]
        predsBoth[0][jd] = np.sum(Yest[0])
        
    return predsCase, predsWW, predsWWip, predsBoth

