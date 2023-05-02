#!/usr/bin/env python
# coding: utf-8

from SEIRWWfiles_R2S.RWest import RWest
from SEIRWWfiles_R2S.SEIR_WW import SEIR_WW
import numpy as np
import math

def paramFit(params,YC,YW,C,gs,es,ns):

    # param
    params['gamma'] = gs
    params['WWexp'] = es

    # Estimate wastewater measurement variance
    RW0 = RWest(YW, params['WWexp'])

    params['RW0'] = RW0
    params['RW'] = params['RW0'] / 10

    if ns > 0:
        params['nu'] = ns
    else:
    # Find nu
        params['nu'] = 1
        SEIR_WW_return = SEIR_WW(params,YC,YW,C,[True,False],1000)
        Yest = SEIR_WW_return[0]
        WWinds = np.where(YW > -0.5)[0]
        Y2 = 1e-5 * YW[WWinds] ** es
        YCaux = np.sort(YC)
        YWaux = np.sort(Y2)
        ccc = np.mean(YCaux[0:math.floor(len(YC)/10)]) / np.mean(YCaux[math.floor(len(YC) / 10) : len(YCaux)])
        aaa = np.mean(YWaux[0:math.floor(len(YWaux) / 10)])
        bbb = np.mean(YWaux[math.floor(len(YWaux) / 10):len(YWaux)])

        if ccc * bbb < aaa:
            Y2 = Y2 - np.min(((aaa - ccc * bbb) / (1 - ccc), np.min(Y2)))


        XX = Yest[1,WWinds]
        nu = Y2 @ XX.T * (XX @ XX.T)**-1
        params['nu'] = nu

    SEIR_WW_return = SEIR_WW(params,YC,YW,C,[False,True],1000)

    Yest = SEIR_WW_return[0]
    Js = np.linalg.norm(Yest[0,:] - YC)**2

    nu = params['nu']

    return Js, nu, RW0

