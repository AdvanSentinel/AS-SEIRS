#!/usr/bin/env python
# coding: utf-8

import numpy as np

def RWest(YW,WWexp):
    inds = np.where(YW > -0.5)[0]
    disc = np.zeros(len(inds)-4)
    _YW = YW.copy()
    _YW[inds] = YW[inds]**WWexp*1e-5

    for jj in range(len(disc)):
        disc[jj] = np.abs(_YW[inds[jj+2]] - np.sum(_YW[inds[jj:jj+5]]) / 5)

    RW0 = np.median(disc)**2
    
    return RW0

