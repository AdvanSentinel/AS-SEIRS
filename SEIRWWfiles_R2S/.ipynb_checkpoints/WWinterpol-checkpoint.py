#!/usr/bin/env python
# coding: utf-8

import numpy as np

def WWinterpol(YW):
    _YW = YW.copy()
    inds = np.where(_YW>-0.5)[0]
    for jt in range(1,len(inds)):
        _YW[inds[jt-1] + 1 : inds[jt]] = _YW[inds[jt-1]] + (list(range(1,inds[jt] - inds[jt-1])) / (inds[jt] - inds[jt-1])) * (_YW[inds[jt]] - _YW[inds[jt-1]])

    return _YW





