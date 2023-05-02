#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
def SEIRWWinit(YC, specialHolidays, darkNumber):
    # Prepare frame C
    C = np.ones(1000)
    included =  np.ones(C.size)
    included[specialHolidays] = 0
    included[np.where(YC<0)] = 0
    
    # Calculate averages over several weeks
    for jd in range(7):
        C[jd:jd+29:7] = np.mean(YC[jd:jd+29:7])/np.mean(YC[0:35])

    for jd in range(35,len(YC)):
        normC = (included[jd-7] + included[jd-14] + included[jd-21] + included[jd-28])
        C[jd] = (included[jd-7] * YC[jd-7] + included[jd-14] * YC[jd-14] + included[jd-21] * YC[jd-21] + included[jd-28] * YC[jd-28])

        if (normC > 0) & (C[jd] > 0):
            C[jd] = C[jd] / normC
            C[jd] = 28 * C[jd] / np.sum(YC[jd-27:jd+1])
        else:
            C[jd] = 1

    for jd in range(len(YC),len(C)):
        C[jd] = (C[jd-7] + C[jd-14] + C[jd-21] + C[jd-28] + C[jd-35]) / 5
        
    # Standardized by moving average
    Cs = pd.Series(C)
    Cm = Cs.rolling(window=7, min_periods=1).mean()
    Cm = np.array(Cm)
    C = C / Cm

    # Adjustment by darknumber
    for jd in range(len(darkNumber)-1):
        C[int(darkNumber[jd,1]):int(darkNumber[(jd+1),1]-1)] = C[int(darkNumber[jd,1]):int(darkNumber[(jd+1),1]-1)] / darkNumber[jd,0]

    C[int(darkNumber[(len(darkNumber)-1),1]):] = C[int(darkNumber[(len(darkNumber)-1),1]):] /darkNumber[(len(darkNumber)-1),0]

    C[specialHolidays] = 0.25 * C[specialHolidays] 
    
    return C

