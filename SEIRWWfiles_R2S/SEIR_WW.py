#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import pandas as pd
from SEIRWWfiles_R2S.SEIRreaction import SEIRreaction

def SEIR_WW(params,YC,YW,C,useData,maxind,S_init):
    
    if type(maxind) == int:
        maxind = np.array([maxind]) 

    if len(maxind) ==1 and np.max(maxind) >= len(YC):
        maxind = np.array([len(YC) - 1])

    # Process the WW data
    WWinds = np.where(YW>-0.5)[0]
    _YW = YW.copy()
    _YW[WWinds] = _YW[WWinds] ** params['WWexp'] *1e-5
    minYW = np.min(_YW[WWinds])
    YCaux = np.sort(YC)
    excl = np.sum(YCaux < 0)
    YWaux = np.sort(_YW[WWinds])
    ccc = np.mean(YCaux[excl:excl + math.floor(len(YC)/10)]) / np.mean(YCaux[excl + math.floor(len(YC) / 10) : len(YCaux)])
    aaa = np.mean(YWaux[0:math.floor(len(YWaux) / 10)])
    bbb = np.mean(YWaux[math.floor(len(YWaux) / 10):len(YWaux)])
    if ccc * bbb < aaa:
        _YW = _YW - np.min([(aaa - ccc * bbb) / (1 - ccc) , minYW])

    # Set parameters
    alpha = params['alpha']
    beta = params['beta']
    tau = params['tau']
    gamma = params['gamma']
    omega = params['omega']
    nu = params['nu'] 
    eta = 1
    CC = params['modelErrorC'] 
    N = params['N']
    params['sigma'] = 1

    OL_limit = 4

    AR = np.array([[-1, 0, 0, 0, 0, 0, 1],
                   [1, -1, 0, 0, 0, 0, 0],   
                   [0, 1, -1, 0, 0, 0, 0],   
                   [1, 0, 0, -1, 0, 0, 0],   
                   [0, 1, 0, 0, 0, 0, 0],    
                   [0, 0, 0, 0, 1, -1, 0],   
                   [0, 0, 0, 0, 0, 0, 0]])

    minWW = 0
    iaux = np.where(_YW[WWinds] < minWW)
    _YW[WWinds[iaux]] = minWW

    # %Time step = 1/N_step (days)
    N_step = 10

    # Initial error variance of beta
    S_beta = params['S_beta']

    # Variance of daily change of beta (initially)
    Q_beta = params['Q_beta0']


    # %Initial state
    # % X(0): S(t)
    # % X(1): E(t)
    # % X(2): I(t)
    # % X(3): A(t)
    # % X(4): D(t)
    # % X(5): W(t)
    # % X(6): beta(t)
    X = np.zeros((7,len(YC) + 1))
    X[:,0] = [(N - params['E_init'] - params['I_init']) * S_init, params['E_init'], params['I_init'], params['E_init'], params['E_init'], 0, beta]

    # Initial state error covariance
    P = np.array([[params['varE_init'] + params['varI_init'], -params['varE_init'], -params['varI_init'], -params['varE_init'], -params['varE_init'], 0, 0],
        [-params['varE_init'], params['varE_init'], 0, params['varE_init'], params['varE_init'], 0, 0],
        [-params['varI_init'], 0, params['varI_init'], 0, 0, 0, 0],
        [-params['varE_init'], params['varE_init'], 0, params['varE_init'], params['varE_init'], 0, 0],
        [-params['varE_init'], params['varE_init'], 0, params['varE_init'], params['varE_init'], 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, S_beta]])

    # %Measurement error variance (for cases), assuming a Binomial distribution for the number of cases
    YCR = YC.rolling(window=7, min_periods=1).mean()
    YCR = np.array(YCR)
    CR = pd.Series(C[0:len(YC)]).rolling(window=7, min_periods=1).mean()
    CR = np.array(CR)

    RC = YCR * C[0:len(YC)] / CR * (1-C[0:len(YC)]) + 1

    # Measurement error variance for WW data
    RW = params['RW']

    # Number of detected cases today depends linearly on the true number of new cases today
    Ccase = np.array([0, 0, 0, 0, 1, 0, 0])
    Cww = np.array([0, 0, 0, nu, 0, 0, 0])

    Ypred = np.zeros((2,len(YC)))
    Yest = np.zeros((2,len(YC)))
    YTest = np.zeros((1,len(YC)))
    errReff = np.zeros((1,len(YC)))
    Ysd = np.zeros((2,len(YC)))
    Xend = np.zeros((7,len(maxind)))

    jaux = 0
    for jday in range(np.max(maxind) + 1):
        # Reduce Q_beta after the first month. Higher Q_beta accounts for
        # errors in the initial estimate.
        if jday > 29.5:
            Q_beta = params['Q_beta1'] 

        # Initialise prediction variables
        Xhat = X[:,jday]
        Phat = P

        # Reset the "cases today" counter and corresponding covariance
        Xhat[4] = 0
        Phat[4,:] = 0
        Phat[:,4] = 0

        # Time loop for one day
        for jt in range(N_step):
            RR, Jf = SEIRreaction(Xhat,N,alpha,tau,gamma,omega,nu,eta,1/N_step)
            Xhat = Xhat + AR @ RR
            Q = CC * AR @ np.diag(RR) @ AR.T
            Q[6,6] = Q_beta / N_step
            Phat = (np.eye(7) + AR @ Jf) @ Phat @ (np.eye(7) + Jf.T @ AR.T) + Q

        Call = np.empty((0,7))
        R = np.array([])
        outInds = np.array([])
        Yday = np.array([])

        # Predicted number of daily new cases and wastewater measurement
        Ypred[:,jday] = [C[jday] * Ccase, Cww] @ Xhat + [0, minWW]

        # Check if there's case data for today
        if (useData[0] == True) & (YC[jday] > -0.5):
            Call = np.vstack([Call, C[jday] * Ccase])
            R = np.append(R, RC[jday])
            Yday = np.append(Yday, YC[jday])
            outInds = np.append(outInds, 0)        
        # Check if there's wastewater data for today
        WWii = -1
        if (useData[1] == True) & (_YW[jday] > -0.00005):
            Call = np.vstack([Call, Cww])
            R = np.append(R, RW)
            Yday = np.append(Yday, _YW[jday])
            outInds = np.append(outInds, 1)    
            WWii = len(Yday) - 1

        # Check if there was new data on this time step
        if len(Call) > 0:

        #   Measurement covariance
            S = Call @ Phat @ Call.T + np.diag(R)

        #   Outlier detection and plateauing
            if WWii > -1:
                discrepancy = (Yday[WWii] - Ypred[1,jday]) / (S[WWii,WWii] + params['RW0'] - params['RW']) ** 0.5
                if np.abs(discrepancy) > OL_limit:
                    Yday[WWii] = Ypred[1,jday] + OL_limit * np.sign(Yday[WWii] - Ypred[1,jday]) * (S[WWii,WWii] + params['RW0'] - params['RW'])**0.5


        # State update based on true and predicted number
            if len(outInds) == 1:
                if outInds[0] == 0:
                    _Ypred = Ypred[0,jday]
                if outInds[0] == 1:
                    _Ypred = Ypred[1,jday]
            else:
                _Ypred = Ypred[:,jday]
            
            try:
                X[:,jday+1] = Xhat + (Phat @ Call.T @ np.linalg.inv(S) @ (Yday - _Ypred)).T
            except:
                X[:,jday+1] = Xhat

        # Covariance update
            try:
                P = Phat - Phat @ Call.T @ np.linalg.inv(S) @ Call @ Phat
            except:
                P = Phat
        else:
        # In case of no new data, skip the update step
            P = Phat
            X[:,jday+1] = Xhat

        # Ensure the states to be non-negative (typically not a problem)
        X[1,jday+1] = np.max((X[1,jday+1],0))
        X[2,jday+1] = np.max((X[2,jday+1],0))

        # Estimated number of daily new cases and wastewater measurement
        Yest[:,jday] = np.array([C[jday] * Ccase, Cww]) @ X[:,jday+1] + np.array([0, minWW])
        
        # Estimated total case
        YTest[0,jday] = X[4,jday+1]

        # Standard deviations for the outputs
        Ysd[0,jday] = C[jday]**2 * Ccase @ P @ Ccase.T + RC[jday]
        Ysd[1,jday] = Cww @ P @ Cww.T + params['RW0']

        # Store the error variance of beta
        errReff[0,jday] = P[6,6]**0.5 * X[0,jday] / N / tau 

        # Store the state estimate of requested times
        if np.any(maxind == jday):
            Xend[:,jaux] = X[:,jday+1]
            jaux = jaux + 1

    # Calculate R_eff
    Reff = X[6,1:len(X[0])].T * X[0,1:len(X[0])] / N / tau

    return Yest, Xend, P, Reff, errReff, Ysd, YTest

