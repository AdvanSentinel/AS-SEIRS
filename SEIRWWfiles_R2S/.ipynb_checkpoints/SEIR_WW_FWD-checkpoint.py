#!/usr/bin/env python
# coding: utf-8

from SEIRWWfiles_R2S.SEIRreaction import SEIRreaction
import numpy as np

def SEIR_WW_FWD(X,C,P,initDay,params,maxind):

    # X: initial state for projection
    # C: daily observation coefficients
    # initDay: number of the first day of simulation
    # params: the coefficients in the SEIR ODE
    # maxInd: length of simulation time

    # Set parameters
    alpha = params['alpha']
    tau = params['tau']
    gamma = params['gamma']
    omega = params['omega']
    nu = params['nu']
    eta = 1
    N = params['N']
    CC = params['modelErrorC']

    # Reaction stoichiometry (w.r.t. SEIRreaction-function)
    AR = np.array([[-1, 0, 0, 0, 0, 0, 1],
                   [1, -1, 0, 0, 0, 0, 0],   
                   [0, 1, -1, 0, 0, 0, 0],   
                   [1, 0, 0, -1, 0, 0, 0],   
                   [0, 1, 0, 0, 0, 0, 0],    
                   [0, 0, 0, 0, 1, -1, 0],   
                   [0, 0, 0, 0, 0, 0, 0]])

    # Time step = 1/N_step (days)
    N_step = 10

    # Number of detected cases today depends linearly on the true number of new cases today
    Ccase = np.array([0, 0, 0, 0, 1, 0, 0])
    Cww = np.array([0, 0, 0, 0, 0, 1, 0])

    Yest = np.zeros((2,maxind))
    err = np.zeros((1,maxind))

    for jday in range(maxind):
        X[4] = 0
        P[4,:] = 0
        P[:,4] = 0

    # Time loop for one day
        for jt in range(N_step):
            RR, Jf = SEIRreaction(X,N,alpha,tau,gamma,omega,nu,eta,1/N_step)
            X = X + AR @ RR
            Q = CC * AR @ np.diag(RR) @ AR.T
            P = (np.eye(7) + AR @ Jf) @ P @ (np.eye(7) + Jf.T @ AR.T) + Q

    # Predicted number of daily new cases and wastewater measurement
        Yest[:,jday] = [C[initDay + jday] * Ccase, Cww] @ X
        err[0,jday] = P[4,4]**.5
    return Yest, err

