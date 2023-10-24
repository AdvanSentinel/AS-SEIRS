#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.optimize import minimize, BFGS, LinearConstraint
from SEIRWWfiles_R2S.paramFit import paramFit

def SEIRWWcalibrate(YC, YW, C, params, S_init):
    """
    # Rate E -> I
    #params['alpha'] = 0.4433
    params['alpha'] = 1/1.5 #20230412 Changed to 1.5
    # Initial rate S -> I
    params['beta'] = 0.44
    # Rate I to R (tau1 in SEIR-ICU model)
    #params['tau'] = 0.32 
    params['tau'] = 1/2.0 #20230412 Changed to 2.0
    # Rate R to S
    params['omega'] = 1/180
    # State noise coefficient (model error)
    params['modelErrorC'] = 4**2
    # Initial error variance of beta
    params['S_beta'] = 0.15**2
    # Variance of daily change of beta (initially)
    params['Q_beta0'] = 0.05**2
    # After 1st month
    params['Q_beta1'] = 0.005**2
    """
    # Estimate the initial sizes of E and I compartments
    params['E_init'] = params['darkNumber'][0,0] / params['alpha'] * (1 + np.mean(YC[0:5]))
    params['I_init'] = params['darkNumber'][0,0] / params['tau'] * (1 + np.mean(YC[0:5]))

    params['varE_init'] = (params['E_init'] / 2)**2
    params['varI_init'] = (params['I_init'] / 2)**2

    # Find initial point by a simpler optimisation
    params['gamma'] = 2
    params['WWexp'] = 0.7
    cost = lambda x:paramFit(params,YC,YW,C,x[0],x[1],-1,S_init)[0]

    Acon = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    Bcon = np.array([4, -0.2, 1, -0.4])

    def cons(x):
        return Bcon - Acon @ x

    cons = (
        {'type': 'ineq', 'fun': cons}
    )

    xopt = minimize(cost, x0 = np.array([params['gamma'], params['WWexp']]), constraints=cons)['x']
    # xopt = minimize(cost, [params['gamma'], params['WWexp']], bounds=((0.2,4),(0.4,1)))['x']
    nuInit = paramFit(params,YC,YW,C,xopt[0],xopt[1],-1,S_init)[1]
    params['gamma'] = xopt[0]
    params['WWexp'] = xopt[1]

    print('Initial point found')
    
    # Estimate gamma, nu, and the exponent in WW-transformation
    cost = lambda x:paramFit(params,YC,YW,C,x[0],x[1],x[2],S_init)[0]
    xopt = minimize(cost, [params['gamma'], params['WWexp'], nuInit], bounds=((0.2,4),(0.4,1),(0,np.inf)))['x']
    J =  paramFit(params,YC,YW,C,xopt[0],xopt[1],xopt[2],S_init)[0]
    RW0 = paramFit(params,YC,YW,C,xopt[0],xopt[1],xopt[2],S_init)[2]
    params['gamma'] = xopt[0]
    params['WWexp'] = xopt[1]
    params['nu'] = xopt[2]
    params['RW0'] = RW0

    print('Parameter estimation complete')
    
    return params

