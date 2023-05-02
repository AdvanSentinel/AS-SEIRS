#!/usr/bin/env python
# coding: utf-8

import numpy as np

def SEIRreaction(X, N, alpha, tau1, gamma, omega, nu, eta, dt):

    # X0: S(t)
    # X1: E(t)
    # X2: I(t)
    # X3: A(t)
    # X4: D(t)
    # X5: W(t)
    # X6: beta(t)

    # Reactions
    R = np.zeros(7)
    R[0] = dt * X[6] * X[0] * X[2] / N     # S to E
    R[1] = dt * alpha * X[1]           # E to I
    R[2] = dt * tau1 * X[2]          # I to R
    R[3] = dt * gamma * X[3]           # A to void
    R[4] = dt * nu * X[3]              # W production
    R[5] = dt * eta * X[5]               # W degradation
    R[6] = dt * omega * (N - X[0] - X[1] - X[2]) # R to S

    R[R < 0] = 0

    # Jacobian of R
    JR = np.zeros((7,7))
    JR[0,:] = [dt * X[6] * X[2] / N, 0, dt * X[6] * X[0] / N, 0, 0, 0, dt * X[0] * X[2] / N]
    JR[1,:] = [0, dt * alpha, 0, 0, 0, 0, 0]
    JR[2,:] = [0, 0, dt * tau1, 0, 0, 0, 0]
    JR[3,:] = [0, 0, 0, dt * gamma, 0, 0, 0]
    JR[4,:] = [0, 0, 0, dt * nu, 0, 0, 0]
    JR[5,:] = [0, 0, 0, 0, 0, dt * eta, 0]
    JR[6,:] = [-dt * omega, -dt * omega, -dt * omega, 0, 0, 0, 0]
    
    return R, JR

