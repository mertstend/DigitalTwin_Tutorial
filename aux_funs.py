# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:23:38 2020

@author: Merten
"""


import numpy as np
from scipy.integrate import odeint
import pickle


# define ODE that represents the "real" system
def ode_dyn_sys(x, t):

    delta = 0.25
    omega = 3
    k3 = 5

    dxdt = [x[1],
            -2 * delta * x[1] - omega**2 * x[0] - k3 * x[0] ]

    return dxdt

def generate_measurement_data():

    # number of samples
    N = 100

    # time vector
    n_timesteps = 200
    T = np.linspace(0, 20, n_timesteps)

    # generate set of initial conditions by uniform random sampling from a
    # [min, max] interval for both states.
    # x(t=0) \in [-3, -1]
    # dx(t=0)/dt \in [-5, -1]
    x0_mins = [-3, -5]
    x0_maxs = [-1, -1]
    X0 = np.random.uniform(low=x0_mins, high=x0_maxs, size=(N,2))

    # define noise level
    noise_ampl = 0.2

    # pre-allocate list where to store the data
    XM = []

    # now generate data by time-integration from the initial conditions
    for idx, x0 in enumerate(X0):

        # some user feedback
        print('time integration ' + str(idx) + '/' + str(N))

        # time integration starting from current x0
        X = odeint(ode_dyn_sys, x0, T)

        # add some uniform random additive noise
        X = X + noise_ampl * np.random.uniform(low=[-1, -1], high=[1, 1], size=[n_timesteps, 2])

        # store in the XM list
        XM.append(X)


    # convert to 3D numpy array for easier handling
    XM = np.array(XM)

    # store as pickle file
    pickle.dump([XM, T], open("historic_data.p", "wb"))




def get_measurement_data(x0, T):

    x = odeint(ode_dyn_sys, x0, T)

    return x