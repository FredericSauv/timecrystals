#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:51:59 2018

@author: FS

Simple example of optimization with the non interacting wavepacket model
Optimization over the width of the initial wavepacket (can be extended to a more
complex optimization)

"""
import sys
# sys.path.append('/home/fred/Desktop/GPyOpt/')
import GPyOpt
sys.path.append('../')
from model.wp_noint import model
import numpy as np
import scipy.optimize as sco

#-----------------------------------------------------------------------------#
# Prepare Optimization
#-----------------------------------------------------------------------------#
# Parameters of the optimizer with some comments
width_range = (0.5, 1.5)
bounds = [{'name': 'width', 'type': 'continuous', 'domain': width_range}]
options_bo ={'initial_design_numdata': 5, 'acquisition_type':'EI', 
             'optim_num_samples':1000, 'optim_num_anchor':10,
             'acquisition_weight':2, 'acquisition_weight_lindec':False,
             'bounds': bounds}
max_iter = 25

# Create the model
s, x_init, L, N, dt, Lambda = 1, 20, 60, 2**9+20, 0.008, 0.04
omega= (s*np.pi)/ (np.sqrt(2*x_init))
T = np.sqrt(8 * x_init)
t_final = 42 * T
mod = model(grid_center = x_init, grid_width = L, grid_resol = N, 
            Lambda = Lambda, omega = omega)
    
#Define the figure of Merit (1-fidelity) to turn it as a minimization problem
def f(x):
    print(x)
    psi0 = mod.gauss_wp(x0 = x_init, delta=x)
    psiT = mod.evol_to_t(t_final+dt, dt, psi0)
    res = mod.fidelity(np.squeeze(psi0), np.squeeze(psiT))
    print(res)
    return 1 - res    


def _get_best_exp_from_BO(bo):
    """ From BO extract the best x"""
    Y_pred = bo.model.predict(bo.X)
    return bo.X[np.argmin(Y_pred[0])], np.min(Y_pred)

#-----------------------------------------------------------------------------#
# Optimization of the noiseless set-up
#-----------------------------------------------------------------------------#
myOpt = GPyOpt.methods.BayesianOptimization(f, bounds, **options_bo)
myOpt.run_optimization(max_iter)
x_no_noise, _ = _get_best_exp_from_BO(myOpt)


