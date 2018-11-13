#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:19:51 2018

@author: fred
"""
import sys
sys.path.append('../')
from model.wp_noint import model

import numpy as np
import matplotlib.pylab as plt
from matplotlib import rc
rc('text', usetex=True)

#--------------------------------------------------#
# Test 1
#--------------------------------------------------#
# parameters of the simulation
s = 1
x_init = 20
L = 60
N = 2**9+20
dt = 0.008
omega= (s*np.pi)/ (np.sqrt(2*x_init))
Lambda = 0.04
T = np.sqrt(8 * x_init)
t_final = 42 * T

# create the model
mod = model(grid_center = x_init, grid_width = 60, grid_resol = N, 
            Lambda = 0.04, omega = omega)
    
#create state init
psi0 = mod.gauss_wp(x0 = x_init, delta=0.702)
    
#evolve it
psiT = mod.evol_to_t(t_final+dt, dt, psi0)

# Plot
plt.plot(mod.grid_x, 200 * mod.density(psiT))
plt.plot(mod.grid_x, 200 * mod.density(psi0))
plt.plot(mod.grid_x, mod.get_potential(t_final))
    

#compute fidelity and overlap with initial state
print(mod.fidelity(psi0, psiT))
print(mod.overlap(psi0, psiT))