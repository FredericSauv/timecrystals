#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:53:41 2018

@author: fred
"""
import sys
sys.path.append('../')
from model.bouncing_atom import model

import numpy as np
import matplotlib.pylab as plt
from matplotlib import rc
rc('text', usetex=True)

#--------------------------------------------------#
# Test 1
# Replicates Arek's Mathematica: scan initial height
#--------------------------------------------------#
mod = model()
mod.scan_h(delta = 1, href = 1/2 * (3*np.pi)**(2/3), omega = np.pi**(2/3)/(3**(1/3)), s=1.0)



#--------------------------------------------------#
# Test 2
# Look at the impact of changing omega when there is 
# noise in the initial height (h_ref). 
# Results are obtained for s = 40
# Takes a little bit of time
#--------------------------------------------------#
h_ref = 328.85
s = 40
print(mod.get_omega_perfect(328.85, 40))
periods = [1, 6]
omega_range = np.linspace(0.8, 1.2, 200)

# to store the results
res_noise_10trials = np.zeros((len(omega_range), len(periods)))
res_noise_100trials = np.zeros((len(omega_range), len(periods)))
res_no_noise = np.zeros((len(omega_range), len(periods)))

omega_ref = s * np.pi/np.sqrt(2 * h_ref)
for n_om, om in enumerate(omega_ref * omega_range):
    res_noise_10trials[n_om, :] = mod.simulate_noisy_h0(
            om, h_ref, noise_h0 = 0.2, nb_repeat = 10, s = s, nb_period = periods)
    res_noise_100trials[n_om, :] = mod.simulate_noisy_h0(
            om, h_ref, noise_h0 = 0.2, nb_repeat = 100, s = s, nb_period = periods)
    res_no_noise[n_om, :] = mod.simulate_noisy_h0(
            om, h_ref, noise_h0 = 0, nb_repeat = 1, s = s, nb_period = periods)

plt.plot(omega_range,res_noise_10trials[:,0]/h_ref, label = 'noise:0.2, 10trials')
plt.plot(omega_range,res_noise_100trials[:,0]/h_ref, label = 'noise:0.2, 100trials')
plt.plot(omega_range,res_no_noise[:,0]/h_ref, label = 'no noise')
plt.legend()
plt.legend(fontsize = 12)
plt.xlabel(r"$\omega / \omega_{ref}$", size = 18)
plt.ylabel(r"$|h(T)-h_{ref}|/ h_{ref}$", size = 18)
#plt.savefig('noise_0_2_h0_329_lambda_0_01.pdf')


plt.plot(omega_range,res_noise_10trials[:,1]/h_ref, label = 'noise:0.2, 10trials')
plt.plot(omega_range,res_noise_100trials[:,1]/h_ref, label = 'noise:0.2, 100trials')
plt.plot(omega_range,res_no_noise[:,1]/h_ref, label = 'no noise')
plt.legend(fontsize = 12)
plt.xlabel(r"$\omega / \omega_{ref}$", size = 18)
plt.ylabel(r"$|h(6T)-h_{ref}|/ h_{ref}$", size = 18)
#plt.savefig('noise_0_2_h0_329_lambda_0_01_6T.pdf',bbox_inches='tight')