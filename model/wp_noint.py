#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:01:02 2018

@author0: arek
@author1: FS

"""

import numpy as np
from numpy.fft import fft, ifft, fftshift
import matplotlib.pylab as plt

class model:    
    """ Non interacting wave packet subject to gravity and driven by a shaken 
    mirror. Evolution is dealt with by split-Fourier technique in a discretized 
    position and momentum grids.
    
    Parameters
    ---------
        grid_center: center of the position grid
        grid_width:  width of the position grid
        grid_resolution: resolution of the position grid
        Lambda: Amplitude of the shaking
        omega: Shaking frequency of the mirror
        h / g / m: Physical scales of the system
        
    Split Fourier Method
    --------------------
    

    """
    def __init__(self, grid_center, grid_width, grid_resol, Lambda, omega, 
                 m = 1, g = 1, h = 1):        
        """Initialization: store parameters and generate the relevant grids"""
        self.grid_center = grid_center
        self.grid_width = grid_width
        self.grid_resol = grid_resol
        self.m = m 
        self.g = g
        self.h = h
        self.Lambda = Lambda
        self.omega = omega
        self.gen_grids()

    def gauss_wp(self, delta, x0=0):
        """Create a gaussian wavepacket with the amplitudes evaluated over 
        self.grid_x
        
        Arguments
        ---------
            delta: float 
                width of the wavepacket
            x0: float
                center of the wave packet
        Output
        ------
            wp: complex np.array
                wp[i] is the amplitude of the wave packet at grid_x[i]
        """
        constant = (delta/np.pi)**(1/4) * (1.+0.j) 
        wp = constant * np.exp((-0.5 * delta * np.square(self.grid_x - x0)))
        return wp

    def gen_grids(self):
        """ Generate several grids for the evolution of the wave packet. All 
        these does not change over the evolution.
            grid_x: position grid (not shifted)
            grid_k: momentum grid (has been rearranged to match the output of fft)
            grid_kin: kinetic energy grid
        """
        self.dx = self.grid_width / self.grid_resol
        self.dk  = 2 * np.pi/self.grid_width
        self.grid_x_shifted = -self.grid_width/2 + self.dx * np.arange(0, self.grid_resol)
        self.grid_x = self.grid_x_shifted + self.grid_center
        self.grid_k = - (np.pi * self.grid_resol)/self.grid_width + self.dk * np.arange(0, self.grid_resol)
        self.grid_k = np.roll(self.grid_k, int((self.grid_resol)/2))
        self.grid_kin =  np.square(self.h)/ (2*self.m) * np.square(self.grid_k)
        
    def get_potential(self,t):
        """ Potential (gravity + mirror) evaluated at each point of grid_x """
        grid_V = self.grid_x * (1 + self.Lambda*np.cos(self.omega*t)) * (self.grid_x >= 0)
        grid_V += 200 * (self.grid_x < 0)
        return grid_V

    def evol_dt(self, psi, t, dt):
        """ Evolve a state |psi(t)> to |psi(t+ dt)> using the split Fourier 
        method """
        V_action = np.exp(- 1.0j * dt * self.get_potential(t))
        kin_action = np.exp(- 1.0j * dt * self.grid_kin) 
        return V_action*self.k_to_x(kin_action * self.x_to_k(psi))

    def evol_to_t(self, t, dt, psi_0, save_freq = 0):
        """ evolve an initial state to a final time by discrete time steps. 
        Save intermediary states if requested
        
        Arguments
        ---------
        t: float
            final time
        dt: float
            time increment
        psi0: complex np.array 
            initial state (i.e at t=0)
        save_freq: int
            frequency for the saving of the data. If 0(1) save nothing(everything)
            
        Output
        ------
        psi: 1d complex np.array
            return the final state (@ T_final)
        (if save > 0):
            self.psi_saved: (nb_save, len_state) array with all the saved states            
            self.t_saved: Times at which the states have been saved
        """
        t_simul = np.concatenate((np.arange(0, t, dt), [t]))
        if (save_freq > 0):
            nb_save = int(np.ceil(len(t_simul) / save_freq)) + 1
            psi_saved = np.zeros(nb_save, len(psi_0), dtype = 'complex128')
            t_saved = np.zeros(nb_save)
        
        psi_tmp = psi_0 * (1. + 0.j)
        for n_t, t in enumerate(t_simul):
            if(n_t > 0):
                dt_tmp = t_simul[n_t] - t_simul[n_t-1]
                psi_tmp = self.evol_dt(psi_tmp, t, dt_tmp)
            if((save_freq>0) and (n_t % save_freq) == 0):
                index_save = n_t // save_freq
                t_saved[index_save] = t
                psi_saved[index_save] = psi_tmp
            
        return psi_tmp
    
    
    #deals with fourrier transfo
    def x_to_k(self, psi_x):
        """ Psi(x) -> Psi(k)"""
        return fft(psi_x, norm = 'ortho')
        
    def k_to_x(self, psi_k):
        """ Psi(k) -> Psi(x)"""
        return ifft(psi_k, norm='ortho')
    
    
    #Extras functions to help computing some figures of merit
    def ip(self, psi_1, psi_2):
        """ Inner product in position"""
        return np.dot(np.conjugate(psi_1), psi_2) * self.dx
    
    def density(self, psi):
        """ Compute the density from amplitudes of a state """
        return np.square(np.abs(psi))
    
    def overlap(self, psi_1, psi_2):
        """ Compute the classical overlap between two states (in position basis)"""
        return self.dx * np.sum(np.min(np.c_[self.density(psi_1), self.density(psi_2)], 1))

    def fidelity(self, psi_1, psi_2):
        """ Compute quantum overlap between two states (in position basis)"""
        return np.abs(self.ip(psi_1, psi_2))**2

    def norm(self, psi):
        """ Compute the norm of a state (in position basis)"""
        return np.sqrt(np.abs(self.ip(psi)))
