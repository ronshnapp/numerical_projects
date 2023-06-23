#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:58:16 2023

@author: ron


This is a code that is used to generate Kinematic simulations (KS) of 
homogeneous isotropic turbulence. We do this by summing random Furier modes,
choosing them in a way that conforms to the continuity equation and that
leads to a Kolmogorov -5/3 spectrum of kinetic energy. 

The method of KS used here is based on the work of:
    
    Osborne, Vassilicos, Sung and Haigh, Fundamentals of pair diffusion in 
    kinematic simulations of turbulence, PRE 2006, 
    DOI: 10.1103/PhysRevE.74.036309

"""

import numpy as np


class kinematic_simulation(object):
    
    def __init__(self, Nk, K1=1.0, KNk=0.1, epsilon=1.0, lmbda=1.0):
        '''
        input - 
        
        Kn (integer) - the number of Furier modes used in the simulation.
        K1 (float, defaul = 1.0) - 
        KNk (float, defaul = 0.1) - 
        epsilon (float, defaul = 1.0) - the dissipation rate of TKE.
        lmbda (float, defaul = 1.0) - the persistence parameter.
        '''
        self.Nk = Nk
        self.epsilon = epsilon
        self.lmbda = lmbda
        
        # the amplitudes of the wave vectors
        self.Kn = [K1 * (KNk / K1)**((n-1)/(Nk-1)) for n in range(1,Nk+1)]
        
        # the directions of the wave vectors
        self.k_hat_n = [np.random.uniform(-1,1,3) for i in range(Nk)]
        for i in range(Nk):
            self.k_hat_n[i] = self.k_hat_n[i] / np.sum(self.k_hat_n[i]**2)**0.5
            
        # a list of the wave vectors
        self.kn = [self.k_hat_n[i] * self.Kn[i] for i in range(Nk)]
        

        # choosing the random Furier coefficients' amplitudes;
        # the spectrum is aaumed to be of the form 
        # E(kn) = epsilon^(2/3)  *  kn^(-5/3)
        Ek = lambda k: self.epsilon**(2./3.) * k**(-5./3.)
        self.dk = []
        for i in range(Nk):
            if i==0: self.dk.append(self.Kn[0]-self.Kn[1])
            elif i==Nk-1: self.dk.append(self.Kn[-2]-self.Kn[-1])
            else: self.dk.append(self.Kn[i-1]-self.Kn[i+1])
        
        self.An = [(Ek(self.Kn[i]) * self.dk[i])**0.5 for i in range(Nk)]
        self.Bn = [(Ek(self.Kn[i]) * self.dk[i])**0.5 for i in range(Nk)]
        
        
        
        # choosing random Furier coefficient unit vectors 
        # that are orthogonal to Kn
        self.an = []
        self.bn = []
        for i in range(Nk):
            k = self.k_hat_n[i]
            a = np.random.uniform(-1,1,3)
            a[2] = (-a[0]*k[0] - a[1]*k[1]) / k[2]
            self.an.append( a / np.sum(a**2)**0.5 * self.An[i])

            b = np.random.uniform(-1,1,3)
            b[2] = (-b[0]*k[0] - b[1]*k[1]) / k[2]
            self.bn.append( b / np.sum(b**2)**0.5 * self.Bn[i])
        
        
        
        # setting the Furier components frequencies
        self.Omega_n = [lmbda * (self.Kn[i]**3 * Ek(self.Kn[i]))**(0.5) 
                        for i in range(Nk)]
        
    
    def __call__(self, x, t):
        '''
        x - An (NX3) array representing a list of N 3D points on which we
            evaluate the velocity field.
        
        t - A float time value

        '''
        modes = []
        
        for i in range(self.Nk):
            pn = np.dot(x, self.kn[i]) + t*self.Omega_n[i]
            modes.append(np.dot(np.reshape(self.an[i],(3,1)), [np.cos(pn)]) + 
                         np.dot(np.reshape(self.bn[i],(3,1)), [np.sin(pn)]))
        
        return np.sum(modes, axis=0).T




#%%
import matplotlib.pyplot as plt

KS = kinematic_simulation(30, K1=10, KNk=0.5)



#%%
X, Y = np.meshgrid(np.linspace(0,2, num=35), np.linspace(0,2, num=35))
U, V = np.zeros(X.shape), np.zeros(Y.shape)

for j in range(X.shape[1]):
    for i in range(X.shape[1]):
        x = [X[i,j],Y[i,j], 0.0]
        u = KS(x,0)
        U[i,j] = u[0]
        V[i,j] = u[1]
        

plt.quiver(X, Y, U,V)



#%%

from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


for x_ in np.linspace(-0.05, 0.05, num=4):
    for y_ in np.linspace(-0.05, 0.05, num=4):
        for z_ in np.linspace(-0.05, 0.05, num=4):
            p = odeint(KS, [x_, y_, z_], np.linspace(0,6, num=100))
            ax.plot(p[:,0], p[:,1], p[:,2], 'k-', alpha=0.4)







