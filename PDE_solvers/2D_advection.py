# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:09:43 2021

@author: Ron
"""

import numpy as np
import matplotlib.pyplot as plt

from fft_derivative_2D import differentiate_2d




def advection_equation(c, t, u, v, X, Y):
    '''Returns the RHS of the heat equation
    
    c is the scalar concentration
    t is time
    u is the velocity field x component (a function of time)
    v is the velocity y component (a function of time)
    '''
    dx = X[0,1] - X[0,0]
    dy = Y[1,0] - Y[0,0]
    #dcx, dcy = np.gradient(c)
    dcx = forward_difference(c, 0) * (u<=0) + backward_difference(c, 0) * (u>0)
    dcy = forward_difference(c, 1) * (v<=0) + backward_difference(c, 1) * (v>0)
    dcdx, dcdy = dcx/dx, dcy/dy
    dcdt = -u * dcdx - v * dcdy #+ 1*(dcdxdx + dcdydy)
    return dcdt




def forward_difference(array2D, axis):
    '''
    given an 2D array, this returns the forward difference along the given
    axis.
    FD = x_i+1 - x_i
    '''
    if axis==0:
        FD = np.append((array2D[1:,:] - array2D[:-1,:]), 
                       np.zeros((1,array2D.shape[0])), axis=0)
    elif axis==1:
        FD = np.append((array2D[:,1:] - array2D[:,:-1]), 
                       np.zeros((array2D.shape[1],1)), axis=1)
    
    else: raise ValueError('axis can be 1 or 0')
    
    return FD





def backward_difference(array2D, axis):
    '''
    given an 2D array, this returns the forward difference along the given
    axis.
    BD = x_i - x_i-1
    '''
    if axis==0:
        BD = np.append(np.zeros((1,array2D.shape[0])),
                       array2D[1:,:] - array2D[:-1,:], axis=0)
    elif axis==1:
        BD = np.append(np.zeros((array2D.shape[1],1)),
                       array2D[:,1:] - array2D[:,:-1], axis=1)
    
    else: raise ValueError('axis can be 1 or 0')
    
    return BD







def advection_solver(c0, u, v, X, Y, dt, steps):
    """Use odeint to solve the advection equation on a periodic domain.
    """
    sol = [c0]
    t = [0]
    for i in range(steps):
        
        dcdt = advection_equation(sol[-1], t[-1], u(t[-1]), v(t[-1]), X, Y)
        sol.append( sol[-1] + dt *  dcdt)
        t.append(t[-1] + dt)
        
    return t, sol




if __name__ == "__main__":
    x, y = np.linspace(0,1,num=400), np.linspace(0,1,num=400)
    X, Y = np.meshgrid(x, y)
    
    u = lambda t: np.cos(np.pi*X) * np.sin(np.pi*Y)
    v = lambda t: -np.sin(np.pi*X) * np.cos(np.pi*Y)
    c0 = (np.arctan( (X -0.5) *250)*2/np.pi + 1)/2
    
    dt = 0.0025
    steps = 5000
    
    t, sol = advection_solver(c0, u, v, X, Y, dt, steps)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    