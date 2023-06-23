#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:47:59 2021

@author: ron

"""

from numpy.fft import fft2, fftfreq, ifft2
from numpy import meshgrid

def differentiate_2d(u, x, y):
    '''a function to estimate derivatives of the 2D array u(x,y) using fft
    
    u - the 2d array to differentiate
    x - the 1d array of values for the variable x
    y - the 1d array of values for the variable y
    '''
    pi = 3.141592653589793
    U = fft2(u)
    X, Y = meshgrid(fftfreq(len(x), d=x[1]-x[0]), fftfreq(len(y), d=y[1]-y[0]))
    dudx = ifft2( pi * 2j * X * U )
    dudy = ifft2( pi * 2j * Y * U )
    return dudx, dudy
    





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import numpy as np
    
    # example for differentiating a Gaussian:
    # generate samples and grid:    
    x_ = np.linspace(-1,1,100)
    y_ = np.linspace(-1,1,100)
    x, y = np.meshgrid(x_, y_)
    u = np.exp( - ((x**2 + y**2)/0.2) )
    
    
    # plot original:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, u)
    
    
    # estimate derivatives:
    dudx, dudy = differentiate_2d(u, x_, y_)
    
    
    # plot derivatie dudx:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, dudx)

    
