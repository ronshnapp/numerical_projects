#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 22:51:46 2021

@author: ron


Here we solve the heat equation:
du/dt = K d/dx(du/dx)
with periodic boundary conditions.

"""

import numpy as np
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff




def heat_equation(u, t, k, L):
    '''Returns the RHS of the heat equation'''
    uxx = psdiff(u, period=L, order=2)
    dudt = k*uxx
    return dudt



def heat_equation_solver(u0, t, k, L):
    """Use odeint to solve the heat equation on a periodic domain.
    
    `u0` is initial condition, 
    `t` is the array of time values at which the solution is to be computed, 
    `L` is the length of the periodic domain.
    """
    sol = odeint(heat_equation, u0, t, args=(k, L), mxstep=5000)
    return sol
    


def animate(sol, dx, dt, play_speed=10.0):
    from moviepy.video.io.bindings import mplfig_to_npimage
    import moviepy.editor as mpy
    
    fig, ax = plt.subplots()
    x_ = np.arange(sol.shape[1])*dx
    mn, mx = np.amin(sol), np.amax(sol)
    xl, xh = mn - 0.05*(mx-mn), mx + 0.05*(mx-mn)
    def mk_frame(t):
        i = int(t / Nframes * play_speed * (Nframes-1))
        ax.plot(x_, sol[i,:])
        ax.set_ylim(xl, xh)
        ax.text(0.01,0.01, 't=%.2f'%(i*dt), transform=ax.transAxes)
        img = mplfig_to_npimage(fig)
        ax.clear()
        return img
    
    Nframes = sol.shape[0]
    animation = mpy.VideoClip(mk_frame, duration = Nframes/play_speed)
    
    animation.write_videofile('test.avi', fps = play_speed, codec='png')
    plt.close()
    return animation






if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # conductivity:
    k = 0.1
    
    # spatial grid:
    L = 1.0
    N = 128
    dx = L / (N-1.0)
    x = np.linspace(0, L-dx, N)
    
    # initial condition
    u0 = np.exp( -((x-L/2)/0.025)**2 )

    
    # temporal grid
    T = 1.0
    t = np.linspace(0, T, 500)
    dt = t[1]-t[0]
    
    # compute the solution:
    sol = heat_equation_solver(u0, t, k, L)
    
    fig, ax = plt.subplots()
    c = ax.imshow(sol, extent=[0,L,T,0])
    fig.colorbar(c)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    






