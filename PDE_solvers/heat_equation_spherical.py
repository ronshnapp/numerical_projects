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




def heat_equation(u, t, x, k, L, Q):
    '''Returns the RHS of the heat equation
    
    t is time
    k is the diffusivity
    L is the length of the domain
    Q is a source term.
    '''
    
    ux = psdiff(u, period=L, order=1)
    tmp = psdiff(ux * 1.0/x**2, period=L, order=1)
    lhs = tmp * x**2 * k
    # add a source term:
    
    dudt = lhs + Q 
    
    return dudt



def heat_equation_solver(u0, t, x, k, L, Q):
    """Use odeint to solve the heat equation on a periodic domain.
    
    `u0` is initial condition, 
    `t` is the array of time values at which the solution is to be computed, 
    `L` is the length of the periodic domain.
    """
    sol = odeint(heat_equation, u0, t, args=(x, k, L, Q), mxstep=5000)
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
    k = 1e-6
    
    # spatial grid:
    L = 10.0
    N = 2**10
    dx = L / (N-1.0)
    x = np.linspace(1.0, L-dx, N)
    
    
    # initial condition
    u0 = np.exp( -(x**2/1.0)**2 )
    
    
    # a source term:
    Q = np.zeros(x.shape)
    
    
    # temporal grid
    T = 1000000.0
    t = np.linspace(0, T, 100)
    dt = t[1]-t[0]
    
    # compute the solution:
    sol = heat_equation_solver(u0, t, x, k, L, Q)
    
    #fig, ax = plt.subplots()
    #c = ax.imshow(sol, extent=[0,L,T,0])
    #fig.colorbar(c)
    #ax.set_xlabel('x')
    #ax.set_ylabel('t')
    
    
    fig, ax = plt.subplots()
    ax.plot(x, Q, 'k-', lw=1)
    for i in [0,10,20,40,60,99]:
        ax.plot(x, sol[i,:])
    





