# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:17:14 2021

@author: Ron


will solve the system 

dudt = -u*u_x + B*B_x
dBdt = B*u_x

with periodic boundary conditions
"""

import numpy as np
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff






def equation(F, t, L):
    """Differential equations for the KdV equation, discretized in x."""
    # Compute the x derivatives using the pseudo-spectral method.
    u, B = F[:int(len(F)/2)], F[int(len(F)/2):]
    ux = psdiff(u, period=L)
    uxx = psdiff(u, period=L, order=2)
    Bx = psdiff(B, period=L)

    # Compute du/dt.    
    dudt = -u*ux + B*Bx + 0.0001*uxx
    dBdt = B*ux - B*(1/12.5)
    dFdt = np.append(dudt, dBdt)

    return dFdt


def eqn_solution(F0, t, L):
    """Use odeint to solve the equation on a periodic domain."""

    sol = odeint(equation, F0, t, args=(L,), mxstep=5000)
    sol_u = sol[:,:int(len(F0)/2)]
    sol_B = sol[:,int(len(F0)/2):]
    return sol_u, sol_B



def animate(sol, dx, dt, play_speed=10.0):
    from moviepy.video.io.bindings import mplfig_to_npimage
    import moviepy.editor as mpy
    
    fig, ax = plt.subplots()
    x_ = np.arange(sol.shape[1])*dx
    mn, mx = np.amin(sol), np.amax(sol)
    
    def mk_frame(t):
        i = int(t / Nframes * play_speed * (Nframes-1))
        ax.plot(x_, sol[i,:])
        ax.set_ylim(mn, mx)
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
    N = 2**12
    L = 5.0
    dx = L/(N-1.0)
    x = np.linspace(0, L-dx, N)
    
    u0 = 1.0 * np.exp(-((x-L/2)/0.1)**2 )
    B0 = 1.0 * np.ones(x.shape)
    F0 = np.append(u0, B0)
    
    T = 2.5
    t = np.linspace(0, T, 500)
    dt = t[1]-t[0]

    u, B = eqn_solution(F0, t, L)
    
    
    
    import matplotlib.pyplot as plt

    # plt.figure(figsize=(6,5))
    # plt.imshow(u[::-1, :], extent=[0,L,0,T])
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('t')
    # #plt.axis('normal')
    # plt.show()
    
    
    # plt.figure(figsize=(6,5))
    # plt.imshow(B[::-1, :], extent=[0,L,0,T])
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('t')
    # plt.show()
    
    plt.figure(figsize=(6,5))
    for i in [0,15,30,60,125,250,499]:
        plt.plot(x, u[i,:])
    









