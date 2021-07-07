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
    dudt = -u*ux + B*Bx + 0.05*uxx
    dBdt = B*ux - B*(1/100.0)
    dFdt = np.append(dudt, dBdt)

    return dFdt


def eqn_solution(F0, t, L):
    """Use odeint to solve the equation on a periodic domain."""

    sol = odeint(equation, F0, t, args=(L,), mxstep=5000)
    sol_u = sol[:,:int(len(F0)/2)]
    sol_B = sol[:,int(len(F0)/2):]
    return sol_u, sol_B



if __name__ == '__main__':
    N = 1024
    L = 50.0
    dx = L/(N-1.0)
    x = np.linspace(0, L-dx, N)
    
    u0 = np.exp(-((x-L/2)/2)**2 )
    B0 = 1.0*np.exp(-((x-L/2)/2)**2 )
    F0 = np.append(u0, B0)
    
    T = 100.
    t = np.linspace(0, T, 500)
    dt = t[1]-t[0]

    u, B = eqn_solution(F0, t, L)
    
    
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,5))
    plt.imshow(u[::-1, :], extent=[0,L,0,T])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    #plt.axis('normal')
    plt.show()
    
    
    plt.figure(figsize=(6,5))
    plt.imshow(B[::-1, :], extent=[0,L,0,T])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.show()
    
    plt.figure(figsize=(6,5))
    for i in [0,100,200,300,400,499]:
        plt.plot(u[i,:])
    









