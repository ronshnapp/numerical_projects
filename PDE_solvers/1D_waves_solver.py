# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:09:43 2021

@author: Ron
"""

import numpy as np
import matplotlib.pyplot as plt


# set up the domain:
# space domain between -1 and 1 and i want x=0; thus Nx = 2/dx+1, 
# thus 1/dx shoud be even.

dx = 0.005
dt = 0.0025
c = 0.5

T = 1
X = 1.0

Nx = int(2*X/dx + 1)
Nt = int(T/dt + 2)

tm = np.arange(-dt,T+dt,dt)
x_ax = np.arange(-X, X+dx, dx)


# set holding variable:
U = np.empty(shape=(Nt, Nx))


# set time boundary condition:
U0 = np.zeros(Nx)
U0[int(Nx/2)]=1.0

U0 = np.exp( -(x_ax/2.0/0.025)**2)
#U0 = np.cos(x_ax*20*np.pi) * np.exp(-np.abs(x_ax)/0.05)



U[0,:] = U0
U[1,:] = U0
U[2,:] = U[0,:]


# set space boundary condition:
U[:,0] = 0.0
U[:,-1] = 0.0



# solve:
for n in range(2,Nt-1):
    for i in range(1,Nx-1):
        fin = (dt/dx*c)**2 * (U[n,i+1] + U[n,i-1] - 2*U[n,i])
        U[n+1,i] = 2*U[n,i] - U[n-1,i] + fin
    U[n+1,0] = U[n+1,1]
    U[n+1,-1] = U[n+1,-2]


fig, ax = plt.subplots()
#ax.contourf(x_ax, tm, U, levels=100)
ax.imshow(U)






