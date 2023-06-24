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
from scipy.integrate import odeint


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
    
    
    def get_dudx(self, x, t):
        '''
        This returns the tensor
        
            A_ij = du_i / dx_j
            
        at a point x and time t.
        '''
        
        modes = []
        
        for i in range(self.Nk):
            pn = np.dot(x, self.kn[i]) + t*self.Omega_n[i]
            
            ankn = np.zeros((3,3))
            bnkn = np.zeros((3,3))
            for n in range(3):
                ankn[:,n] = self.an[i] * self.kn[i][n]
                bnkn[:,n] = self.bn[i] * self.kn[i][n]
                
            #print()
            
            modes.append(-ankn * np.sin(pn) + bnkn * np.cos(pn))

        return np.sum(modes, axis=0)
    
    
    
    def get_particle_trajectory(self, x0, t):
        '''
        This method will interate the flow velocity in time to obtain
        the Lagrangian trajectory of a particle that started at time
        t0 and position x0. The particles position is returned at a 
        list of times t.
        '''
        p = odeint(self.__call__, x0, t)
        return p
        
    
    
    def get_jeffrey_fiber_trajectory(self, x0, s0, t, chi=10.0):
        '''
        This function integrates the flow field and the Jeffrey equation 
        to get the position and orientation of a small fiber in the flow
        field.
        
        x0 - initial position
        s0 - initial orientation (unit vector pointing along the fiber)
        t - list of times for which the trajectory is needed
        chi - the aspect ratio of the fiber (default=10)
        
        returns
        x - trajectory of center point of the fiber
        s - orientation of the fiber (fiber-parallel unit vector)
        '''
        
        def rate_of_change(X, t):
            x = X[:3]
            s = X[3:]
            
            dxdt = self.__call__(x, t)
            
            A = self.get_dudx(x, t)
            E = 0.5*(A + A.T)
            R = 0.5*(A - A.T)
            gamma = (chi**2 + 1) / (1 - chi**2)    
            J = E + gamma*R
            dsdt = 1/gamma * (np.dot(J, s) - s * np.dot(np.dot(E, s), s))
            
            return np.append(dxdt, dsdt)
        
        X0 = np.append(x0, s0)
        p = odeint(rate_of_change, X0, t)
        
        x = p[:,:3]
        s = p[:,3:]
        
        return x, s
        
        
 
        
 
    
 
    
 

def line_dist(O1, r1, O2, r2):
    '''
    2 lines are defined as (O1 + a r1) and  (O2 + b r2), where O are origins,
    r are direction vectors, and a,b are variables (in n dimensions). 
    This utility calculates the minimal distance between these 2 lines.
    
    input - 
    O1,O2,r1,r2 (arrays, n) - line parameters
    
    output - 
    dist (float) -the minimum distance between the lines
    x (array, n)- the point that is nearest to the two points crossing
    '''
    
    # find the a,b that minimize the distance:
    # A = array([[dot(r1,r1), -dot(r1,r2)],
    #            [dot(r1,r2), -dot(r2,r2)]])
    
    r1r2 = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2]
    r12 = r1[0]**2 + r1[1]**2 + r1[2]**2
    r22 = r2[0]**2 + r2[1]**2 + r2[2]**2

    #Ainv = [[-r22, r1r2],
    #        [-r1r2, r12]]
    
    dO = O2-O1
    B = [r1[0]*dO[0] + r1[1]*dO[1] + r1[2]*dO[2],
         r2[0]*dO[0] + r2[1]*dO[1] + r2[2]*dO[2]]
    
    try:
        #a,b = dot(Ainv, B)
        a = (-r22*B[0] + r1r2*B[1])/(r1r2**2 - r12 * r22)
        b = (-r1r2*B[0] + r12*B[1])/(r1r2**2 - r12 * r22)
    except:
        a,b = 0.0, 0.0
    
    # use the a,b to calc the minimum distance:
    l1,l2 = O1 + a*r1 , O2 + b*r2
    dist = sum((l1 - l2)**2)**0.5
    x = (l1+l2)*0.5
    
    return dist, x
        
        
        
        


#%%
import matplotlib.pyplot as plt

KS = kinematic_simulation(30, K1=10, KNk=0.5)



#%%


from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


x0 = [-1.38494392, -1.42215316,  0.5079383 ] 
s0 = [-0.37767053, -0.32207659, -0.86811949]
t = np.linspace(-2,2,num=200)
x1, s1 = KS.get_jeffrey_fiber_trajectory(x0, s0, t)



x0 = [-1.61017578, -1.03790838,  0.37058359] 
s0 = [-0.65376448, 0.5897513, 0.47411548 ]
t = np.linspace(-2,2,num=200)
x2, s2 = KS.get_jeffrey_fiber_trajectory(x0, s0, t)


L = 0.1


for i in range(t):
    x_ = [x1[i,0]-s1[i,0] * L, x1[i,0] + s1[i,0] * L]
    y_ = [x1[i,1]-s1[i,1] * L, x1[i,1] + s1[i,1] * L]
    z_ = [x1[i,2]-s1[i,2] * L, x1[i,2] + s1[i,2] * L]
    ax.plot(x_, y_, z_, 'b-')
    
    x_ = [x2[i,0]-s2[i,0] * L, x2[i,0] + s2[i,0] * L]
    y_ = [x2[i,1]-s2[i,1] * L, x2[i,1] + s2[i,1] * L]
    z_ = [x2[i,2]-s2[i,2] * L, x2[i,2] + s2[i,2] * L]
    ax.plot(x_, y_, z_, 'r-')




#%%


