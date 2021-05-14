#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:13:19 2021

@author: ron
"""

import numpy as np
import matplotlib.pyplot as plt


class pair(object):
    
    def __init__(self, x10, x20, y10, y20, d0, k, m, dt):
        self.x1 = [x10]
        self.x2 = [x20]
        self.y1 = [y10]
        self.y2 = [y20]
        
        self.vx1 = [0.0]
        self.vx2 = [0.0]
        self.vy1 = [0.5]
        self.vy2 = [-0.5]
        
        self.d0 =d0
        self.k_m = k/m
        self.t = [0.0]
        self.dt = dt
        
        
    def step(self):
        
        d = ((self.x1[-1]-self.x2[-1])**2 + (self.y1[-1]-self.y2[-1])**2)**0.5
        A = -self.k_m*(1.0 - self.d0/d)
        
        z = np.array([self.x1[-1],
                      self.x2[-1],
                      self.y1[-1],
                      self.y2[-1],
                      self.vx1[-1],
                      self.vx2[-1],
                      self.vy1[-1],
                      self.vy2[-1]])
        
        dzm0 = np.array([z[4],
                         z[5],
                         z[6],
                         z[7],
                         A*(z[0]-z[1]),
                         A*(z[1]-z[0]),
                         A*(z[2]-z[3]),
                         A*(z[3]-z[2])])
        
        zm1 = z + self.dt * dzm0 
        
        dzm1 = np.array([zm1[4],
                         zm1[5],
                         zm1[6],
                         zm1[7],
                         A*(zm1[0]-zm1[1]),
                         A*(zm1[1]-zm1[0]),
                         A*(zm1[2]-zm1[3]),
                         A*(zm1[3]-zm1[2])])
        
        dz = 0.5*(dzm0 + dzm1)
                        
        
        self.x1.append( z[0] + self.dt * dz[0] )
        self.x2.append( z[1] + self.dt * dz[1] )
        self.y1.append( z[2] + self.dt * dz[2] )
        self.y2.append( z[3] + self.dt * dz[3] )
        self.vx1.append( z[4] + self.dt * dz[4] )
        self.vx2.append( z[5] + self.dt * dz[5] )
        self.vy1.append( z[6] + self.dt * dz[6] )
        self.vy2.append( z[7] + self.dt * dz[7] )
        self.t.append(self.t[-1] + self.dt)
        
        
    def plot_frame(self, i, ax):
        x_ = [self.x1[i], self.x2[i]]
        y_ = [self.y1[i], self.y2[i]]
        ax.plot(x_, y_, 'o-k', lw=0.3)
        




if __name__ == '__main__':
        
    p = pair(-1.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.002)
    steps = 5000
    
    for i in range(steps):
        p.step()
    
    # fig, ax = plt.subplots()
    
    # for e,i in enumerate(range(0, len(p.t), 50)):
    #     p.plot_frame(i, ax)
    #     ax.set_xlim(-3,3)
    #     ax.set_ylim(-3,3)
    #     ax.set_aspect('equal')
    #     fig.savefig('./img/im%02d.jpg'%e)
    #     ax.clear()
    # plt.close()










