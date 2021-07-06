#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:08:45 2021

@author: ron
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



def plot_square(x1,x2,y1,y2, ax, color):
    rc = Rectangle((x1,y1), x2-x1, y2-y1, color=color)
    ax.add_patch(rc)
    return rc



def plot_2by6_cross(center, edge, ax):
    small_edge = edge/6.0
    x0 = center[0] - edge/2.0
    y0 = center[1] - edge/2.0
    
    for i in range(6):
        for j in range(6):
            
            if i in [2,3] or j in [2,3]: 
                color='k'
                plot_square(x0 + small_edge*i,
                            x0 + small_edge*(i+1),
                            y0 + small_edge*j,
                            y0 + small_edge*(j+1), 
                            ax, 
                            color)

            #else: color='w'
                



def get_next_iter_centers(center, edge):
    '''
    this will return the centers and edges of the next five crosses given a
    mother cross's center and edge 
    '''
    centers = []
    centers.append( [center[0]+edge/3, center[1]] )
    centers.append( [center[0]-edge/3, center[1]] )
    centers.append( [center[0], center[1]+edge/3] )
    centers.append( [center[0], center[1]-edge/3] )
    centers.append( [center[0], center[1]] )
    edges = [edge/3 for i in range(5)]
    return centers, edges




def plot_Vicsek_fractal_iter_n(n=2):
    fig, ax=plt.subplots()
    
    if n==0: plot_square(-1,1,-1,1, ax, 'k')
    
    for i in range(1,n+1):
        if i == 1:
            centers_lst = [[0.0, 0.0]]
            edges_lst = [2.0]
        else: 
            tmpc = []
            tmpe = []
            for ii in range(len(centers_lst)):
                tmp = get_next_iter_centers(centers_lst[ii],
                                              edges_lst[ii])
                
                tmpc += tmp[0]
                tmpe += tmp[1]
                
            centers_lst = tmpc
            edges_lst = tmpe
        
        if i==n:
            for ii in range(len(centers_lst)):
                plot_2by6_cross(centers_lst[ii], edges_lst[ii], ax)
            
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_aspect('equal')
    fig.set_size_inches((8, 8))
    return fig, ax


n = 4  #(don't go above 6 due to memory consumption..)
fig, ax = plot_Vicsek_fractal_iter_n(n)
fig.savefig('im%02d.jpg'%n, dpi=150)

