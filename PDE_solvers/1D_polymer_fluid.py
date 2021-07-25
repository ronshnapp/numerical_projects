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
    B0 = -0.05
    U0 = 0.4
    tau = 1.0  #12.1
    nu = 0.0001   #0.00009
    dudt = -(u+U0)*ux + (B+B0)*Bx + nu*(uxx)
    dBdt = -(u+U0)*Bx + (B+B0)*ux - B*(1/tau)
    dFdt = np.append(dudt, dBdt)

    return dFdt


def eqn_solution(F0, t, L):
    """Use odeint to solve the equation on a periodic domain."""

    sol = odeint(equation, F0, t, args=(L,), mxstep=5000)
    sol_u = sol[:,:int(len(F0)/2)]
    sol_B = sol[:,int(len(F0)/2):]
    return sol_u, sol_B



def animate(sol, dx, dt, play_speed=10.0, ylim=None):
    from moviepy.video.io.bindings import mplfig_to_npimage
    import moviepy.editor as mpy
    
    fig, ax = plt.subplots()
    x_ = np.arange(sol.shape[1])*dx
    
    if ylim == None:
        mn, mx = np.amin(sol), np.amax(sol)
    else:
        mn, mx = ylim
    
    def mk_frame(t):
        i = int(t / Nframes * play_speed * (Nframes-1))
        ax.plot(x_, sol[i,:])
        ax.set_ylim(mn, mx)
        ax.text(0.01,0.01, 't=%.2f'%(i*dt), transform=ax.transAxes)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        img = mplfig_to_npimage(fig)
        ax.clear()
        return img
    
    Nframes = sol.shape[0]
    animation = mpy.VideoClip(mk_frame, duration = Nframes/play_speed)
    
    animation.write_videofile('test.avi', fps = play_speed, codec='png')
    plt.close()
    return animation




def plot_multicolor_line(ax, x, y, c, vmin=None, vmax=None):
    '''will plot the line x vs y with a color parameter c'''
    
    #import matplotlib as mpl
    #from matplotlib.colors import ListedColormap
    from matplotlib import cm
    
    cmap = cm.get_cmap('viridis', 100)
    
    T = 0.5*(c[1:]+c[:-1])
    
    if vmax==None: vmax=np.amax(T)
    if vmin==None: vmin=np.amin(T)
    
    T = T + vmin
    T = T / vmax
    #T[T>1] = 1.0
    #T[T<0] = 0.0
    
    
    for i in range(len(x)-1):
        ax.plot( x[i:i+2], y[i:i+2], color=cmap(T[i])) #color=(T[i], 0.5*T[i], 0.5*(1-T[i])) )
    
    clr_map = cm.ScalarMappable(norm=None, cmap=cmap)
    return clr_map
    
    

if __name__ == '__main__':
    N = 2**9
    L = 1.0
    dx = L/(N-1.0)
    x = np.linspace(0, L-dx, N)
    
    
    # two strong and fast shocks:
    x0 = 0.1
    d = 0.03
    u0 = 0.1 * np.exp(-((x-x0)/d)**2 )
    B0 = 0.0 * np.ones(x.shape)
    
    #x0 = 0.02
    #u0 = 0.01 * np.exp(-((x-x0)/0.005)**2 )
    #B0 = -0.0015 * np.ones(x.shape)
    
    
    # two slow shocks, one very weak:
    #u0 = 1 * np.exp(-((x-L/2)/0.005)**2 )
    #B0 = 0.025 * np.ones(x.shape) #-  0.1 * np.exp(-((x-L/2)/0.005)**2 )
    F0 = np.append(u0, B0)
    
    T = 2
    t = np.linspace(0, T, 500)
    dt = t[1]-t[0]

    u, B = eqn_solution(F0, t, L)
    
    
    
    import matplotlib.pyplot as plt

    # plt.figure(figsize=(6,5))
    # # plt.imshow(u[::-1, :], extent=[0,L,0,T])
    # plt.imshow(u[::-1, :], vmax=0.5)
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('t')
    # #plt.axis('normal')
    # plt.show()
    
    
    plt.figure(figsize=(6,5))
    plt.imshow(B[::-1, :], extent=[0,L,0,T])
    #plt.imshow(B[::-1, :])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.show()
    
    #plt.figure(figsize=(6,5))
    #for i in [0,15,30,60,125,250,499]:
    #    plt.plot(x-x0, u[i,:])
    


    

    # import matplotlib.pylab as pylab
    # from matplotlib.legend import Legend
    
    # params = {'legend.fontsize': 'x-large',
    #           'figure.figsize': (6, 4.5),
    #          'axes.labelsize': 'x-large',
    #          'axes.titlesize':'x-large',
    #          'xtick.labelsize':'x-large',
    #          'ytick.labelsize':'x-large',
    #          'font.family': ['serif'],
    #          'font.sans-serif' : 'Computer Modern Roman',
    #          'grid.color': '#c1c1c1',
    #          'text.usetex' : 'false',
    #          'legend.framealpha' : '1.0',
    #          'lines.markersize': '7.0',
    #          'lines.markeredgewidth': '0.5',
    #          'text.usetex':        'true'}
    # pylab.rcParams.update(params)


    # fig, ax = plt.subplots(4,1, sharex=True)
    # fig.subplots_adjust(right=0.82, left=0.1, bottom=0.12, top=0.99)
    # e=0
    # vmax = np.amax(np.abs(B))
    # for i in [75,200,325,450]:
    #     c = plot_multicolor_line(ax[len(ax)-e-1], x-x0, u[i,:], 
    #                          np.abs(B[i,:]), vmax = 0.05, vmin=0.0)
        
    #     ax[e].plot(x-x0, u[0,:], 'k-', lw=.6, alpha=0.5)
    #     ax[e].set_ylabel(r'$u_x$')
    #     ax[len(ax)-e-1].text(0.85, 0.75, r'$t=%.1f$'%t[i],
    #                          transform=ax[len(ax)-e-1].transAxes,fontsize=15)
    #     e+=1
        
    # ax[-1].set_xlabel(r'$x-x_0$')
    
    # cbar_ax = fig.add_axes([0.87, 0.18, 0.0175, 0.75])
    # cbar = fig.colorbar(c, cax=cbar_ax)
    # cbar.set_label(r'$|b_x/B|$')

    #fig.savefig('1D_model.pdf')
    #plt.tight_layout()






