#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:35:57 2022

@author: scottfeehan

Comparing theoretical critical velocity to highly constrainted field observations of incipient motion

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from scipy.stats import truncnorm

#%% 

filepath = 'Helley_Table_1_all.txt'
Helley_1969_table_1 = np.loadtxt(filepath,skiprows=1) #particle_num	C	B	A	rho_s	Vol	calc_v 
Helley_1969_table_1[:,(1,2,3,6)] = Helley_1969_table_1[:,(1,2,3,6)]*0.3048 # Convert to meters 
Helley_1969_table_1[:,4] = Helley_1969_table_1[:,4]*1000 # Convert mass to kg/m^3
Helley_1969_table_1[:,5] = Helley_1969_table_1[:,5]*0.0283 # Convert volume to m^3

#%% 

filepath  = 'Helley_1969_fig_9.txt'
Helley_1969_fig_9 = np.loadtxt(filepath,skiprows=1) # measured, calculated 
Helley_1969_fig_9 = Helley_1969_fig_9*0.3048

Helley_1969_all_data = Helley_1969_table_1[Helley_1969_table_1[:, -1].argsort()]
Helley_1969_all_data[:,6] = np.sort(Helley_1969_fig_9[:,0])

#%% Assumed force balance parameter mean, minimum, maximum, and standard deviation

#Sediment density 
rho_s_mean = np.mean(Helley_1969_all_data[:,4])  

# Fluid density 
rho_w_mean = 1000 
rho_w_min = 990 
rho_w_max = 1200 
rho_w_stdv = 30

# Drag coefficient
C_d_mean = 0.76 
C_d_min = 0.1
C_d_max = 3 
C_d_stdv = 0.29

# Lift coefficient 
C_l_mean = 0.38
C_l_min = 0.06
C_l_max = 2 
C_l_stdv = 0.29

# Grain protrusion 
p_mean = 0.7
p_min = 0.1
p_max = 1
p_stdv = 0.4

# Coefficient of friction
mu_mean = 0.58 # Grains are sitting on the top of the bed
mu_min = 0.27
mu_max =  11.4
mu_stdv = 0.27 

# Converting mu mean and standard deviation to lognormal space
ln_mu_stdv = np.sqrt( np.log( mu_stdv**2 / mu_mean**2 + 1))
ln_mu = np.log(mu_mean/np.exp(0.5*ln_mu_stdv**2))

# Assume all distributions have a truncated normal except mu which has a lognormal distribution (Booth et. al., 2014)

#%% Create function to generate truncated normal distributions for force balaance parameters 

def get_truncated_normal(upp,low, mean, sd): # Upper bound # Lower bound # Mean # Standard deviation
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#%% 
    
monte_carlo_step = 100000 # Monte Carlo iteration length

X = get_truncated_normal(rho_w_max,rho_w_min,rho_w_mean,rho_w_stdv)
rho_w = X.rvs(monte_carlo_step) 

X = get_truncated_normal(C_l_max,C_l_min,C_l_mean,C_l_stdv)
C_l = X.rvs(monte_carlo_step) 

X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
C_d = X.rvs(monte_carlo_step) 

X = get_truncated_normal(p_max ,p_min,p_mean,p_stdv)  
p = X.rvs(monte_carlo_step) 

mu = np.random.lognormal(ln_mu,ln_mu_stdv,monte_carlo_step)

# Resample distribution if beyond the assumed minimum and maximum until the value within the specified range
# Have to do this because of the lognormal distribution

while True: 
    temp = np.where((0>mu) | (mu>mu_max))
    mu[temp] = np.random.lognormal(ln_mu,ln_mu_stdv,len(temp[0]))
    if len(temp[0]) <1:
        break 

#%% Constants
        
g = 9.81 # Gravity 
theta = 0.006 # reported local slope
V_w_V = 1 # Fully submerged grain  
k_von = 0.407 # Von Karman constant

#%% Using simplified force balance due to high assumed protrusion and non-spherical grains 

def ForceBalance(rho_s,rho_w,g,mu,C_l,C_d,theta,A_axis,B_axis,C_axis,V_w_V,V,A_n,A_p,p):
    v_c = ((2*g*V*(rho_s/rho_w - 1*(V_w_V))*(mu*np.cos(theta) - np.sin(theta)))/(C_d*A_n*p + mu*C_l*A_p))**0.5
    return v_c

#%% Measurements of grain long (A), intermediate (B), and short (C) axes 

A_B = np.mean(Helley_1969_all_data[:,3]/Helley_1969_all_data[:,2])
C_B = np.mean(Helley_1969_all_data[:,1]/Helley_1969_all_data[:,2])

#%% Generate power law using physical constraints from observations 

grain_range = np.arange(0.001,1.001,0.001)

v_ForceBalance = np.zeros([int(monte_carlo_step),len(grain_range)])

B_axis = grain_range
A_axis = B_axis*A_B
C_axis = B_axis*C_B
V = (A_axis/2)*(B_axis/2)*(C_axis/2)*(4/3)*np.pi
A_n = (B_axis/2)*(C_axis/2)*np.pi
A_p = (A_axis/2)*(B_axis/2)*np.pi
theta = 10**-3

for i in range(0,len(grain_range)):
        
    v_ForceBalance[:,i] = ForceBalance(rho_s_mean,rho_w_mean,g,mu,C_l,C_d,theta,A_axis[i],B_axis[i],C_axis[i],V_w_V,V[i],A_n[i],A_p[i],p)

v_FB_median = np.zeros(len(grain_range))
v_FB_iqr = np.zeros(len(grain_range))
v_FB_90 = np.zeros(len(grain_range))

for i in range(0,len(grain_range)):
    
    v_FB_median[i] = np.nanmedian(v_ForceBalance[:,i])
    v_FB_iqr[i] = iqr(v_ForceBalance[:,i],nan_policy='omit')
    v_FB_90[i] = iqr(v_ForceBalance[:,i],rng=(5, 95),nan_policy='omit')

v_FB_iqr = v_FB_iqr/2
v_FB_90 = v_FB_90/2

Grain_size_plot_median = v_FB_median
Grain_size_plot_iqr = v_FB_iqr
Grain_size_plot_90 = v_FB_90

v_c_1_1_iqr_90 = np.array([Grain_size_plot_median,Grain_size_plot_iqr,Grain_size_plot_90])
v_c_1_1_iqr_90 = np.transpose(v_c_1_1_iqr_90)

#%% Monte carlo simulaiton to determine critical velocity of observed grains 

B_axis = Helley_1969_all_data[:,2]
A_axis = Helley_1969_all_data[:,1]
C_axis = Helley_1969_all_data[:,3]
V = Helley_1969_all_data[:,5]
rho_s = Helley_1969_all_data[:,4]
rho_w = 1000
A_n = (B_axis/2)*(C_axis/2)*np.pi
A_p = (A_axis/2)*(B_axis/2)*np.pi
theta = 0.006

v_ForceBalance = np.zeros([int(monte_carlo_step),len(B_axis)])

for i in range(0,len(B_axis)):
        
    v_ForceBalance[:,i] = ForceBalance(rho_s[i],rho_w,g,mu,C_l,C_d,theta,A_axis[i],B_axis[i],C_axis[i],V_w_V,V[i],A_n[i],A_p[i],p)

v_FB_median = np.zeros(len(B_axis))
v_FB_iqr = np.zeros(len(B_axis))

for i in range(0,len(B_axis)):
    
    v_FB_median[i] = np.nanmedian(v_ForceBalance[:,i])
    v_FB_iqr[i] = iqr(v_ForceBalance[:,i],nan_policy='omit')

v_FB_iqr = v_FB_iqr/2

#%% 

plt.figure(figsize=(9,5))
plt.fill_between(v_c_1_1_iqr_90[:,0],v_c_1_1_iqr_90[:,0]+v_c_1_1_iqr_90[:,1],v_c_1_1_iqr_90[:,0]-v_c_1_1_iqr_90[:,1],color='r',alpha=0.2,label='IQR')
plt.scatter(v_FB_median,Helley_1969_all_data[:,6],c=Helley_1969_all_data[:,1]/np.sqrt(Helley_1969_all_data[:,2]*Helley_1969_all_data[:,3]),cmap = 'PuOr',marker='d',s=150,edgecolor='k',label='Helley (1969)')
plt.plot(v_c_1_1_iqr_90[:,0],v_c_1_1_iqr_90[:,0],'r',label='1:1')
plt.plot(v_c_1_1_iqr_90[:,0],v_c_1_1_iqr_90[:,0] + v_c_1_1_iqr_90[:,2],'r--',label='95 to 5')
plt.plot(v_c_1_1_iqr_90[:,0],v_c_1_1_iqr_90[:,0] - v_c_1_1_iqr_90[:,2],'r--')
plt.xlim([1,3.5])
plt.ylim([1,3.5])
plt.ylabel('Reported critical velocity, $u_c$ ($m/s$)',fontsize=14)
plt.xlabel('Theoretical critical velocity, $u_c$ ($m/s$)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.legend(fontsize=14)#,loc=0)
plt.loglog()
cbar = plt.colorbar(orientation='vertical',pad=0.05)
cbar.set_label(label=r'Tabular $\leftarrow$ CSF (C/$\sqrt{A*B}$) $\rightarrow$ Spherical',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.tight_layout()























