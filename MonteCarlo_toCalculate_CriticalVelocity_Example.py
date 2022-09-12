#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 09:12:13 2022

@author: scottfeehan

Example code for using deterministic force balance model and Monte Carlo 
simulations to estimate critical velocity and associated uncertainty for an 
individual and range of grain sizes. 

"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm
from scipy.stats import iqr
import seaborn as sns

#%% Constants and parameter

monte_carlo_step = 100000 # Monte Carlo iteration length
g = 9.81 # Gravity 
theta = 0.001 # Constant slope 
V_w_V = 1 # Fully submerged grain  
k_von = 0.41 # Von Karman constant
Grain_size = 0.1 # Example grain size 

#Sediment density 
rho_s_mean = 2650 # Assumed mean
rho_s_min = 2500 # Assumed minimium 
rho_s_max = 3000 # Assumed maximum
rho_s_stdv = 100 # Assumed standard deviation

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
mu_mean = 2.75
mu_min = 0.27
mu_max =  11.4
mu_stdv = 0.27 

# Converting mu mean and standard deviation to lognormal space
ln_mu_stdv = np.sqrt( np.log( mu_stdv**2 / mu_mean**2 + 1))
ln_mu = np.log(mu_mean/np.exp(0.5*ln_mu_stdv**2))

#%% Create function to generate truncated normal distributions for force balaance parameters 

def get_truncated_normal(upp,low, mean, sd): # Upper bound # Lower bound # Mean # Standard deviation
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#%% Generate force balance parameter distributions 
    
# Generate distribution of respective force balance parameter
X = get_truncated_normal(rho_s_max ,rho_s_min,rho_s_mean,rho_s_stdv ) 
rho_s = X.rvs(monte_carlo_step) 

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
    
#%% Define deterministic force balance 
# Wiberg and Smith (1987)
    
def ForceBalance(rho_s,rho_w,g,mu,C_l,C_d,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p):
    r = B_axis/2
    A_e = r**2*np.arccos((r - (B_axis-(p*B_axis)))/r) - (r - (B_axis-(p*B_axis)))*np.sqrt(2*r*(B_axis-(p*B_axis)) - (B_axis-(p*B_axis))**2)
    A_e = A - A_e
    A_p = np.ones(len(p))*A
    h = B_axis*p[np.where(p < 0.5)]
    a = np.sqrt(r**2 - ((r - h)**2))
    A_p[np.where(p < 0.5)] = np.pi * (a**2)
    v_c = ((2*g*V*(rho_s/rho_w - 1*(V_w_V))*(mu*np.cos(theta) - np.sin(theta)))/(C_d*A_e + mu*C_l*A_p))**0.5
    return v_c
    
#%% Calculate critical velocity (u_c) for grain entrainment using a Monte Carlo method for a single grain size

# Calculate grain dimensions
B_axis = Grain_size # Intermediate axis
A_axis = B_axis # Long axis 
C_axis = B_axis # Short axis 
V = (A_axis/2)*(B_axis/2)*(C_axis/2)*(4/3)*np.pi # Volume of a sphere 
A = (B_axis/2)*(C_axis/2)*np.pi # Area

# Monte Carlo simulation for single grain size 
v_ForceBalance = ForceBalance(rho_s,rho_w,g,mu,C_l,C_d,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p) 

# Calculate statistics of distribution 
v_FB_median = np.nanmedian(v_ForceBalance) # Median while excluding NaN values 
v_FB_iqr = iqr(v_ForceBalance,nan_policy='omit')/2 # Interquartile range, divide by 2 so it can be added to the median value in a plot
v_FB_90 = iqr(v_ForceBalance,rng=(5,95),nan_policy='omit')/2 # 5th to 95th percentile

#%% Plot distribution of the estimated critical velocity and show median, interquartile range (IQR), and the 5th to 95th percentile range

plt.figure(figsize=(6,5))
sns.kdeplot(v_ForceBalance,linewidth=3,color='grey',label='$u_c$') # Full distribution of values 
plt.axvline(v_FB_median,linewidth=3,color='r',label='Median') # Median of distribution 
plt.axvspan(v_FB_median - v_FB_iqr,v_FB_median + v_FB_iqr,color='r',alpha=0.2,label='IQR') # Interquartile range 
plt.axvline(v_FB_median - v_FB_90,linewidth=3,color='r',linestyle='--',label='5$^{th}$ to 95$^{th}$') # 5th to 95th percentile
plt.axvline(v_FB_median + v_FB_90,linewidth=3,color='r',linestyle='--') 
plt.legend(fontsize=14)
plt.xlabel('Critical velocity, $u_c$ (m/s)',fontsize=14)
plt.ylabel('Probability density',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)

#%% Perform Monte Carlo simulation for grain sizes 1 mm to 1 m at a 1 mm resolution

Grain_size_range = np.arange(0.001,1.001,0.001) # Range of grain sizes to sample over

v_ForceBalance = np.zeros([int(monte_carlo_step),len(Grain_size_range)]) # Empty array to store Monte Carlo velocity estimates

# Calculate grain dimensions
B_axis = Grain_size_range
A_axis = B_axis
C_axis = B_axis
V = (A_axis/2)*(B_axis/2)*(C_axis/2)*(4/3)*np.pi
A = (B_axis/2)*(C_axis/2)*np.pi


for i in range(0,len(Grain_size_range)): # Monte Carlo simulation across grain sizes
    v_ForceBalance[:,i] = ForceBalance(rho_s,rho_w,g,mu,C_l,C_d,theta,A_axis[i],B_axis[i],C_axis[i],V_w_V,V[i],A[i],p)

# Calculate statistics of distribution 
v_FB_median = np.zeros(len(Grain_size_range))
v_FB_iqr = np.zeros(len(Grain_size_range))
v_FB_90 = np.zeros(len(Grain_size_range))

for i in range(0,len(Grain_size_range)):
    
    v_FB_median[i] = np.nanmedian(v_ForceBalance[:,i])
    v_FB_iqr[i] = iqr(v_ForceBalance[:,i],nan_policy='omit')
    v_FB_90[i] = iqr(v_ForceBalance[:,i],rng=(5,95),nan_policy='omit')

v_FB_iqr = v_FB_iqr/2
v_FB_90 = v_FB_90/2

#%% Plot calculated statistics across grain size range

plt.figure(figsize=(6,5))
plt.plot(Grain_size_range,v_FB_median,'r',label='Median')
plt.fill_between(Grain_size_range,v_FB_median - v_FB_iqr,v_FB_median + v_FB_iqr,color='r',alpha=0.2,label='IQR')
plt.plot(Grain_size_range,v_FB_median + v_FB_90,'r',linestyle='--',label='5$^{th}$ to 95$^{th}$')
plt.plot(Grain_size_range,v_FB_median - v_FB_90,'r',linestyle='--')
plt.legend(fontsize=14)
plt.xlabel('Grain size, $D$ ($m$)',fontsize=14)
plt.ylabel('Critical velocity, $u_c$ ($m/s$)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.xlim([min(Grain_size_range),max(Grain_size_range)])
plt.tight_layout()

