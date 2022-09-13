#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:57:46 2022

@author: scottfeehan

This code generates parts of figure 4

Comparing theoretical estiamtes of critical velocity to idealized experiments from Wu and Shih (2012).
Theoretical estimates are refined using identical constraints when avaliable (e.g. Prostrusion and Coefficeint of friction) 
or inferred from other laboratory experiments (e.g. Drag coefficient)

"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm
from scipy.stats import iqr


#%% Uploading data

# Experimental flume data
filepath = 'Wu_Shih_2012_Experiment_1.csv'
Wu_dataset_1 = np.loadtxt(filepath,delimiter=',',skiprows=1)

filepath = 'Wu_Shih_2012_Experiment_2.csv'
Wu_dataset_2 = np.loadtxt(filepath,delimiter=',',skiprows=1)

# Near grain velocity - Drag coefficient observations
filepath = filepath = 'Schmeeckle_fig_10c.txt'
Schmeeckle_2007_CD_u = np.loadtxt(filepath,skiprows=1) # Drag coefficient, grain proximal velocity 
Schmeeckle_2007_CD_u[:,0] = Schmeeckle_2007_CD_u[:,0]/100 # Convert velocity to m/s

#%% Constraints and parameters 

monte_carlo_step = 100000 # Monte Carlo iteration length
grain = 0.008 # Diameter of mobilized grain
bed = 0.008 # Diameter of grains in the bed
V_w_V = 1 # Fully submerged grain 
p1_protrusion = 0.86 # Protrusion of grain in pocket 1 
p2_protrusion = 0.86 # Protrusion of grain in pocket 2 
rho_s_2500 = 2500
rho_s_1430 = 1430
rho_w = 1000 
g = 9.81
theta = 0.0001

#%% Experimental constraints for unconstrained parameters 

C_l_max = 2 
C_l_min = 0.06
C_l_stdv = 0.29
C_l_mean = 0.19 # Assumed lower lift coefficient b/c low flow velocity

C_d_max = 3 
C_d_min = 0.1
C_d_stdv = 0.29

#%% Loading data and differentiate into pre- and post-entrainment.

# Experiment 2
Wu_data_1_t = Wu_dataset_1[:,0] # Store time array 

neg = np.where(Wu_data_1_t < 0) # Negative time = time prior to mobilization
pos = np.where(Wu_data_1_t > 0) # Positive time = time prior to mobilization

Wu_mean_data_1_neg = Wu_dataset_1[neg,1].reshape(-1) # Corresponding data 
Wu_mean_data_1_pos = Wu_dataset_1[pos,1].reshape(-1)

# Experiment 3
Wu_data_2_t = Wu_dataset_2[:,0]

neg = np.where(Wu_data_2_t < 0)
pos = np.where(Wu_data_2_t > 0)

Wu_mean_data_2_neg = Wu_dataset_2[neg,1].reshape(-1)
Wu_mean_data_2_pos = Wu_dataset_2[pos,1].reshape(-1)

#%% Experimental data from Wu and Shih (2012) table 1 

data_1_u_pre = np.round(np.mean(Wu_mean_data_1_neg),decimals=3) # Averaged mean is same as found in Wu and Shih (2012) table 1
data_1_sigma_pre = 0.04 # sigma From Wu and Shih (2012) table 1

data_1_u_post = np.round(np.mean(Wu_mean_data_1_pos),decimals=3)
data_1_sigma_post = 0.043

data_2_u_pre = np.round(np.mean(Wu_mean_data_2_neg),decimals=3)
data_2_sigma_pre = 0.023

data_2_u_post = np.round(np.mean(Wu_mean_data_2_pos),decimals=3)
data_2_sigma_post = 0.026 

#%% Calculating the mean coefficient of friction tan(phi) = mu of the idealized pocket using method from Kirchner et. al. (1990). Dataset 2 pivots between two spheres. 
# Dataset 3 pivots either up and over particle directly downstream or diagonal to flow direction through pocket. 

# Coefficient of friction  Pocket 1 - Dataset 2
mu_p1_mean = (1/(np.sqrt(3)))/(np.sqrt((grain/bed)**2  + 2*(grain/bed) - (1/3))) # Eq 1, Kirchner et al. 1990 # tan(phi) = this equation
deg_5 = np.tan(np.deg2rad(5)) # Effective friction angle of 5 degrees
deg_10 = np.tan(np.deg2rad(10)) # Effective friction angle of 10 degrees 

# Narrow distribution of effective friction angles 
mu_p1_max = mu_p1_mean + deg_10 # Maximum is 10 degrees larger than the mean 
mu_p1_min = deg_5
mu_p1_stdv =  deg_5 # Narrow distribution

# Convert to lognormal form 
ln_mu_stdv = np.sqrt( np.log( mu_p1_stdv**2 / mu_p1_mean**2 + 1))
ln_mu = np.log(mu_p1_mean/np.exp(0.5*ln_mu_stdv**2))

mu_p1 = np.random.lognormal(ln_mu,ln_mu_stdv,monte_carlo_step) # Coefficient of friction distribution

while True: # Truncate at specified minimum and maximum 
    temp = np.where((0>mu_p1) | (mu_p1>mu_p1_max))
    mu_p1[temp] = np.random.lognormal(ln_mu,ln_mu_stdv,len(temp[0]))
    if len(temp[0]) <1:
        break 
        
# Coefficient of friction  Pocket 2 - Dataset 3
mu_p2_mean = (2/(np.sqrt(3)))/(np.sqrt((grain/bed)**2  + 2*(grain/bed) - (1/3)))
mu_p2_mean = mu_p2_mean

mu_p2_max =  mu_p2_mean + deg_10
mu_p2_min =  mu_p2_mean - deg_10
mu_p2_stdv = deg_5

ln_mu_stdv = np.sqrt( np.log( mu_p2_stdv**2 / mu_p2_mean**2 + 1))
ln_mu = np.log(mu_p2_mean/np.exp(0.5*ln_mu_stdv**2))

mu_p2 = np.random.lognormal(ln_mu,ln_mu_stdv,monte_carlo_step) 

while True:
    temp = np.where((0>mu_p2) | (mu_p2>mu_p2_max))
    mu_p2[temp] = np.random.lognormal(ln_mu,ln_mu_stdv,len(temp[0]))
    if len(temp[0]) <1:
        break 
    
#%% Physical properties of the idealized experiments 
        
B_axis = grain
V = (B_axis/2)*(B_axis/2)*(B_axis/2)*(4/3)*np.pi # Volume of test particle
A = (B_axis/2)*(B_axis/2)*np.pi # Area 


#%% Using independent grain proximal-$C_D$ observations from Schmeeckle et. al. (2007) to calibrate mean $C_D$ for the respective experimental Monte Carlo simulation for the respective experiments.

# Find rolling mean and standard deviation 
velocity_bins = np.linspace(min(Schmeeckle_2007_CD_u[:,0]),max(Schmeeckle_2007_CD_u[:,0])+0.2,21)

cd_mean = np.zeros(len(velocity_bins))
std_mean = np.zeros(len(velocity_bins))

for i in range(0,len(velocity_bins)-1):
        
    temp = np.where((Schmeeckle_2007_CD_u[:,0] > velocity_bins[i]) & (Schmeeckle_2007_CD_u[:,0] < velocity_bins[i+1])) # Calculate mean Cd for each velocity bin 
    cd_mean[i] = np.mean(Schmeeckle_2007_CD_u[temp,1])
    std_mean[i] = np.std(Schmeeckle_2007_CD_u[temp,1]) 


#%% Plot interpolated location 

plt.figure(figsize=(7,5))
plt.plot(Schmeeckle_2007_CD_u[:,0],Schmeeckle_2007_CD_u[:,1],'k.',markersize=10,alpha=0.5,label='Measurements') # Observations
plt.ylabel('$C_D$')
plt.xlabel('Velocity (m/s)')
plt.plot(velocity_bins,cd_mean,'r-',linewidth=3,label='$\overline{C}_D$') # Rolling mean 
plt.plot(velocity_bins,cd_mean+(std_mean),'r--',linewidth=3,label=r'$\sigma_{C_D}$') # Rolling standard deviation
plt.plot(velocity_bins,cd_mean-(std_mean),'r--',linewidth=3)
plt.axvline(data_1_u_post,linewidth=3,linestyle=':',color='royalblue',label='Experiment 2') # Mean velocity post grain entrainment for experiment 2
plt.axvline(data_2_u_post,linewidth=3,linestyle=':',color='darkorange',label='Experiment 3') # Experiment 3
plt.ylabel('Instantaneous $C_D$',fontsize=14)
plt.xlabel('Instantaneous downstream velocity ($m/s$)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.legend(fontsize=14)
plt.xlim([0.1,0.55])
plt.tight_layout()

#%% Interpolate mean drag coefficient from experimental measurements 

Cd_data_1_mean_pre = np.interp(data_1_u_pre,velocity_bins,cd_mean)
Cd_data_1_mean_post = np.interp(data_1_u_post,velocity_bins,cd_mean)
Cd_mean_2 = Cd_data_1_mean_post 

Cd_data_1_mean = (Cd_data_1_mean_pre+Cd_data_1_mean_post)/2
Cd_data_1_std = np.round(np.std(Schmeeckle_2007_CD_u[:,1]),decimals=2) # Data too sparse for interpolation. Make generalized asumption using all data

Cd_data_2_mean_pre = np.interp(data_2_u_pre,velocity_bins,cd_mean)
Cd_data_2_mean_post = np.interp(data_2_u_post,velocity_bins,cd_mean)
Cd_data_2_mean = (Cd_data_2_mean_pre+Cd_data_2_mean_post)/2
Cd_data_2_std = np.interp(data_2_u_post,velocity_bins,std_mean)
Cd_mean_3 = Cd_data_2_mean_post 

#%% 

Wu_data_1_RMS_upper = np.append(Wu_mean_data_1_neg + data_1_sigma_pre/2,Wu_mean_data_1_pos + data_1_sigma_post/2)
Wu_data_1_RMS_lower = np.append(Wu_mean_data_1_neg - data_1_sigma_pre/2,Wu_mean_data_1_pos - data_1_sigma_post/2)

Wu_data_2_RMS_upper = np.append(Wu_mean_data_2_neg + data_2_sigma_pre/2,Wu_mean_data_2_pos + data_2_sigma_post/2)
Wu_data_2_RMS_lower = np.append(Wu_mean_data_2_neg - data_2_sigma_pre/2,Wu_mean_data_2_pos - data_2_sigma_post/2)

#%% Plot experimental data from Wu and Shih (2012)

x_lim = [-2,2]
y_lim = [0.07,0.3]

plt.figure(figsize=(7,9))
#Experiment 2
ax = plt.subplot(211)
plt.scatter(Wu_data_1_t[0:len(Wu_mean_data_1_neg)],Wu_mean_data_1_neg,color='w',edgecolor='k',linewidth=2,s=75) # Data before mobilizaiton
plt.scatter(Wu_data_1_t[len(Wu_mean_data_1_neg):],Wu_mean_data_1_pos,color='w',edgecolor='k',linewidth=2,s=75) # Data after mobilizaiton
plt.fill_between(Wu_data_1_t,Wu_data_1_RMS_upper,Wu_data_1_RMS_lower,color='grey',alpha=0.4) # RMS of observed velocity fluctiations
plt.axvline(0,color='k',label='Initial Motion',linewidth = 2)
plt.xlim(x_lim)
plt.ylabel('Grain proximal velocity (m/s)',fontsize=14)
plt.xlabel('Time relative to incipient motion (s)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)

ax1 = plt.subplot(212)
#Experiment 3
plt.scatter(Wu_data_2_t[0:len(Wu_mean_data_2_neg)],Wu_mean_data_2_neg,color='w',edgecolor='k',linewidth=2,s=75)
plt.scatter(Wu_data_2_t[len(Wu_mean_data_2_neg):],Wu_mean_data_2_pos,color='w',edgecolor='k',linewidth=2,s=75)
plt.fill_between(Wu_data_2_t,Wu_data_2_RMS_upper,Wu_data_2_RMS_lower,color='grey',alpha=0.4)
plt.axvline(0,color='k',label='Initial Motion',linewidth = 2)
plt.ylabel('Grain proximal velocity (m/s)',fontsize=14)
plt.xlabel('Time relative to incipient motion (s)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.xlim(x_lim)
plt.tight_layout()

#%% Create function to generate truncated normal distributions for force balaance parameters 

def get_truncated_normal(upp,low, mean, sd): # Upper bound # Lower bound # Mean # Standard deviation
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


#%% Monte Carlo simulation using Wu and Shih (2012) experimental constraints 

X = get_truncated_normal(C_l_max,C_l_min,C_l_mean,C_l_stdv)
C_l = X.rvs(monte_carlo_step)

X = get_truncated_normal(C_d_max,C_d_min,Cd_mean_2,Cd_data_1_std)
C_d_data_1 = X.rvs(monte_carlo_step)

X = get_truncated_normal(C_d_max,C_d_min,Cd_mean_3,Cd_data_2_std)
C_d_data_2 = X.rvs(monte_carlo_step)

#%% Generate force balance parameter distributions for fluid-grain interactions not experimentally controlled 
    
# Generate distribution of respective force balance parameter
X = get_truncated_normal(C_l_max,C_l_min,C_l_mean,C_l_stdv)
C_l = X.rvs(monte_carlo_step)

X = get_truncated_normal(C_d_max,C_d_min,Cd_mean_2,Cd_data_1_std)
C_d_data_1 = X.rvs(monte_carlo_step)

X = get_truncated_normal(C_d_max,C_d_min,Cd_mean_3,Cd_data_2_std)
C_d_data_2 = X.rvs(monte_carlo_step)

#%% Define deterministic force balance 
# Wiberg and Smith (1987)
# Single protrusion value so do not need calculation for area parallel to the flow     

def ForceBalance(rho_s,rho_w,g,mu,C_l,C_d,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p):
    r = B_axis/2
    A_e = r**2*np.arccos((r - (B_axis-(p*B_axis)))/r) - (r - (B_axis-(p*B_axis)))*np.sqrt(2*r*(B_axis-(p*B_axis)) - (B_axis-(p*B_axis))**2)
    A_e = A - A_e
    v_c = ((2*g*V*(rho_s/rho_w - 1*(V_w_V))*(mu*np.cos(theta) - np.sin(theta)))/(C_d*A_e + mu*C_l*A))**0.5
    return v_c

#%% Monte Carlo simulation for respective simulation
    
v_data_1 = ForceBalance(rho_s_2500,rho_w,g,mu_p1,C_l,C_d_data_1,theta,B_axis,B_axis,B_axis,V_w_V,V,A,p1_protrusion)
v_data_2 = ForceBalance(rho_s_1430,rho_w,g,mu_p2,C_l,C_d_data_2,theta,B_axis,B_axis,B_axis,V_w_V,V,A,p2_protrusion)

v_data_1_median = np.nanmedian(v_data_1)
v_data_2_median = np.nanmedian(v_data_2)

v_data_1_iqr = iqr(v_data_1,nan_policy='omit')/2
v_data_2_iqr = iqr(v_data_2,nan_policy='omit')/2

v_data_1_90 = iqr(v_data_1,rng=(5,95),nan_policy='omit')/2
v_data_2_90 = iqr(v_data_2,rng=(5,95),nan_policy='omit')/2

#%% Plot velocity estimated from Monte Carlo on top of experimental observations

x_lim = [-2.1,2.1]
plt.figure(figsize=(7,9))
ax = plt.subplot(211)
plt.scatter(Wu_data_1_t[0:len(Wu_mean_data_1_neg)],Wu_mean_data_1_neg,color='w',edgecolor='k',linewidth=2,s=75)
plt.scatter(Wu_data_1_t[len(Wu_mean_data_1_neg):],Wu_mean_data_1_pos,color='w',edgecolor='k',linewidth=2,s=75)
plt.fill_between(Wu_data_1_t,Wu_data_1_RMS_upper,Wu_data_1_RMS_lower,color='grey',alpha=0.4)
plt.axhline(v_data_1_median,color='r',linewidth=3,label='Median')
plt.axhspan(v_data_1_median+v_data_1_iqr,v_data_1_median-v_data_1_iqr,color='r',alpha=0.2,label='IQR')
plt.axhline(v_data_1_median+v_data_1_90,color='r',linestyle='--',linewidth=3,label='$5^{th}$ to $95^{th}$')
plt.axhline(v_data_1_median-v_data_1_90,color='r',linestyle='--',linewidth=3)
plt.axvline(0,color='k',label='Initial Motion',linewidth = 2)
plt.legend(fontsize=14)
plt.xlim(x_lim)
plt.ylabel('Grain proximal velocity (m/s)',fontsize=14)
plt.xlabel('Time relative to incipient motion (s)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)

ax1 = plt.subplot(212)
plt.scatter(Wu_data_2_t[0:len(Wu_mean_data_2_neg)],Wu_mean_data_2_neg,color='w',edgecolor='k',linewidth=2,s=75)
plt.scatter(Wu_data_2_t[len(Wu_mean_data_2_neg):],Wu_mean_data_2_pos,color='w',edgecolor='k',linewidth=2,s=75)
plt.fill_between(Wu_data_2_t,Wu_data_2_RMS_upper,Wu_data_2_RMS_lower,color='grey',alpha=0.4)
plt.axhline(v_data_2_median,color='r',linewidth=3)
plt.axhspan(v_data_2_median+v_data_2_iqr,v_data_2_median-v_data_2_iqr,color='r',alpha=0.2)
plt.axhline(v_data_2_median+v_data_2_90,color='r',linestyle='--',linewidth=3)
plt.axhline(v_data_2_median-v_data_2_90,color='r',linestyle='--',linewidth=3)
plt.axvline(0,color='k',label='Initial Motion',linewidth = 2)
plt.ylabel('Grain proximal velocity (m/s)',fontsize=14)
plt.xlabel('Time relative to incipient motion (s)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.xlim(x_lim)
plt.tight_layout()
