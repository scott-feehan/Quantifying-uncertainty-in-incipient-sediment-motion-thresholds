#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:51:08 2022

@author: scottfeehan

This code generates figure 5.

This code is used to determine how changing assumed physical parameters will alter estimated entrainment thresholds. 
We vary effective firction coefficient (tan(phi) = mu) and grain protrusion (p) and determine the influence of assumed 
or measured physical parameters on estimating mathematical fit coefficients. 

Plots are of power law coefficients for critical velocity (u_c), shear velocity (u_s), shear stress (u_t), and Critcal 
Shields (u_tc) with the associated uncertainty when assuming different physical parameters. 

This code takes exceptionally long to run (20+ hr), be aware of the monte carlo iteration length and the number of grains 
tested when running script.  

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from scipy.stats import truncnorm

#%% Constants and parameters 

monte_carlo_step = 100000 # Monte Carlo iteration length
g = 9.81 # Gravity 
theta = 0.001 # Constant slope 
V_w_V = 1 # Fully submerged grain  
k_von = 0.407 # Von Karman constant

# Range of grain sizes tested 
Grain_size = np.arange(0.001,1.001,0.001)

# Range of roughness locations tested
ks_len = 10
k_s_range = np.arange(1,6.2,0.1)

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

# Range of mu and p tested 
mu_range = np.tan(np.deg2rad(np.arange(15,86,1)))
p_range = np.arange(0.1,1.1,0.1)
mu_stdv = 0.27  # from Booth
finalmu = mu_range 
finalsigma = mu_stdv
mu_stdv = np.sqrt( np.log( finalsigma**2 / finalmu**2 + 1))
mu_range = np.log(finalmu/np.exp(0.5*ln_mu_stdv**2))

# Assume all distributions have a truncated normal except mu which has a lognormal distribution (Booth et. al., 2014)
#%% Create function to generate truncated normal distributions for force balaance parameters 

def get_truncated_normal(upp,low, mean, sd): # Upper bound # Lower bound # Mean # Standard deviation
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#%% Generate distribution of respective force balance parameter

X = get_truncated_normal(rho_s_max ,rho_s_min,rho_s_mean,rho_s_stdv ) 
rho_s = X.rvs(monte_carlo_step) 

X = get_truncated_normal(rho_w_max,rho_w_min,rho_w_mean,rho_w_stdv)
rho_w = X.rvs(monte_carlo_step) 

X = get_truncated_normal(C_l_max,C_l_min,C_l_mean,C_l_stdv)
C_l = X.rvs(monte_carlo_step) 

X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
C_d = X.rvs(monte_carlo_step) 

#%% Define force balance 

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

#%% Calculate u_c and other threshold parameters 

v_FB_median = np.zeros([len(Grain_size),len(mu_range),len(p_range)])
v_FB_iqr = np.zeros([len(Grain_size),len(mu_range),len(p_range)])
v_FB_90 = np.zeros([len(Grain_size),len(mu_range),len(p_range)])

B_axis = Grain_size
A_axis = B_axis
C_axis = B_axis
V = (A_axis/2)*(B_axis/2)*(C_axis/2)*(4/3)*np.pi # assume perfect sphere volume 
A = (B_axis/2)*(C_axis/2)*np.pi # area 

u_shear_median = np.zeros([len(Grain_size),len(mu_range),len(p_range)])
u_shear_iqr = np.zeros(np.shape(u_shear_median))
u_shear_90 = np.zeros(np.shape(u_shear_median))

t_shear_stress_median = np.zeros(np.shape(u_shear_median))
t_shear_stress_iqr = np.zeros(np.shape(u_shear_median))
t_shear_stress_90 = np.zeros(np.shape(u_shear_median))

t_shields_stress_median = np.zeros(np.shape(u_shear_median))
t_shields_stress_iqr = np.zeros(np.shape(u_shear_median))
t_shields_stress_90 = np.zeros(np.shape(u_shear_median))

for i in range(0,len(Grain_size)):
    for j in range(0,len(mu_range)): 
        for h in range(0,len(p_range)):
            
            X = get_truncated_normal(p_max ,p_min,p_range[h],p_stdv) 
            p = X.rvs(monte_carlo_step)
            
            mu = np.random.lognormal(mu_range[j],mu_stdv[j],monte_carlo_step)            
            while True:
                temp = np.where((0>mu) | (mu>mu_max))
                mu[temp] = np.random.lognormal(mu_range[j],mu_stdv[j],len(temp[0]))
                if len(temp[0]) <1:
                    break 

            v_ForceBalance = ForceBalance(rho_s,rho_w,g,mu,C_l,C_d,theta,A_axis[i],B_axis[i],C_axis[i],V_w_V,V[i],A[i],p)
            v_FB_median[i,j,h] = np.nanmedian(v_ForceBalance)
            v_FB_iqr[i,j,h] = iqr(v_ForceBalance,nan_policy='omit')
            v_FB_90[i,j,h] = iqr(v_ForceBalance,rng=(5,95),nan_policy='omit')
            
            Velocity = v_ForceBalance
                       
            k_s = np.random.choice(k_s_range,len(Velocity))
            k_s = k_s*Grain_size[i]
            
            z_2 = np.zeros(np.size(k_s))

            for k in range(0,len(k_s)):
            
                z_2[k] = np.random.uniform(k_s[k]/30 + Grain_size[i],k_s[k],1)
            
            z_1 = z_2 - Grain_size[i]
            
            u_shear = (Velocity*k_von*(z_2 - z_1))/(((((k_s)/30)+z_2)*np.log(1 + ((30*z_2)/(k_s))) - z_2)
                     - ((((k_s)/30)+z_1)*np.log(1 + ((30*z_1)/(k_s))) - z_1)) # solve for shear value at each rougness layer 
            
            t_shear_stress = rho_w*(u_shear**2)
            t_shields_stress = (rho_w*(u_shear**2))/((rho_s - rho_w)*g*Grain_size[i])
    
            u_shear_median[i,j,h] = np.nanmedian(u_shear)
            u_shear_iqr[i,j,h] = iqr(u_shear,nan_policy='omit')
            u_shear_90[i,j,h] = iqr(u_shear,rng=(5,95),nan_policy='omit')
             
            t_shear_stress_median[i,j,h] = np.nanmedian(t_shear_stress)
            t_shear_stress_iqr[i,j,h] = iqr(t_shear_stress,nan_policy='omit')
            t_shear_stress_90[i,j,h] = iqr(t_shear_stress,rng=(5,95),nan_policy='omit')
                     
            t_shields_stress_median[i,j,h] = np.nanmedian(t_shields_stress)
            t_shields_stress_iqr[i,j,h] = iqr(t_shields_stress,nan_policy='omit')
            t_shields_stress_90[i,j,h] = iqr(t_shields_stress,rng=(5,95),nan_policy='omit')

v_FB_iqr = v_FB_iqr/2
v_FB_90 = v_FB_90/2
u_shear_iqr = u_shear_iqr/2
u_shear_90 = u_shear_90/2

t_shear_stress_iqr = t_shear_stress_iqr/2
t_shear_stress_90 = t_shear_stress_90/2
         
t_shields_stress_iqr = t_shields_stress_iqr/2
t_shields_stress_90 = t_shields_stress_90/2

#%% Determine power law coefficient for each mathematical fit of the threshold parameters

m_phi_p = np.zeros([len(mu_range),len(p_range)])
m_phi_p_iqr = np.zeros([len(mu_range),len(p_range)])

m_u_shear_phi_p = np.zeros([len(mu_range),len(p_range)])
m_u_shear_phi_p_iqr = np.zeros([len(mu_range),len(p_range)])

m_t_shear_stress_phi_p = np.zeros([len(mu_range),len(p_range)])
m_t_shear_stress_phi_p_iqr = np.zeros([len(mu_range),len(p_range)])

m_t_shields_stress_phi_p = np.zeros([len(mu_range),len(p_range)])
m_t_shields_stress_phi_p_iqr = np.zeros([len(mu_range),len(p_range)])

x = Grain_size

for j in range(0,len(mu_range)): 
    for h in range(0,len(p_range)):
        
        # Velocity 
        y = v_FB_median[:,j,h]
        y_iqr = v_FB_median[:,j,h] + v_FB_iqr[:,j,h]
        
        a,m = np.polyfit(np.log(x),np.log(y),1)
        m_phi_p[j,h] = np.round(np.e**m,decimals=3)
        
        a,m1 = np.polyfit(np.log(x),np.log(y_iqr),1)
        m_phi_p_iqr[j,h] = np.round(np.e**m1,decimals=3)
        
        # Shear velocity 
        y = u_shear_median[:,j,h]
        y_iqr = u_shear_median[:,j,h] + u_shear_iqr[:,j,h]
        
        a,m = np.polyfit(np.log(x),np.log(y),1)
        m_u_shear_phi_p[j,h] = np.round(np.e**m,decimals=3)
        
        a,m1 = np.polyfit(np.log(x),np.log(y_iqr),1)
        m_u_shear_phi_p_iqr[j,h] = np.round(np.e**m1,decimals=3) 
                
        # Shear stress 
        y = t_shear_stress_median[:,j,h]
        y_iqr = t_shear_stress_median[:,j,h] + t_shear_stress_iqr[:,j,h]
        
        a,m = np.polyfit(np.log(x),np.log(y),1)
        m_t_shear_stress_phi_p[j,h] = np.round(np.e**m,decimals=3)
        
        a,m1 = np.polyfit(np.log(x),np.log(y_iqr),1)
        m_t_shear_stress_phi_p_iqr[j,h] = np.round(np.e**m1,decimals=3) 
              
        # Shields stress 
        y = t_shields_stress_median[:,j,h]
        y_iqr = t_shields_stress_median[:,j,h] + t_shields_stress_iqr[:,j,h]
        
        a,m = np.polyfit(np.log(x),np.log(y),1)
        m_t_shields_stress_phi_p[j,h] = np.round(np.e**m,decimals=3)
        
        a,m1 = np.polyfit(np.log(x),np.log(y_iqr),1)
        m_t_shields_stress_phi_p_iqr[j,h] = np.round(np.e**m1,decimals=3) 

m_phi_p_iqr = m_phi_p_iqr - m_phi_p
m_u_shear_phi_p_iqr = m_u_shear_phi_p_iqr - m_u_shear_phi_p
m_t_shear_stress_phi_p_iqr = m_t_shear_stress_phi_p_iqr - m_t_shear_stress_phi_p
m_t_shields_stress_phi_p_iqr = m_t_shields_stress_phi_p_iqr - m_t_shields_stress_phi_p

#%% Smoothing the Shields estimates 

m_t_shields_stress_phi_p_iqr_smoothed = np.zeros(np.shape(m_t_shields_stress_phi_p_iqr))

for i in range(0,len(p_range)):
    data = m_t_shields_stress_phi_p_iqr[:,i]
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(data, kernel, mode='same')
    data_convolved[0] = data[0]
    data_convolved[1] = data[1]
    data_convolved[-1] = data[-1]
    data_convolved[-2] = data[-2]
    m_t_shields_stress_phi_p_iqr_smoothed[:,i] = data_convolved
    
m_t_shields_stress_phi_p_smoothed = np.zeros(np.shape(m_t_shields_stress_phi_p))

for i in range(0,len(p_range)):
    data = m_t_shields_stress_phi_p[:,i]
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(data, kernel, mode='same')
    data_convolved[0] = data[0]
    data_convolved[1] = data[1]
    data_convolved[-1] = data[-1]
    data_convolved[-2] = data[-2]
    m_t_shields_stress_phi_p_smoothed[:,i] = data_convolved

#%% Plot mathematical fit and uncertainty for u_c, u_s, tau, and tau_c 

colors = plt.cm.viridis(np.linspace(0,1,len(p_range))) 

mu_range = np.tan(np.deg2rad(np.arange(15,86,1))) # Change label back to degrees

text_x = 60.5
text_y = 0.25
label_y = 0.9

plt.figure(figsize=(16,18))

# Critical velocity 
plt.subplot(421)
for i in range(0,len(p_range)):
    
   ax =  plt.plot(np.rad2deg(np.arctan(mu_range)),m_phi_p[:,i],color=colors[i],linewidth=3)
    
ylim_high = np.max(m_phi_p)+((np.max(m_phi_p) - np.min(m_phi_p))*0.05)
ylim_low = np.min(m_phi_p)-((np.max(m_phi_p) - np.min(m_phi_p))*0.05)
plt.ylim([ylim_low,ylim_high])
plt.text(17.5,(ylim_high - ylim_low)*label_y + ylim_low,'a',fontsize=14,fontweight='bold')
plt.ylabel('Critical velocity coefficient, $m_c$',fontsize=14)
plt.xlabel('Effective friction angle, $\phi$ ($\N{DEGREE SIGN}$)',fontsize=14)
plt.xlim([min(np.rad2deg(np.arctan(mu_range))),max(np.rad2deg(np.arctan(mu_range)))])
plt.text(text_x,(ylim_high - ylim_low)*text_y + ylim_low,'$u_c = m_c D^{0.5}$',fontsize=18,bbox=dict(facecolor='white',alpha=0.5,edgecolor='none'))
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.tick_params(bottom=True,top=True,left=True,right=True,which='both')
plt.minorticks_on()

plt.subplot(422)
for i in range(0,len(p_range)):
    
    plt.plot(np.rad2deg(np.arctan(mu_range)),m_phi_p_iqr[:,i],label=np.round(p_range[i],decimals=2),color=colors[i],linewidth=3)

lg = plt.legend(fontsize=14,ncol=1,frameon=True,edgecolor='k',bbox_to_anchor=(1.04,1.03), loc="upper left",)
lg.set_title('Grain protrusion, $p_*$',prop={'size': 14})

ylim_high = np.max(m_phi_p_iqr)+((np.max(m_phi_p_iqr) - np.min(m_phi_p_iqr))*0.05)
ylim_low = np.min(m_phi_p_iqr)-((np.max(m_phi_p_iqr) - np.min(m_phi_p_iqr))*0.05)
plt.ylim([ylim_low,ylim_high])
plt.text(17.5,(ylim_high - ylim_low)*label_y + ylim_low,'b',fontsize=14,fontweight='bold')
plt.ylabel('$m_c$ uncertainty, $+/-$',fontsize=14)
plt.xlabel('Effective friction angle, $\phi$ ($\N{DEGREE SIGN}$)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.tick_params(bottom=True,top=True,left=True,right=True,which='both')
plt.minorticks_on()
plt.xlim([min(np.rad2deg(np.arctan(mu_range))),max(np.rad2deg(np.arctan(mu_range)))])

# Shear velocity 
plt.subplot(423)
for i in range(0,len(p_range)):
    
   ax =  plt.plot(np.rad2deg(np.arctan(mu_range)),m_u_shear_phi_p[:,i],color=colors[i],linewidth=3)

ylim_high = np.max(m_u_shear_phi_p)+((np.max(m_u_shear_phi_p) - np.min(m_u_shear_phi_p))*0.05)
ylim_low = np.min(m_u_shear_phi_p)-((np.max(m_u_shear_phi_p) - np.min(m_u_shear_phi_p))*0.05)
plt.ylim([ylim_low,ylim_high])
plt.text(17.5,(ylim_high - ylim_low)*label_y + ylim_low,'c',fontsize=14,fontweight='bold')
plt.ylabel('Shear velocity coefficient, $m_s$',fontsize=14)
plt.xlabel('Effective friction angle, $\phi$ ($\N{DEGREE SIGN}$)',fontsize=14)
plt.xlim([min(np.rad2deg(np.arctan(mu_range))),max(np.rad2deg(np.arctan(mu_range)))])
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.text(text_x,(ylim_high - ylim_low)*text_y + ylim_low,'$u_s = m_s D^{0.5}$',fontsize=18,bbox=dict(facecolor='white',alpha=0.5,edgecolor='none'))
plt.tick_params(bottom=True,top=True,left=True,right=True,which='both')
plt.minorticks_on()

plt.subplot(424)
for i in range(0,len(p_range)):
    
    plt.plot(np.rad2deg(np.arctan(mu_range)),m_u_shear_phi_p_iqr[:,i],color=colors[i],linewidth=3)
    
plt.ylabel('$m_s$ uncertainty, $+/-$',fontsize=14)
plt.xlabel('Effective friction angle, $\phi$ ($\N{DEGREE SIGN}$)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
ylim_high = np.max(m_u_shear_phi_p_iqr)+((np.max(m_u_shear_phi_p_iqr) - np.min(m_u_shear_phi_p_iqr))*0.05)
ylim_low = np.min(m_u_shear_phi_p_iqr)-((np.max(m_u_shear_phi_p_iqr) - np.min(m_u_shear_phi_p_iqr))*0.05)
plt.ylim([ylim_low,ylim_high])
plt.text(17.5,(ylim_high - ylim_low)*label_y + ylim_low,'d',fontsize=14,fontweight='bold')
plt.tick_params(bottom=True,top=True,left=True,right=True,which='both')
plt.minorticks_on()
plt.xlim([min(np.rad2deg(np.arctan(mu_range))),max(np.rad2deg(np.arctan(mu_range)))])


# Shear stress
plt.subplot(425)
for i in range(0,len(p_range)):
    
   ax =  plt.plot(np.rad2deg(np.arctan(mu_range)),m_t_shear_stress_phi_p[:,i],color=colors[i],linewidth=3)

ylim_high = np.max(m_t_shear_stress_phi_p)+((np.max(m_t_shear_stress_phi_p) - np.min(m_t_shear_stress_phi_p))*0.05)
ylim_low = np.min(m_t_shear_stress_phi_p)-((np.max(m_t_shear_stress_phi_p) - np.min(m_t_shear_stress_phi_p))*0.05)
plt.ylim([ylim_low,ylim_high])
plt.text(17.5,(ylim_high - ylim_low)*label_y + ylim_low,'e',fontsize=14,fontweight='bold')
plt.text(text_x,(ylim_high - ylim_low)*text_y + ylim_low,r'$\tau_b = m_{\tau} D$',fontsize=18,bbox=dict(facecolor='white',alpha=0.5,edgecolor='none'))
plt.ylabel(r'Shear stress coefficient, $m_{\tau}$',fontsize=14)
plt.xlabel('Effective friction angle, $\phi$ ($\N{DEGREE SIGN}$)',fontsize=14)
plt.xlim([min(np.rad2deg(np.arctan(mu_range))),max(np.rad2deg(np.arctan(mu_range)))])
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.tick_params(bottom=True,top=True,left=True,right=True,which='both')
plt.minorticks_on()

plt.subplot(426)
for i in range(0,len(p_range)):
    
    plt.plot(np.rad2deg(np.arctan(mu_range)),m_t_shear_stress_phi_p_iqr[:,i],color=colors[i],linewidth=3)

ylim_high = np.max(m_t_shear_stress_phi_p_iqr)+((np.max(m_t_shear_stress_phi_p_iqr) - np.min(m_t_shear_stress_phi_p_iqr))*0.05)
ylim_low = np.min(m_t_shear_stress_phi_p_iqr)-((np.max(m_t_shear_stress_phi_p_iqr) - np.min(m_t_shear_stress_phi_p_iqr))*0.05)
plt.ylim([ylim_low,ylim_high])
plt.text(17.5,(ylim_high - ylim_low)*label_y + ylim_low,'f',fontsize=14,fontweight='bold')
plt.ylabel(r'$m_{\tau}$ uncertainty, $+/-$',fontsize=14)
plt.xlabel('Effective friction angle, $\phi$ ($\N{DEGREE SIGN}$)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.tick_params(bottom=True,top=True,left=True,right=True,which='both')
plt.minorticks_on()
plt.xlim([min(np.rad2deg(np.arctan(mu_range))),max(np.rad2deg(np.arctan(mu_range)))])
    
# Shields stress 
plt.subplot(427)
for i in range(0,len(p_range)):
   ax =  plt.plot(np.rad2deg(np.arctan(mu_range)),m_t_shields_stress_phi_p_smoothed[:,i],color=colors[i],linewidth=3)#,label=np.round(p_range[i],decimals=2)

ylim_high = np.max(m_t_shields_stress_phi_p_smoothed)+((np.max(m_t_shields_stress_phi_p_smoothed) - np.min(m_t_shields_stress_phi_p_smoothed))*0.05)
ylim_low = np.min(m_t_shields_stress_phi_p_smoothed)-((np.max(m_t_shields_stress_phi_p_smoothed) - np.min(m_t_shields_stress_phi_p_smoothed))*0.05)
plt.ylim([ylim_low,ylim_high])
plt.text(17.5,(ylim_high - ylim_low)*label_y + ylim_low,'g',fontsize=14,fontweight='bold')
plt.ylabel(r'Shields stress, $\tau^*_c$',fontsize=14)
plt.xlabel('Effective friction angle, $\phi$ ($\N{DEGREE SIGN}$)',fontsize=14)
plt.xlim([min(np.rad2deg(np.arctan(mu_range))),max(np.rad2deg(np.arctan(mu_range)))])
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.tick_params(bottom=True,top=True,left=True,right=True,which='both')
plt.minorticks_on()

plt.subplot(428)
for i in range(0,len(p_range)):
    
    plt.plot(np.rad2deg(np.arctan(mu_range)),m_t_shields_stress_phi_p_iqr_smoothed[:,i],color=colors[i],linewidth=3)
    
ylim_high = np.max(m_t_shields_stress_phi_p_iqr_smoothed)+((np.max(m_t_shields_stress_phi_p_iqr_smoothed) - np.min(m_t_shields_stress_phi_p_iqr_smoothed))*0.05)
ylim_low = np.min(m_t_shields_stress_phi_p_iqr_smoothed)-((np.max(m_t_shields_stress_phi_p_iqr_smoothed) - np.min(m_t_shields_stress_phi_p_iqr_smoothed))*0.05)
plt.ylim([ylim_low,ylim_high])
plt.text(17.5,(ylim_high - ylim_low)*label_y + ylim_low,'h',fontsize=14,fontweight='bold')
plt.ylabel(r'$\tau^*_c$ uncertainty, $+/-$',fontsize=14)
plt.xlabel('Effective friction angle, $\phi$ ($\N{DEGREE SIGN}$)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.tick_params(bottom=True,top=True,left=True,right=True,which='both')
plt.minorticks_on()
plt.xlim([min(np.rad2deg(np.arctan(mu_range))),max(np.rad2deg(np.arctan(mu_range)))])
plt.tight_layout()


