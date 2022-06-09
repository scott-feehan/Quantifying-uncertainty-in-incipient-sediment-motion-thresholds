#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:33:18 2022

@author: scottfeehan

Comparison of field and flume compliation with theoretical estimates of shear velocity 

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import truncnorm
from scipy.stats import iqr

#%% 

filepath = 'Shields_comp_field.csv' 
Shields_comp_field = pd.read_csv(filepath)

filepath = 'Shields_comp_flume.csv' 
Shields_comp_flume = pd.read_csv(filepath)

#%% 

Shields_stress_field = Shields_comp_field['Shields stress'].to_numpy()
Slope_field = Shields_comp_field['slope (m/m)'].to_numpy()
D50_field = Shields_comp_field['median grain size (cm)'].to_numpy()/100
Flow_depth_field = Shields_comp_field['flow depth (cm)'].to_numpy()/100
Sediment_density_field = Shields_comp_field['density (g/cm3)'].to_numpy()*1000
Reynolds_field = Shields_comp_field['Re (reported)'].to_numpy()
D84_field = (Shields_comp_field['median grain size (cm)'].to_numpy() + Shields_comp_field['phi'].to_numpy())/100

Shields_stress_flume = Shields_comp_flume['Shields stress'].to_numpy()
Slope_flume = Shields_comp_flume['slope (m/m)'].to_numpy()
D50_flume = Shields_comp_flume['median grain size (cm)'].to_numpy()/100
Flow_depth_flume = Shields_comp_flume['flow depth (cm)'].to_numpy()/100
Sediment_density_flume = Shields_comp_flume['density (g/cm3)'].to_numpy()*1000
Reynolds_flume = Shields_comp_flume['Re (reported)'].to_numpy()
D84_flume = (Shields_comp_flume['median grain size (cm)'].to_numpy() + Shields_comp_flume['phi'].to_numpy())/100

#%% Thresholding with slope and grain size 

temp = np.where((Slope_field <= 0.05) & (D50_field >= 0.001))

Shields_stress_field = Shields_stress_field[temp[0]]
Slope_field = Slope_field[temp]
D50_field = D50_field[temp]
Flow_depth_field = Flow_depth_field[temp]
Sediment_density_field = Sediment_density_field[temp]
Reynolds_field = Reynolds_field[temp]
D84_field = D84_field[temp]


temp = np.where((Slope_flume <= 0.05) & (D50_flume >= 0.001))

Shields_stress_flume = Shields_stress_flume[temp[0]]
Slope_flume = Slope_flume[temp]
D50_flume = D50_flume[temp]
Flow_depth_flume = Flow_depth_flume[temp]
Sediment_density_flume = Sediment_density_flume[temp]
Reynolds_flume = Reynolds_flume[temp]
D84_flume = D84_flume[temp]


#%% Assumed force balance parameter mean, minimum, maximum, and standard deviation

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
mu_mean = 2.74
mu_min = 0.27
mu_max =  11.4
mu_stdv = 0.58

# Converting mu mean and standard deviation to lognormal space
ln_mu_stdv = np.sqrt( np.log( mu_stdv**2 / mu_mean**2 + 1))
ln_mu = np.log(mu_mean/np.exp(0.5*ln_mu_stdv**2))

#%% Create function to generate truncated normal distributions for force balaance parameters 

def get_truncated_normal(upp,low, mean, sd): # Upper bound # Lower bound # Mean # Standard deviation
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#%% 
    
monte_carlo_step = 100000 # Monte Carlo iteration length

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

#%% Constants
        
g = 9.81 # Gravity 
theta = 0.001 # Constant slope 
V_w_V = 1 # Fully submerged grain  
k_von = 0.407 # Von Karman constant

#%% Convert reported tau_c to u_*

Shear_velocity_field = ((Shields_stress_field*(Sediment_density_field - rho_w_mean)*g*D50_field)/rho_w_mean)**0.5

Shear_velocity_flume = ((Shields_stress_flume*(Sediment_density_flume - rho_w_mean)*g*D50_flume)/rho_w_mean)**0.5

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

#%% Estimate theoretical threshold for D50 of each compilation using a Monte Carlo simulation and compilation constraints

v_ForceBalance_field = np.zeros([int(monte_carlo_step),len(D50_field)])

B_axis_field = D50_field
A_axis_field = B_axis_field
C_axis_field = B_axis_field
V_field = (A_axis_field/2)*(B_axis_field/2)*(C_axis_field/2)*(4/3)*np.pi
A_field = (A_axis_field/2)*(B_axis_field/2)*np.pi
theta_field = Slope_field
rho_s_field = Sediment_density_field

v_ForceBalance_flume = np.zeros([int(monte_carlo_step),len(D50_flume)])

B_axis_flume = D50_flume
A_axis_flume = B_axis_flume
C_axis_flume = B_axis_flume
V_flume = (A_axis_flume/2)*(B_axis_flume/2)*(C_axis_flume/2)*(4/3)*np.pi
A_flume = (A_axis_flume/2)*(B_axis_flume/2)*np.pi
theta_flume = Slope_flume
rho_s_flume = Sediment_density_flume

for i in range(0,len(D50_flume)):
        
    v_ForceBalance_flume[:,i] = ForceBalance(rho_s_flume[i],rho_w,g,mu,C_l,C_d,theta_flume[i],A_axis_flume[i],B_axis_flume[i],C_axis_flume[i],V_w_V,V_flume[i],A_flume[i],p)
    
    if i < len(D50_field):
        
        v_ForceBalance_field[:,i] = ForceBalance(rho_s_field[i],rho_w,g,mu,C_l,C_d,theta_field[i],A_axis_field[i],B_axis_field[i],C_axis_field[i],V_w_V,V_field[i],A_field[i],p)
    else:
        continue
    
v_FB_median_field = np.zeros(len(D50_field))
v_FB_iqr_field = np.zeros(len(D50_field))

v_FB_median_flume = np.zeros(len(D50_flume))
v_FB_iqr_flume = np.zeros(len(D50_flume))

for i in range(0,len(D50_flume)):
    
    v_FB_median_flume[i] = np.nanmedian(v_ForceBalance_flume[:,i])
    v_FB_iqr_flume[i] = iqr(v_ForceBalance_flume[:,i],rng=(5, 95),nan_policy='omit')

    if i < len(D50_field):

        v_FB_median_field[i] = np.nanmedian(v_ForceBalance_field[:,i])
        v_FB_iqr_field[i] = iqr(v_ForceBalance_field[:,i],rng=(5, 95),nan_policy='omit')

    else:
        continue

v_FB_iqr_field = v_FB_iqr_field/2
v_FB_iqr_flume = v_FB_iqr_flume/2

#%% Calculate u_s, tau, and tau_c from simulated critical velocity

k_s_location = 1
Velocity = v_ForceBalance_field
Grain_size = D50_field
u_shear = np.zeros([len(Grain_size),monte_carlo_step]) # Place to store shear velocity values 
u_shear_median = np.zeros([len(Grain_size)])
u_shear_iqr = np.zeros([len(Grain_size)])

for i in range(0,len(D50_field)):  
    
    k_s = Grain_size[i]
    
    z_2 = k_s + k_s/30 # top of the flow field is reported depth
    z_1 = z_2 - Grain_size[i] # bottom of the roughness layer is base of the roughness layer 

    u_shear[i,:] = (Velocity[:,i]*k_von*(z_2 - z_1))/(((((k_s)/30)+z_2)*np.log(1 + ((30*z_2)/(k_s))) - z_2)
             - ((((k_s)/30)+z_1)*np.log(1 + ((30*z_1)/(k_s))) - z_1)) # solve for shear value at each rougness layer 

    u_shear_median[i] = np.nanmedian(u_shear[i,:])
    u_shear_iqr[i] = iqr(u_shear[i,:],nan_policy='omit')

u_shear_iqr = u_shear_iqr/2

tau_shear_field = ((u_shear_median**2)*rho_w_mean)/((rho_s_field - rho_w_mean)*g*Grain_size)
u_shear_median_field = u_shear_median 
u_shear_iqr_field = u_shear_iqr/2

Velocity = v_ForceBalance_flume

Grain_size = D50_flume
u_shear = np.zeros([len(Grain_size),monte_carlo_step]) # Place to store shear velocity values 
u_shear_high = np.zeros([len(Grain_size),monte_carlo_step]) # Place to store shear velocity values 
u_shear_low = np.zeros([len(Grain_size),monte_carlo_step]) # Place to store shear velocity values 

u_shear_median = np.zeros([len(Grain_size)])
u_shear_median_high = np.zeros([len(Grain_size)])
u_shear_median_low = np.zeros([len(Grain_size)])

u_shear_iqr = np.zeros([len(Grain_size)])

for i in range(0,len(D50_flume)): 
    
    k_s = Grain_size[i]
    
    z_2 = k_s + k_s/30
    z_1 = z_2 - Grain_size[i]

    u_shear[i,:] = (Velocity[:,i]*k_von*(z_2 - z_1))/(((((k_s)/30)+z_2)*np.log(1 + ((30*z_2)/(k_s))) - z_2)
             - ((((k_s)/30)+z_1)*np.log(1 + ((30*z_1)/(k_s))) - z_1)) 

    u_shear_median[i] = np.nanmedian(u_shear[i,:])
    u_shear_iqr[i] = iqr(u_shear[i,:],nan_policy='omit')

u_shear_iqr = u_shear_iqr/2

tau_shear_flume = ((u_shear_median**2)*rho_w_mean)/((rho_s_flume - rho_w_mean)*g*Grain_size)
tau_shear_flume_plus = (((u_shear_median+u_shear_iqr)**2)*rho_w_mean)/((rho_s_flume - rho_w_mean)*g*Grain_size)
u_shear_median_flume = u_shear_median
u_shear_iqr_flume = u_shear_iqr/2

#%% Calculate theoretical critical velocity and other flow parameters to determine where the reported data sits relative to theory

grain_range = np.arange(0.0001,1.001,0.001)

v_ForceBalance = np.zeros([int(monte_carlo_step),len(grain_range)])

B_axis = grain_range
A_axis = B_axis
C_axis = B_axis
V = (A_axis/2)*(B_axis/2)*(C_axis/2)*(4/3)*np.pi
A = (A_axis/2)*(B_axis/2)*np.pi
theta = 10**-3

for i in range(0,len(grain_range)):
        
    v_ForceBalance[:,i] = ForceBalance(rho_s,rho_w,g,mu,C_l,C_d,theta,A_axis[i],B_axis[i],C_axis[i],V_w_V,V[i],A[i],p)

v_FB_median = np.zeros(len(grain_range))
v_FB_iqr = np.zeros(len(grain_range))

for i in range(0,len(grain_range)):
    
    v_FB_median[i] = np.nanmedian(v_ForceBalance[:,i])
    v_FB_iqr[i] = iqr(v_ForceBalance[:,i],rng=(5, 95),nan_policy='omit')

v_FB_iqr = v_FB_iqr/2

Grain_size_plot_median = v_FB_median
Grain_size_plot_iqr = v_FB_iqr

#%% 

Velocity = v_ForceBalance
k_s_location = 1 

Grain_size = grain_range
u_shear = np.zeros([len(Grain_size),monte_carlo_step]) 
u_shear_high = np.zeros([len(Grain_size),monte_carlo_step])  
u_shear_low = np.zeros([len(Grain_size),monte_carlo_step]) 

u_shear_median = np.zeros([len(Grain_size)])
u_shear_median_high = np.zeros([len(Grain_size)])
u_shear_median_low = np.zeros([len(Grain_size)])

u_shear_iqr = np.zeros([len(Grain_size)])
u_shear_90 = np.zeros([len(Grain_size)])

u_shear_iqr_high = np.zeros([len(Grain_size)])
u_shear_90_high = np.zeros([len(Grain_size)])

u_shear_iqr_low = np.zeros([len(Grain_size)])
u_shear_90_low = np.zeros([len(Grain_size)])

for i in range(0,len(Grain_size)): 
    
    k_s = Grain_size[i]
    
    z_2 = k_s*k_s_location 
    z_1 = z_2 - Grain_size[i]

    u_shear[i,:] = (Velocity[:,i]*k_von*(z_2 - z_1))/(((((k_s)/30)+z_2)*np.log(1 + ((30*z_2)/(k_s))) - z_2)
             - ((((k_s)/30)+z_1)*np.log(1 + ((30*z_1)/(k_s))) - z_1)) 

    u_shear_median[i] = np.nanmedian(u_shear[i,:])
    u_shear_iqr[i] = iqr(u_shear[i,:],nan_policy='omit')
    u_shear_90[i] = iqr(u_shear[i,:],nan_policy='omit',rng=(5, 95))

    z_2 = k_s
    z_1 = z_2 - Grain_size[i]

    u_shear_high[i,:] = (Velocity[:,i]*k_von*(z_2 - z_1))/(((((k_s)/30)+z_2)*np.log(1 + ((30*z_2)/(k_s))) - z_2)
        - ((((k_s)/30)+z_1)*np.log(1 + ((30*z_1)/(k_s))) - z_1)) 

    u_shear_median_high[i] = np.nanmedian(u_shear_high[i,:])
    u_shear_iqr_high[i] = iqr(u_shear_high[i,:],nan_policy='omit')
    u_shear_90_high[i] = iqr(u_shear_high[i,:],nan_policy='omit',rng=(5, 95))
    
    z_2 = k_s/30 + Grain_size[i] 
    z_1 = z_2 - Grain_size[i]

    u_shear_low[i,:] = (Velocity[:,i]*k_von*(z_2 - z_1))/(((((k_s)/30)+z_2)*np.log(1 + ((30*z_2)/(k_s))) - z_2)
        - ((((k_s)/30)+z_1)*np.log(1 + ((30*z_1)/(k_s))) - z_1)) 
    
    u_shear_median_low[i] = np.nanmedian(u_shear_low[i,:])
    u_shear_iqr_low[i] = iqr(u_shear_low[i,:],nan_policy='omit')#,rng=(5, 95))
    u_shear_90_low[i] = iqr(u_shear_low[i,:],nan_policy='omit',rng=(5, 95))

u_shear_1_1 = u_shear_median
u_shear_1_1_iqr = u_shear_iqr/2
u_shear_1_1_90 =u_shear_90/2

u_shear_1_1_high = u_shear_median_high
u_shear_1_1_iqr_high = u_shear_iqr_high/2
u_shear_1_1_90_high = u_shear_90_high/2

u_shear_1_1_low = u_shear_median_low
u_shear_1_1_iqr_low = u_shear_iqr_low/2
u_shear_1_1_90_low = u_shear_90_low/2

u_shear_1_1_iqr = u_shear_iqr/2

#%% 

plt.figure(figsize=(9,5)) 
plt.fill_between(u_shear_1_1,u_shear_1_1+u_shear_1_1_iqr,u_shear_1_1-u_shear_1_1_iqr,color='r',alpha=0.2,label='IQR')
plt.scatter(u_shear_median_flume ,Shear_velocity_flume,c=Shields_stress_flume,edgecolors='k',marker='o',label='Field',s=75)
plt.scatter(u_shear_median_field ,Shear_velocity_field,c=Shields_stress_field,edgecolors='k',marker='^',label='Flume',s=75)
plt.plot(u_shear_1_1,u_shear_1_1,'r',label='1:1')
plt.plot(u_shear_1_1,u_shear_1_1+u_shear_1_1_90,'r--',alpha=1,label='95 to 5')
plt.plot(u_shear_1_1,u_shear_1_1-u_shear_1_1_90,'r--',alpha=1)
plt.ylabel('Reported shear velocity ($u_*$)',fontsize=14)
plt.xlabel('Theoretical shear velocity ($u_*$)',fontsize=14)
plt.clim(min(np.append(Shields_stress_field,Shields_stress_flume)),0.09)
cbar = plt.colorbar(orientation="vertical",pad=0.05)
cbar.ax.tick_params(labelsize=14,size=6)
cbar.ax.set_yticklabels(labels=('0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','>0.09'),fontsize=14)
cbar.set_label(r'$\tau^*_c$', fontsize=16)
plt.legend(fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.loglog()
plt.xlim([0.014,0.44])
plt.ylim([0.009,0.7])
plt.tight_layout()

#%% Determine the number of grains within the IQR and 5th to 95th confidince interal

# Flume iqr
IQR_upper = np.interp(u_shear_median_flume,u_shear_1_1, u_shear_1_1 + u_shear_1_1_iqr)
IQR_lower = np.interp(u_shear_median_flume,u_shear_1_1, u_shear_1_1 - u_shear_1_1_iqr)
temp = np.where((Shear_velocity_flume <= IQR_upper) & (Shear_velocity_flume >= IQR_lower))
flume_len_iqr = len(temp[0])/len(Shear_velocity_flume)

# Flume 90 percentile
upper_90 = np.interp(u_shear_median_flume,u_shear_1_1, u_shear_1_1 + u_shear_1_1_90)
lower_90 = np.interp(u_shear_median_flume,u_shear_1_1, u_shear_1_1 - u_shear_1_1_90)
temp = np.where((Shear_velocity_flume <= upper_90) & (Shear_velocity_flume >= lower_90))
flume_len_90 = len(temp[0])/len(Shear_velocity_flume)

# Field iqr 
IQR_upper = np.interp(u_shear_median_field,u_shear_1_1, u_shear_1_1 + u_shear_1_1_iqr)
IQR_lower = np.interp(u_shear_median_field,u_shear_1_1, u_shear_1_1 - u_shear_1_1_iqr)
temp = np.where((Shear_velocity_field <= IQR_upper) & (Shear_velocity_field >= IQR_lower))
field_len_iqr = len(temp[0])/len(Shear_velocity_field)

#field 90
upper_90 = np.interp(u_shear_median_field,u_shear_1_1, u_shear_1_1 + u_shear_1_1_90)
lower_90 = np.interp(u_shear_median_field,u_shear_1_1, u_shear_1_1 - u_shear_1_1_90)
temp = np.where((Shear_velocity_field <= upper_90) & (Shear_velocity_field >= lower_90))
field_len_90 = len(temp[0])/len(Shear_velocity_field)

print('Flume within IQR = '+str(np.round(flume_len_iqr*100,decimals=2))+'%')
print('Flume within 90 = '+str(np.round(flume_len_90*100,decimals=2))+'%')

print('Field within IQR = '+str(np.round(field_len_iqr*100,decimals=2))+'%')
print('Field within 90 = '+str(np.round(field_len_90*100,decimals=2))+'%')

