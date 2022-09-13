#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:07:24 2022

@author: scottfeehan

This code generates figure 2.

Converting  critical velocty (u_c) to shear velocity (u_s), shear stress (tau) and Shield's stress (tau_c). 
Assuming that the roughness layer (k_s) thickness is equally probable to be from  k_s  = 1 *  D  to  k_s  = 6.1 *  D  
where  D  is grain size. A k_s value is randomly selected from that range for each iteration of the Monte Carlo 
simulation. Within that k_s, the top of the grain is randomly selected, with equal probability, to be located 
from  k_s/30 +  D  to  k_s.

Note: This code may take multiple hours to run at current monte carlo iteration length (monte_carlo_step)
and the number of grains (Grain_size) tested when running code. To reduce run time, decrease one or both 
of these parameters. 

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from scipy.stats import truncnorm
import matplotlib as mpl

#%% Constants and parameters 

monte_carlo_step = 100000 # Monte Carlo iteration length
g = 9.81 # Gravity 
theta = 0.001 # Constant slope 
V_w_V = 1 # Fully submerged grain  
k_von = 0.41 # Von Karman constant
Grain_size = np.arange(0.001,1.001,0.001)
Slope_value = 10**-3
Slope = np.ones(np.shape(Grain_size))*Slope_value

ks_len = 10 # Discritization of locations within the roughness layer where the grain can sit
k_s_range = np.arange(1,6.2,0.1) # Range of roughness layer "ks" multiplier values

# Sediment density 
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

# Assume all distributions have a truncated normal except mu which has a lognormal distribution (Booth et. al., 2014)

#%% Create function to generate truncated normal distributions for force balaance parameters 

def get_truncated_normal(upp,low, mean, sd): # Upper bound # Lower bound # Mean # Standard deviation
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#%% 
    
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

#%% 

v_ForceBalance = np.zeros([int(monte_carlo_step),len(Grain_size)])

B_axis = Grain_size
A_axis = B_axis
C_axis = B_axis
V = (A_axis/2)*(B_axis/2)*(C_axis/2)*(4/3)*np.pi
A = (A_axis/2)*(B_axis/2)*np.pi
theta = Slope

for i in range(0,len(Grain_size)): # Monte Carlo simulation across grain sizes
    
    v_ForceBalance[:,i] = ForceBalance(rho_s,rho_w,g,mu,C_l,C_d,theta[i],A_axis[i],B_axis[i],C_axis[i],V_w_V,V[i],A[i],p)
    
# Calculate statistics of distribution 
v_FB_median = np.zeros(len(Grain_size))
v_FB_iqr = np.zeros(len(Grain_size))
v_FB_90 = np.zeros(len(Grain_size))

for i in range(0,len(Grain_size)):
    
    v_FB_median[i] = np.nanmedian(v_ForceBalance[:,i])
    v_FB_iqr[i] = iqr(v_ForceBalance[:,i],nan_policy='omit')
    v_FB_90[i] = iqr(v_ForceBalance[:,i],rng=(5,95),nan_policy='omit')

v_FB_iqr = v_FB_iqr/2
v_FB_90 = v_FB_90/2

#%% 

u_shear = np.zeros([len(Grain_size),len(v_ForceBalance)])
t_shear_stress = np.zeros(np.shape(u_shear))
t_shields_stress = np.zeros(np.shape(u_shear))

for h in range(0,len(Grain_size)):

    Velocity = v_ForceBalance[:,h] # Select single grain size Monte Carlo simulation
    
    for i in range (0,len(Velocity)): # For each estimated velocity in the Monte Carlo
        
        k_s = np.random.choice(k_s_range,1) # Make no assumption of probable roughness layer, select from uniform distribution
        k_s = k_s*Grain_size[h] # Multiply by the individual grain size to determine roughness layer thickness
        
        z_2 = np.random.uniform(k_s/30 + Grain_size[h],k_s,1) # Determine location of the top of the grain within the rouhgness layer
        
        z_1 = z_2 - Grain_size[h] # Location of the bottom of the grain is top of grain minus grain size
        
        u_shear[h,i] = (Velocity[i]*k_von*(z_2 - z_1))/(((((k_s)/30)+z_2)*np.log(1 + ((30*z_2)/(k_s))) - z_2)
                 - ((((k_s)/30)+z_1)*np.log(1 + ((30*z_1)/(k_s))) - z_1)) # solve for shear value at each randomly selected rougness layer 
        
        t_shear_stress[h,i] = rho_w[i]*(u_shear[h,i]**2) # Convert to shear stress 
        t_shields_stress[h,i] = (rho_w[i]*(u_shear[h,i]**2))/((rho_s[i] - rho_w[i])*g*Grain_size[h]) # Convert to Shield's stress 

#%% Calculate median, IQR, and 5th to 95th percent confidence interval for u_s, tau, and tau_c across grain size range specified 
        
u_shear_median = np.zeros(np.size(Grain_size))
u_shear_iqr = np.zeros(np.size(Grain_size))
u_shear_90 = np.zeros(np.size(Grain_size))

t_shear_stress_median = np.zeros(np.size(Grain_size))
t_shear_stress_iqr = np.zeros(np.size(Grain_size))
t_shear_stress_90 = np.zeros(np.size(Grain_size))

t_shields_stress_median = np.zeros(np.size(Grain_size))
t_shields_stress_iqr = np.zeros(np.size(Grain_size))
t_shields_stress_90 = np.zeros(np.size(Grain_size))

for i in range(0,len(Grain_size)):
    
    u_shear_median[i] = np.nanmedian(u_shear[i,:])
    u_shear_iqr[i] =  iqr(u_shear[i,:],nan_policy='omit')
    u_shear_90[i] =  iqr(u_shear[i,:],rng=(5,95),nan_policy='omit')
    
    t_shear_stress_median[i] = np.nanmedian(t_shear_stress[i,:])
    t_shear_stress_iqr[i] = iqr(t_shear_stress[i,:],nan_policy='omit')
    t_shear_stress_90[i] = iqr(t_shear_stress[i,:],rng=(5,95),nan_policy='omit')


    t_shields_stress_median[i] = np.nanmedian(t_shields_stress[i,:])
    t_shields_stress_iqr[i] = iqr(t_shields_stress[i,:],nan_policy='omit')
    t_shields_stress_90[i] = iqr(t_shields_stress[i,:],rng=(5,95),nan_policy='omit')

u_shear_iqr = u_shear_iqr/2
u_shear_90 = u_shear_90/2
t_shear_stress_iqr = t_shear_stress_iqr/2
t_shear_stress_90 = t_shear_stress_90/2
t_shields_stress_iqr = t_shields_stress_iqr/2
t_shields_stress_90 = t_shields_stress_90/2
      

#%% Determine relationship between grain size and the median specified flow parameter and uncertainty (IQR) 
# Critical velocity

x = Grain_size
y = v_FB_median
y_iqr = v_FB_median + v_FB_iqr

a,m = np.polyfit(np.log(x),np.log(y),1) # Fit between log of grain size and flow parameter 
m_v_median = np.round(np.e**m,decimals=3) # Convert from log-log linear space to calculate Power Law coefficeint 

a,m = np.polyfit(np.log(x),np.log(y_iqr),1) # Repeat fit for uncertainty in flow parameter
m_v_iqr= np.round(np.e**m,decimals=3)

#%% Shear velocity

y = u_shear_median
y_iqr = u_shear_median + u_shear_iqr

a,m = np.polyfit(np.log(x),np.log(y),1)
m_shear = np.round(np.e**m,decimals=3)

a,m = np.polyfit(np.log(x),np.log(y_iqr),1)
m_shear_iqr = np.round(np.e**m,decimals=3)
   
#%% Shear stress

y = t_shear_stress_median
y_iqr = t_shear_stress_median + t_shear_stress_iqr

a,m = np.polyfit(np.log(x),np.log(y),1)
m_t_shear = np.round(np.e**m,decimals=3)

a,m = np.polyfit(np.log(x),np.log(y_iqr),1)
m_t_shear_iqr = np.round(np.e**m,decimals=3)

#%% Shields Stress

y = t_shields_stress_median
y_iqr = t_shields_stress_median + t_shields_stress_iqr

a,m = np.polyfit(np.log(x),np.log(y),1)
m_t_shields = np.round(np.e**m,decimals=3)

a,m = np.polyfit(np.log(x),np.log(y_iqr),1)
m_t_shields_iqr = np.round(np.e**m,decimals=3)

#%% Plotting all the estimated u_c across the range of roughness layer heights and grain locations within the roughness layer. 

Grain_size_matrix = np.ones(np.shape(v_ForceBalance)) # Empty array the size of the Monte Carlo Simulation for force balance

for i in range(0,len(Grain_size)): 
    Grain_size_matrix[:,i] = Grain_size_matrix[:,i]*Grain_size[i] # Grain size for each iteration to plot velocity value as a heatmap
    
temp_boulder_matrix = Grain_size_matrix.reshape(-1) # Reshape array of grain size for plotting 
temp_v_matrix = v_ForceBalance.reshape(-1) # Reshape array of estimated velocity for plotting 

v_critical = np.zeros([len(temp_v_matrix),2]) # Combine grain size and velocity to single array
v_critical[:,0] = temp_boulder_matrix 
v_critical[:,1] = temp_v_matrix 

x_space = Grain_size # Create bin spacing for x axis# Create bin spacing for x axis
y_space = np.linspace((min(v_critical[:,1])), (max(v_critical[:,1])),len(Grain_size)) # Create bin spacing for y axis with same number of bins as x axis

#%% 

u_shear_matrix = np.zeros([len(Grain_size),len(u_shear[0,:].reshape(-1))])

t_shear_matrix = np.zeros([len(Grain_size),len(t_shear_stress[0,:].reshape(-1))])

t_shields_matrix = np.zeros([len(Grain_size),len(t_shields_stress[0,:].reshape(-1))])

for i in range(0,len(Grain_size)):
    u_shear_matrix[i,:] = u_shear[i,:].reshape(-1)
    t_shear_matrix[i,:] = t_shear_stress[i,:].reshape(-1)
    t_shields_matrix[i,:] = t_shields_stress[i,:].reshape(-1)

#%%
    
Grain_size_matrix = np.ones(np.shape(u_shear_matrix)) # Generate an array to store grain sizes in for each 

for i in range(0,len(Grain_size)):
    Grain_size_matrix[i,:] = Grain_size_matrix[i,:]*Grain_size[i] 

temp_boulder_matrix = Grain_size_matrix.reshape(-1)
    
# Shear velocity
temp_v_matrix = u_shear_matrix.reshape(-1)
u_shear_heatmap = np.zeros([len(temp_v_matrix),2])
u_shear_heatmap[:,0] = temp_boulder_matrix
u_shear_heatmap[:,1] = temp_v_matrix

u_shear_x_space = Grain_size
u_shear_y_space = np.linspace((min(u_shear_heatmap[:,1])), (max(u_shear_heatmap[:,1])),len(Grain_size))

# Shear stress 
temp_v_matrix = t_shear_matrix.reshape(-1)
t_shear_heatmap = np.zeros([len(temp_v_matrix),2])
t_shear_heatmap[:,0] = temp_boulder_matrix
t_shear_heatmap[:,1] = temp_v_matrix

t_shear_x_space = Grain_size
t_shear_y_space = np.linspace((min(t_shear_heatmap[:,1])), (max(t_shear_heatmap[:,1])),len(Grain_size))

# Shields stress 
temp_v_matrix = t_shields_matrix.reshape(-1)
t_shields_heatmap = np.zeros([len(temp_v_matrix),2])
t_shields_heatmap[:,0] = temp_boulder_matrix
t_shields_heatmap[:,1] = temp_v_matrix

t_shields_x_space = Grain_size
t_shields_y_space = np.linspace((min(t_shields_heatmap[:,1])), (max(t_shields_heatmap[:,1])),len(Grain_size))

#%% finding fit to the other flow parameters 

line_color = '#F89C20'

colormap = 'Greys'
min_counts = 10 # Minimum number of points within a bin to color
max_counts = 10**3 # Maximum color threshold for number of points within a bin. 
x_text = 0.025 # Location for text on figure
y_text = 0.9
x_lim = [0.001,1]  

plt.figure(figsize=(9,15))
# Critical velocity
plt.subplot(411)
plt.hist2d(v_critical[:,0],v_critical[:,1],bins=(x_space,y_space),cmin=min_counts,norm=mpl.colors.Normalize(vmax=max_counts), cmap=colormap) # Grain size X, critival velocity Y
plt.plot(Grain_size,v_FB_median,color=line_color,linewidth=3,label='Median') # Median fit of critical velocity for each grain size
plt.plot(Grain_size,v_FB_median+v_FB_iqr,c=line_color,linestyle='--',linewidth=3,label='IQR') # Upper IQR bound 
plt.plot(Grain_size,v_FB_median-v_FB_iqr,c=line_color,linestyle='--',linewidth=3) # Lower IQR bound 
plt.plot(Grain_size,v_FB_median+v_FB_90,c=line_color,linestyle=':',linewidth=3) # 95th percentile bound 
plt.plot(Grain_size,v_FB_median-v_FB_90,c=line_color,linestyle=':',label='$5^{th}$ to $95^{th}$',linewidth=3) # 5th percentile bound
plt.ylabel('Critival velocity, $u_c$ (m/s)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.legend(fontsize=14,loc='lower right',frameon=True,edgecolor='k',facecolor='white', framealpha=1)
y_lim = [0,9]
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.text(x_lim[0] + (x_lim[1] - x_lim[0])*x_text , y_lim[0] + (y_lim[1] - y_lim[0])*y_text,'a',horizontalalignment ='left',fontsize=14,fontweight='bold')

# Shear velocity
plt.subplot(412)
plt.hist2d(u_shear_heatmap[:,0],u_shear_heatmap[:,1],bins=(u_shear_x_space,u_shear_y_space),cmin=min_counts,norm=mpl.colors.Normalize(vmax=max_counts), cmap=colormap)
plt.plot(Grain_size,u_shear_median,color=line_color,linewidth=3,label='m = '+str(np.round(m_shear,decimals=3))+' +/- '+str(np.round(m_shear_iqr - m_shear,decimals=3)))
plt.plot(Grain_size,u_shear_median+u_shear_iqr,c=line_color,linestyle='--',linewidth=3)
plt.plot(Grain_size,u_shear_median-u_shear_iqr,c=line_color,linestyle='--',linewidth=3)
plt.plot(Grain_size,u_shear_median+u_shear_90,c=line_color,linestyle=':',linewidth=3)
plt.plot(Grain_size,u_shear_median-u_shear_90,c=line_color,linestyle=':',linewidth=3)
plt.ylabel('Shear velocity, $u_*$ (m/s)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
y_lim = [0,1.5]
plt.ylim(y_lim)
plt.text(x_lim[0] + (x_lim[1] - x_lim[0])*x_text , y_lim[0] + (y_lim[1] - y_lim[0])*y_text,'b',horizontalalignment ='left',fontsize=14,fontweight='bold')

# Shear stress
plt.subplot(413)
plt.hist2d(t_shear_heatmap[:,0],t_shear_heatmap[:,1],bins=(t_shear_x_space,t_shear_y_space),cmin=min_counts,norm=mpl.colors.Normalize(vmax=max_counts), cmap=colormap)
plt.plot(Grain_size,t_shear_stress_median,color=line_color,linewidth=3,label='m = '+str(np.round(m_t_shear,decimals=3))+' +/- '+str(np.round(m_t_shear_iqr - m_t_shear,decimals=3)))
plt.plot(Grain_size,t_shear_stress_median+t_shear_stress_iqr,c=line_color,linestyle='--',linewidth=3)
plt.plot(Grain_size,t_shear_stress_median-t_shear_stress_iqr,c=line_color,linestyle='--',linewidth=3)
plt.plot(Grain_size,t_shear_stress_median+t_shear_stress_90,c=line_color,linestyle=':',linewidth=3)
plt.plot(Grain_size,t_shear_stress_median-t_shear_stress_90,c=line_color,linestyle=':',linewidth=3)
plt.ylabel(r'Shear stress, $\tau$ (Pa)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
y_lim = [0,2000]
plt.ylim(y_lim)
plt.text(x_lim[0] + (x_lim[1] - x_lim[0])*x_text , y_lim[0] + (y_lim[1] - y_lim[0])*y_text,'c',horizontalalignment ='left',fontsize=14,fontweight='bold')

# Shields stress
plt.subplot(414)
im = plt.hist2d(t_shields_heatmap[:,0],t_shields_heatmap[:,1],bins=(t_shields_x_space,t_shields_y_space),cmin=min_counts,norm=mpl.colors.Normalize(vmax=max_counts), cmap=colormap)
plt.plot(Grain_size,t_shields_stress_median,color=line_color,linewidth=3,label='m = '+str(np.round(m_t_shields,decimals=3))+' +/- '+str(np.round(m_t_shields_iqr - m_t_shields,decimals=3)))
plt.plot(Grain_size,t_shields_stress_median+t_shields_stress_iqr,Grain_size,t_shields_stress_median-t_shields_stress_iqr,c=line_color,linestyle='--',linewidth=3)
plt.plot(Grain_size,t_shields_stress_median+t_shields_stress_90,c=line_color,linestyle=':',linewidth=3)
plt.plot(Grain_size,t_shields_stress_median-t_shields_stress_90,c=line_color,linestyle=':',linewidth=3)
plt.xlabel('Grain size (m)',fontsize=14)
plt.ylabel(r'Critical Shields, $\tau^*_c$',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.show()
plt.tight_layout()
y_lim = [0,0.2]
plt.ylim(y_lim)
plt.text(x_lim[0] + (x_lim[1] - x_lim[0])*x_text , y_lim[0] + (y_lim[1] - y_lim[0])*y_text,'d',horizontalalignment ='left',fontsize=14,fontweight='bold')
plt.tight_layout(h_pad=1)

# Add colorbar to the bottom of the plot 
cbar = plt.colorbar(orientation="horizontal")
cbar.ax.tick_params(labelsize=14,size=6)
cbar.ax.set_xticklabels(labels=('0','0.2','0.4','0.6','0.8','1.0'),fontsize=14)
cbar.set_label('Relative probability', fontsize=14)
