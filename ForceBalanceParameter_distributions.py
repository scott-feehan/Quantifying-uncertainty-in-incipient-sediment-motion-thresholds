#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 18:23:02 2022

@author: scottfeehan

This code generates parts of figure 1.

Plotting probability density functions of the force balance parameters (FBP) that 
are used in critical velocity calculation. FBP values divided by the mean show 
relative range of each parameter. 

Calculate the relative influence of each parameter on the critical velocity by 
holding all except one variable constant and comparing to the result if all 
parameters are able to vary across their full distribution. 
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm

#%% Constants and parameters 
        
g = 9.81 # Gravity 
theta = 0.001 # Constant slope , which at these low slopes is basically np.arctan(theta)
V_w_V = 1 # Fully submerged grain  
monte_carlo_step = 10000000 # Monte Carlo iteration length
var_bins= 200 # Bin number to discretize probability density estimate
B_axis = 0.1
A_axis = B_axis
C_axis = B_axis

# Assumed force balance parameter mean, minimum, maximum, and standard deviation
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
C_l_mean = 0.85*C_d_mean
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
X = get_truncated_normal(rho_s_max ,rho_s_min,rho_s_mean,rho_s_stdv) 
rho_s = X.rvs(monte_carlo_step) 

X = get_truncated_normal(rho_w_max,rho_w_min,rho_w_mean,rho_w_stdv)
rho_w = X.rvs(monte_carlo_step) 

X = get_truncated_normal(C_l_max,C_l_min,C_l_mean,C_l_stdv)
C_l = X.rvs(monte_carlo_step) 
C_l = np.sort(C_l)

X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
C_d = X.rvs(monte_carlo_step) 
C_d = np.sort(C_d)

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

#%% 

def ForceBalance(rho_s,rho_w,g,mu,C_l,C_d,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p):
    r = B_axis/2
    A_e = r**2*np.arccos((r - (B_axis-(p*B_axis)))/r) - (r - (B_axis-(p*B_axis)))*np.sqrt(2*r*(B_axis-(p*B_axis)) - (B_axis-(p*B_axis))**2)
    A_e = A - A_e
    A_p = np.ones(np.size(p))*A
    if np.size(p)>1 :
        h = B_axis*p[np.where(p < 0.5)]
        a = np.sqrt(r**2 - ((r - h)**2))
        A_p[np.where(p < 0.5)] = np.pi * (a**2)
    
    if (np.size(p)==1 and p < 0.5):     
        h = B_axis*p
        a = np.sqrt(r**2 - ((r - h)**2))
        A_p = np.pi * (a**2)
        
    v_c = ((2*g*V*(rho_s/rho_w - 1*(V_w_V))*(mu*np.cos(theta) - np.sin(theta)))/(C_d*A_e + mu*C_l*A_p))**0.5
    return v_c

#%% Calculte probability density function for each force balance parameter such that the area under the curve integrates to one

temp = plt.hist(C_d,bins=var_bins,density='True') # Store counts for each bin proportional to the parameter range
C_d_bins = temp[1] # Number of parameters generated within each bin or "counts"
C_d_counts = temp[0] # Store normalized count for each bin 
C_d_bins = np.append(C_d_bins[0],C_d_bins) # Store ends of distribution twice to plot distribution down to zero
C_d_bins = np.append(C_d_bins,C_d_bins[-1])
C_d_counts = np.append(0,C_d_counts) # Correlated zeros
C_d_counts = np.append(C_d_counts,0)

temp = plt.hist(C_l,bins=var_bins,density='True')
C_l_bins = temp[1]
C_l_counts = temp[0]
C_l_bins = np.append(C_l_bins[0],C_l_bins)
C_l_bins = np.append(C_l_bins,C_l_bins[-1])
C_l_counts = np.append(0,C_l_counts)
C_l_counts = np.append(C_l_counts,0)

temp = plt.hist(rho_s/1000,bins=var_bins,density='True')
rho_s_bins = temp[1]
rho_s_counts = temp[0]
rho_s_bins = np.append(rho_s_bins[0],rho_s_bins)
rho_s_bins = np.append(rho_s_bins,rho_s_bins[-1])
rho_s_counts = np.append(0,rho_s_counts)
rho_s_counts = np.append(rho_s_counts,0)

temp = plt.hist(rho_w/1000,bins=var_bins,density='True')
rho_w_bins = temp[1]
rho_w_counts = temp[0]
rho_w_bins = np.append(rho_w_bins[0],rho_w_bins)
rho_w_bins = np.append(rho_w_bins,rho_w_bins[-1])
rho_w_counts = np.append(0,rho_w_counts)
rho_w_counts = np.append(rho_w_counts,0)

temp = plt.hist(p,bins=var_bins,density='True')
p_bins = temp[1]
p_counts = temp[0]
p_bins = np.append(p_bins[0],p_bins)
p_bins = np.append(p_bins,p_bins[-1])
p_counts = np.append(0,p_counts)
p_counts = np.append(p_counts,0)

temp = plt.hist(mu,bins=var_bins,density='True')
mu_bins = temp[1]
mu_counts = temp[0]
mu_bins = np.append(mu_bins[0],mu_bins)
mu_bins = np.append(mu_bins,mu_bins[-1])
mu_counts = np.append(0,mu_counts)
mu_counts = np.append(mu_counts,0)
plt.close()


#%% Calculate critical velocity when varying all parameters and compare to when a each parameter is varied while all others are held constant. 

V = (A_axis/2)*(B_axis/2)*(C_axis/2)*(4/3)*np.pi # assume perfect sphere volume 
A = (B_axis/2)*(C_axis/2)*np.pi # area normal to flow 

v_ForceBalance_all = ForceBalance(rho_s,rho_w,g,mu,C_l,C_d,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p) # Varying all parameters 
v_ForceBalance_none = ForceBalance(rho_s_mean,rho_w_mean,g,mu_mean,C_l_mean,C_d_mean,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p_mean) # Keeping all parameters constant
v_ForceBalance_rho_s = ForceBalance(rho_s,rho_w_mean,g,mu_mean,C_l_mean,C_d_mean,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p_mean) # Vary sediment density 
v_ForceBalance_rho_w = ForceBalance(rho_s_mean,rho_w,g,mu_mean,C_l_mean,C_d_mean,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p_mean) # Vary fluid density 
v_ForceBalance_mu = ForceBalance(rho_s_mean,rho_w_mean,g,mu,C_l_mean,C_d_mean,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p_mean) # Vary friction coefficient
v_ForceBalance_C_d = ForceBalance(rho_s_mean,rho_w_mean,g,mu_mean,C_l_mean,C_d,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p_mean) # Vary drag coefficient 
v_ForceBalance_C_l = ForceBalance(rho_s_mean,rho_w_mean,g,mu_mean,C_l,C_d_mean,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p_mean) # Vary lift coefficient
v_ForceBalance_p = ForceBalance(rho_s_mean,rho_w_mean,g,mu_mean,C_l_mean,C_d_mean,theta,A_axis,B_axis,C_axis,V_w_V,V,A,p) # Vary protrusion

#%% Calculate probability density function for each varied force balance parameter such that the area under the curve integrates to one

plt.figure()
temp = plt.hist(v_ForceBalance_all,bins=var_bins,density='True')
v_all_bins = temp[1]
v_all_counts = temp[0]
v_all_bins = np.append(v_all_bins[0],v_all_bins)
v_all_bins = np.append(v_all_bins,v_all_bins[-1])
v_all_counts = np.append(0,v_all_counts)
v_all_counts = np.append(v_all_counts,0)

temp = plt.hist(v_ForceBalance_C_d,bins=var_bins,density='True')
v_C_d_bins = temp[1]
v_C_d_counts = temp[0]
v_C_d_bins = np.append(v_C_d_bins[0],v_C_d_bins)
v_C_d_bins = np.append(v_C_d_bins,v_C_d_bins[-1])
v_C_d_counts = np.append(0,v_C_d_counts)
v_C_d_counts = np.append(v_C_d_counts,0)

temp = plt.hist(v_ForceBalance_C_l,bins=var_bins,density='True')
v_C_l_bins = temp[1]
v_C_l_counts = temp[0]
v_C_l_bins = np.append(v_C_l_bins[0],v_C_l_bins)
v_C_l_bins = np.append(v_C_l_bins,v_C_l_bins[-1])
v_C_l_counts = np.append(0,v_C_l_counts)
v_C_l_counts = np.append(v_C_l_counts,0)

temp = plt.hist(v_ForceBalance_mu,bins=var_bins,density='True')
v_mu_bins = temp[1]
v_mu_counts = temp[0]
v_mu_bins = np.append(v_mu_bins[0],v_mu_bins)
v_mu_bins = np.append(v_mu_bins,v_mu_bins[-1])
v_mu_counts = np.append(0,v_mu_counts)
v_mu_counts = np.append(v_mu_counts,0)

temp = plt.hist(v_ForceBalance_rho_s,bins=var_bins,density='True')
v_rho_s_bins = temp[1]
v_rho_s_counts = temp[0]
v_rho_s_bins = np.append(v_rho_s_bins[0],v_rho_s_bins)
v_rho_s_bins = np.append(v_rho_s_bins,v_rho_s_bins[-1])
v_rho_s_counts = np.append(0,v_rho_s_counts)
v_rho_s_counts = np.append(v_rho_s_counts,0)

temp = plt.hist(v_ForceBalance_rho_w,bins=var_bins,density='True')
v_rho_w_bins = temp[1]
v_rho_w_counts = temp[0]
v_rho_w_bins = np.append(v_rho_w_bins[0],v_rho_w_bins)
v_rho_w_bins = np.append(v_rho_w_bins,v_rho_w_bins[-1])
v_rho_w_counts = np.append(0,v_rho_w_counts)
v_rho_w_counts = np.append(v_rho_w_counts,0)

temp = plt.hist(v_ForceBalance_p,bins=var_bins,density='True')
v_p_bins = temp[1]
v_p_counts = temp[0]
v_p_bins = np.append(v_p_bins[0],v_p_bins)
v_p_bins = np.append(v_p_bins,v_p_bins[-1])
v_p_counts = np.append(0,v_p_counts)
v_p_counts = np.append(v_p_counts,0)
plt.close()

#%% Plotting each calculated value 

colors = ['#5790FC', '#F89C20', '#E42536', '#964A8B', '#9C9CA1', '#7A21DD']

y_lim = [0,5]
x_lim = [-0.05,4.8] 
x_text = 0.02
y_text = 0.89

plt.figure(figsize=(7.5,13))
# Probability density function of FBP values 
plt.subplot(311)
plt.plot(C_d_bins[:-1],C_d_counts,linewidth=2,label='$C_D$',color=colors[0], linestyle='-.')
plt.plot(C_l_bins[:-1],C_l_counts,linewidth=2,label='$C_L$',color=colors[1], linestyle=(0, (3, 1, 1, 1, 1, 1)))
plt.plot(mu_bins[:-1],mu_counts,linewidth=2,label=r'$\mu$ = $tan(\phi)$',color=colors[2], linestyle= '-')
plt.plot(rho_s_bins[:-1],rho_s_counts,linewidth=2,label=r'$\rho_s$ (g/cm$^3$)',color=colors[3], linestyle='-.')
plt.plot(rho_w_bins[:-1],rho_w_counts,linewidth=2,label=r'$\rho_f$ (g/cm$^3$)',color=colors[4], linestyle=':')
plt.plot(p_bins[:-1],p_counts,linewidth=2,label='$p_*$', linestyle='--',color=colors[5])
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.text(x_lim[0] + (x_lim[1] - x_lim[0])*x_text , y_lim[0] + (y_lim[1] - y_lim[0])*y_text,'b',horizontalalignment ='left',fontsize=14,fontweight='bold')
plt.ylabel('Probability density',fontsize=14)
plt.xlabel('Parameter value',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.legend(fontsize=14,frameon=True,edgecolor='k')

plt.subplot(312)
# Probability density function of FBP values normalized by the respective mean
plt.plot(C_d_bins[:-1]/C_d_mean,C_d_counts,linewidth=2,label='$C_D$ / $\overline{C}_D$', linestyle='-.',color=colors[0])
plt.plot(C_l_bins[:-1]/C_l_mean,C_l_counts,linewidth=2,label='$C_L$ / $\overline{C}_L$', linestyle=(0, (3, 1, 1, 1, 1, 1)),color=colors[1])
plt.plot(mu_bins[:-1]/mu_mean,mu_counts,linewidth=2,label=r'$\mu$ / $\overline{\mu}$', linestyle= '-',color=colors[2])
plt.plot(rho_s_bins[:-1]/(rho_s_mean/1000),rho_s_counts,linewidth=2,label=r'$\rho_s$ / $\overline{\rho}_s$', linestyle='-.',color=colors[3])
plt.plot(rho_w_bins[:-1]/(rho_w_mean/1000),rho_w_counts,linewidth=2,label=r'$\rho_f$ / $\overline{\rho}_f$', linestyle=':',color=colors[4])
plt.plot(p_bins[:-1]/p_mean,p_counts,linewidth=2,label='$p_*$ / $\overline{p}_*$', linestyle='--',color=colors[5])
plt.ylabel('Probability density',fontsize=14)
plt.xlabel('Parameter value normalizes by its mean',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.text(x_lim[0] + (x_lim[1] - x_lim[0])*x_text , y_lim[0] + (y_lim[1] - y_lim[0])*y_text,'c',horizontalalignment ='left',fontsize=14,fontweight='bold')
plt.legend(fontsize=14,frameon=True,edgecolor='k') 

y_lim = [0,4]
x_lim = [1,3.5] 
x_text = 0.02
y_text = 0.89

plt.subplot(313)
# Probability density function of critical velocity calculated by varying each parameter and maintaining all others at a constant value
plt.plot(v_all_bins[:-1],v_all_counts,linewidth=2,label='All', linestyle='-',color='k')
plt.plot(v_C_d_bins[:-1],v_C_d_counts,linewidth=2,label='$C_D$', linestyle='-.',color=colors[0])
plt.plot(v_C_l_bins[:-1],v_C_l_counts,linewidth=2,label='$C_L$', linestyle=(0, (3, 1, 1, 1, 1, 1)),color=colors[1])
plt.plot(v_mu_bins[:-1],v_mu_counts,linewidth=2,label=r'$\mu$', linestyle= '-',color=colors[2])
plt.plot(v_rho_s_bins[:-1],v_rho_s_counts,linewidth=2,label=r'$\rho_s$', linestyle='-.',color=colors[3])
plt.plot(v_rho_w_bins[:-1],v_rho_w_counts,linewidth=2,label=r'$\rho_f$', linestyle=':',color=colors[4])
plt.plot(v_p_bins[:-1],v_p_counts,linewidth=2,label='$p_*$', linestyle='--',color=colors[5])
plt.ylabel('Probability Density',fontsize=14)
plt.xlabel('Estimate critical velocity ($m/s$)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
lg = plt.legend(fontsize=14,frameon=True,title='Varied FBP',edgecolor='k')
lg.get_title().set_fontsize('14') 
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.text(x_lim[0] + (x_lim[1] - x_lim[0])*x_text , y_lim[0] + (y_lim[1] - y_lim[0])*y_text,'d',horizontalalignment ='left',fontsize=14,fontweight='bold')
plt.tight_layout()





