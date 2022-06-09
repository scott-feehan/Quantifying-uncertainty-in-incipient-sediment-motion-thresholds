#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:00:56 2022

@author: scottfeehan

Determining the effect of covariability between the drag and lift coefficient on estimating critical flow  

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from scipy.stats import truncnorm
import seaborn as sns
from scipy.stats import pearsonr

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
    
monte_carlo_step = 1000000 # Monte Carlo iteration length

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

#%% Sginle grain size

GrainSize = 0.1

B_axis = GrainSize
A_axis = B_axis
C_axis = B_axis
V = (A_axis/2)*(B_axis/2)*(C_axis/2)*(4/3)*np.pi # assume perfect sphere volume 
A_n = (B_axis/2)*(C_axis/2)*np.pi # area normal to flow 
A_p = (A_axis/2)*(B_axis/2)*np.pi # area parallel to flow

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

#%% No Covariance 

# Typical 
X = get_truncated_normal(C_l_max,C_l_min,C_l_mean,C_l_stdv) 
C_l_typical = X.rvs(monte_carlo_step) # Generate typical distribution 

X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
C_d_typical = X.rvs(monte_carlo_step) # Generate typical distribution

C_d_C_l_pR_typical = pearsonr(C_d_typical,C_l_typical) # Calculate Pearson correlation coefficiont (r)

C_d_C_l_line_typical = np.polyfit(C_d_typical,C_l_typical,1) # Linear fit between drag and lift 
C_d_C_l_typical_x = np.linspace(min(C_d_typical),max(C_d_typical),10) # Generate independent variables to plot line  
C_d_C_l_typical_y = C_d_C_l_line_typical[0]*C_d_C_l_typical_x + C_d_C_l_line_typical[1] # Calculate dependent variables to plot line on data

v_ForceBalance_typical = ForceBalance(rho_s,rho_w,g,mu,C_l_typical,C_d_typical,theta,A_axis,B_axis,C_axis,V_w_V,V,A_p,p)

#%% Example Cd values to plot a line 

C_d_x = np.arange(0.001,2.1,0.1)

#%% Generate Cl values that are linearly related to Cd with some scatter 
# Cl = a * Cd + b 
# randomly sampling m from a normal distribution for each generated Cd value
# positive relationship with wide scatter 

partial_stdv = 0.5 # Standard deviation of the m value
m_all = 0.5 # mean 

X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
C_d = X.rvs(monte_carlo_step) # Generate typical C_d distribution

C_d_stdv_05 = C_d 

m_partial = np.random.normal(m_all,partial_stdv,monte_carlo_step) # generate normal distribution of linear coefficient values 

C_l_stdv_05 = m_partial*C_d_stdv_05 + C_l_min # Generate linearly related Cl values with associated scatter 

# Make sure no values exist beyond assumed randes 
while True:  
    temp = np.where((C_l_min>C_l_stdv_05) | (C_l_stdv_05>C_l_max)) # Find where values are beyond bounds 
    X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv) # Generate distribution of Cd values 
    C_d = X.rvs(len(temp[0])) # Make the distribution the length of the number of values that are beyond outside the specified bound   
    m_partial = np.random.normal(m_all,partial_stdv,len(temp[0])) # generate distribution of linear coefficient values of the same lenght 

    C_l_stdv_05[temp] = (m_partial*C_d + C_l_min) # generate linearly related Cl values  
    if len(temp[0]) <1: # once there are no values left beyond the bounds 
        break # leave the loop 

C_l_C_d_fit_stdv_05 = np.polyfit(C_d_stdv_05,C_l_stdv_05,1) # fit linear equaiton to the relationship 
C_l_y_stdv_05 = C_l_C_d_fit_stdv_05[0]*C_d_x  + C_l_C_d_fit_stdv_05[1] # Create line for plot 
C_d_C_l_pR_stdv_05 = pearsonr(C_d_stdv_05,C_l_stdv_05) # calculate pearson r 
v_ForceBalance_stdv_05 = ForceBalance(rho_s,rho_w,g,mu,C_l_stdv_05,C_d_stdv_05,theta,A_axis,B_axis,C_axis,V_w_V,V,A_p,p) # Used monte carlo force balance to calculate critial velocity

#%% Positive relationship with narrow scatter

partial_stdv = 0.1
m_all = 0.5

X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
C_d = X.rvs(monte_carlo_step) 

C_d_stdv_01 = C_d

m_partial = np.random.normal(m_all,partial_stdv,monte_carlo_step)

C_l_stdv_01 = m_partial*C_d_stdv_01 + C_l_min

while True: 
    temp = np.where((C_l_min>C_l_stdv_01) | (C_l_stdv_01>C_l_max))
    X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
    C_d = X.rvs(len(temp[0]))
    m_partial = np.random.normal(m_all,partial_stdv,len(temp[0]))

    C_l_stdv_01[temp] = (m_partial*C_d + C_l_min) 
    if len(temp[0]) <1:
        break 

C_l_C_d_fit_stdv_01 = np.polyfit(C_d_stdv_01,C_l_stdv_01,1)
C_l_y_stdv_01 = C_l_C_d_fit_stdv_01[0]*C_d_x  + C_l_C_d_fit_stdv_01[1]
C_d_C_l_pR_stdv_01 = pearsonr(C_d_stdv_01,C_l_stdv_01)
v_ForceBalance_stdv_01 = ForceBalance(rho_s,rho_w,g,mu,C_l_stdv_01,C_d_stdv_01,theta,A_axis,B_axis,C_axis,V_w_V,V,A_p,p)

#%% Negative relationship with wide scatter 

partial_stdv = 0.5
m_all = 0.5

X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
C_d = X.rvs(monte_carlo_step) 

C_d_stdv_05_neg = C_d

m_partial = np.random.normal(m_all,partial_stdv,monte_carlo_step)

C_l_stdv_05_neg = -m_partial*C_d_stdv_05_neg + 1 # negative linear relationship

while True: 
    temp = np.where((C_l_min>C_l_stdv_05_neg) | (C_l_stdv_05_neg>C_l_max))
    X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
    C_d = X.rvs(len(temp[0])) 
    m_partial = np.random.normal(m_all,partial_stdv,len(temp[0]))
    C_l_stdv_05_neg[temp] = (m_partial*C_d + C_l_min) 
    if len(temp[0]) <1:
        break 

C_l_C_d_fit_stdv_05_neg = np.polyfit(C_d_stdv_05_neg,C_l_stdv_05_neg,1)
C_l_y_stdv_05_neg = C_l_C_d_fit_stdv_05_neg[0]*C_d_x  + C_l_C_d_fit_stdv_05_neg[1]
C_d_C_l_pR_stdv_05_neg = pearsonr(C_d_stdv_05_neg,C_l_stdv_05_neg)
v_ForceBalance_stdv_05_neg = ForceBalance(rho_s,rho_w,g,mu,C_l_stdv_05_neg,C_d_stdv_05_neg,theta,A_axis,B_axis,C_axis,V_w_V,V,A_p,p)

#%% Negative relationship with narrow scatter

partial_stdv = 0.1
m_all = 0.5

X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
C_d = X.rvs(monte_carlo_step) 

C_d_stdv_01_neg = C_d

m_partial = np.random.normal(m_all,partial_stdv,monte_carlo_step)

C_l_stdv_01_neg = -m_partial*C_d_stdv_01_neg + 1

while True: 
    temp = np.where((C_l_min>C_l_stdv_01_neg) | (C_l_stdv_01_neg>C_l_max))
    X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
    C_d = X.rvs(len(temp[0])) 
    m_partial = np.random.normal(m_all,partial_stdv,len(temp[0]))

    C_l_stdv_01_neg[temp] = (m_partial*C_d + C_l_min) 
    if len(temp[0]) <1:
        break 

C_l_C_d_fit_stdv_01_neg = np.polyfit(C_d_stdv_01_neg,C_l_stdv_01_neg,1)
C_l_y_stdv_01_neg = C_l_C_d_fit_stdv_01_neg[0]*C_d_x  + C_l_C_d_fit_stdv_01_neg[1]
C_d_C_l_pR_stdv_01_neg = pearsonr(C_d_stdv_01_neg,C_l_stdv_01_neg)
v_ForceBalance_stdv_01_neg = ForceBalance(rho_s,rho_w,g,mu,C_l_stdv_01_neg,C_d_stdv_01_neg,theta,A_axis,B_axis,C_axis,V_w_V,V,A_p,p)

#%% Generate positive and negative relationship with no scatter 
 
covariance_scale = 0.5 # Cl is equal to 1/2 of Cd
C_d_x = np.arange(0.01,2.1,0.1)

# Positive 
X = get_truncated_normal(C_l_max,C_l_min,C_l_mean,C_l_stdv)
C_l = X.rvs(monte_carlo_step)

X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
C_d = X.rvs(monte_carlo_step)

C_d_mathfit_pos = C_d
C_l = covariance_scale*C_d + C_l_min
C_l_y_pos = covariance_scale*C_d_x + C_l_min
C_l_mathfit_pos = C_l

# Negative 
X = get_truncated_normal(C_l_max,C_l_min,C_l_mean,C_l_stdv)
C_l = X.rvs(monte_carlo_step)

X = get_truncated_normal(C_d_max ,C_d_min,C_d_mean,C_d_stdv )
C_d = X.rvs(monte_carlo_step)

C_d_mathfit_neg = C_d
C_l = -covariance_scale*C_d + 1
C_l_y_neg = -covariance_scale*C_d_x + 1
C_l_mathfit_neg = C_l

v_ForceBalance_covary_pos = ForceBalance(rho_s,rho_w,g,mu,C_l_mathfit_pos,C_d_mathfit_pos,theta,A_axis,B_axis,C_axis,V_w_V,V,A_p,p)
v_ForceBalance_covary_neg = ForceBalance(rho_s,rho_w,g,mu,C_l_mathfit_neg,C_d_mathfit_neg,theta,A_axis,B_axis,C_axis,V_w_V,V,A_p,p)

C_d_C_l_pR_covary_pos = pearsonr(C_d_mathfit_pos,C_l_mathfit_pos)
C_d_C_l_pR_covary_neg = pearsonr(C_d_mathfit_neg,C_l_mathfit_neg)

#%% Plot relationship between Cd and Cl

colors = ['#1845FB', '#FF5E02', '#C91F16', '#C849A9', '#ADAD7D', '#86C8DD', '#578DFF', '#656364'] 

x_lim = [0.05,1.6]
y_lim = [0.0,1.6]
text_loc = 0.9
r_xloc = 0.25
a_xloc = 0.8
b_xloc = 0.25
r_a_yloc = 1.45
b_yloc = 1.3

plt.figure(figsize=(9,9))

# Positive correlations 
# Typical assumption 
ax = plt.subplot(331)
sns.kdeplot(C_d_typical,C_l_typical,shade=True,shade_lowest=False,alpha=0.5,color='slategrey')
plt.plot(C_d_C_l_typical_x,C_d_C_l_typical_y,linewidth=3,label='$r_p$ = 0.0',color='slategrey')
plt.text(r_xloc,r_a_yloc,'$r_p$ = 0.0',fontsize=12)
plt.text(x_lim[1]*text_loc,y_lim[1]*text_loc,'a',fontsize=14,fontweight='bold')
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.xlabel('$C_D$',fontsize=14)
plt.ylabel('$C_L$',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)

# Positive correlations 
plt.subplot(332,sharex=ax,sharey=ax)
sns.kdeplot(C_d_stdv_05,C_l_stdv_05,shade=True,shade_lowest=False,alpha=0.5,color=colors[0])
plt.plot(C_d_x,C_l_y_stdv_05,linewidth=3,label='$r_p$ = '+str(np.round(C_d_C_l_pR_stdv_05[0],decimals=2)),color=colors[0])
plt.text(r_xloc,r_a_yloc,'$r_p$ = '+str(np.round(C_d_C_l_pR_stdv_05[0],decimals=2)),fontsize=12)
plt.xlabel('$C_D$',fontsize=14)
plt.ylabel('$C_L$',fontsize=14)
plt.text(x_lim[1]*text_loc,y_lim[1]*text_loc,'b',fontsize=14,fontweight='bold')
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.ylim(y_lim)
plt.xlim(x_lim)

plt.subplot(333,sharex=ax,sharey=ax)
sns.kdeplot(C_d_stdv_01,C_l_stdv_01,shade=True,shade_lowest=False,alpha=0.5,color=colors[1]) 
plt.plot(C_d_x,C_l_y_stdv_01,linewidth=3,label='$r_p$ = '+str(np.round(C_d_C_l_pR_stdv_01[0],decimals=2)),color=colors[1]) 
plt.text(r_xloc,r_a_yloc,'$r_p$ = '+str(np.round(C_d_C_l_pR_stdv_01[0],decimals=2)),fontsize=12)
plt.xlabel('$C_D$',fontsize=14)
plt.ylabel('$C_L$',fontsize=14)
plt.text(x_lim[1]*text_loc,y_lim[1]*text_loc,'c',fontsize=14,fontweight='bold')
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.ylim(y_lim)
plt.xlim(x_lim)

# Negative correlations 
plt.subplot(335,sharex=ax,sharey=ax)
sns.kdeplot(C_d_stdv_05_neg,C_l_stdv_05_neg,shade=True,shade_lowest=False,alpha=0.5,color=colors[2])
plt.plot(C_d_x,C_l_y_stdv_05_neg,linewidth=3,label='$r_p$ = '+str(np.round(C_d_C_l_pR_stdv_05_neg[0],decimals=2)),color=colors[2]) 
plt.text(r_xloc,r_a_yloc,'$r_p$ = '+str(np.round(C_d_C_l_pR_stdv_05_neg[0],decimals=2)),fontsize=12)
plt.xlabel('$C_D$',fontsize=14)
plt.ylabel('$C_L$',fontsize=14)
plt.text(x_lim[1]*text_loc,y_lim[1]*text_loc,'e',fontsize=14,fontweight='bold')
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.ylim(y_lim)
plt.xlim(x_lim)


plt.subplot(336,sharex=ax,sharey=ax)
sns.kdeplot(C_d_stdv_01_neg,C_l_stdv_01_neg,shade=True,shade_lowest=False,alpha=0.5,color=colors[3]) 
plt.plot(C_d_x,C_l_y_stdv_01_neg,linewidth=3,label='$r_p$ = '+str(np.round(C_d_C_l_pR_stdv_01_neg[0],decimals=2)),color=colors[3]) 
plt.text(r_xloc,r_a_yloc,'$r_p$ = '+str(np.round(C_d_C_l_pR_stdv_01_neg[0],decimals=2)),fontsize=12)
plt.xlabel('$C_D$',fontsize=14)
plt.ylabel('$C_L$',fontsize=14)
plt.text(x_lim[1]*text_loc,y_lim[1]*text_loc,'f',fontsize=14,fontweight='bold')
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.ylim(y_lim)
plt.xlim(x_lim)

# Perfect correlations
plt.subplot(334)
plt.plot(C_d_mathfit_pos,C_l_mathfit_pos,linewidth=3,color=colors[4],label='$r_p$ = 1')
plt.plot(C_d_mathfit_neg,C_l_mathfit_neg,linewidth=3,color=colors[5],label='$r_p$ = -1')
plt.text(r_xloc,r_a_yloc,'$r_p$ = 1 and -1',fontsize=12)
plt.xlabel('$C_D$',fontsize=14)
plt.ylabel('$C_L$',fontsize=14)
plt.text(x_lim[1]*text_loc,y_lim[1]*text_loc,'d',fontsize=14,fontweight='bold')
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.tight_layout()


y_lim = [1.3,2.5]
x_lim = [-1.1,1.1]
text_loc = 0.95

plt.subplot(313)

plt.axhline(np.nanmedian(v_ForceBalance_typical) - iqr(v_ForceBalance_typical,nan_policy='omit')/2,linestyle=(0, (5, 10)),color='k',linewidth=1)
plt.axhline(np.nanmedian(v_ForceBalance_typical) + iqr(v_ForceBalance_typical,nan_policy='omit')/2,linestyle=(0, (5, 10)),color='k',linewidth=1)

# Typical
plt.plot(C_d_C_l_pR_typical[0],np.nanmedian(v_ForceBalance_typical),marker='o',markersize=10,mfc='slategrey',mec='k')
plt.errorbar(C_d_C_l_pR_typical[0],np.nanmedian(v_ForceBalance_typical),yerr=iqr(v_ForceBalance_typical,nan_policy='omit')/2,color='k',capthick=1,capsize=5)

# Positive 
plt.plot(C_d_C_l_pR_stdv_05[0],np.nanmedian(v_ForceBalance_stdv_05),marker='o',markersize=10,mfc=colors[0],mec='k')
plt.errorbar(C_d_C_l_pR_stdv_05[0],np.nanmedian(v_ForceBalance_stdv_05),yerr=iqr(v_ForceBalance_stdv_05,nan_policy='omit')/2,color='k',capthick=1,capsize=5)

plt.plot(C_d_C_l_pR_stdv_01[0],np.nanmedian(v_ForceBalance_stdv_01),marker='o',markersize=10,mfc=colors[1],mec='k')
plt.errorbar(C_d_C_l_pR_stdv_01[0],np.nanmedian(v_ForceBalance_stdv_01),yerr=iqr(v_ForceBalance_stdv_01,nan_policy='omit')/2,color='k',capthick=1,capsize=5)

# Negative
plt.plot(C_d_C_l_pR_stdv_05_neg[0],np.nanmedian(v_ForceBalance_stdv_05_neg),marker='o',markersize=10,mfc=colors[2],mec='k')
plt.errorbar(C_d_C_l_pR_stdv_05_neg[0],np.nanmedian(v_ForceBalance_stdv_05_neg),yerr=iqr(v_ForceBalance_stdv_05_neg,nan_policy='omit')/2,color='k',capthick=1,capsize=5)

plt.plot(C_d_C_l_pR_stdv_01_neg[0],np.nanmedian(v_ForceBalance_stdv_01_neg),marker='o',markersize=10,mfc=colors[3],mec='k')
plt.errorbar(C_d_C_l_pR_stdv_01_neg[0],np.nanmedian(v_ForceBalance_stdv_01_neg),yerr=iqr(v_ForceBalance_stdv_01_neg,nan_policy='omit')/2,color='k',capthick=1,capsize=5)

plt.plot(1,np.nanmedian(v_ForceBalance_covary_pos),marker='o',markersize=10,mfc=colors[4],mec='k')
plt.errorbar(1,np.nanmedian(v_ForceBalance_covary_pos),yerr=iqr(v_ForceBalance_covary_pos,nan_policy='omit')/2,color='k',capthick=1,capsize=5)

plt.plot(-1,np.nanmedian(v_ForceBalance_covary_neg),marker='o',markersize=10,mfc=colors[5],mec='k')
plt.errorbar(-1,np.nanmedian(v_ForceBalance_covary_neg),yerr=iqr(v_ForceBalance_covary_neg,nan_policy='omit')/2,color='k',capthick=1,capsize=5)

plt.text(-0.85,2.15,'Range of $u_c$ for uncorrelated case',fontsize=12)
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.text(x_lim[1]*0.94,y_lim[1]*0.96,'g',fontsize=14,fontweight='bold')
plt.xlabel('$C_D$ and $C_L$ correlation coefficient, $r_p$',fontsize=14)
plt.ylabel('Critical velocity, $u_c$ (m/s)',fontsize=14)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.tight_layout()

