#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:14:45 2023

@author: scottfeehan

This code uses bedload transport data from Meyer-Peter and Muller (1948) and
the modified dimensionless bedload transport rate equation from Wong and 
Parker (2006) to estimate the range of data that would be incorporated within
expected uncertainty from general uncertainty of force balance parameters 
promoting and resisting grain entrainment. 

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Uploading data

filepath = 'Meyer-Peter_Muller_1948_data.csv'
MPM_data = pd.read_csv(filepath)

#%% Setting threshold conditions 

tau_c = 0.047 # From Wong and Parker (2010) 
tau_c_uncertainty = 0.022 # From general force balance parameter uncertainty 

#%% Calculate dimmensionless bedload transport rate using Wong and Parker (2006)

tau_shear_imposed = np.linspace(min(MPM_data['Shields stress'])-(min(MPM_data['Shields stress'])*0.1) ,max(MPM_data['Shields stress'])*1.1,1000) # range of imposed Shields' stress

q_star = 4.93*(tau_shear_imposed - tau_c)**(1.60) # Calculate commonly applied bedload transport equation
q_star_high = 4.93*(tau_shear_imposed - (tau_c + tau_c_uncertainty))**(1.60) # Threshold on upper end of uncertainty 
q_star_low = 4.93*(tau_shear_imposed - (tau_c - tau_c_uncertainty))**(1.60) # Threshold on lower end of uncertainty 

#%% Plot data and estimated sediment transport rate

# limits 
x_lim = [0.045,0.26] 
y_lim = [0.00001,0.7] 


plt.figure(figsize=(7,4))
plt.plot(tau_shear_imposed,q_star,color='darkorange',linewidth=2,label=r'$q_* = 4.93(\tau^* - \tau^*_c)^{1.60}$') # Typical dimensionless transport rate per unit width from Wong and Parker (2006)
plt.plot(tau_shear_imposed,q_star_high,linestyle='--',color='darkorange',linewidth=2,label=r'IQR uncertainty on $\tau^*_c$') # High threshold 
plt.plot(tau_shear_imposed,q_star_low,linestyle='--',color='darkorange',linewidth=2) # Low threshold 
plt.scatter(MPM_data['Shields stress'],MPM_data['dimensionless bedload flux per unit width'],color='w',edgecolors='k',label='Meyer-Peter & Müller (1948)')  # Meyer-Peter and Müller observational data
plt.ylabel('Bedload flux per unit width, $q_*$',fontsize=14)
plt.xlabel(r'Shields stress, $\tau_*$',fontsize=14)
plt.legend(fontsize=14)
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.tick_params(axis='both',which='both',direction='in',labelsize=14)
plt.tick_params(which='major',length=10)
plt.tick_params(which='minor',length=5)
plt.loglog()
plt.tight_layout()

#%% Data within uncertainty

q_star_upper_bound = np.interp(MPM_data['Shields stress'],tau_shear_imposed,q_star_low) # Upper uncertainty bound for each observed Sheilds stress
q_star_lower_bound = np.interp(MPM_data['Shields stress'],tau_shear_imposed,q_star_high) # Lower uncertainty bound for each observed Sheilds stress
temp = np.where((MPM_data['dimensionless bedload flux per unit width'] >= q_star_upper_bound) | (MPM_data['dimensionless bedload flux per unit width'] <= q_star_lower_bound)) # Where points sit beyond the upper and lower uncertainty
q_star_in_uncertainty = np.round(len(temp[0])/len(MPM_data['dimensionless bedload flux per unit width']),decimals=2)*100 # Percentage of points beyond uncertainty
print('MPM within Shields uncertaitny = '+str(q_star_in_uncertainty)+'%') # Print percentage of points beyond uncertainty

