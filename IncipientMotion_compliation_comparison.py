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

#%% 

filepath = '/Users/scottfeehan/Box Sync/Spyder/Power Law (V-GS)/Data/Buffington/Shields_comp_field.csv' 
Shields_comp_field = pd.read_csv(filepath)

filepath = '/Users/scottfeehan/Box Sync/Spyder/Power Law (V-GS)/Data/Buffington/Shields_comp_flume.csv' 
Shields_comp_flume = pd.read_csv(filepath)

# Field observations
Shields_stress_field = Shields_comp_field['Shields stress'].to_numpy() # Reported Shield's stress 
Slope_field = Shields_comp_field['slope (m/m)'].to_numpy() # Slope 
D50_field = Shields_comp_field['median grain size (cm)'].to_numpy()/100 # Convert grain size to meters 
Flow_depth_field = Shields_comp_field['flow depth (cm)'].to_numpy()/100 # Convert flow depth to meters
Sediment_density_field = Shields_comp_field['density (g/cm3)'].to_numpy()*1000 # Convert sediment density to kg/m^3
Reynolds_field = Shields_comp_field['Re (reported)'].to_numpy() # Reynolds number 
D84_field = (Shields_comp_field['median grain size (cm)'].to_numpy() + Shields_comp_field['phi'].to_numpy())/100 # Calculate D84 using sorting coefficient

# Flume observations
Shields_stress_flume = Shields_comp_flume['Shields stress'].to_numpy()
Slope_flume = Shields_comp_flume['slope (m/m)'].to_numpy()
D50_flume = Shields_comp_flume['median grain size (cm)'].to_numpy()/100
Flow_depth_flume = Shields_comp_flume['flow depth (cm)'].to_numpy()/100
Sediment_density_flume = Shields_comp_flume['density (g/cm3)'].to_numpy()*1000
Reynolds_flume = Shields_comp_flume['Re (reported)'].to_numpy()
D84_flume = (Shields_comp_flume['median grain size (cm)'].to_numpy() + Shields_comp_flume['phi'].to_numpy())/100

#%% Clip data compilation at less than a 5% slope and grain size greater than 1 mm

temp = np.where((Slope_field <= 0.05) & (D50_field >= 0.001)) # Slope >= 5% and D50 >= 1 mm
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

#%% Shear velocity 

#Shear_velocity_field = (Shields_stress_field/rho_w_true)**0.5
Shear_velocity_field = ((Shields_stress_field*(Sediment_density_field - rho_w_true)*g*D50_field)/rho_w_true)**0.5

#Shear_velocity_flume = (Shields_stress_flume/rho_w_true)**0.5
Shear_velocity_flume = ((Shields_stress_flume*(Sediment_density_flume - rho_w_true)*g*D50_flume)/rho_w_true)**0.5




