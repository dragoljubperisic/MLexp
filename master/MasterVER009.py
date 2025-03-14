# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 00:24:58 2025

@author: drago
8.3.2025. 14:08
002
"""


## 001 kreiranje klase

import pandas as pd

class Asteroid:
    def __init__(self, *args):
        field_names = [
            "SMOC_ID", "OBJ_ID_RUN", "OBJ_ID_COL", "OBJ_ID_FIELD", "OBJ_ID_OBJ",
            "ROWC", "COLC", "JD_ZERO", "RA", "DEC", "LAMBDA", "BETA", "PHI",
            "VMU", "VMU_ERROR", "VNU", "VNU_ERROR", "VLAMBDA", "VBETA", 
            "U_MAG", "U_ERR", "G_MAG", "G_ERR", "R_MAG", "R_ERR", "I_MAG", "I_ERR",
            "Z_MAG", "Z_ERR", "A_MAG", "A_ERR", "V_MAG", "B_MAG", "IDFLAG", "AST_NUMBER",
            "PROV_ID", "D_COUNTER", "TOTAL_D_COUNT", "RA_COMPUTED", "DEC_COMPUTED", 
            "V_MAG_COMPUTED", "R_DIST", "G_DIST", "PHASE", "OSC_CAT_ID", "H", "G",
            "ARC", "EPOCH_OSC", "A_OSC", "E_OSC", "I_OSC", "LON_OSC", "AP_OSC", 
            "M_OSC", "PROP_CAT_ID", "A_PROP", "E_PROP", "SIN_I_PROP"
        ]
        
        for field, value in zip(field_names, args):
            setattr(self, field, value)
    
    def __repr__(self):
        return f"Asteroid({', '.join([f'{field}={getattr(self, field)}' for field in vars(self)])})"

'''
## 002 ucitavanje objekata
def LoadAst(file_path):
    asteroids = []
    
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) == 59:
                asteroid = Asteroid(*data)
                asteroids.append(asteroid)
    
    return asteroids
'''

##005 modifikovana verzija za ucitavanje 
## ovde pazimo na velicinu magnituda asteroida
## kao i da je velika poluosa veca od 0.5


### 009 ubaceni i filteri za rayliku filtera  GR RI i IZ
def LoadAst(file_path):
    asteroids = []
    
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) == 59:
                try:
                    U_MAG, G_MAG, R_MAG, I_MAG, Z_MAG = map(float, [data[19], data[21], data[23], data[25], data[27]])
                    A_OSC = float(data[49])
                    
                    # Apply filtering conditions
                    if any(mag > 35 for mag in [U_MAG, G_MAG, R_MAG, I_MAG, Z_MAG]) or A_OSC < 0.5:
                        continue  # Skip asteroid if any magnitude is greater than 35 or A_OSC < 0.5
                    
                    GR_diff = G_MAG - R_MAG
                    RI_diff = R_MAG - I_MAG
                    IZ_diff = I_MAG - Z_MAG
                    

## orig                    if (0.3 < GR_diff < 0.7) or (0.1 < RI_diff < 0.4) or (0.01 < IZ_diff < 0.3):
   ##                 if (0.1 < GR_diff < 1.0) or (0.1 < RI_diff < 1.0) or (0.01 < IZ_diff < 1.0):                        
                        
  ##                      continue  # Skip asteroid if it falls within restricted magnitude difference ranges
                    
                    asteroid = Asteroid(*data)
                    asteroids.append(asteroid)
                except ValueError:
                    continue  # Skip invalid entries
    
    return asteroids




# Example usage
asteroid_list = LoadAst("sdssmocadr4.tab")
## asteroid_list = LoadAst("SDSSSample.txt")

print(asteroid_list[:5])  # Print first 5 asteroids with all fields
print("ucitano "+ str(len(asteroid_list)))  # Print first 5 asteroids with all fields

komada = len(asteroid_list)

AllPe=[]

##003 crtanje grafika avs e 

import matplotlib.pyplot as plt

def extract_and_plot(asteroid_list):
    global AllPe  # Ensure AllPe is accessible globally
    # Extract A_OSC and E_OSC values
    AllPa = [float(asteroid.A_OSC) for asteroid in asteroid_list if asteroid.A_OSC.replace('.', '', 1).isdigit()]
    AllPe = [float(asteroid.E_OSC) for asteroid in asteroid_list if asteroid.E_OSC.replace('.', '', 1).isdigit()]
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(AllPa, AllPe, alpha=0.5)
    plt.xlabel("A_OSC (Semi-Major Axis)")
    plt.ylabel("E_OSC (Eccentricity)")
    plt.title("Asteroid Orbital Parameters")
    plt.xlim(0, 6)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


# Run the function
extract_and_plot(asteroid_list)



##004 crtanje svih filtera



def extract_magnitudes(asteroid_list):
    AllPa = [float(asteroid.A_OSC) for asteroid in asteroid_list if asteroid.A_OSC.replace('.', '', 1).isdigit()]
    AllFilU = [float(asteroid.U_MAG) for asteroid in asteroid_list if asteroid.U_MAG.replace('.', '', 1).isdigit()]
    AllFilG = [float(asteroid.G_MAG) for asteroid in asteroid_list if asteroid.G_MAG.replace('.', '', 1).isdigit()]
    AllFilR = [float(asteroid.R_MAG) for asteroid in asteroid_list if asteroid.R_MAG.replace('.', '', 1).isdigit()]
    AllFilI = [float(asteroid.I_MAG) for asteroid in asteroid_list if asteroid.I_MAG.replace('.', '', 1).isdigit()]
    AllFilZ = [float(asteroid.Z_MAG) for asteroid in asteroid_list if asteroid.Z_MAG.replace('.', '', 1).isdigit()]
    
    return AllPa, AllFilU, AllFilG, AllFilR, AllFilI, AllFilZ

def plot_magnitudes(AllPa, AllFilU, AllFilG, AllFilR, AllFilI, AllFilZ):
    filters = [AllFilU, AllFilG, AllFilR, AllFilI, AllFilZ]
    labels = ['U_MAG', 'G_MAG', 'R_MAG', 'I_MAG', 'Z_MAG']
    
    for i in range(5):
        plt.figure(figsize=(8, 6))
        plt.scatter(AllPa, filters[i], alpha=0.5)
        plt.xlabel("A_OSC (Semi-Major Axis)")
        plt.ylabel(labels[i])
        plt.title(f"Asteroid {labels[i]} vs A_OSC")
        plt.xlim(0, 6)
        plt.grid(True)
        plt.show()



# Extract magnitudes
AllPa, AllFilU, AllFilG, AllFilR, AllFilI, AllFilZ = extract_magnitudes(asteroid_list)

# Plot magnitudes
plot_magnitudes(AllPa, AllFilU, AllFilG, AllFilR, AllFilI, AllFilZ)


## provera broja po listama

print("ucitano ast list  "+ str(len(asteroid_list))) 
print("ucitano a "+   str(len(AllPa))) 
print("ucitano e "+   str(len(AllPe))) 
print("ucitano FilU "+ str(len(AllFilU))) 
print("ucitano FilG "+ str(len(AllFilG)))  
print("ucitano FilR "+ str(len(AllFilR))) 
print("ucitano FilI "+ str(len(AllFilI)))  
print("ucitano FilZ "+ str(len(AllFilZ))) 


##007 uvodimo razlike filtera i crtamo:
    
    
    


def CreateFilterDifferences(AllPa, AllFilG, AllFilR, AllFilI, AllFilZ):
    # Compute differences
    ListGR = [g - r for g, r in zip(AllFilG, AllFilR)]
    ListRI = [r - i for r, i in zip(AllFilR, AllFilI)]
    ListIZ = [i - z for i, z in zip(AllFilI, AllFilZ)]
    ListGZ = [g - z for g, z in zip(AllFilG, AllFilZ)]
    ListRZ = [r - z for r, z in zip(AllFilR, AllFilZ)]
    
    # Plot differences
    differences = [(ListGR, 'G-R'), (ListRI, 'R-I'), (ListIZ, 'I-Z'), (ListGZ, 'G-Z'), (ListRZ, 'R-Z')]
    
    for diff_list, label in differences:
        plt.figure(figsize=(8, 6))
        plt.scatter(AllPa, diff_list, alpha=0.5)
        plt.xlabel("A_OSC (Semi-Major Axis)")
        plt.ylabel(f"{label} Magnitude Difference")
        plt.title(f"Asteroid {label} vs A_OSC")
        plt.grid(True)
        plt.show()

# Call the function with extracted data
CreateFilterDifferences(AllPa, AllFilG, AllFilR, AllFilI, AllFilZ)
    
    

##008 odredjujemo PC i crtamo prve grafike  ... btw : theta je preuzet iz rada .. ne znam kako se odredjuje

import math


def CreateListPC(AllFilG, AllFilR, AllFilI):
    theta = 0.370  # Theta in radians
    
    ListGR = [g - r for g, r in zip(AllFilG, AllFilR)]
    ListRI = [r - i for r, i in zip(AllFilR, AllFilI)]
    ListPC = [math.cos(theta) * gr + math.sin(theta) * ri for gr, ri in zip(ListGR, ListRI)]
    
    return ListPC, ListGR, ListRI

def plot_scatter(ListGR, ListRI, ListPC, ListIZ):
    plt.figure(figsize=(8, 6))
    plt.scatter(ListGR, ListRI, alpha=0.5)
    plt.xlabel("G-R")
    plt.ylabel("R-I")
    plt.title("Scatter Plot: G-R vs R-I")
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(ListPC, ListIZ, alpha=0.5)
    plt.xlabel("PC")
    plt.ylabel("I-Z")
    plt.title("Scatter Plot: PC vs I-Z")
    plt.grid(True)
    plt.show()

# Generate ListPC and plot
ListPC, ListGR, ListRI = CreateListPC(AllFilG, AllFilR, AllFilI)
plot_scatter(ListGR, ListRI, ListPC, AllFilZ)



