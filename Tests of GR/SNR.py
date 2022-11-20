#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
​
# using upper limit as f_isco or 700 in f_data gives the same SNR because the integral goes only till f_isco anyway
# 
# SNR vs total mass for 3 differnet mass ratios and SNR vs total distance for equal masses
​
# In[3]:
​
​
#importing the required packages
​
import math
import numpy as np
import scipy.constants
import cmath
import matplotlib
from matplotlib import pyplot as plt
​
​
# In[861]:
​
​
m_1_raw = 25
m_2_raw = 3 * m_1_raw #mass of body 2 in SM
m_g_raw = 10 ** (-32) #rest mass of graviton
D_L_raw = D_raw = 100 #distance to event in Mpc
​
​
# In[862]:
​
​
# input masses have to be in solar masses
# using 1 solar mass = 1.9891 * 10**30 kg
# assuming raw mass is in SM
​
m_1_kg = m_1_raw * 1.9891 * 10**30 #mass in kg
m_1 = m_1_kg * scipy.constants.G / scipy.constants.c ** 3 #mass of 1st coalescing body in seconds
print(m_1)
​
m_2_kg = m_2_raw * 1.9891 * 10**30 #mass in kg
m_2 = m_2_kg * scipy.constants.G / scipy.constants.c ** 3 #mass of second coalescing body in seconds
print(m_2)
​
​
# In[863]:
​
​
m_g_kg = m_g_raw * 1.78266192 * 10 ** (-36) #mass of graviton in kg using 1 ev/c**2 relation
m_g = m_g_kg * scipy.constants.G / scipy.constants.c ** 3 #mass of graviton in seconds
print(m_g)
​
​
# In[864]:
​
​
#assuming raw distance is in MPc
#using 1 Mpc = 3.08567758 * 10**22 m
​
D_L_meter = D_L_raw * 3.08567758 * 10**22 #distance to event in m
D_meter = D_L_meter
D_L = D_L_meter / scipy.constants.c
D = D_L
print(D_L)
​
​
# In[865]:
​
​
Z = 1 #redshift
t_c = 0 #fiducial time
phi_c = 0 #fiducial phase
lambda_g = scipy.constants.h/m_g * scipy.constants.c #graviton compton wavelength
​
​
# In[866]:
​
​
m = m_1 + m_2 #total mass is the sum of individual masses of coalescing bodies
#since m_1 and m_2 are in s, m is in s
#m_1 and m_2 are individual masses
​
mu = (m_1 * m_2) / m #reduced mass of system
#mu is unitless
​
​
# In[867]:
​
​
f_isco = (6**(3/2) * math.pi * (m_1 + m_2))**(-1)
print(f_isco)
​
​
# In[868]:
​
​
f_data = np.arange(50, f_isco)
print(f_data)
​
​
# In[869]:
​
​
eta = mu / m #reduced mass parameter 
M_e = eta ** (3/5) * m #source chirp mass
M = (1 + Z) * M_e #measured chirp mass
​
​
# In[870]:
​
​
def u(f):
    U = math.pi * M * f
    return U
​
​
# In[871]:
​
​
beta = (math.pi * D * M) / (lambda_g ** 2 * (1 + Z))
print(beta)
​
​
# In[872]:
​
​
def amplitude(f):
    A = math.sqrt(math.pi / 30) * (M ** 2 / D_L) * u(f) ** (-7/6)
    return A
​
​
# In[873]:
​
​
def psi(f): 
       phase = (2 * math.pi * f * t_c) - phi_c - math.pi/4. + (3./128. * u(f)**(-5./3.)) 
       - (beta * u(f)**(-1.)) 
       + (5./96. * (743./336.) + (11./4. * eta) * eta**(-2./5.) * u(f)**(-1.)) 
       - (3. * math.pi/8. * eta**(-3./5.) * u(f)**(-2./3.))
       return phase
​
​
# In[874]:
​
​
def h(f):
        H = amplitude(f) * scipy.constants.e**(1j * psi(f))
        return H
​
​
# In[875]:
​
​
def psd(f):
    x = f/245.4
    return 10**(-48.) * (0.0512*x**(-4) + 0.2935 * x**(9/4) + 2.7951* x**(3/2) - 6.5080 * x**(3/4) + 17.7622)
    
​
​
# In[876]:
​
​
from scipy.integrate import quad
​
def integrand(f):
    return 4 * h(f) * np.conj(h(f)) / psd(f)
​
​
# In[877]:
​
​
ans = quad(integrand, 20, f_isco) #lower cut off is 20 Hz for detectors and upper value is f_ISCO
print(ans)
​
​
# In[878]:
​
​
snr = np.sqrt(np.real(ans))
print(snr)
​
​
# In[881]:
​
​
#Storing M and SNR values for equal mass bodies
# f is from 50 to f_isco in this case
totalmass = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
TM2 = np.linspace(15, 105, 7)
TM3 = np.linspace(20, 100, 5)
print(TM3)
SNR = np.array([5.86171277e+01, 
                9.91994535e+01, 
                1.29905765e+02, 
                1.51870675e+02, 
                1.65718363e+02, 
                1.72419327e+02, 
                1.73439609e+02, 
                1.70445776e+02, 
                1.64907779e+02, 
                1.57890847e+02])
​
SNR2 = np.array([7.56936627e+01, 
                 1.22476330e+02, 
                 1.50624233e+02,
                 1.62558500e+02,
                 1.62487746e+02,
                 1.55476545e+02,
                 1.45238524e+02
                ])
​
SNR3 = np.array([8.59092468e+01,
                 1.31523863e+02,
                 1.49319517e+02,
                 1.47610372e+02,
                 1.36737484e+02
                ])
​
plt.plot(totalmass, SNR, label = 'Mass Ratio = 1')
plt.plot(TM2, SNR2, label = 'Mass Ratio = 2')
plt.plot(TM3, SNR3, label = 'Mass Ratio = 3')
plt.xlabel("Total Mass (solar mass)")
plt.ylabel("SNR")
plt.legend()
​
​
# In[650]:
​
​
# distance vs SNR for equal mass bodies (m1 = m2 = 30 SM)
dist = np.linspace(100, 1000, 19)
SNR1 = np.array([1.72419327e+02, 
                1.14946218e+02,
                8.62096635e+01,
                6.89677308e+01,
                5.74731090e+01,
                4.92626648e+01,
                4.31048317e+01,
                3.83154060e+01,
                3.44838654e+01,
                3.13489685e+01,
                2.87365545e+01,
                2.65260503e+01,
                2.46313324e+01,
                2.29892436e+01,
                2.15524159e+01,
                2.02846267e+01,
                1.91577030e+01,
                1.81494028e+01,
                1.72419327e+01
               ])
print(dist, SNR1)
​
plt.plot(dist, SNR1)
plt.xlabel("Distance (MPc)")
plt.ylabel("SNR")
​
​
# In[ ]:

