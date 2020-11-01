"""
Author: Raquel García Bellés
Last update: 28/08/2020
Quantum Computing Technology Group @ IFAE
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy as sc

def function_fit(x_res, y_res, function, param_ini, plot, plotTrial, title = '', xlabel= '', ylabel= '', bounds = None):
    if bounds is None:
        bounds = (-np.inf, np.inf)
    parameters, par_errors = curve_fit(function, x_res, y_res, param_ini, maxfev = 10000, bounds = bounds)
    y_fit = function(x_res, *parameters)
    
    if plot == True:
        plt.figure(figsize=(9,6))
        plt.plot(x_res, y_fit, label="Calibration", color = 'red')
        plt.plot(x_res, y_res, '.', label = 'Results',color = 'black')
        if plotTrial == True:
            y_trial = function(x_res, *param_ini)
            plt.plot(x_res, y_trial, '-', label = "Calib. initial guess", color = 'blue')
        plt.ylim(np.amin(y_res), np.amax(y_res))
        plt.xlim(np.amin(x_res), np.amax(x_res))
        plt.xlabel(xlabel)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        plt.show()
    return parameters, y_fit



def exponential_T1(x_res, y_res, plot = True, plotTrial = True, title = '', xlabel= '', ylabel= ''):
    
    def function(t, A, B, T1):
        return A + B*np.exp(-t/T1)
    
    if isinstance(y_res[0], complex):
        y_res = np.absolute(y_res) # We want to fit the amplitude, not the real and imag part
    max_x = np.amax(x_res)
    min_x = np.amin(x_res)
    A_trial = y_res[np.argmax(x_res)]
    B_trial = y_res[np.argmin(x_res)] - A_trial
    T1_trial = max_x/2
    
    param_ini = [A_trial, B_trial, T1_trial]
    parameters, y_fit = function_fit(x_res, y_res, function, param_ini, plot, plotTrial, title = title, xlabel= xlabel, ylabel= ylabel)
    A = parameters[0]
    B = parameters[1]
    T1 = parameters[2]
    
    return A, B, T1



def cosine_squared(x_res, y_res, plot=True, plotTrial=True, title = '', xlabel= '', ylabel= ''):
    """
    Fit the calibration result to a general cosinus
    """
    def function(x, period, phi, A, c):
        return A*np.cos(2*np.pi/period*x + phi)**2+c
    
    # First some initial trials are obtained so that the fit is succesful
    if isinstance(y_res[0], complex):
        y_res = np.absolute(y_res) # We want to fit the amplitude, not the real and imag part
    max_y = np.amax(y_res)
    min_y = np.amin(y_res)
    A_trial = max_y - min_y
    c_trial = min_y
    
    #getting an estimate of the period using the FFT
    yf = sc.fft(y_res)
    absyf = 2/len(x_res)*np.abs(yf[0:len(x_res)//2])
    xf = np.linspace(0, 1/(2*(x_res[len(x_res)-1]/len(x_res))), len(x_res)//2)
    i = np.argmax(absyf)
    if xf[i] == 0:
        absyf = np.delete(absyf, i)
        xf = np.delete(xf, i)
        i = np.argmax(absyf)
    if x_res[0]<0:
        period_trial = 4/xf[i] #If x_res ranges from negative to positive values we have to use 4/xf[i]
    else:
        period_trial = 2/xf[i]
        #If x_res only has positive values, then we have to use 2/xf[i], because it is a cosine squared
    phi_0_trial = 0

    param_ini = [period_trial, phi_0_trial, A_trial, c_trial]
    parameters, y_fit = function_fit(x_res, y_res, function, param_ini, plot, plotTrial, title = title, xlabel= xlabel, ylabel= ylabel)

    # mapping parameters
    period = parameters[0]
    phi = parameters[1]
    A = parameters[2]
    c = parameters[3]
    
    #pipulse will be period/4 because we are fitting a cosine squared
    
    return period, phi, A, c, A_trial



def exponentialcos_T2(x_res, y_res, plot = True, plotTrial = True, title = '', xlabel= '', ylabel= ''):
    
    def function(t, A, B, phi, period, T2):
        return A + B*np.cos(2*np.pi/period*t + phi)*np.exp(-t/T2)
    
    if isinstance(y_res[0], complex):
        y_res = np.absolute(y_res) # We want to fit the amplitude, not the real and imag part
    max_x = np.amax(x_res)
    min_x = np.amin(x_res)
    A_trial = y_res[np.argmax(x_res)]
    B_trial = y_res[np.argmin(x_res)] - A_trial
    T2_trial = max_x/2
    
    #getting an estimate of the period using the FFT
    yf = sc.fft(y_res)
    absyf = 2/len(x_res)*np.abs(yf[0:len(x_res)//2])
    xf = np.linspace(0, 1/(2*(x_res[len(x_res)-1]/len(x_res))), len(x_res)//2)
    i = np.argmax(absyf)
    if xf[i] == 0:
        absyf = np.delete(absyf, i)
        xf = np.delete(xf, i)
        i = np.argmax(absyf)
        
    if x_res[0]<0:
        period_trial = 2/xf[i] #If x_res ranges from negative to positive values we have to use 2/xf[i]
    else:
        period_trial = 1/xf[i]
        #If x_res only has positive values, then we have to use 1/xf[i]
   
    phi_trial = 0
    
    param_ini = [A_trial, B_trial, phi_trial, period_trial, T2_trial]
    parameters, y_fit = function_fit(x_res, y_res, function, param_ini, plot, plotTrial, title = title, xlabel= xlabel, ylabel= ylabel)
    A = parameters[0]
    B = parameters[1]
    phi = parameters[2]
    period = parameters[3]
    T2 = parameters[4]
    
    return A, B, phi, period, T2


def gaussiancos_TphiG(x_res, y_res, T1, plot = True, plotTrial = True, title = '', xlabel= '', ylabel= ''):
    # Fit the Ramsey measurement taking into account the 1/f Gaussian noise
    # See page 16 of A Quantum Engineer's Guide to Supercomputing Qubits
    def function(t, A, B, phi, period, TphiG):
        return A + B*np.cos(2*np.pi/period*t + phi)*np.exp(-t**2/TphiG**2)
     
    T1 = T1
    TphiG_trial = T1
    
    if isinstance(y_res[0], complex):
        y_res = np.absolute(y_res) # We want to fit the amplitude, not the real and imag part
    
    A_trial, B_trial, phi_trial, period_trial, T2 = exponentialcos_T2(
        x_res, y_res, plot, plotTrial, title = title, xlabel= xlabel, ylabel= ylabel) 
    
    param_ini = [0, B_trial, phi_trial, period_trial, TphiG_trial]
     
    #print('T2 = ', T2) 
    
    # Re-scale the data
    E = y_res - A_trial # Eliminate the off-set
    Y = np.zeros(len(x_res))   
    for i in range(0, len(x_res)):
        Y[i] = E[i]*np.exp(x_res[i]/T1/2) # Divide the data by exp(-t/2T1)
        
    parameters, y_fit = function_fit(x_res, Y, function, param_ini, plot, plotTrial, title = title, xlabel= xlabel, ylabel= ylabel)
    A = parameters[0]
    B = parameters[1]
    phi = parameters[2]
    period = parameters[3]
    TphiG = parameters[4]
    
    return A, B, phi, period, abs(TphiG), T2


def lorentzian(x_res, y_res, x_res1, y_res1, plot = True, plotTrial = True, title = '', xlabel= '', ylabel= ''):
    # It can be used to calibrate the resonator frequency
    def function(x, A, kappa, C, f, A1, kappa1, f1):
        return A/((x-f)**2+kappa**2/4) + A1/((x-f1)**2+kappa1**2/4) + C
    # We consider the sum of two lorentzian functions because there is some residual population
    # in the state 0 or 1
    
    if isinstance(y_res[0], complex):
        y_res = np.absolute(y_res) # We want to fit the amplitude, not the real and imag part
    if isinstance(y_res1[0], complex):
        y_res1 = np.absolute(y_res1) 
        
    # we are assuming that the peak of the lorentzian function goes upwards
    # the state 0 and state 1 results should have the same dimensions
    C_trial = np.amin(y_res)
    f_trial = x_res[np.argmax(y_res)]
    f1_trial = x_res1[np.argmax(y_res1)]
    kappa_trial = (1.E7)*2*np.pi
    kappa1_trial = (1.E7)*2*np.pi
    A_trial = (np.amax(y_res) - C_trial)*kappa_trial**2/4
    A1_trial = (y_res[np.argmax(y_res1)] - C_trial)*kappa_trial**2/4
    param_ini = [A_trial, kappa_trial, C_trial, f_trial, A1_trial, kappa1_trial, f1_trial]
    
    parameters, y_fit = function_fit(x_res, y_res, function, param_ini, plot, plotTrial, title = title, xlabel= xlabel, ylabel= ylabel)
    A = parameters[0]
    kappa = parameters[1]
    C = parameters[2]
    f = parameters[3]
    
    return A, kappa, C, f


def linear(x_res, y_res, plot = True, plotTrial = True, title = '', xlabel= '', ylabel= ''):
    
    def function(t, A, B):
        return A + B*t
    
    if isinstance(y_res[0], complex):
        y_res = np.absolute(y_res) # We want to fit the amplitude, not the real and imag part
    
    A_trial = np.mean(y_res)
    B_trial = (y_res[len(y_res)-1]-y_res[0])/(x_res[len(x_res)-1]-x_res[0])
    param_ini = [A_trial, B_trial]
    parameters, y_fit = function_fit(x_res, y_res, function, param_ini, plot, plotTrial, title = title, xlabel= xlabel, ylabel= ylabel)
    A = parameters[0] 
    B = parameters[1]
    return A, B

def ramseyPhotons(x_res, y_res, ResonatorKappa, DispersiveShift, T2Echo, ArtificialDetuning, Amax, Amin, bounds, plot = True, plotTrial = True, title = '', xlabel= '', ylabel= ''):
    
    def function(t, phi, n0):
        tau = (1-np.exp(-(ResonatorKappa*2*np.pi + 2*1j*DispersiveShift*2*np.pi)*t))/(ResonatorKappa + 2*1j*DispersiveShift)
        return 1/2*(1 - np.imag(np.exp(-(1/T2Echo + 2*np.pi*ArtificialDetuning*1j)*t + (phi - 2*n0*DispersiveShift*tau)*1j)))
     
    
    if isinstance(y_res[0], complex):
        y_res = np.absolute(y_res) # We want to fit the amplitude, not the real and imag part
    
    y_rescaled = []
    for i in range(0, len(y_res)):
        y_resc = (y_res[i] - Amin)/(Amax - Amin)
        y_rescaled.append(y_resc)
    phi_trial = 0
    n0_trial = 2
    param_ini = [phi_trial, n0_trial]
        
    parameters, y_fit = function_fit(x_res, y_rescaled, function, param_ini, plot, plotTrial, bounds = bounds, title = title, xlabel= xlabel, ylabel= ylabel)

    phi = parameters[0]
    n0 = parameters[1]
    
    return phi, n0