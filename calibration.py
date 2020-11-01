"""
Author: Raquel García Bellés
Last update: 31/08/2020
Quantum Computing Technology Group @ IFAE
"""


import fittings as fit
import numpy as np
from Labber import ScriptTools
import Labber
import os
import json
import matplotlib.pyplot as plt
import math

currentPath = os.path.dirname(os.path.abspath(__file__))

class Calibration:
    
    def __init__(self, file = None, Reset = True, StableFile = 'StableParameters.txt', plot = True): 
        self._Reset = Reset
        self._Plot = plot
        with open(StableFile) as json_stable:
            stable = json.load(json_stable)
            
        self._QuickSynPower = stable['QuickSyn Power'] # in dBm
        self._RFSourcePower = stable['RF Source Power'] # in dBm
        self._SSBFreq = stable['SSBFreqCavity'] # modulation frequency cavity in Hz
        self._SSBFreqQubit = stable['SSBFreqQubit'] # modulation frequency qubit in Hz
        self._PulseWidth = stable['PulseWidth'] # width of the pulses in seconds. #Note that pulses may overlap if this value is increased from 7ns.
        self._ReadoutAmplitude = stable['ReadoutAmplitude'] #in V
        self._ReadoutDuration = stable['ReadoutDuration'] #in seconds
        # The file 'Reset Rabi Qutrit-pp.hdf5' has to be modified manually if the Qubit Frequency or the 1-2 transition frequency change significantly 
        
        if file is None:      
            # pre-stored values in case no file is given
            self._QubitFreq = 4.825E9 #The SSB freq is 70MHz (LO freq is 4.895E9)
            self._PiPulse = 0.340
            self._Amax = 0.0035 # Maximum value of the rabi oscilations
            self._Amin = 0.0005 # Minimum value of the rabi oscilations
            self._T1 = 1.569854926866105e-05
            self._T2Ramsey = 9.074181775468289e-06
            self._T2Echo = 4.890671631676906e-06
            self._TphiG = 1.7352252683597397e-05
            self._ResonatorFreq0 = 7754604532687.694 #mHz resonator frequency when qubit is in state 0 
            self._ResonatorFreq1 = 7753025774183.293 #mHz resonator frequency when qubit is in state 1
            self._ResetRabiAmplitude = 3E-3
            self._ResetCavityAmplitude = 120E-3
            self._ResetLength = 2E-6
            self._ResetSpacing = 2E-6
            self._DRAG = -590E-12
            self._DispersiveShift = (self._ResonatorFreq0 - self._ResonatorFreq1)*10**(-3)/2
            self._ResonatorKappa = 1.86E6 #From Rafel Luque Master thesis
        else:
            with open(file) as json_file:
                parameters = json.load(json_file)
            
            self._QubitFreq = parameters['QubitFreq']
            self._PiPulse = parameters['PiPulse']
            self._Amax = parameters['Amax']
            self._Amin = parameters['Amin']
            self._T1 = parameters['T1']
            self._T2Ramsey = parameters['T2Ramsey']
            self._T2Echo = parameters['T2Echo']
            self._TphiG = parameters['TphiG']
            self._ResonatorFreq0 = parameters['ResonatorFreq0']
            self._ResonatorFreq1 = parameters['ResonatorFreq1']
            self._ResetRabiAmplitude = parameters['ResetRabiAmplitude']
            self._ResetCavityAmplitude = parameters['ResetCavityAmplitude']
            self._ResetLength = parameters['ResetLength']
            self._ResetSpacing = parameters['ResetSpacing']
            self._DRAG = parameters['DRAG']
            self._DispersiveShift = parameters['DispersiveShift']
            self._ResonatorKappa = parameters['ResonatorKappa']
            
    def toFile(self, output_file):
        # save new parameters to output_file
        parametersOut = {'QubitFreq': self._QubitFreq,
                         'PiPulse' : self._PiPulse,
                         'Amax' : self._Amax,
                         'Amin' : self._Amin,
                        'T1' : self._T1,
                        'T2Ramsey' : self._T2Ramsey,
                        'T2Echo' : self._T2Echo,
                        'TphiG' : self._TphiG,
                        'ResonatorFreq0' : self._ResonatorFreq0,
                        'ResonatorFreq1' : self._ResonatorFreq1,
                        'ResetRabiAmplitude': self._ResetRabiAmplitude, 
                         'ResetCavityAmplitude': self._ResetCavityAmplitude,
                        'ResetLength': self._ResetLength,
                        'ResetSpacing': self._ResetSpacing,
                        'DRAG': self._DRAG, 
                        'DispersiveShift': self._DispersiveShift,
                        'ResonatorKappa': self._ResonatorKappa}
        with open(output_file, 'w') as output:
            json.dump(parametersOut, output)
        
    def getPiPulse(self, plot = None, Reset = None):
        if plot is None:
            plot = self._Plot
        
        log_name = 'Rabi Test with Reset_5.hdf5'
        log_output = 'Rabi Test with Reset_5_Results.hdf5X'
        measurement = ScriptTools.MeasurementObject(
            os.path.join(currentPath,log_name), os.path.join(currentPath, log_output))
        #measurement.updateValue('Control Pulse - Width #3', 8E-9)
        measurement.updateValue('QubitFreq', self._QubitFreq)
        measurement.updateValue('CavityFreq', self._ResonatorFreq0)
        measurement.updateValue('SSBFreq', self._SSBFreq) #cavity
        measurement.updateValue('SSBFreqQubit', self._SSBFreqQubit) #qubit
        measurement.updateValue('RS IQ Source - Power', self._RFSourcePower)
        measurement.updateValue('QS RF Source - Power', self._QuickSynPower)
        measurement.updateValue('Control Pulse - DRAG scaling', self._DRAG)
        measurement.updateValue('Control Pulse - Width #3', self._PulseWidth)
        measurement.updateValue('Control Pulse - Plateau #4', self._ReadoutDuration)
        measurement.updateValue('Control Pulse - Amplitude #4', self._ReadoutAmplitude)
        
        if Reset is None:
            Reset = self._Reset
        if Reset == True:
            #use reset sequence
            measurement.updateValue('Reset', 1)
            measurement.updateValue('ResetLength', self._ResetLength)
            measurement.updateValue('ResetSpacing', self._ResetSpacing)
            measurement.updateValue('ResetRabiAmplitude', self._ResetRabiAmplitude)
            measurement.updateValue('ResetCavityAmplitude', self._ResetCavityAmplitude)
        if Reset == False:
            measurement.updateValue('Reset', 0)
        (x, y) = measurement.performMeasurement()
        period, phi, A, c, A_trial = fit.cosine_squared(x, y, plot, title = 'Rabi oscillations', xlabel= 'Pulse #3 Amplitude (V)', ylabel= 'Digitizer voltage')
        
        #returns the amplitude of a PiPulse (Pulse width is fixed)
        self._PiPulse = period/4 #update PiPulse
        self._Amax = A + c
        self._Amin = c
        
        # Since we use the frequency of the cavity corresponding to the qubit in state 0 for the readout pulse,
        # the maximum amplitude corresponds to the state 0 and the minimum to state 1. Therefore, 
        # given an amplitude A, the population of the excited state will be: 
        # Pe = (A_max - A)/(A_max - A_min)
        # and the population of the ground state will be:
        # Pg = (A - A_min)/(A_max - A_min)
        return self._PiPulse, self._Amax, self._Amin
        
    def getT1(self, plot = None, Reset = None):
        if plot is None:
            plot = self._Plot
        #it is recommended to run getPiPulse() first
        log_name = 'T1 with Reset.hdf5'
        log_output = 'T1 with Reset_Results.hdf5X'
        
        measurement = ScriptTools.MeasurementObject(
            os.path.join(currentPath,log_name), os.path.join(currentPath, log_output))
        measurement.updateValue('PiPulse', self._PiPulse)
        measurement.updateValue('QubitFreq', self._QubitFreq)
        measurement.updateValue('QS RF Source - Frequency', self._ResonatorFreq0 + self._SSBFreq*10**3)
        measurement.updateValue('SSBFreq', self._SSBFreq) #cavity
        measurement.updateValue('SSBFreqQubit', self._SSBFreqQubit) #qubit
        measurement.updateValue('RS IQ Source - Power', self._RFSourcePower)
        measurement.updateValue('QS RF Source - Power', self._QuickSynPower)
        measurement.updateValue('Control Pulse - DRAG scaling', self._DRAG)
        measurement.updateValue('Control Pulse - Width #3', self._PulseWidth)
        measurement.updateValue('Control Pulse - Plateau #4', self._ReadoutDuration)
        measurement.updateValue('Control Pulse - Amplitude #4', self._ReadoutAmplitude)
        
        if Reset is None:
            Reset = self._Reset
        if Reset == True:
            #use reset sequence
            measurement.updateValue('Reset', 1)
            measurement.updateValue('ResetLength', self._ResetLength)
            measurement.updateValue('ResetSpacing', self._ResetSpacing)
            measurement.updateValue('ResetRabiAmplitude', self._ResetRabiAmplitude)
            measurement.updateValue('ResetCavityAmplitude', self._ResetCavityAmplitude)
        if Reset == False:
            measurement.updateValue('Reset', 0)
        (x, y) = measurement.performMeasurement()
        A, B, T1 = fit.exponential_T1(x, y, plot, title = 'T1 calibration', xlabel= 'Time after Pi Pulse (s)', ylabel= 'Digitizer voltage')
        
        self._T1 = T1 #update T1
        
        return self._T1
    
    def getQubitFreq(self, plot = None, Reset = None, ArtificialDetuning = 2E6):
        if plot is None:
            plot = self._Plot
        # use Ramsey oscillations to find the detuning
           
        log_name = 'Ramsey with Reset QubitFreq.hdf5'
        log_output = 'Ramsey with Reset QubitFreq_Results.hdf5X'
            
        measurement = ScriptTools.MeasurementObject(
            os.path.join(currentPath,log_name), os.path.join(currentPath, log_output))
        
        measurement.updateValue('PiPulse', self._PiPulse)
        measurement.updateValue('RS IQ Source - Frequency', self._QubitFreq + self._SSBFreqQubit)
        measurement.updateValue('QS RF Source - Frequency', self._ResonatorFreq0 + self._SSBFreq*10**3)
        measurement.updateValue('SSBFreq', self._SSBFreq) #cavity
        measurement.updateValue('SSBFreqQubit', self._SSBFreqQubit) #qubit
        measurement.updateValue('RS IQ Source - Power', self._RFSourcePower)
        measurement.updateValue('QS RF Source - Power', self._QuickSynPower)
        measurement.updateValue('Control Pulse - DRAG scaling', self._DRAG)
        measurement.updateValue('Control Pulse - Width #3', self._PulseWidth)
        measurement.updateValue('Control Pulse - Width #4', self._PulseWidth)
        measurement.updateValue('Control Pulse - Plateau #5', self._ReadoutDuration)
        measurement.updateValue('Control Pulse - Amplitude #5', self._ReadoutAmplitude)
        
        if Reset is None:
            Reset = self._Reset
        if Reset == True:
            #use reset sequence
            measurement.updateValue('Reset', 1)
            measurement.updateValue('ResetLength', self._ResetLength)
            measurement.updateValue('ResetSpacing', self._ResetSpacing)
            measurement.updateValue('ResetRabiAmplitude', self._ResetRabiAmplitude)
            measurement.updateValue('ResetCavityAmplitude', self._ResetCavityAmplitude)
        if Reset == False:
            measurement.updateValue('Reset', 0)
            
        DetuningIn = ArtificialDetuning #Artificially introduced detuning
        measurement.updateValue('Artificial detuning', DetuningIn)
        
        (x, y) = measurement.performMeasurement()
        A, B, phi, period, T2 = fit.exponentialcos_T2(x, y, plot, title = 'Qubit Frequency Calibration', xlabel= 'Time between Pi/2 pulses (s)', ylabel= 'Digitizer voltage')
        
        Detuning = 1/period
        print('Detuning = ', Detuning)
        Delta = DetuningIn - Detuning #if negative means the qubit frequency was lower, if positive it was higher
        self._QubitFreq = self._QubitFreq + Delta

        return self._QubitFreq
        
    def getT2Ramsey(self, plot = None, TphiG = False, Reset = None, ArtificialDetuning = 2E6):
        if plot is None:
            plot = self._Plot
            
        log_name = 'T2 Ramsey with Reset.hdf5'
        log_output = 'T2 Ramsey with Reset.hdf5X'
        
        measurement = ScriptTools.MeasurementObject(
            os.path.join(currentPath,log_name), os.path.join(currentPath, log_output))
        
        measurement.updateValue('PiPulse', self._PiPulse)
        measurement.updateValue('RS IQ Source - Frequency', self._QubitFreq + self._SSBFreqQubit)
        measurement.updateValue('QS RF Source - Frequency', self._ResonatorFreq0 + self._SSBFreq*10**3)
        measurement.updateValue('SSBFreq', self._SSBFreq) #cavity
        measurement.updateValue('SSBFreqQubit', self._SSBFreqQubit) #qubit
        measurement.updateValue('RS IQ Source - Power', self._RFSourcePower)
        measurement.updateValue('QS RF Source - Power', self._QuickSynPower)
        measurement.updateValue('Control Pulse - DRAG scaling', self._DRAG)
        measurement.updateValue('Control Pulse - Width #3', self._PulseWidth)
        measurement.updateValue('Control Pulse - Width #4', self._PulseWidth)
        measurement.updateValue('Control Pulse - Plateau #5', self._ReadoutDuration)
        measurement.updateValue('Control Pulse - Amplitude #5', self._ReadoutAmplitude)
        
        if Reset is None:
            Reset = self._Reset
        if Reset == True:
            #use reset sequence
            measurement.updateValue('Reset', 1)
            measurement.updateValue('ResetLength', self._ResetLength)
            measurement.updateValue('ResetSpacing', self._ResetSpacing)
            measurement.updateValue('ResetRabiAmplitude', self._ResetRabiAmplitude)
            measurement.updateValue('ResetCavityAmplitude', self._ResetCavityAmplitude)
        if Reset == False:
            measurement.updateValue('Reset', 0)
            
        DetuningIn = ArtificialDetuning #Artificially introduced detuning
        measurement.updateValue('Artificial detuning', DetuningIn)
        
        (x, y) = measurement.performMeasurement()
        
        if TphiG == False:
            A, B, phi, period, T2 = fit.exponentialcos_T2(x, y, plot, title = 'T2 Ramsey Calibration', xlabel= 'Time between Pi/2 pulses (s)', ylabel= 'Digitizer voltage')
            self._T2Ramsey = T2
            return self._T2Ramsey
        
        if TphiG == True:
            A, B, phi, period, TphiG, T2 = fit.gaussiancos_TphiG(x, y, self._T1, plot, title = 'T2 and TphiG Calibration', xlabel= 'Time between Pi/2 pulses (s)', ylabel= 'Digitizer voltage (TphiG: re-scaled)')
            self._TphiG = TphiG
            self._T2Ramsey = T2
        
            return self._T2Ramsey, self._TphiG
        
    def getT2Echo(self, plot = None, Reset = None, ArtificialDetuning = 2E6):
        if plot is None:
            plot = self._Plot
            
        log_name = 'T2 Echo with Reset.hdf5'
        log_output = 'T2 Echo with Reset_Results.hdf5X'
        
        measurement = ScriptTools.MeasurementObject(
        os.path.join(currentPath,log_name), os.path.join(currentPath, log_output))
        
        measurement.updateValue('PiPulse', self._PiPulse)
        measurement.updateValue('RS IQ Source - Frequency', self._QubitFreq + self._SSBFreqQubit)
        measurement.updateValue('QS RF Source - Frequency', self._ResonatorFreq0 + self._SSBFreq*10**3)
        measurement.updateValue('SSBFreq', self._SSBFreq) #cavity
        measurement.updateValue('SSBFreqQubit', self._SSBFreqQubit) #qubit
        measurement.updateValue('RS IQ Source - Power', self._RFSourcePower)
        measurement.updateValue('QS RF Source - Power', self._QuickSynPower)
        measurement.updateValue('Control Pulse - DRAG scaling', self._DRAG)
        measurement.updateValue('Control Pulse - Width #3', self._PulseWidth)
        measurement.updateValue('Control Pulse - Width #4', self._PulseWidth)
        measurement.updateValue('Control Pulse - Width #5', self._PulseWidth)
        measurement.updateValue('Control Pulse - Plateau #6', self._ReadoutDuration)
        measurement.updateValue('Control Pulse - Amplitude #6', self._ReadoutAmplitude)
        
        if Reset is None:
            Reset = self._Reset
        if Reset == True:
            #use reset sequence
            measurement.updateValue('Reset', 1)
            measurement.updateValue('ResetLength', self._ResetLength)
            measurement.updateValue('ResetSpacing', self._ResetSpacing)
            measurement.updateValue('ResetRabiAmplitude', self._ResetRabiAmplitude)
            measurement.updateValue('ResetCavityAmplitude', self._ResetCavityAmplitude)
        if Reset == False:
            measurement.updateValue('Reset', 0)
            
        DetuningIn = ArtificialDetuning #Artificially introduced detuning
        measurement.updateValue('Artificial detuning', DetuningIn)
        
        (x, y) = measurement.performMeasurement()
        A, B, T2echo = fit.exponential_T1(x, y, plot, title = 'T2 Echo Calibration', xlabel= 'Time between Pi/2 pulses (s)', ylabel= 'Digitizer voltage')
        
        self._T2Echo = T2echo
        
        return self._T2Echo
    
    def getResonatorFreq(self, plot = None, Reset = None):
        if plot is None:
            plot = self._Plot
            
        log_name = 'Resonator Frequency with Reset.hdf5'
        log_output = 'Resonator Frequency with Reset_Results.hdf5X'
        
        measurement = ScriptTools.MeasurementObject(
            os.path.join(currentPath,log_name), os.path.join(currentPath, log_output))
        measurement.updateValue('PiPulse', self._PiPulse)
        measurement.updateValue('RS IQ Source - Frequency', self._QubitFreq + self._SSBFreqQubit)
        measurement.updateValue('CavityFreq', self._ResonatorFreq0)
        measurement.updateValue('SSBFreq', self._SSBFreq) #cavity
        measurement.updateValue('SSBFreqQubit', self._SSBFreqQubit) #qubit
        measurement.updateValue('RS IQ Source - Power', self._RFSourcePower)
        measurement.updateValue('QS RF Source - Power', self._QuickSynPower)
        measurement.updateValue('Control Pulse - DRAG scaling', self._DRAG)
        measurement.updateValue('Control Pulse - Width #3', self._PulseWidth)
        measurement.updateValue('Control Pulse - Plateau #4', self._ReadoutDuration)
        measurement.updateValue('Control Pulse - Amplitude #4', self._ReadoutAmplitude)
        
        if Reset is None:
            Reset = self._Reset
        if Reset == True:
            #use reset sequence
            measurement.updateValue('Reset', 1)
            measurement.updateValue('ResetLength', self._ResetLength)
            measurement.updateValue('ResetSpacing', self._ResetSpacing)
            measurement.updateValue('ResetRabiAmplitude', self._ResetRabiAmplitude)
            measurement.updateValue('ResetCavityAmplitude', self._ResetCavityAmplitude)
        if Reset == False:
            measurement.updateValue('Reset', 0)
        
        #state 0
        measurement.updateValue('State', 0)
        (x0, y0) = measurement.performMeasurement()
        
        #state 1
        measurement.updateValue('State', 1)
        (x1, y1) = measurement.performMeasurement()       
        
        # Now we fit the results
        # We use the values from the two measurements to predict the trial parameters
        A0, kappa0, C0, f0= fit.lorentzian(x0, y0, x1, y1, plot, title = 'Qubit in the ground state', xlabel= 'QS RF Source - Frequency (mHz)', ylabel= 'Digitizer voltage')     
        print('f0 = ', f0)
        print('kappa0 =', kappa0)
   
        A1, kappa1, C1, f1= fit.lorentzian(x1, y1, x0, y0, plot, title = 'Qubit in the excited state', xlabel = 'QS RF Source - Frequency', ylabel = 'Digitizer voltage')     
        
        print('kappa1 =', kappa1)     
        print('f1 = ', f1)
        SSBFreq = self._SSBFreq*10**3 #unit is mHz
        self._ResonatorFreq0 = f0 - SSBFreq #the real cavity frequency is stored
        self._ResonatorFreq1 = f1 - SSBFreq
        self._DispersiveShift = (self._ResonatorFreq0 - self._ResonatorFreq1)*10**(-3)
        
        return self._ResonatorFreq0, self._ResonatorFreq1
    
    
    def getDRAG(self, RandomizedBenchmarking = True, points = 20, Cliffords = 50, plot = None):
        if Cliffords > 50:
            print('Using more than 50 Clifford gates might require to change the Test Rand Bench_550scaling_2.hdf5 Labber file.')
        if plot is None:
            plot = self._Plot
            
        log_name = 'DRAG.hdf5'
        log_output = 'DRAG_Results.hdf5X'
        
        measurement = ScriptTools.MeasurementObject(
            os.path.join(currentPath,log_name), os.path.join(currentPath, log_output))
        measurement.updateValue('PiPulse', self._PiPulse)
        measurement.updateValue('RS IQ Source - Frequency', self._QubitFreq + self._SSBFreqQubit)
        measurement.updateValue('QuickSyn Signal Generator - Frequency', self._ResonatorFreq0 + self._SSBFreq*10**3)
        measurement.updateValue('SSBFreq', self._SSBFreq) #cavity
        measurement.updateValue('SSBFreqQubit', self._SSBFreqQubit) #qubit
        measurement.updateValue('RS IQ Source - Power', self._RFSourcePower)
        measurement.updateValue('QuickSyn Signal Generator - Power', self._QuickSynPower)
        measurement.updateValue('Control Pulse - Width #1', self._PulseWidth)
        measurement.updateValue('Control Pulse - Width #2', self._PulseWidth)
        measurement.updateValue('Control Pulse - Plateau #3', self._ReadoutDuration)
        measurement.updateValue('Control Pulse - Amplitude #3', self._ReadoutAmplitude)
        
        #Xpi/2Xpi
        (x, y) = measurement.performMeasurement()
        if isinstance(y[0], complex):
            y = np.absolute(y)
        A0 = np.mean(y)
        if plot == True:
            y_fit = np.empty(len(x))
            y_fit.fill(A0)
            plt.figure(figsize = (9,6))
            plt.scatter(x, y, label = 'Results', color = 'black')
            plt.plot(x, y_fit, label="Calibration", color = 'red')
            plt.ylim(np.amin(y), np.amax(y))
            plt.xlim(np.amin(x), np.amax(x))
            plt.grid()
            plt.ylabel('Digitizer voltage')
            plt.xlabel('DRAG scaling constant (s)')
            plt.title('Xpi/2Xpi')
            plt.legend()
            plt.show()      
        
        #Xpi/2Ypi
        measurement.updateValue('Y Pi', 1)
        (x, y1) = measurement.performMeasurement()
        A1, B1 = fit.linear(x,y1, plot, title = 'Xpi/2Ypi', xlabel = 'DRAG scaling constant (s)', ylabel = 'Digitizer voltage')
        
        #Xpi/2Y-pi
        measurement.updateValue('Y Pi', 0)
        measurement.updateValue('Y-Pi', 1)
        (x, y2) = measurement.performMeasurement()
        A2, B2 = fit.linear(x,y2, plot, title = 'Xpi/2Y-pi', xlabel = 'DRAG scaling constant (s)', ylabel = 'Digitizer voltage')
        
        #finding intersection points
        P = (A1-A2)/(B2-B1)
        Q = (A0-A1)/B1
        Z = (A0-A2)/B2
        if plot == True:
            plt.plot(x, y, label = 'Xpi/2Xpi')
            y1 = np.absolute(y1)
            plt.plot(x,y1, label = 'Xpi/2Ypi')
            y2 = np.absolute(y2)
            plt.plot(x,y2, label = 'Xpi/2Y-pi')
            plt.xlabel('DRAG scaling constant')
            plt.ylabel('Digitizer voltage')
            plt.legend()
            plt.show()
        print('Intersection points:', P, ', ',Q, ',',Z)
        
        if RandomizedBenchmarking == False:
            self._DRAG = P
            if Q<Z:
                return [Q,Z], P
            if Q>=Z:
                return [Z,Q], P
        if RandomizedBenchmarking == True:
            log_name1 = 'Test Rand Bench_550scaling_2.hdf5'
            log_output1 = 'Test Rand Bench_550scaling_2_Results.hdf5X'
        
            measurement1 = ScriptTools.MeasurementObject(
                os.path.join(currentPath,log_name1), os.path.join(currentPath, log_output1))
            measurement1.updateValue('PiPulse', self._PiPulse)
            measurement1.updateValue('SSBFreqQubit', self._SSBFreqQubit)
            measurement1.updateValue('Multi - Number of Cliffords', Cliffords)
            measurement.updateValue('RS IQ Source - Frequency', self._QubitFreq + self._SSBFreqQubit)
            measurement.updateValue('QS RF Source - Frequency', self._ResonatorFreq0 + self._SSBFreq*10**3)
            measurement.updateValue('SSBFreq', self._SSBFreq) #cavity
            measurement.updateValue('RS IQ Source - Power', self._RFSourcePower)
            measurement.updateValue('QS RF Source - Power', self._QuickSynPower)
            measurement.updateValue('Multi - Width', self._PulseWidth)
            measurement.updateValue('Multi - Readout duration #1', self._ReadoutDuration)
            measurement.updateValue('Multi - Readout amplitude #1', self._ReadoutAmplitude)
        
            if Q<Z:
                drag_list = np.linspace(Q,Z,points)
            if Q>=Z:
                drag_list = np.linspace(Z,Q,points)
            print('Calibrating the DRAG scaling in the interval:',[drag_list[0],drag_list[len(drag_list)-1]])
            voltages = np.zeros(len(drag_list))
            for i in range(0, len(drag_list)):
                measurement1.updateValue('Multi - DRAG scaling #1', drag_list[i])
                (averages, ydrag) = measurement1.performMeasurement()
                if isinstance(ydrag[0], complex):
                    ydrag = np.absolute(ydrag)
                voltages[i] = np.mean(ydrag)
            if plot == True:
                plt.figure(figsize = (9,6))
                plt.grid()
                plt.plot(drag_list, voltages)
                plt.xlabel('DRAG scaling constant (s)')
                plt.ylabel('Digitizer voltage')
                plt.title('%i Cliffords' %Cliffords)
            index = np.argmax(voltages)
            self._DRAG = drag_list[index]
            print('DRAG scaling = ', self._DRAG)
            return self._DRAG
        
        
    def getPhotonNumber(self, plot = None, Reset = None, ArtificialDetuning = 10E6):
        if plot is None:
            plot = self._Plot
             
        log_name = 'Ramsey Reset Spacing.hdf5'
        log_output = 'Ramsey Reset Spacing_Results.hdf5X'
            
        measurement = ScriptTools.MeasurementObject(
            os.path.join(currentPath,log_name), os.path.join(currentPath, log_output))
        
        measurement.updateValue('PiPulse', self._PiPulse)
        measurement.updateValue('RS IQ Source - Frequency', self._QubitFreq + self._SSBFreqQubit)
        measurement.updateValue('QS RF Source - Frequency', self._ResonatorFreq0 + self._SSBFreq*10**3)
        measurement.updateValue('SSBFreq', self._SSBFreq) #cavity
        measurement.updateValue('SSBFreqQubit', self._SSBFreqQubit) #qubit
        measurement.updateValue('RS IQ Source - Power', self._RFSourcePower)
        measurement.updateValue('QS RF Source - Power', self._QuickSynPower)
        measurement.updateValue('Control Pulse - DRAG scaling', self._DRAG)
        measurement.updateValue('Control Pulse - Width #3', self._PulseWidth)
        measurement.updateValue('Control Pulse - Width #4', self._PulseWidth)
        measurement.updateValue('Control Pulse - Plateau #5', self._ReadoutDuration)
        measurement.updateValue('Control Pulse - Amplitude #5', self._ReadoutAmplitude)
        
        
        if Reset is None:
            Reset = self._Reset
        if Reset == True:
            #use reset sequence
            measurement.updateValue('Reset', 1)
            measurement.updateValue('ResetLength', self._ResetLength)
            measurement.updateValue('ResetSpacing', self._ResetSpacing)
            measurement.updateValue('ResetRabiAmplitude', self._ResetRabiAmplitude)
            measurement.updateValue('ResetCavityAmplitude', self._ResetCavityAmplitude)
        if Reset == False:
            measurement.updateValue('Reset', 0)
            
        DetuningIn = ArtificialDetuning #Artificially introduced detuning
        measurement.updateValue('Artificial detuning', DetuningIn)
        
        (x, y) = measurement.performMeasurement()
        phi, n0 = fit.ramseyPhotons(x, y, ResonatorKappa = self._ResonatorKappa, DispersiveShift = self._DispersiveShift, T2Echo = self._T2Echo, ArtificialDetuning = DetuningIn, Amax = self._Amax, Amin = self._Amin , bounds = ((-np.inf, 0),(np.inf, 8)), plot = plot, title = 'Ramsey trace for finding Photon number', xlabel = 'Time between Pi/2 pulses (s)', ylabel = 'Digitizer voltage')
        print('phi =', phi)
        print('n0 = ', n0)
        print('Parameters used:')
        if Reset == True:
            print('ResetSpacing = ', self._ResetSpacing)
        print('ArtificialDetuning = ', DetuningIn)
        print('T2Echo = ', self._T2Echo)
        print('DispersiveShift =', self._DispersiveShift)
        print('ResonatorKappa =', self._ResonatorKappa)
        

        return n0
    
        
    def getReset(self, plot = None, Amplitudes = False, RabiAmplitudes = None,
                 CavityAmplitudes = None, Length = False, ResetLengths = None, 
                 Spacing = False, ResetSpacings = None, SpacingPopulationThreshold = None, PiPulse = True):
        
        if plot is None:
            plot = self._Plot
        if RabiAmplitudes is None:
            RabiAmplitudes = self._ResetRabiAmplitude
        if CavityAmplitudes is None:
            CavityAmplitudes = self._ResetCavityAmplitude
        if ResetLengths is None:
            ResetLengths = self._ResetLength
        if ResetSpacings is None:
            ResetSpacings = self._ResetSpacing
            
        log_name = 'Reset Rabi Qutrit-pp.hdf5'
        log_output = 'Reset Rabi Qutrit-pp_Results.hdf5X'
        
        measurement = ScriptTools.MeasurementObject(
            os.path.join(currentPath,log_name), os.path.join(currentPath, log_output)) 
        
        ResetLengths = ResetLengths
        ResetSpacings = ResetSpacings
        RabiAmplitudes = RabiAmplitudes
        CavityAmplitudes = CavityAmplitudes

        measurement.updateValue('PiPulse', self._PiPulse)
        # The frequency that goes to the RS IQ Source has been reduced to 4.87E9, to reach the 1-2 transition
        # SSBFreqQubit is 45E6 Hz
        # SSBFreq1-2 is 380E6 Hz
        # Modify manually in the Labber file if needed
        measurement.updateValue('QS RF Source - Frequency', self._ResonatorFreq0 + self._SSBFreq*10**3)
        measurement.updateValue('SSBFreq', self._SSBFreq) #cavity
        measurement.updateValue('RS IQ Source - Power', self._RFSourcePower)
        measurement.updateValue('QS RF Source - Power', self._QuickSynPower)
        measurement.updateValue('Control Pulse - DRAG scaling', self._DRAG)
        measurement.updateValue('Control Pulse - Width #1', self._PulseWidth)
        measurement.updateValue('Control Pulse - Width #4', self._PulseWidth)
        measurement.updateValue('Control Pulse - Width #5', self._PulseWidth)
        measurement.updateValue('Control Pulse - Width #6', self._PulseWidth)
        measurement.updateValue('Control Pulse - Plateau #7', self._ReadoutDuration)
        measurement.updateValue('Control Pulse - Amplitude #7', self._ReadoutAmplitude)
        
        if PiPulse == True: #add an additional pi pulse before reset
            measurement.updateValue('Population Control', 1)
        if PiPulse == False:
            measurement.updateValue('Population Control', 0)
        
        if Amplitudes == True: # we want to calibrate the amplitudes
            if Length == True:
                print('Error. Length must be set to False when calibrating the amplitudes.')
                return 0
            if Spacing == True:
                print('Error. Spacing must be set to False when calibrating the amplitudes.')
                return 0
            if np.size(ResetLengths)>1 or np.size(ResetSpacings)>1:
                print('Dimension error, ResetLengths and ResetSpacings must be just one number when calibrating the amplitudes.\n For example leave by default, or set ResetLengths = 5E-6 and ResetSpacings = 1E-6.')
                return 0
            if np.size(RabiAmplitudes)<2 or np.size(CavityAmplitudes)<2:
                print('Dimension error, RabiAmplitudes and CavityAmplitudes must be arrays of length at least 2 when calibrating the amplitudes. \n For example set RabiAmplitudes = np.linspace(1E-3, 11E-3, 11) and CavityAmplitudes = np.linspace(100E-3, 200E-3, 11)')
                return 0
            else:
                #seems fine
                measurement.updateValue('Reset Length', ResetLengths)
                measurement.updateValue('Reset Spacing', ResetSpacings)
        
                Z = np.zeros((len(CavityAmplitudes), len(RabiAmplitudes))) #here we will save the excited state populations
                A = np.zeros((len(CavityAmplitudes), len(RabiAmplitudes))) #here we will keep the ground state rabi osc. amplitudes
        
                for i in range(0, len(CavityAmplitudes)):
                    CavityAmplitude = CavityAmplitudes[i]
                    for j in range(0, len(RabiAmplitudes)):
                        RabiAmplitude = RabiAmplitudes[j]
          
                #RabiAmplitude - Amplitude of the reset pulse that produces Rabi oscillations between |g,0> and |e,0>
                #CavityAmplitude - Amplitude of the reset pulse that adds photons: |0,0> to |0, alpha>
        
                        measurement.updateValue('Rabi Amplitude', RabiAmplitude)
                        measurement.updateValue('Resonator Amplitude', CavityAmplitude)
        
        
                # We will measure the population of the excited state after the reset pulse.
                # The purpose of the reset pulse is to minimize this population.
        
                # Rabi oscillations of ground state population:
                        measurement.updateValue('Control', 1) # A Xpi rotation is performed at the beginning
                        (x0, y0) = measurement.performMeasurement()
        
                # Rabi oscillations of excited state population:
                        measurement.updateValue('Control', 0)
                        (x1, y1) = measurement.performMeasurement()
                    
                        if plot == True:
                            plt.plot(x0, y0, label = 'ground')
                            plt.plot(x1, y1, label = 'excited')
                            plt.xlabel('Pulse #5 amplitude (Volts)')
                            plt.ylabel('Digitizer voltage')
                            plt.title('RPM measurement')
                            plt.legend()
                            plt.show()
        
                        period0, phi0, A0, c0, A_trial0 = fit.cosine_squared(x0, y0, plot=False)
                        period1, phi1, A1, c1, A_trial1 = fit.cosine_squared(x1, y1, plot=False)
        
                # Excited state population:
                        if abs(A1)>3*abs(A_trial1): #sometimes fitting errors occur and A1 is much bigger than it should
                            A1 = A_trial1
                        Pe = abs(A1)/(abs(A1)+abs(A0))
                        
                        Z[i,j] = Pe
                        A[i,j] = abs(A0)
                        print('Pe =', Pe)
                        print('A0 =', A0)
                        print('RabiAmplitude = ', RabiAmplitude)
                        print('CavityAmplitude = ', CavityAmplitude)
                
                X, Y = np.meshgrid(RabiAmplitudes, CavityAmplitudes)
                cp = plt.pcolormesh(X, Y, Z)
                plt.colorbar(cp)
                plt.title('Excited state population') #we want to minimize this
                plt.ylabel('Cavity Amplitude')
                plt.xlabel('Rabi Amplitude')
                plt.show()
        
                cp2 = plt.pcolormesh(X, Y, A)
                plt.colorbar(cp2)
                plt.title('Amplitude ground state oscillation') #we want to maximize this
                plt.ylabel('Cavity Amplitude')
                plt.xlabel('Rabi Amplitude')
                plt.show()
        
                flat_Z = Z.flatten()
                k = len(RabiAmplitudes)*len(CavityAmplitudes)//4 # we will choose the 25% smallest excited state populations
                idx = np.argpartition(flat_Z, k)
                Amax = []
                for t in range(0, len(idx[:k])):
                    i = idx[t]//len(RabiAmplitudes)
                    j = idx[t]-idx[t]//len(RabiAmplitudes)*len(RabiAmplitudes)
                    print('The', t+1, 'th  minimum Pe is:', Z[i,j],'with ground amplitude:',A[i,j], ' which corresponds to Rabi Amplitude = ', RabiAmplitudes[j], 'and Resonator Amplitude = ',CavityAmplitudes[i])
                    Amax.append(A[i,j])
            
                q = np.argmax(Amax) #out of the 25% with smallest Pe, 
                #we will finally choose the one with the biggest ground state rabi oscillation amplitude
                i = idx[q]//len(RabiAmplitudes)
                j = idx[q]-idx[q]//len(RabiAmplitudes)*len(RabiAmplitudes)
                print('The minimum Pe with biggest amplitude:', Amax[q], 'is', Z[i,j], 
                      ' which corresponds to Rabi Amplitude = ' ,RabiAmplitudes[j], 'and Resonator Amplitude = ', CavityAmplitudes[i])
                print('Length used: ',ResetLengths)
                print('Spacing used: ',ResetSpacings )
                self._ResetRabiAmplitude = RabiAmplitudes[j]
                self._ResetCavityAmplitude = CavityAmplitudes[i]
                return self._ResetRabiAmplitude, self._ResetCavityAmplitude
        
        if Length == True: # we want to calibrate the amplitudes
            if Amplitudes == True:
                print('Error. Amplitudes must be set to False when calibrating the length.')
                return 0
            if Spacing == True:
                print('Error. Spacing must be set to False when calibrating the length.')
                return 0
            if np.size(ResetSpacings)>1 or np.size(RabiAmplitudes)>1 or np.size(CavityAmplitudes)>1:
                print('Dimension error, RabiAmplitudes, CavityAmplitudes and ResetSpacings must be just one number when calibrating the length.\n For example leave by default, or set RabiAmplitudes = 3E-3, CavityAmplitudes = 120E-3 and ResetSpacings = 1E-6.')
                return 0
            if np.size(ResetLengths)<2:
                print('Dimension error, ResetLengths must be an array of length at least 2 when calibrating the length. \n For example set ResetLengths = np.linspace(0E-6, 3E-6, 51)')
                return 0
            else:
                #seems fine
                print('To calibrate the reset length, PiPulse has to be set to True. Default is True. For the current measurement PiPulse is:', PiPulse)
                measurement.updateValue('Reset Spacing', ResetSpacings)
                measurement.updateValue('Rabi Amplitude', RabiAmplitudes)
                measurement.updateValue('Resonator Amplitude', CavityAmplitudes)
                Z = np.zeros(len(ResetLengths)) #here we will save the excited state populations
                
        
                for i in range(0, len(ResetLengths)):
                    ResetLength = ResetLengths[i] 
                    measurement.updateValue('Reset Length', ResetLength)
                    
                    # Rabi oscillations of ground state population:
                    measurement.updateValue('Control', 1) # A Xpi rotation is performed at the beginning
                    (x0, y0) = measurement.performMeasurement()
        
                    # Rabi oscillations of excited state population:
                    measurement.updateValue('Control', 0)
                    (x1, y1) = measurement.performMeasurement()
                    
                    if plot == True:
                        plt.plot(x0, y0, label = 'ground')
                        plt.plot(x1, y1, label = 'excited')
                        plt.xlabel('Pulse #5 amplitude (Volts)')
                        plt.ylabel('Digitizer voltage')
                        plt.title('RPM measurement')
                        plt.legend()
                        plt.show()
        
                    period0, phi0, A0, c0, A_trial0 = fit.cosine_squared(x0, y0, plot=False)
                    period1, phi1, A1, c1, A_trial1 = fit.cosine_squared(x1, y1, plot=False)
        
                    # Excited state population:
                    if abs(A1)>3*abs(A_trial1): #sometimes fitting errors occur and A1 is much bigger than it should
                        A1 = A_trial1
                    Pe = abs(A1)/(abs(A1)+abs(A0))
                    
                    Z[i] = Pe
                    print('Pe =', Pe)
                    print('ResetLength', ResetLength)
                
                plt.figure(figsize=(9,6)) 
                plt.grid()
                plt.plot(ResetLengths, Z)
                plt.xlabel('Reset Pulse Duration')
                plt.ylabel('Excited state population')
                plt.show()
                print('Amplitudes used:\n RabiAmplitude: ',RabiAmplitudes, '\n CavityAmplitude: ', CavityAmplitudes)
                print('Spacing used: ',ResetSpacings )
                if PiPulse == True:
                    A, B, T = fit.exponential_T1(ResetLengths, Z)
                    a=math.modf(4*T*10**6)
                    if a[0]<=0.5:
                        self._ResetLength = (a[1] + 0.5)*10**(-6)
                    if a[0]> 0.5:
                        self._ResetLength = (a[1] + 1)*10**(-6)
                    return self._ResetLength
                if PiPulse == False:
                    print('Set PiPulse = True to obtain a fit.')
                    return 0
            
        if Spacing == True: # we want to calibrate the amplitudes
            if Amplitudes == True:
                print('Error. Amplitudes must be set to False when calibrating the spacing.')
                return 0
            if Length == True:
                print('Error. Length must be set to False when calibrating the spacing.')
                return 0
            if np.size(ResetLengths)>1 or np.size(RabiAmplitudes)>1 or np.size(CavityAmplitudes)>1:
                print('Dimension error, RabiAmplitudes, CavityAmplitudes and ResetLengths must be just one number when calibrating the spacing.\n For example leave by default, or set ResetLengths = 5E-6, RabiAmplitudes = 3E-3 and CavityAmplitudes = 120E-3.')
                return 0
            if np.size(ResetSpacings)<2:
                print('Dimension error, ResetSpacings must be an array of length at least 2 when calibrating the spacing. \n For example set ResetSpacings = np.linspace(0E-6, 2E-6, 31)')
                return 0
            else:
                #seems fine
                if SpacingPopulationThreshold is None:
                    SpacingPopulationThreshold = 5 #we will set as default 5% maximum excited population
                else:
                    SpacingPopulationThreshold = SpacingPopulationThreshold
                    
                measurement.updateValue('Reset Length', ResetLengths)
                measurement.updateValue('Rabi Amplitude', RabiAmplitudes)
                measurement.updateValue('Resonator Amplitude', CavityAmplitudes)
                Z = [] #here we will save the excited state populations
                A = [] #here we will save the ground state rabi oscillations
                Spacings = [] #here we will append the spacings with excited state population smaller than threshold
                Pe = 0
                for i in range(0,len(ResetSpacings)):
                  
                    ResetSpacing = ResetSpacings[i] 
                    measurement.updateValue('Reset Spacing', ResetSpacing)
                    
                        # Rabi oscillations of ground state population:
                    measurement.updateValue('Control', 1) # A Xpi rotation is performed at the beginning
                    (x0, y0) = measurement.performMeasurement()
        
                        # Rabi oscillations of excited state population:
                    measurement.updateValue('Control', 0)
                    (x1, y1) = measurement.performMeasurement()
                    
                    if plot == True:
                        plt.plot(x0, y0, label = 'ground')
                        plt.plot(x1, y1, label = 'excited')
                        plt.xlabel('Pulse #5 amplitude (Volts)')
                        plt.ylabel('Digitizer voltage')
                        plt.title('RPM measurement')
                        plt.legend()
                        plt.show()
        
                    period0, phi0, A0, c0, A_trial0 = fit.cosine_squared(x0, y0, plot=False)
                    period1, phi1, A1, c1, A_trial1 = fit.cosine_squared(x1, y1, plot=False)
        
                        # Excited state population:
                    if abs(A1)>3*abs(A_trial1): #sometimes fitting errors occur and A1 is much bigger than it should
                        A1 = A_trial1
                    Pe = abs(A1)/(abs(A1)+abs(A0))
                    print('Pe =', Pe)
                    print('A0 =', A0)
                    print('ResetSpacing', ResetSpacing)
                    if Pe > SpacingPopulationThreshold*0.01:
                        break
                    A.append(abs(A0))
                    Z.append(Pe)
                    Spacings.append(ResetSpacing)
                        
                        
                    
                plt.figure(figsize=(9,6)) 
                plt.grid()
                plt.plot(Spacings, Z, '-o')
                plt.xlabel('Spacing after Reset Pulse (s)',fontsize = 14, fontname = 'Arial')
                plt.ylabel('Excited state population',fontsize = 14, fontname = 'Arial')
                plt.show()
                
                plt.figure(figsize=(9,6)) 
                plt.grid()
                plt.plot(Spacings, A)
                plt.xlabel('Spacing after Reset Pulse')
                plt.ylabel('Ground state amplitude')
                plt.show()
                
                #we need to calibrate other parameters first:
                print('Calibrating PiPulse without Reset...')
                self.getPiPulse(Reset = False)
                print('Calibrating QubitFreq without Reset...')
                self.getQubitFreq(Reset = False)
                print('Calibrating T2Echo without Reset...')
                self.getT2Echo(Reset = False)
                PiPulse_noReset = self._PiPulse
                #we will adjust the photon number threshold:
                self._ResetSpacing = Spacings[-1] #greatest spacing
                print('Calibrating Amax and Amin...')
                self.getPiPulse(Reset = True) #we will adjust the Amax and Amin
                PiPulse_Reset = self._PiPulse
                self._PiPulse = PiPulse_noReset
                print('Calibrating threshold photon number (photon number with no reset)...')
                tol = 0.1
                n0threshold = self.getPhotonNumber(Reset = False) + tol
                self._PiPulse = PiPulse_Reset
                print('tolerance =', tol, 'photons')
                print('n0threshold = ', n0threshold)
                photons = []
                for i in range(0, len(Spacings)):
                    self._ResetSpacing = Spacings[i]
                    n0 = self.getPhotonNumber(Reset = True)
                    photons.append(n0)
                    if n0 <= n0threshold:
                        plt.figure(figsize=(9,6)) 
                        plt.grid()
                        plt.plot(Spacings[:(i+1)], photons, '-o')
                        plt.xlabel('Spacing after Reset Pulse (s)', fontsize = 14, fontname = 'Arial')
                        plt.ylabel('Photon Number', fontsize = 14, fontname = 'Arial')
                        plt.tight_layout()
                        plt.show()
                        return self._ResetSpacing
                print('Spacing could not be found within the specified parameters, try to increase the SpacingPopulationThreshold (default is 5) or the maximum Spacing.')
                return 0
                
               
            
          
    def getAll(self):
        self.getPiPulse()
        self.getT1()
        self.getQubitFreq()
        self.getT2Ramsey(TphiG = True)
        self.getT2Echo()
        self.getResonatorFreq()

        dictio = {'QubitFreq': self._QubitFreq,
                  'PiPulse' : self._PiPulse,
                  'Amax' : self._Amax,
                  'Amin' : self._Amin,
                  'T1' : self._T1,
                  'T2Ramsey' : self._T2Ramsey,
                  'T2Echo' : self._T2Echo,
                  'TphiG' : self._TphiG,
                  'ResonatorFreq0' : self._ResonatorFreq0,
                  'ResonatorFreq1' : self._ResonatorFreq1}
        return dictio