Author: Raquel García Bellés

Last update: 31/08/2020

# Qubit-Calibration
This package is intended for the calibration of a single transmon qubit coupled to a 3D cavity in the dispersive regime. In particular, the parameters that can be calibrated are:
- The qubit frequency.
- The coherence times: T1, T2* (using Ramsey or Hahn Echo measurements), and additionally, TphiG (see page 16 of Ref. [1](#ref1)).
- The amplitude in volts of a Pi Pulse.
- The cavity frequencies when the qubit is in state 1 and in state 0 respectively.

Two additional functions have been included:
- One to calibrate a Reset pulse, see Refs. [3](#ref3) and [4](#ref4).
- Other to calibrate the DRAG pulse.  

## Contents
1. [Requirements](#Requirements)
2. [Files](#files)
3. [How to run the calibration script](#Howto)
4. [Reset Pulse Calibration](#resetcal)
5. [DRAG Pulse Calibration](#DRAG)
6. [References](#references)

<a name="Requirements"></a>
## Requirements
The python modules required to execute the script are:
- os
- numpy
- matplotlib
- json
- scipy
- math

It is also necessary to have the program "Labber" installed and to be able to acces the "Labber API".
For more information on how to install the "Labber API", as well as Labber's own python requirements see: `https://labber.org/online-doc/api/Installation.html`

<a name="files"></a>
## Files
The files contained in this package are:

- **calibration.py** 
    Main file, it is where the Calibration class and all the functions that execute Labber are defined.    
- **fittings.py**
    This file contains the functions to fit the experimental data results.
- **parameters.txt**
    Text file in json format. It stores previous calibration results.
- **StableParameters.txt**
    Text file in json format. It stores parameters kept fixed. 
- **Labber measurement configurations**
    Several hdf5 files that contain the instructions and the configuration of the instruments to perform a given measurement.


<a name="Howto"></a>
## How to run the calibration script
A simple way to run this script is, for instance, by using a "jupyter notebook".
First we have to import the calibration module:

    from calibration import *
we can also add the line:

    %matplotlib inline
if we want the plots to appear in the notebook.
Then we have to generate a calibration object:

    cal = Calibration('parameters.txt')
Since we have specified the file 'parameters.txt', the pre-stored values will be read from this file. Note that the file must be written in json format.
We could have also written:

    cal = Calibration()
In this case the program will use its own pre-stored values. 

By default, the class Calibration sets `Reset = True`, which means that a Reset Pulse will be inserted at the beginning of all the calibration measurements, unless otherwise specified. The parameters used for this Reset Pulse are the ones stored in the class, in this example they would be: `cal._ResetRabiAmplitude`, `cal._ResetCavityAmplitude`, `cal._ResetLength`, `cal._ResetSpacing`. For more information see section: [Reset Pulse Calibration](#resetcal). If we do not want to use the Reset Pulse in any calibration, we can generate the calibration object as: `cal = Calibration('parameters.txt', Reset = False)`. By default, a plot showing the measurement results and the fitting will be shown, if we don't want to see any plot we can generate the calibration object as: `cal = Calibration('parameters.txt', plot = False)`.

Then we can start with the calibration measurements. We can calibrate the voltage amplitude for a Pi Pulse by running:

    cal.getPiPulse()
It is recommended to calibrate the Pi Pulse from the beginning, since this value will then be updated, and it is used in all the other calibrations. If we leave the calibration object as default, a plot showing the rabi oscillations results and the fitting function will be generated, but if we don't want to see this plot in particular, we can instead type:

    cal.getPiPulse(plot = False)
Apart from returning the voltage amplitude of a Pi Pulse, this function also returns the maximum and minimum amplitude of the Rabi oscillation, and stores them in `cal._Amax` and `cal._Amin` respectively. Since we use the cavity frequency when the qubit is in the zero state for the readout pulse, the maximum of the rabi oscillation corresponds to the qubit in state zero, and the minimum to the qubit in state one --we are using the magnitude of the integrated signal of the digitizer corresponding to the readout pulse to determine the qubit state population, not the phase--.

Other calibrations can be done by running:

    cal.getQubitFreq() # Calibrate the Qubit Frequency estimating the detuning in a Ramsey measurement
    cal.getT2Ramsey() # Find the T2* time using Ramsey
    cal.getT2Ramsey(TphiG = True) # Find the T2* and the TphiG times using Ramsey
    cal.getT2Echo() # Find the T2* time using Hahn Echo
This functions above send pulses detuned from the qubit frequency. By default, the artificial detuning introduced is 2E6Hz, but it can be modified by setting: `ArtificialDetuning = 1E6` (for example) when calling the function.

    cal.getT1() # Find the T1 time
    cal.getResonatorFreq() # Find the cavity frequencies when the qubit is in states 0 and 1.
    cal.getAll() # Execute all the above functions to calibrate all.

If we have left the calibration object as default (with `Reset = True`), but for a particular calibration we do not want to use the Reset Pulse, we will have to specify it by setting `Reset = False` when running the function. For instance, finding the T1 time without Reset would be: `cal.getT1(Reset = False)`.

We can acces the stored values in the calibration object by writing:

    cal._PiPulse            # Amplitude in volts of a Pi Pulse
    cal._Amax               # Maximum amplitude of the Rabi oscillation
    cal._Amin               # Minimum amplitude of the Rabi oscillation
    cal._QubitFreq          # Qubit Frequency in Hz
    cal._T2Ramsey           # T2 time in seconds found using Ramsey
    cal._T2Echo             # T2 time in seconds found using Hahn Echo
    cal._TphiG              # TphiG time in seconds found using Ramsey
    cal._T1                 # T1 time in seconds
    cal._ResonatorFreq0     # Frequency of the cavity in mHz when the qubit is in state 0
    cal._ResonatorFreq1     # Frequency of the cavity in mHz when the qubit is in state 1
Other parameters stored are:

    cal._ResetRabiAmplitude             # Amplitude in volts of the Reset pulse at the frequency of the qubit
    cal._ResetCavityAmplitude           # Amplitude in volts of the Reset pulse at the cavity frequency
    cal._ResetLength                    # Duration in seconds of the reset pulse
    cal._ResetSpacing                   # Waiting time in seconds after the reset pulse
    cal._DRAG                           # DRAG scaling in seconds
    cal._DispersiveShift                # in Hz, it is defined as: (self._ResonatorFreq0 - self._ResonatorFreq1)*10**(-3)/2
    cal._ResonatorKappa                 # linewidth of the cavity in Hz
When we are done with the calibration, we can store the parameters in a file by typing:

    cal.toFile('output.txt')
and a .txt file in json format will be generated.
By running:

    cal.toFile('parameters.txt')
we would update the 'parameters.txt' file.

To see an example of this functions' output, take a look at the files `Example with Reset.html` and `Example without Reset.html` in the Examples folder.

<a name="resetcal"></a>
## Reset Pulse Calibration
Usually, we start the measurements assuming that the qubit is in the ground state. To be able to assume this, we can simply wait a couple of times the T1 time after the previous measurement, and then, with high probability (around 70-85%), the qubit will have spontaneously relaxed. There is a way in which this process can be made faster and the probability of the qubit being in the ground state can be increased (to around 95-99%), this is done by using a Reset Pulse (see Ref. [3](#ref3)).

The Reset Pulse that we will implement is the same as the one described in [3](#ref3), and it consists of two simultaneous pulses: one applied at the cavity frequency when the qubit is in the zero state, and the other applied at the qubit frequency. After the Reset Pulse, there will be many photons left inside of the cavity, and we will have to wait a certain time for them to disappear. Therefore, we have four parameters to calibrate: the amplitude (in V) of the pulse at the frequency of the cavity, the amplitude of the pulse at the qubit frequency, the duration of the pulses, and the waiting time after the pulses for the photons to leave the cavity. Following the example above, this parameters will be stored respectively in: `cal._ResetCavityAmplitude`, `cal._ResetRabiAmplitude`, `cal._ResetLength`, and `cal._ResetSpacing`, to be used in the other calibrations.

The Labber files used for the Reset calibration are:
- `Reset Rabi Qutrit-pp.hdf5`: we use this file to obtain the Rabi oscillations for the Rabi population measurement (RPM), which makes use of the 1 - 2 transition. See Ref. [3](#ref3).
- `Ramsey Reset Spacing.hdf5`: we use this file to obtain a Ramsey trace after a given Reset Pulse. The number of photons in the cavity can be obtained from the Ramsey trace, see Ref. [4](#ref4).

Assuming that the Pi Pulse and the Cavity Frequency when the qubit is in the ground state are properly calibrated in the calibration object*, we can start the calibration of the Reset Pulse as follows:

*Note that the modulation frequency of the qubit and the 1-2 pulses have to be manually modified in the Labber files if this parameters changed substancially from the current ones.*

### Calibrating the Amplitude of the Pulses
We can run the function:

    cal.getReset(Amplitudes = True, RabiAmplitudes = np.arange(1E-3, 12E-3, 1E-3), 
                CavityAmplitudes = np.arange(100E-3, 210E-3, 10E-3), ResetLengths = 5E-6,
                ResetSpacings = 2E-6)
For the chosen parameters in this example, the function will take the 100 possible combinations of the RabiAmplitudes (amplitude of the pulse at the qubit frequency) and the CavityAmplitudes (amplitude of the pulse at the cavity frequency), and do an RPM measurement to estimate the excited state population after the Reset Pulse. Then, out of the pulses' amplitudes that gave around 25% of the smallest excited state population, the combination that yields the largest rabi oscillation amplitude of the ground state in the RPM measurement will be the one selected. In this example, for all the trial Reset pulses the duration is set to 5E-6s and the spacing to 2E-6s. By default, a Pi Pulse is applied before the Reset, but to calibrate the Amplitudes it is optional and can be removed by setting: `PiPulse = False` when calling the function. The results of this calibration will be stored in `cal._ResetRabiAplitude` and `cal._ResetCavityAmplitude`.

### Calibrating the Duration of the Pulses
An example of how to calibrate the duration of the Reset is:

     cal.getReset(Length = True, ResetLengths = np.arange(0, 7E-6, 0.5E-6), ResetSpacings = 2E-6)
This function will estimate the excited state population for each of the durations specified in ResetLengths by doing an RPM measurement. When a Pi Pulse is applied before the Reset, the excited state population as a function of the Reset duration takes approximately the shape of a decaying exponential. Therefore, by fitting the results to a decaying exponential we can estimate the Reset duration for which the excited state population has decayed by a factor of exp(4) with respect to no Reset. We store this value in `cal._ResetLength`. Note that in this case leaving `PiPulse = True` (default) is necessary to calibrate the duration. Nevertheless, we can also set `PiPulse = False` to see a plot of the evolution of the excited state population as a function of the Reset duration when no Pi Pulse is applied before the reset, as in figure 2 of Ref. [3](#ref3) (no value will be stored in `cal._ResetLength`). The amplitudes of the pulses used by default will be the ones stored in `cal._ResetRabiAplitude` and `cal._ResetCavityAmplitude`, but it is also possible to use a different ones by specifying them when calling the function, for example setting: `RabiAmplitudes = 0.003` and `CavityAmplitudes = 0.120`.

### Calibrating the Spacing after the Pulses
The waiting time after the Reset pulse can be calibrated by running:

    cal.getReset(Spacing = True, ResetSpacings = np.arange(0.5E-6, 10.5E-6, 0.5E-6))
This function will do an RPM measurement for each spacing specified in ResetSpacings, until the excited state population is above a given threshold --Note that when we stop applying the Reset pulse, the excited state population will increase with time until it reaches an equilibrium (wich is around 14%), but we also need to wait a certain time to get rid of the photons in the cavity, thus we need to reach a trade-off between the excited state population and the photons in the cavity--. By default, this threshold is: `SpacingPopulationThreshold = 5`, which means that only spacings for which the excited state population is below 5% will be considered. This threshold can also be modified when calling the function. The number of photons in the cavity is obtained using a function called `self.getPhotonNumber()`, which fits a Ramsey trace (as in ref. [4](#ref4)) obtained after a Reset pulse generated with the parameters stored in the class. First, a threshold for the number of photons is defined by calculating the number of photons in the cavity when no Reset pulse is applied and adding a tolerance of 0.1 photons. Then, we calculate the photon number for Reset pulses with different spacings until it is below the threshold photon number defined, and we store this spacing in `cal._ResetSpacing`. The Reset pulses' amplitudes and duration used by default are the ones stored in the calibration object, but it is also possible to use different ones by modifying the values of `CavityAmplitudes`, `RabiAmplitudes` and `ResetLengths` when calling the function.

To see an example of the Reset calibration, take a look at the file `ResetCalibration.html` in the Examples folder.

<a name="DRAG"></a>
## DRAG Pulse Calibration
In order to prevent leakage errors that can cause excitations of the qubit outside of the computational subspace, and phase errors (acumulation of a relative phase between the ground and the excited state), the envelope of the quadratures sent to the qubit is modified according to the so called DRAG procedure, see Ref. [1](#ref1). If *s(t)* was the envelope of the original pulse, then applying the DRAG procedure consists on leaving the envelope of the I quadrature as it was (i.e. as *s(t)*), and setting the envelope of the Q quadrature as *q·s'(t)*, where *s'(t)* is the time derivative of *s(t)* and *q* is a constant that we call DRAG scaling (as Labber does). Following the examples above, this DRAG scaling can be calibrated by running the function:

    cal.getDRAG(RandomizedBenchmarking = True, points = 20, Cliffords = 50)
The result is stored in `cal._DRAG`. What this function does, is to apply the gates *Xpi/2 Xpi*, *Xpi/2 Ypi* and *Xpi/2 Y-pi* (that are supposed to leave the qubit in the same state population of 0.5) for different DRAG scaling constants, see Ref. [5](#ref5). The experimental results can be fit to three lines (one for each gate), and the DRAG scaling lies approximately in the region where these lines meet. If `RandomizedBenchmarking = False`, the value we store in `cal._DRAG` is the point where the lines corresponding to the *Xpi/2 Ypi* and *Xpi/2 Y-pi* gates meet. The other two intersection points are returned as an interval. By default, `RandomizedBenchmarking = True` as in this example. In this case, we will do a Randomized Benchmarking experiment to find the optimal value of the DRAG scaling among 20 points (`points = 20`) in the interval given by the intersections of the previous lines. The Multi-Qubit Pulse Generator instrument in Labber generates a Randomized Benchmarking sequence automatically --sequence of Clifford gates with the last gate returning the qubit to the original state--, we only need to specify the parameters of a Pi Pulse, the number of Clifford gates and the number of Randomizations. In general, as we increase the number of gates, the final state of the qubit will be more different from the initial state, because each gate accumulates a certain error. We choose the DRAG scaling constant for which less error is accumulated after applying 50 Clifford gates (`Cliffords = 50`).

The Labber files used in this calibration are:
- `DRAG.hdf5`: we use this file to measure the qubit state (digitizer voltage) as a function of the DRAG scaling after applying each of the gates: *Xpi/2 Xpi*, *Xpi/2 Ypi* and *Xpi/2 Y-pi*.
- `Test Rand Bench_550scaling_2.hdf5`: we use this file to measure the qubit state (digitizer voltage) after a Randomized Benchmarking sequence for different randomizations.

To see an example of the output of this function, take a look at the file `DRAGCalibration.html` in the Examples folder.
<a name="references"></a>
## References
<a name="ref1"></a>
[1] - P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and
    W. D. Oliver, A quantum engineer’s guide to superconducting qubits, 
    Appl. Phys. Rev. 6, 021318 (2019).
 
<a name="ref2"></a>
[2] - Naghiloo, M. (2019). Introduction to experimental quantum measurement with superconducting qubits. arXiv preprint arXiv:1904.09291.

<a name="ref3"></a>
[3] - Geerlings, K., Leghtas, Z., Pop, I. M., Shankar, S., Frunzio, L., Schoelkopf, R. J., ... & Devoret, M. H. (2013). Demonstrating a driven reset protocol for a superconducting qubit. Physical review letters, 110(12), 120501.

<a name="ref4"></a>
[4] - McClure, D. T., Paik, H., Bishop, L. S., Steffen, M., Chow, J. M., & Gambetta, J. M. (2016). Rapid driven reset of a qubit readout resonator. Physical Review Applied, 5(1), 011001.

<a name="ref5"></a>
[5] - Matthias  Baur.  “Realizing  quantum  gates  and  algorithms  with  three  superconducting qubits”.  PhD  thesis.  ETH  Zurich,  Mar.  2012.


    
