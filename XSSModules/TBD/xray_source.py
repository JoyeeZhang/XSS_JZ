import numpy as np
import os
import matplotlib.pyplot as plt
import XSSModules.xray_spectra as spk
'''============================================================================
List of Classes
    IdealSource
    IdealTube
============================================================================'''


'''============================================================================
Class       
    IdealSource
Parent
    None
Model
    Monoenergy
Attributes
    energy_kev: this could be a scalar or ndarray
    photon_counts: this could be a scalar or ndarray                  
Methods
            
Notes
    This class could be used for reconstruction algorithm verificaiton only 
    concerning noise impact, but without concerning any non-ideal physical
    effects such as beam hardening, etc.
    The unit of photon_counts is dependent on the context. In most cases, photon
    counts could be the number of photons per view, ignoring integration time, 
    detector pixe size and source to detector distance.
History [yyyy/mm/dd BY whom ABOUT what]
============================================================================'''
class IdealSource():    
    def __init__(self,mono_energy_kev,photon_counts_per_ray):
        self.mono_energy_kev = mono_energy_kev
        self.photon_counts_per_ray = photon_counts_per_ray

'''============================================================================
Class       
    IdealTube
Parent Class
    None
Parent Coordiante System
    Gantry coordinate system
Modeling
    Focal spot: Point source
    Anode angle: 0
    Heel affect: No
    Heat capacity: No
Inputs
    focal_spot_position_mm - np.array([x,y,z])
    tube_current_mA
    pulse_width_ms
    fn_spectrum - spectrum filename
Attributes
    focal_spot_position_mm - np.array([x,y,z])
    tube_current_mA: scalar
    tube_potential_kVp: scalar
    pulse_width_ms: scalar
    pulse_density_mAs
    xray_spectrum: SpekCalc
Methods   
    None    
Notes
    - Mono-energetic x-ray can be configured by inputting a single energy spectrum
    - Tube potential is defined by spectrum
    - Currently supported spectrum types: SpekCalc (.spec)
    
History [yyyy/mm/dd BY whom ABOUT what]
============================================================================'''
class IdealTube():
    def __init__(self, focal_spot_position_mm,tube_current_mA,pulse_width_ms,fn_spectrum):
        # Direct attributes
        self.focal_spot_position_mm = focal_spot_position_mm
        self.tube_current_mA = tube_current_mA
        self.pulse_width_ms = pulse_width_ms
        self.pulse_density_mAs = tube_current_mA*pulse_width_ms*0.001
 
        # Read spectrum information
        fnext = os.path.splitext(fn_spectrum)
        # Case I: SpekCalc: Scale up with mAs
        if fnext[-1] == '.spec':
            SpekCalc = spk.SpekCalc(fn_spectrum)
            self.xray_spectrum = SpekCalc
            self.spectrum_counts_per_kevmm2_1m = SpekCalc.spectrum_counts_per_kevmm2mAs_1m*self.pulse_density_mAs
            self.tube_potential_kVp = SpekCalc.peak_energy_kev
        else:
            raise Exception('Invalid spectrum type!!')
        
        