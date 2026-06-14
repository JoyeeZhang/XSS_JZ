import csv
import os
import numpy as np
from scipy.interpolate import pchip
#from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import XSSModules.cross_section  as CS

'''============================================================================
Class       
    SpecCalc
Parent
    None
Attributes
    peak_energy_kev
    minimum_energy_kev
    energy_bin_space_kev
    energy_kev
    spectrum_counts_per_kevmm2mAs_1m
    spectrum_normalized
    anode_angle_deg
    air_thickness_mm
    beryllium_thickness_mm
    aluminium_thickness_mm
    copper_thickness_mm
    tin_thickness_mm
    tungsten_thickness_mm
    tantalum_thickness_mm
    water_thickness_mm
    Nf
    P
    bremsstrahlung_uGypmAs_1m
    characteristic_uGypmAs_1m
    HVL1_Al_mm
    HVL2_Al_mm
    HVL1_Cu_mm
    HVL2_Cu_mm
    mean_energy_kev
    effective_energy_kev_Al
    effective_energy_kev_Cu
Methods
    __init__(fn_spectrum)
    normalize_spectrum(energy_kev = None, spectrum = None)
    plot_spectrum(spectrum_type = 'Counts')
    interp_spectrum(energy_kev, spectrum_type = 'Counts', plot_tag = False)
    apply_filter(material_id,thickness_mm,plot_tag = False)
    interp_spectrum_flux(energy_kev, interp_kind = 'cubic', plot_switch = False)
    interp_spectrum_flux(energy_kev, spectrum_type = 'Counts', plot_tag = False)        
Notes
    SpekCalc uses uniform kev space
History: Created by ZYu prior to 20250614
============================================================================'''
class SpekCalc():    
    def __init__(self,fn_spectrum):
        self.fn_spectrum = fn_spectrum
        path_data_base = os.environ['XSSLibraries']
        path_data_base = os.path.join(path_data_base,'SpekCalc')
        filename = os.path.join(path_data_base,fn_spectrum)
        
        with open(filename) as spectrum_file:
            spectrum_reader = csv.reader(spectrum_file)
            spectrum_definition = [row for row in spectrum_reader]
            
            # -----------------------------------------------------------------
            # Input parameters
            kv_info = spectrum_definition[4][0].split()
            self.peak_energy_kev = float(kv_info[0])
            self.minimum_energy_kev = float(kv_info[1])
            self.energy_bin_space_kev= float(kv_info[2])
            # Energy indices
            self.energy_kev = np.arange(self.minimum_energy_kev, self.peak_energy_kev + 1, self.energy_bin_space_kev)
            num_energy_bin = len(self.energy_kev)
            self.spectrum_counts_per_kevmm2mAs_1m = np.zeros(num_energy_bin)
            self.spectrum_normalized = np.zeros(num_energy_bin)
            self.anode_angle_deg= float(spectrum_definition[6][0])
            thickness_info = spectrum_definition[8][0].split()
            self.air_thickness_mm = float(thickness_info[0])
            self.beryllium_thickness_mm = float(thickness_info[1])
            self.aluminium_thickness_mm = float(thickness_info[2])
            self.copper_thickness_mm = float(thickness_info[3])
            self.tin_thickness_mm = float(thickness_info[4])
            self.tungsten_thickness_mm = float(thickness_info[5])
            self.tantalum_thickness_mm = float(thickness_info[6])
            self.water_thickness_mm = float(thickness_info[7])
            method_para_info = spectrum_definition[10][0].split()
            self.Nf = float(method_para_info[0])
            self.P = float(method_para_info[1])
            
            #------------------------------------------------------------------
            # Output parameters
            output_para_info = spectrum_definition[13][0].split()
            self.bremsstrahlung_uGypmAs_1m = float(output_para_info[0])
            self.characteristic_uGypmAs_1m = float(output_para_info[1])
            # Convert all unit from cm to mm by x 10
            beam_quality_info = spectrum_definition[15][0].split()
            self.HVL1_Al_mm = float(beam_quality_info[0])*10
            self.HVL2_Al_mm = float(beam_quality_info[1])*10
            self.HVL1_Cu_mm = float(beam_quality_info[2])*10
            self.HVL2_Cu_mm = float(beam_quality_info[3])*10
            self.mean_energy_kev = float(beam_quality_info[4])
            self.effective_energy_kev_Al = float(beam_quality_info[5])
            self.effective_energy_kev_Cu = float(beam_quality_info[6])
            
            
            # Extract the spectrum - spectrum
            index_offset = 18
            for k in range(num_energy_bin):
                quanta_info = spectrum_definition[index_offset + k][0].split()
                self.spectrum_counts_per_kevmm2mAs_1m[k] = float(quanta_info[1])*0.01      
            
            # Normalized spectrum
            # self.spectrum_normalized = self.normalize_spectrum(self.energy_kev,self.spectrum_counts_per_kevmm2mAs_1m)
            self.spectrum_normalized = self.normalize_spectrum()

    '''------------------------------------------------------------------------
    [F] - normalize_spectrum(self,energy_kev = None, spectrum = None)
    [I] - energy_kev [optional], default is self.energy_kev
    [I] - spectrum [optional], default is self.spectrum_counts...
    [O] - generate normalized spectrum such that sum(phi(E))dE = 1
    [N] - **ssume energy_kev is the center of energy bin
          **non-uniform kev space is allowed
          **Here, self is needed as an input, otherwise, num of input error would
          **show up. Alternatively, one can declare @staticmethod before the 
            funciton and remove self as input.
    [H] - created by ZYu on 20250614
    ------------------------------------------------------------------------'''
    def normalize_spectrum(self,energy_kev = None,spectrum = None):
        if (energy_kev is None) and (spectrum is None):
            energy_kev = self.energy_kev
            spectrum = self.spectrum_counts_per_kevmm2mAs_1m
        dkev = np.diff(energy_kev)
        dkevL= np.append(dkev[0],dkev)
        dkevR= np.append(dkev,dkev[-1])
        dkev = 0.5*(dkevL+dkevR)
        spectrum_ttl = np.sum(dkev*spectrum)
        spectrum_normalized = spectrum/spectrum_ttl
        return spectrum_normalized
    
    '''------------------------------------------------------------------------
    [F] - plot_spectrum_flux(self,spectrum_type = 'Counts'):
    [I] - spectrum_type [optional]
            default: 'Counts'
            other option: 'Normalized'
    [O] - plotted figure of spectrum in terms of photon counts/[kev*mm^2*mAs] @ 1m
    [H] - created by ZYu on 20250614
    ------------------------------------------------------------------------'''
    def plot_spectrum(self,spectrum_type = 'counts'):
        # Address Inputs
        energy_kev = self.energy_kev
        if spectrum_type.lower() == 'counts':
            spectrum = self.spectrum_counts_per_kevmm2mAs_1m
        elif spectrum_type.lower() == 'normalized':
            spectrum = self.spectrum_normalized
        else:
            raise Exception('In valid spectrum type!!')
        
        # Plot
        plt.figure()
        plt.plot(energy_kev,spectrum)
        plt.grid(True)
        plt.xlabel('Energy [kev]')
        plt.ylabel(spectrum_type.capitalize())
        plt.title('Spectrum')
        plt.show()
    
    '''------------------------------------------------------------------------
    [F] - interp_spectrum(energy_kev, spectrum_type = 'Counts', plot_tag = False)
    [I] - energy_kev:  1D array representing the desired energy bin locaiton
    [I] - spectrum_type [optional]
            default: 'Counts'
            other option: 'Normalized'
    【I】 - plot_tag [optional]: False [default]; True: plot
    [O] - spectrum_interpolated
    [N] - Method: scipy.interpolate.pchip
    [H] - created by ZYu on 20250614
    ------------------------------------------------------------------------'''
    def interp_spectrum(self,energy_kev,spectrum_type = 'Counts', plot_tag = False):
        # Address Inputs
        energy_kev_origin = self.energy_kev
        if spectrum_type.lower() == 'counts':
            spectrum_origin = self.spectrum_counts_per_kevmm2mAs_1m
        elif spectrum_type.lower() == 'normalized':
            spectrum_origin = self.spectrum_normalized
        
        # Perform Interpolation
        interp = pchip(energy_kev_origin,spectrum_origin)
        spectrum_interpolated = interp(energy_kev)
        
        # Plot
        if plot_tag:
                plt.figure()
                plt.plot(energy_kev,spectrum_interpolated,'D')
                plt.plot(energy_kev_origin,spectrum_origin)
                plt.gca().legend(('interpolated - pchip', 'original'))
                plt.xlabel('energy - kev')
                plt.ylabel(spectrum_type.capitalize())
                plt.grid(True)
                plt.show()
        
        return spectrum_interpolated
    
    '''------------------------------------------------------------------------
    [F] - apply_filter(material_id,thickness_mm)
    [I] - material_id: material specified in XSSLibrary CrossSection
    [I] - thickness_mm: a scalar or 1d array
    [O] - energy_kev: energy grids defined by spectrum
    [O] - spectrum after filter for each element specified by thickness_mm
          spectrum is a 2d array of size length(thickness_mm) x length(energy_kev)
    [H] - created by ZYu on 20250614
    ------------------------------------------------------------------------'''
    def apply_filter(self,material_id,thickness_mm,plot_tag = False):
        material = CS.XcomCrossSection(material_id)
        interp = pchip(material.energy_kev,material.LAC_pmm)
        LAC_pmm = interp(self.energy_kev)
        # interp outputs a column vector, convert LAC_pmm to a row vector
        LAC_pmm = LAC_pmm.reshape([1,-1])
        # make sure thickness is a single column vector
        thickness_mm = thickness_mm.reshape([-1,1]);
        spectrum_original = np.ones([len(thickness_mm),1])*self.spectrum_counts_per_kevmm2mAs_1m
        # spectrum_output is a matrix of len(thickness_mm)xlen(LAC_pmm_original)
        spectrum_output = spectrum_original*np.exp(-1*thickness_mm*LAC_pmm)
        
        # Plot only for the 1st thickness_mm if it is a 1darray
        if plot_tag:
                plt.figure()
                plt.plot(self.energy_kev,spectrum_output[0,])
                plt.plot(self.energy_kev,self.spectrum_counts_per_kevmm2mAs_1m)
                plt.gca().legend(('attenuated spectrum', 'original spectrum'))
                plt.xlabel('energy - kev')
                plt.ylabel('counts_per_kevmm2mAs_1m')
                plt.grid(True)
                plt.show()
            
        return spectrum_output
            
    '''------------------------------------------------------------------------
    [F] - interp_spectrum_flux(energy_kev)
    [I] - energy_kev:  1D array representing the desired energy bin locaiton
    [I] - interp_kind [optional]: 'cubic'[default]
    【I】 - plot_swtich [optional]: False [default]; True: plot
    [O] - photon_flux_interpolated: interpolated flux @ desired energy location
    ------------------------------------------------------------------------'''
#    def interp_spectrum_flux(self, energy_kev, interp_kind = 'cubic', plot_switch = False):
#        if ( (energy_kev[0] < self.energy_kev[0])|(energy_kev[-1] > self.energy_kev[-1]) ):
#            print('Error: requested energy out of range')
#            return 0
#        else:
#            # set up the interpolator
#            # set_interp = interp1d(self.energy_kev, self.spectrum_counts_per_kevmm2mAs_1m, kind = interp_kind)
#            # photon_flux_per_kevmm2mAs_1m = set_interp(energy_kev) 
#            num_energy = len(energy_kev)
#            photon_flux_per_kevmm2mAs_1m = np.zeros(num_energy)
#            for k in range(num_energy):
#                ii = np.diff(self.energy_kev > energy_kev[k])
#                eng_lb = self.energy_kev[:-1][ii][0]
#                eng_ub = self.energy_kev[1:][ii][0]
#                deng = 1.0/(eng_ub - eng_lb)
#                dis2lb = (energy_kev[k] - eng_lb)*deng
#                dis2ub = (eng_ub - energy_kev[k])*deng
#                photon_flux_per_kevmm2mAs_1m[k] = dis2lb*self.spectrum_counts_per_kevmm2mAs_1m[1:][ii][0] \
#                                                + dis2ub*self.spectrum_counts_per_kevmm2mAs_1m[:-1][ii][0]
#                                              
#            if plot_switch:
#                plt.figure()
#                plt.plot(energy_kev,photon_flux_per_kevmm2mAs_1m*0.001,'--D')
#                plt.plot(self.energy_kev,self.spectrum_counts_per_kevmm2mAs_1m*0.001)
#                plt.gca().legend(('interpolated - ' + interp_kind, 'original'))
#                plt.xlabel('energy - kev')
#                plt.ylabel('photon flux (k) / (kev*mm^2*mAs)@ 1 meter')
#                plt.grid(True)
#                plt.show()
#        
#            return photon_flux_per_kevmm2mAs_1m

    
        
