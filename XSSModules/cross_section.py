import csv
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import pchip

'''============================================================================
Class       
    XcomCrossSection
Parent
    None
Attributes
    fn_material
    material
    density_gpmm3
    energy_kev
    MAC_coherent_scattering_mm2pg
    MAC_incoherent_scattering_mm2pg
    MAC_photo_electric_mm2pg
    MAC_pair_production_nuclear_mm2pg
    MAC_pair_production_electron_mm2pg
    MAC_mm2pg   
    LAC_coherent_scattering_pmm
    LAC_incoherent_scattering_pmm
    LAC_photo_electric_pmm
    LAC_pair_production_nuclear_pmm
    LAC_pair_production_electron_pmm
    LAC_pmm
Methods
    plot_LAC: plot linear attenuation coefficient
    interp_LAC: output interpolated linear attenuation coefficient    
    interp_MAC: output interpolated mass attenuation coefficient   
Notes
    Motion
History: Created by ZYu prior to 20250614
============================================================================'''
class XcomCrossSection():
    def __init__(self,fn_material):
        self.fn_material = fn_material
        
        path_data_base = os.environ['XSSLibraries']
        path_data_base = os.path.join(path_data_base,'CrossSection')
        filename = os.path.join(path_data_base,fn_material)
        
        with open(filename) as material_file:
            material_reader = csv.reader(material_file)
            material_definition = [row for row in material_reader]
            # Material name: Element or Compound
            self.material = material_definition[0][0]
            # The original density is in g/cm^3
            self.density_gpmm3 = float(material_definition[1][0])*0.001
            num_energy_bin = int(material_definition[2][0])
            self.energy_kev = np.zeros(num_energy_bin)
            # Mass attenuation coefficinets; original in cm^2/g
            self.MAC_coherent_scattering_mm2pg = np.zeros(num_energy_bin)
            self.MAC_incoherent_scattering_mm2pg = np.zeros(num_energy_bin)
            self.MAC_photo_electric_mm2pg = np.zeros(num_energy_bin)
            self.MAC_pair_production_nuclear_mm2pg = np.zeros(num_energy_bin)
            self.MAC_pair_production_electron_mm2pg = np.zeros(num_energy_bin)
            self.MAC_mm2pg = np.zeros(num_energy_bin)
            for k in range(num_energy_bin):   
                # converting energy from mev to kev
                MAC = material_definition[k + 3][0].split() 
                self.energy_kev[k] = float(MAC[0])*1000
                
                # converting mass attenuation coefficient from cm^2/g to mm^2/g
                self.MAC_coherent_scattering_mm2pg[k] = float(MAC[1])*100
                self.MAC_incoherent_scattering_mm2pg[k] = float(MAC[2])*100
                self.MAC_photo_electric_mm2pg[k] = float(MAC[3])*100
                self.MAC_pair_production_nuclear_mm2pg[k] = float(MAC[4])*100
                self.MAC_pair_production_electron_mm2pg[k] = float(MAC[5])*100
                self.MAC_mm2pg = self.MAC_coherent_scattering_mm2pg \
                               + self.MAC_incoherent_scattering_mm2pg \
                               + self.MAC_photo_electric_mm2pg \
                               + self.MAC_pair_production_nuclear_mm2pg \
                               + self.MAC_pair_production_electron_mm2pg 
                
            
            self.LAC_coherent_scattering_pmm = self.MAC_coherent_scattering_mm2pg*self.density_gpmm3
            self.LAC_incoherent_scattering_pmm = self.MAC_incoherent_scattering_mm2pg*self.density_gpmm3
            self.LAC_photo_electric_pmm = self.MAC_photo_electric_mm2pg*self.density_gpmm3
            self.LAC_pair_production_nuclear_pmm = self.MAC_pair_production_nuclear_mm2pg*self.density_gpmm3
            self.LAC_pair_production_electron_pmm = self.MAC_pair_production_electron_mm2pg*self.density_gpmm3
            
            self.LAC_pmm = self.MAC_mm2pg*self.density_gpmm3
            
            # address K edges so that energy is mono increasing
            denergy = np.diff(self.energy_kev)
            ii_kedge = (denergy == 0)
            if any(ii_kedge):
                self.energy_kev[:-1][ii_kedge] -= sys.float_info.epsilon
                self.energy_kev[1:][ii_kedge]  += sys.float_info.epsilon
                
    '''------------------------------------------------------------------------
    [F] - plot_LAC(self,energy_low_kev=0, energy_high_kev=100000):
    [I] - energy_low_kev: the lower bound of energy for requested display
    [I] - energy_high_kev: the high bound of energy for requested display
    [I] - detail_tag: default False, when True plot all compnents of LAC
    [O] - plotted figure with all LAC activities includded
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''           
    def plot_LAC(self,energy_low_kev=0, energy_high_kev=100000, detail_tag = False):
        idx = (self.energy_kev >= energy_low_kev) & (self.energy_kev <= energy_high_kev) 
        energy = self.energy_kev[idx]
        plt.figure()
        if detail_tag:
            plt.plot(energy,self.LAC_coherent_scattering_pmm[idx])
            plt.plot(energy,self.LAC_incoherent_scattering_pmm[idx])
            plt.plot(energy,self.LAC_photo_electric_pmm[idx])
            plt.plot(energy,self.LAC_pair_production_nuclear_pmm[idx])
            plt.plot(energy,self.LAC_pair_production_electron_pmm[idx])
            plt.plot(energy,self.LAC_pmm[idx])
            plt.gca().legend(('coherent_scattering',\
                       'incoherent_scattering',\
                       'photo_electric',\
                       'pair_production_nuclear', \
                       'pair_production_electron',\
                       'Total'),loc='lower left')
    
            plt.grid(True)
            plt.xlabel('kev')
            plt.title('LAC (1/mm): ' + self.material)
            # plt.xscale('log')
            plt.yscale('log')
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.show()
        else:
            plt.plot(energy,self.LAC_pmm[idx])
            plt.grid(True)
            plt.xlabel('kev')
            plt.title('LAC (1/mm): ' + self.material)
            # plt.xscale('log')
            plt.yscale('log')
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.show()
        
    '''------------------------------------------------------------------------
    [F] - interp_LAC(energy_kev)
    [I] - energy_kev:  1D array representing the desired energy bin locaiton
    [I] - interp_kind [optional]: 'cubic'[default]
    [I] - plot_swtich [optional]: False [default]; True: plot
    [O] - LAC_pmm_interp: Interpolated LAC at the desired energy bin location
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def interp_LAC(self, energy_kev, plot_tag=False):
        energy_kev = np.array(energy_kev)
        if energy_kev.size == 1:
            energy_kev = np.array([energy_kev]).flatten()


        if ( (energy_kev[0] < self.energy_kev[0])|(energy_kev[-1] > self.energy_kev[-1]) ):
            print('Error: requested energy out of range')
            return 0
        else:  
            # Perform Interpolation
            # Address kedge if existed
            kedges = (self.energy_kev[1:] == self.energy_kev[:-1])
            kedge_idx = np.where(kedges)
            energy_kev_tuned = self.energy_kev.copy()
            energy_kev_tuned[kedge_idx] -= np.finfo(np.float32).eps 
            
            interp = pchip(energy_kev_tuned,self.LAC_pmm)
            LAC_pmm_interpolated = interp(energy_kev)
            
        if plot_tag:   
            energy_kev_min = energy_kev[0]*0.5
            energy_kev_max = energy_kev[-1]*2    
            idx = (self.energy_kev >= energy_kev_min) & (self.energy_kev <= energy_kev_max) 
            plt.figure()
            plt.plot(energy_kev,LAC_pmm_interpolated,'--D')
            plt.plot(self.energy_kev[idx],self.LAC_pmm[idx],'--o')
            plt.gca().legend(('interpolated', 'original'))
            plt.xlabel('energy - kev')
            plt.ylabel('LAC (1/mm)')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            plt.show()
        
        return LAC_pmm_interpolated
    
    '''------------------------------------------------------------------------
    [F] - interp_MAC(energy_kev)
    [I] - energy_kev:  1D array representing the desired energy bin locaiton
    [I] - interp_kind [optional]: 'cubic'[default]
    [I] - plot_swtich [optional]: False [default]; True: plot
    [O] - MAC_mm2pg_interp: Interpolated MAC at the desired energy bin location
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def interp_MAC(self, energy_kev, plot_tag=False):
        if ( (energy_kev[0] < self.energy_kev[0])|(energy_kev[-1] > self.energy_kev[-1]) ):
            print('Error: requested energy out of range')
            return 0
        else:  
            # Perform Interpolation
            interp = pchip(self.energy_kev,self.MAC_mm2pg)
            MAC_mm2pg_interpolated = interp(energy_kev)
            
        if plot_tag:   
            energy_kev_min = energy_kev[0]*0.5
            energy_kev_max = energy_kev[-1]*2    
            idx = (self.energy_kev >= energy_kev_min) & (self.energy_kev <= energy_kev_max) 
            plt.figure()
            plt.plot(energy_kev,MAC_mm2pg_interpolated,'--D')
            plt.plot(self.energy_kev[idx],self.MAC_mm2pg[idx],'--o')
            plt.gca().legend(('interpolated', 'original'))
            plt.xlabel('energy - kev')
            plt.ylabel('MAC (mm2/g)')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            plt.show()
        
        return MAC_mm2pg_interpolated
                
'''============================================================================
Class       
    NISTCrossSection
Parent
    None
Attributes
    fn_material
    material
    density_gpmm3
    energy_kev
    MAC_mm2pg   
    MEAC_mm2pg
    LAC_pmm
Methods
    plot_LAC: plot linear attenuation coefficient
    interp_LAC: output interpolated linear attenuation coefficient    
    interp_MAC: output interpolated mass attenuation coefficient   
Notes
    Motion
History: Created by ZYu prior to 20250614
============================================================================'''
class NISTCrossSection():
    def __init__(self,fn_material):
        self.fn_material = fn_material
        
        path_data_base = os.environ['XSSLibraries']
        path_data_base = os.path.join(path_data_base,'CrossSection')
        filename = os.path.join(path_data_base,fn_material)
        
        with open(filename) as material_file:
            material_reader = csv.reader(material_file)
            material_definition = [row for row in material_reader]
            # Material name: Element or Compound
            self.material = material_definition[0][0]
            # The original density is in g/cm^3
            self.density_gpmm3 = float(material_definition[1][0])*0.001
            num_energy_bin = int(material_definition[2][0])
            self.energy_kev = np.zeros(num_energy_bin)
            # Mass attenuation coefficinets; original in cm^2/g
            self.MAC_mm2pg = np.zeros(num_energy_bin)
            self.MEAC_mm2pg = np.zeros(num_energy_bin)
            self.LAC_pmm = np.zeros(num_energy_bin)
            for k in range(num_energy_bin):   
                # converting energy from mev to kev
                MAC = material_definition[k + 3][0].split() 
                self.energy_kev[k] = float(MAC[0])*1000
                
                # converting mass attenuation coefficient from cm^2/g to mm^2/g
                self.MAC_mm2pg[k] = float(MAC[1])*100
                self.MEAC_mm2pg[k] = float(MAC[2])*100                
                
            self.LAC_pmm = self.MAC_mm2pg*self.density_gpmm3
            
            # address K edges so that energy is mono increasing
            denergy = np.diff(self.energy_kev)
            ii_kedge = (denergy == 0)
            if any(ii_kedge):
                self.energy_kev[:-1][ii_kedge] -= sys.float_info.epsilon
                self.energy_kev[1:][ii_kedge]  += sys.float_info.epsilon            

    '''------------------------------------------------------------------------
    [F] - plot_LAC(self,energy_low_kev=0, energy_high_kev=100000):
    [I] - energy_low_kev: the lower bound of energy for requested display
    [I] - energy_high_kev: the high bound of energy for requested display
    [O] - plotted figure with all LAC activities includded
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''           
    def plot_LAC(self,energy_low_kev=0, energy_high_kev=100000):
        idx = (self.energy_kev >= energy_low_kev) & (self.energy_kev <= energy_high_kev) 
        energy = self.energy_kev[idx]
        plt.figure()
        plt.plot(energy,self.LAC_pmm[idx])
        plt.grid(True)
        plt.xlabel('kev')
        plt.title('Linear Attenuation Coefficient (1/mm): ' + self.material)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
    '''------------------------------------------------------------------------
    [F] - interp_LAC(energy_kev)
    [I] - energy_kev:  1D array representing the desired energy bin locaiton
    [I] - interp_kind [optional]: 'cubic'[default]
    [I] - plot_swtich [optional]: False [default]; True: plot
    [O] - LAC_pmm_interp: Interpolated LAC at the desired energy bin location
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def interp_LAC(self, energy_kev, plot_tag=False):
        if ( (energy_kev[0] < self.energy_kev[0])|(energy_kev[-1] > self.energy_kev[-1]) ):
            print('Error: requested energy out of range')
            return 0
        else:  
            # Perform Interpolation
            interp = pchip(self.energy_kev,self.LAC_pmm)
            LAC_pmm_interpolated = interp(energy_kev)
            
        if plot_tag:   
            energy_kev_min = energy_kev[0]*0.5
            energy_kev_max = energy_kev[-1]*2    
            idx = (self.energy_kev >= energy_kev_min) & (self.energy_kev <= energy_kev_max) 
            plt.figure()
            plt.plot(energy_kev,LAC_pmm_interpolated,'--D')
            plt.plot(self.energy_kev[idx],self.LAC_pmm[idx],'--o')
            plt.gca().legend(('interpolated', 'original'))
            plt.xlabel('energy - kev')
            plt.ylabel('LAC (1/mm)')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            plt.show()
        
        return LAC_pmm_interpolated
    
    '''------------------------------------------------------------------------
    [F] - interp_MAC(energy_kev)
    [I] - energy_kev:  1D array representing the desired energy bin locaiton
    [I] - interp_kind [optional]: 'cubic'[default]
    [I] - plot_swtich [optional]: False [default]; True: plot
    [O] - MAC_mm2pg_interp: Interpolated MAC at the desired energy bin location
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def interp_MAC(self, energy_kev, plot_tag=False):
        if ( (energy_kev[0] < self.energy_kev[0])|(energy_kev[-1] > self.energy_kev[-1]) ):
            print('Error: requested energy out of range')
            return 0
        else:  
            # Perform Interpolation
            interp = pchip(self.energy_kev,self.MAC_mm2pg)
            MAC_mm2pg_interpolated = interp(energy_kev)
            
        if plot_tag:   
            energy_kev_min = energy_kev[0]*0.5
            energy_kev_max = energy_kev[-1]*2    
            idx = (self.energy_kev >= energy_kev_min) & (self.energy_kev <= energy_kev_max) 
            plt.figure()
            plt.plot(energy_kev,MAC_mm2pg_interpolated,'--D')
            plt.plot(self.energy_kev[idx],self.MAC_mm2pg[idx],'--o')
            plt.gca().legend(('interpolated', 'original'))
            plt.xlabel('energy - kev')
            plt.ylabel('MAC (mm2/g)')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            plt.show()
        
        return MAC_mm2pg_interpolated

    '''------------------------------------------------------------------------
    [F] - interp_MEAC(energy_kev)
    [I] - energy_kev:  1D array representing the desired energy bin locaiton
    [I] - interp_kind [optional]: 'cubic'[default]
    [I] - plot_swtich [optional]: False [default]; True: plot
    [O] - MEAC_mm2pg_interp: Interpolated MAC at the desired energy bin location
    [H] - Created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def interp_MEAC(self, energy_kev, plot_tag=False):
        if ( (energy_kev[0] < self.energy_kev[0])|(energy_kev[-1] > self.energy_kev[-1]) ):
            print('Error: requested energy out of range')
            return 0
        else:  
            # Perform Interpolation
            interp = pchip(self.energy_kev,self.MEAC_mm2pg)
            MEAC_mm2pg_interpolated = interp(energy_kev)
            
        if plot_tag:   
            energy_kev_min = energy_kev[0]*0.5
            energy_kev_max = energy_kev[-1]*2    
            idx = (self.energy_kev >= energy_kev_min) & (self.energy_kev <= energy_kev_max) 
            plt.figure()
            plt.plot(energy_kev,MEAC_mm2pg_interpolated,'--D')
            plt.plot(self.energy_kev[idx],self.MEAC_mm2pg[idx],'--o')
            plt.gca().legend(('interpolated', 'original'))
            plt.xlabel('energy - kev')
            plt.ylabel('MEAC (mm2/g)')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            plt.show()
        
        return MEAC_mm2pg_interpolated