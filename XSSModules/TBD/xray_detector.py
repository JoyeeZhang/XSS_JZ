import numpy as np

'''============================================================================
Class       
    IdealFPD (pair with IdealTube)
Parent
    None
Modeling
    Electronic noise 
        White noise with 0 mean
        single frame for all views
Assumptions
    Ideal detector pixel size
    Ideal detector response
        - No signal loss (DQE = 1)
        - Ideal spatial resolution response (blooding factor = 1)
        - Ignore the following processes
            photon->light
            light->electronic (electron per quanta)
            AD process (no bit limitation
Inputs
    channel_space_mm (du)
    num_channel (nu)
    channel_off (uoff, corresponding to orthogonal projection of source)
    (* u(k) = (k-uoff)xdu k = 0,1,2,...)
    row_space_mm (dv)
    num_row (nv)
    row_off (voff, corresponding to orthogonal projeciton of source)
    (* v(k) = (k-voff)xdv k = 0,1,2,...)
    detector_crd_axis_normal (ew)
    detector_crd_axis_channel (eu)
    detector_crd_axis_channel (ev)
    source-to_detector_distance_mm
    detector_noise_floor: set 0 [default] to turn off electronic noise
    detector_type - string
        - 'EID' [default]: energy-integrating-detector
        - 'PCD': photon-counting-detector
Attributes
    channel_space_mm, num_channel, channel_off
    row_space_mm, num_row, row_off
    detector_crd_axis_normal
    detector_crd_axis_channel
    detector_crd_axis_row
    source_to_detector_distance_mm
    detector_offset_frame: white noise with mean 0
    detector_data_frame: 
    detector_type: 'EID' or 'PCD'
Methods   
        
Notes
    
History [yyyy/mm/dd BY whom ABOUT what]
============================================================================'''
class IdealFPD():
    def __init__(self, 
                 channel_space_mm, num_channel,channel_off\ # du, nu, uoff
                 row_space_mm, num_row,row_off\ # dv, nv, voff
                 detector_crd_axis_normal,\ # ew
                 detector_crd_axis_channel, \ # eu
                 detector_crd_axis_row, \ # ev
                 source_to_detector_distance_mm, \ # D
                 detector_noise_floor = 0, \ # average electronic noise 
                 detector_type = 'EID'
                 ):
        # detector channel grid definition
        self.channel_space_mm = channel_space_mm
        self.num_channel = num_channel
        self.channel_off = channel_off
        # detector row grid definition
        self.row_space_mm = row_space_mm
        self.num_row = num_row
        self.row_off = row_off
        # basis vectors of the detector coordinate system
        self.detector_crd_axis_normal = detector_crd_axis_normal
        self.detector_crd_axis_channel = detector_crd_axis_channel
        self.detector_crd_axis_row = detector_crd_axis_row
        # source-to-detector distance in mm
        self.source_to_detector_distance_mm = source_to_detector_distance_mm
        # single offset frame [nv,nu]
        self.detector_offset_frame = np.random.normal(0,detector_noise_floor,[num_row,num_channel])
        # projection data in attenuation [0, 1)
        self.detector_data_frame = np.zeros([num_row,num_channel])
        # detector type
        self.detector_type = detector_type

'''class FlatPanelDetector():
    def __init__(self,channel_space_mm, row_space_mm, channel_num, row_num, ADC_bit = 16, gain_level = 5, enum_per_photon = 0.002, enum_per_ADU_base_gain = 28.57, vendor='Varex', product='XRD 4343CT', scintillator='CsI'):
        # Mandatory Information
        self.channel_space_mm = channel_space_mm
        self.row_space_mm = row_space_mm
        self.channel_num = channel_num
        self.row_num = row_num
        
        # Optional Information
        self.ADC_bit = ADC_bit
        self.gain_level = gain_level
        self.vendor = vendor
        self.product = product
        self.scintillator = scintillator
        self.enum_per_ADU_base_gain = enum_per_ADU_base_gain
        self.enum_per_photon = enum_per_photon
        # Derived parameters
        self.enum_per_ADU = np.round(self.enum_per_ADU_base_gain*np.power(2,gain_level))
        
        if gain_level == 1:
            self.offset_bias_ADU = 600
            self.offset_noise_ADU = 6
        elif gain_level == 2:
            self.offset_bias_ADU = 900
            self.offset_noise_ADU = 7
        elif gain_level == 3:
            self.offset_bias_ADU = 1100
            self.offset_noise_ADU = 8
        elif gain_level == 4:
            self.offset_bias_ADU = 1300
            self.offset_noise_ADU = 9
        elif gain_level == 5:
            self.offset_bias_ADU = 1800
            self.offset_noise_ADU = 10
        elif gain_level == 6:
            self.offset_bias_ADU = 3000
            self.offset_noise_ADU = 15
        else:
            self.offset_bias_ADU = 3000
            self.offset_noise_ADU = 15
            
    # Function: Generae a single offset image in ADU    
    def generate_single_offset_image_ADU(self)   :       
        offset_image = np.random.randint(self.offset_noise_ADU,size=(self.row_num,self.channel_num)) + self.offset_bias_ADU;
        return offset_image
    
    # Function: Generate a single projection image in ADU
    # [I] - photon_num: number of incident photons @ xmAs/SDD/mm2
    #       size: 2D array [row_num, channel_num]
    # [I] - num_photons_per_ADU: conversion rate from number of photon to ADU
    #       this includes lots of linear(assumed) physics and ee operations
    # [O] - scan_image: projection image convered to ADU image   
    #       size: 2D array [row_num, channel_num]
    # [N] - Store sequence: row by row  
    def generate_single_scan_image_ADU(self,photon_num):
        detector_pixel_area = self.channel_space_mm*self.row_space_mm;
        scan_image = photon_num*detector_pixel_area
 '''
      
        
	


