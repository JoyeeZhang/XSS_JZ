import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import copy
import shutil
import matplotlib.pyplot as plt
from XSSModules import geometry as geo
from XSSModules.xray_spectra import SpekCalc as spk
from XSSModules.cross_section import XcomCrossSection as cs
from XSSModules.image import Volume3D as v3d
from XSSModules import support as spt
from XSSModules import enum_list as em
from XSSModules import analytical_phantom as ap
from XSSModules import file_io as fio

'''============================================================================
Class       
  CTCBFPDHelicalLAC()
Parent
  n.a.
Attributes
  fn_scan_protocol: XML file containing scan protocol
  device_name: name of the scanner
  xml_version: version of the XML file
  raw_data_folder: folder to store raw data
  phantom_type: type of phantom used
  phantom_name: name of the phantom
  coordinate_system_type: type of coordinate system used
  trajectory_type: type of trajectory used
  geometry_type: type of geometry used
  focal_spot_model: model of focal spot
  detector_pixel_model: model of detector pixel
  gantry_tilt_angle_deg: gantry tilt angle in degrees
  source_starting_angle_deg: starting angle of source in degrees
  source_starting_position_mm: starting position of source in mm
  source_rotation_direction: direction of source rotation (RHR or LHR)
  total_number_of_projections: total number of projections
  views_per_rotation: number of views per rotation
  source_translation_per_rotation_mm: translation distance per rotation in mm
  source_to_isocenter_distance_mm: distance from source to isocenter in mm
  source_to_piercing_point_distance_mm: distance from source to piercing point in mm
  detector_layout: layout of the detector (U1V1, U1V0, U0V1, U0V0)
  detector_channel_space_mm: space between detector channels in mm
  detector_channel_number: number of detector channels
  detector_channel_offset: offset for detector channels in mm
  detector_row_space_mm: space between detector rows in mm
  detector_row_number: number of detector rows
  detector_row_offset: offset for detector rows in mm
  detector_rotation_around_waxis_deg: rotation around w-axis in degrees
  detector_rotation_around_vaxis_deg: rotation around v-axis in degrees
  detector_rotation_around_uaxis_deg: rotation around u-axis in degrees
  detector_rotation_deployment_order: order of deployment for rotations (wvu)                  
Methods
  load_scan_protocol_xml(fn_scan_protocol): Load scan protocol from XML file
  generate_CBFPD_geometry_angle_zero(): Generate CBFPD geometry at angle zero
  generate_CBFPD_geometry_single_view(idx_view): Generate CBFPD geometry for a single view
  display_source_trajectory(disp_angular_interval_deg): Display source trajectory
  generate_CBFPD_projections(): Generate cone beam projections and store data
  generate_CBFPD_projections_beam_avarage(nups_channel=1, nups_row=1): Generate projections with beam averaging             
  generate_CBFPD_sinogram(nups_channel=1, nups_row=1): Generate sinogram and store data
Notes
    
History: Created by ZYu prior to 20250614
         Revised by JZhang prior to 20260404
         {Add method - generate_CBFPD_sinogram()}
============================================================================'''
class CTCBFPDHelicalLAC():         
    def __init__(self,fn_scan_protocol):  
        self.fn_scan_protocol = fn_scan_protocol 
        self.load_scan_protocol_xml(fn_scan_protocol)
        self.generate_CBFPD_geometry_angle_zero()
        # self.generate_source_trajectory(fn_scan_protocol)
        # self.generate_scan_geometry(fn_scan_protocol)
        # self.generate_phantom(fn_scan_protocol)
        # self.generate_projection_data()
    '''------------------------------------------------------------------------
    [F] - load_scan_protocol_xml from XML file
    [I] - fn_scan_protocol
    [O] - self.attributes including info of 
          General info, phantom, geometry, and imaging chain
    [H] - created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def load_scan_protocol_xml(self,fn_scan_protocol = None):    
      if fn_scan_protocol is None:
        fn_scan_protocol = self.fn_scan_protocol
                
      root = ET.parse(fn_scan_protocol)
      XSS = root.getroot()
      
      #------ general
      self.device_name = XSS.find('device_name').text
      self.xml_version = XSS.find('xml_version').text
      self.raw_data_folder = XSS.find('raw_data_folder').text
      
      #------ phantom
      phantom = XSS.find('phantom')
      self.phantom_type = phantom.find('phantom_type').text
      self.phantom_name = phantom.find('phantom_name').text

      #------ Geometry
      geometry = XSS.find('scan_geometry')
      self.coordinate_system_type = geometry.find('coordinate_system_type').text
      self.trajectory_type = geometry.find('trajectory_type').text
      self.geometry_type = geometry.find('geometry_type').text
      self.focal_spot_model = geometry.find('focal_spot_model').text
      self.detector_pixel_model = geometry.find('detector_pixel_model').text
      self.gantry_tilt_angle_deg = np.float32(geometry.find('gantry_tilt_angle_deg').text)
      self.source_starting_angle_deg = np.float32(geometry.find('source_starting_angle_deg').text)
      self.source_starting_position_mm = np.float32(geometry.find('source_starting_position_mm').text)
      self.source_rotation_direction = geometry.find('source_rotation_direction').text
      self.total_number_of_projections = np.int32(geometry.find('total_number_of_projections').text)
      self.views_per_rotation = np.int32(geometry.find('views_per_rotation').text)
      self.source_translation_per_rotation_mm = np.float32(geometry.find('source_translation_per_rotation_mm').text)
      self.source_to_isocenter_distance_mm = np.float32(geometry.find('source_to_isocenter_distance_mm').text)
      self.source_to_piercing_point_distance_mm = np.float32(geometry.find('source_to_piercing_point_distance_mm').text)
      self.detector_layout = geometry.find('detector_layout').text.upper()
      self.detector_channel_space_mm = np.float32(geometry.find('detector_channel_space_mm').text)
      self.detector_channel_number = np.int32(geometry.find('detector_channel_number').text)
      self.detector_channel_offset = np.float32(geometry.find('detector_channel_offset').text)
      self.detector_row_space_mm = np.float32(geometry.find('detector_row_space_mm').text)
      self.detector_row_number = np.int32(geometry.find('detector_row_number').text)
      self.detector_row_offset = np.float32(geometry.find('detector_row_offset').text)
      self.detector_rotation_around_waxis_deg = np.float32(geometry.find('detector_rotation_around_waxis_deg').text)
      self.detector_rotation_around_vaxis_deg = np.float32(geometry.find('detector_rotation_around_vaxis_deg').text)
      self.detector_rotation_around_uaxis_deg = np.float32(geometry.find('detector_rotation_around_uaxis_deg').text)
      self.detector_rotation_deployment_order = geometry.find('detector_rotation_deployment_order').text
      
      #------ Imaging Chain
      imaging_chain = XSS.find('imaging_chain')
      self.imaging_chain_type = imaging_chain.find('imaging_chain_type').text
      self.forward_projection_method = imaging_chain.find('forward_projection_method').text
      self.projection_type = imaging_chain.find('projection_type').text.upper()
      self.projection_data_type = imaging_chain.find('projection_data_type').text.upper()
      self.energy_kev = np.float32(imaging_chain.find('energy_kev').text)
      self.photon_counts_per_ray = np.float32(imaging_chain.find('photon_counts_per_ray').text)
      self.electronic_noise_std = np.float32(imaging_chain.find('electronic_noise_std').text)
      self.poisson_noise = spt.str_to_bool(imaging_chain.find('poisson_noise').text)
      
      #------ Derive key parameters
      self.gantry_tilt_angle_rad = np.deg2rad(self.gantry_tilt_angle_deg)
      self.source_starting_angle_rad = np.deg2rad(self.source_starting_angle_deg)
      self.detector_width_mm = self.detector_channel_space_mm*self.detector_channel_number
      self.detector_height_mm = self.detector_row_space_mm*self.detector_row_number
      self.magnification_factor = self.source_to_piercing_point_distance_mm/self.source_to_isocenter_distance_mm  
      self.nominal_helical_pitch = self.source_translation_per_rotation_mm*self.magnification_factor/self.detector_width_mm
      self.source_translation_per_view_mm = self.source_translation_per_rotation_mm/self.views_per_rotation 
      
      # angular step size in rad
      angular_space_rad = 2 * np.pi / self.views_per_rotation

      try:        
        self.source_angular_space_rad = angular_space_rad if self.source_rotation_direction.upper() == em.RotationDirection.RHR.name else -angular_space_rad
      except KeyError:
        print('Invalid rotation direction!!')        
      
      # gantry title and rotation axis
      etax = np.deg2rad(self.gantry_tilt_angle_deg)
      self.ve_rotation_axis = np.array([0,-np.sin(etax),np.cos(etax)])
      self.ve_translation_axis = np.array([0, 0, 1])
      
    '''------------------------------------------------------------------------
    [F] - generate_CBFPD_geometry_angle_zero()
    [I] - self
    [N] - Angle zero: source on the x-axis of the CT scan coordinate system
        - Generate CBFPD object at view angle zero. Because gantry tilt is
          around x-axis, it has no impact on detector basis or source vector
        - Supported configuratoins
            detector_layout: 'U1V1','U1V0','U0V1','U0V0'
            detector orientation rotation order (nominal->general): 'wvu'
    [H] - created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def generate_CBFPD_geometry_angle_zero(self):
      # Define basis vectors for the nominal detector coordiante system
      ve_normal = np.array([1, 0, 0])   # detector normal vector (vew)
      match self.detector_layout:
        case em.DetectorLayout.U1V1.name:          
          ve_channel = np.array([0, 1, 0]) # basis vector for channel index (veu)
          ve_row = np.array([0, 0, 1])     # basis vector for row index (vev)
        case em.DetectorLayout.U1V0.name:          
          ve_channel = np.array([0, 1, 0]) # basis vector for channel index (veu)
          ve_row = -np.array([0, 0, 1])     # basis vector for row index (vev)     
        case em.DetectorLayout.U0V1.name:          
          ve_channel = -np.array([0, 1, 0]) # basis vector for channel index (veu)
          ve_row = np.array([0, 0, 1])     # basis vector for row index (vev)
        case em.DetectorLayout.U0V0.name:          
          ve_channel = -np.array([0, 1, 0]) # basis vector for channel index (veu)
          ve_row = -np.array([0, 0, 1])     # basis vector for row index (vev)                   
        case _:
          print('Undefined detector layout!')
            
      nchannel  = self.detector_channel_number    # number of channels
      dchannel  = self.detector_channel_space_mm  # channel space (def in mm)
      ochannel  = self.detector_channel_offset    # channel offset
      nrow  = self.detector_row_number            # number of rows
      drow  = self.detector_row_space_mm          # row space (default in mm)
      orow  = self.detector_row_offset            # row offset
      SID   = self.source_to_isocenter_distance_mm# source to isocenter distance
      SPD   = self.source_to_piercing_point_distance_mm # source to piercing point distance
      vsource  = SID*ve_normal                    # assume in mm
      hs = self.source_starting_position_mm
      dh = self.source_translation_per_view_mm
      dlam = self.source_angular_space_rad
      lams = self.source_starting_angle_rad
      h = hs - lams*dh/dlam # source position at lambda = 0
      vsource = vsource + h*self.ve_translation_axis
          
      SDD   = SPD
      etau  = self.detector_rotation_around_uaxis_deg
      etav  = self.detector_rotation_around_vaxis_deg
      etaw  = self.detector_rotation_around_waxis_deg
          
      # Detector orientation adjustment with rotation:
      #   - The rotation around different det axes are applied one after 
      #     another, rather than applied relative to the old ones
      CBGeo = geo.CBFPD(ve_channel,ve_row,ve_normal,
                        nchannel,dchannel,ochannel,
                        nrow, drow, orow,          
                        SDD, vsource)
      if self.detector_rotation_deployment_order.lower() == 'wvu':
          CBGeo.rotate_detector_around_normal_axis_at_piercing_RH(etaw)
          CBGeo.rotate_detector_around_row_axis_at_piercing_RH(etav)
          CBGeo.rotate_detector_around_channel_axis_at_piercing_RH(etau)
      else:
          print('Undefined detector rotation deployment order!')          
          
      self.cone_beam_geometry_angle_zero = CBGeo
      
      return CBGeo
    
    '''------------------------------------------------------------------------
    [F] - generate_single_view_CBFPD_geometry(idx_view)
    [N] - one application of this function is to generate the CB projection 
          geometry for each view angle around the rotation axis
        - Supported configuratoins
            coordinatey system type: 'CT'    
    [H] - created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def generate_CBFPD_geometry_single_view(self,idx_view):
      if hasattr(self,'cone_beam_geometry_angle_zero') == False:
        self.generate_CBFPD_geometry_angle_zero()
            
      lams = self.source_starting_angle_rad
      dlam = self.source_angular_space_rad
      lam  = lams + idx_view*dlam
      lam_deg = np.rad2deg(lam)
      
      hs = self.source_starting_position_mm
      dh = self.source_translation_per_view_mm
      h = hs + idx_view*dh

      CBGeo = copy.deepcopy(self.cone_beam_geometry_angle_zero)
      # detector basis vectors
      CBGeo.ve_channel = geo.rotate_around_axis_RH(lam_deg, CBGeo.ve_channel,self.ve_rotation_axis)
      CBGeo.ve_row = geo.rotate_around_axis_RH(lam_deg, CBGeo.ve_row,self.ve_rotation_axis)
      CBGeo.ve_normal = geo.rotate_around_axis_RH(lam_deg, CBGeo.ve_normal,self.ve_rotation_axis)
      
      # x-ray source position rotation (first) + translation (second)
      vsource = CBGeo.vsource
      vsource[2] = 0
      vsource = geo.rotate_around_axis_RH(lam_deg, vsource,self.ve_rotation_axis)
      CBGeo.vsource = vsource + h*self.ve_translation_axis      

      return CBGeo

    '''------------------------------------------------------------------------
    [F] - display_source_trajectory()
    [I] - disp_angular_interval_deg: angular step size used for display
    [N] - displays scan geometry every 10 degrees
    [H] - created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def display_source_trajectory(self,disp_angular_interval_deg=None):
      if disp_angular_interval_deg == None:
        disp_angular_interval_deg = 10
        
      # CB projection geo at view angle 0
      CBGeo = copy.deepcopy(self.generate_CBFPD_geometry_angle_zero())
      CBGeo.vsource[2] = 0
      
      # gantry rotation
      ve_rotation_axis = self.ve_rotation_axis
      lams = self.source_starting_angle_rad
      dlam = self.source_angular_space_rad
      nlam = self.total_number_of_projections
      lam  = lams + np.arange(nlam)*dlam
      lam_deg = np.rad2deg(lam)
      
      # source translation
      ve_translation_axis = self.ve_translation_axis
      hs = self.source_starting_position_mm
      dh = self.source_translation_per_view_mm
      nh = nlam
      h  = hs + np.arange(nh)*dh
      
      # source positions: first rotation then translation
      # the order matters when the rotation and translation axes are diff
      vsource = np.outer(np.ones(nlam),CBGeo.vsource) #[nview,3]
      vsource = geo.rotate_around_axis_RH(lam_deg, vsource, ve_rotation_axis)
      vsource = vsource + np.outer(h,ve_translation_axis)
            
      # plot: source trajectory, detector boundary, piercing point
      nlam_skip = np.int32(disp_angular_interval_deg/np.rad2deg(dlam))
      fig = plt.figure()
      ax = fig.add_subplot(projection='3d')
      
      for k in np.arange(0,nlam,nlam_skip):
        ax.clear()
        # source trajectory
        ax.plot3D(vsource[:,0],vsource[:,1],vsource[:,2],label='Source Trajectory')
        
        # CB geometry at view k
        CBGeok = self.generate_CBFPD_geometry_single_view(k)
        vak = CBGeok.vsource
        uoff = CBGeok.ochannel
        nu = CBGeok.nchannel
        du = CBGeok.dchannel
        voff = CBGeok.orow
        nv = CBGeok.nrow
        dv = CBGeok.drow
        uoffp = CBGeok.ochannel_piercing
        voffp = CBGeok.orow_piercing
        veu = CBGeok.ve_channel
        vev = CBGeok.ve_row
        vew = CBGeok.ve_normal
        D = CBGeok.SDD
        
        umin = -uoff*du
        umax = (nu-uoff-1)*du
        vmin = -voff*dv
        vmax = (nv-voff-1)*dv
        upiercing = umin + uoffp*du
        vpiercing = vmin + voffp*dv
        
        # detector origin (ortho projection of source)
        vod = vak - D*vew
        v00 = vod + umin*veu + vmin*vev 
        v01 = vod + umin*veu + vmax*vev
        v10 = vod + umax*veu + vmin*vev
        v11 = vod + umax*veu + vmax*vev
        # piercing point: intersection between det and line connect src and iso
        vdp = vod + upiercing*veu + vpiercing*vev
        
        # source point
        ax.scatter3D(vak[0],vak[1],vak[2],color='red')
        # source to piercing
        vtmp = np.vstack((vak,vdp))
        ax.plot3D(vtmp[:,0],vtmp[:,1],vtmp[:,2])
        # detector
        vtmp = np.vstack((v00,v01,v11,v10,v00))
        ax.plot3D(vtmp[:,0],vtmp[:,1],vtmp[:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-2000, 2000)
        ax.set_ylim(-2000, 2000)
        ax.set_zlim(-2000, 2000)
        ax.set_box_aspect([1, 1, 1])
        # ax.set_aspect('auto')
        ttl = str(np.int32(lam_deg[k]))+'deg'
        ax.set_title(ttl)
        plt.tight_layout()
        # plt.axis('image') does not work for 3D plot
        plt.legend()
        plt.pause(0.2)
        
      plt.show()
    
    '''------------------------------------------------------------------------
    [F] - generate_CBFPD_projections()
    [N] - Generate cone beam projections and store data
    [H] - created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''
    def generate_CBFPD_projections(self):
      # Generate phantom
      phantom_name = self.phantom_name
      phantom = ap.StaticLACPhantom(phantom_name)
      
      # Generate projection data header
      nview = self.total_number_of_projections
      nrow = self.detector_row_number
      nchannel = self.detector_channel_number
      angular_space_per_view = self.source_angular_space_rad
      drow = self.detector_row_space_mm
      dchannel = self.detector_channel_space_mm
      starting_angular_position = self.source_starting_angle_rad
      orow = self.detector_row_offset
      ochannel = self.detector_channel_offset
      projection_data_type_numpy = em.DataTypeDtype[self.projection_data_type].value
      projection_type_string = self.projection_type.upper()
      starting_axial_position = self.source_starting_position_mm
      axial_movement_per_view = self.source_translation_per_view_mm
      SDD = self.source_to_piercing_point_distance_mm
      SID = self.source_to_isocenter_distance_mm
      
      header = fio.create_header_dict_projection_basic(nview,nrow,nchannel,
                              angular_space_per_view,drow,dchannel,
                              starting_angular_position,orow,ochannel,
                              projection_data_type_numpy,
                              projection_type_string,
                              starting_axial_position,
                              axial_movement_per_view,
                              SDD,
                              SID)
      
      os.makedirs(self.raw_data_folder, exist_ok=True)
      fnProjection = phantom_name + '_' + str(nchannel) + 'x' + str(nrow) + 'x' + str(nview) +'.prj'
      fnProjection = os.path.join(self.raw_data_folder, fnProjection)
      fio.write_header(fnProjection,header)
 
      # Generate projection data
      fig = plt.figure()
      ax = fig.add_subplot()
      for k in range(nview):
        CBGeo = self.generate_CBFPD_geometry_single_view(k)
        gk = phantom.generate_line_integral_single_projection_FPD(CBGeo)
        fio.append_slices(fnProjection,gk,projection_data_type_numpy)

        if np.mod(k,10) == 0:
          print(f'Completed projection {k} of {nview}')
          ax.clear()  # clear previous content
          ax.set_title(f'View #: {k}')
          ax.imshow(gk, cmap='gray')
          plt.tight_layout()
          plt.axis('image')
          plt.pause(0.2)
      plt.show()
      
      # Save scan protocol XML file
      fn_scan_protocol_source = self.fn_scan_protocol
      fld_scan_protocol_target = self.raw_data_folder
      shutil.copy(fn_scan_protocol_source, fld_scan_protocol_target)
      
    '''------------------------------------------------------------------------
    [F] - generate_CBFPD_projections_beam_average()
    [I] - nups_channel, nups_row: Average of upsampled rays
          -- Both shall be odd numbers, default is 1
          -- nunew = nups_channel*nu
          -- dunew = du/nunew
          -- uoffnew = nups_channel*nu + (nu-1)/2
    [N] - Generate cone beam projections and store data
    [H] - created by ZYu prior to 20250510
    ------------------------------------------------------------------------'''
    def generate_CBFPD_projections_beam_average(self,nups_channel=1,nups_row=1):
      # Generate phantom
      phantom_name = self.phantom_name
      phantom = ap.StaticLACPhantom(phantom_name)
      
      # Generate projection data header
      nview = self.total_number_of_projections
      nrow = self.detector_row_number
      nchannel = self.detector_channel_number
      angular_space_per_view = self.source_angular_space_rad
      drow = self.detector_row_space_mm
      dchannel = self.detector_channel_space_mm
      starting_angular_position = self.source_starting_angle_rad
      orow = self.detector_row_offset
      ochannel = self.detector_channel_offset
      projection_data_type_numpy = em.DataTypeDtype[self.projection_data_type].value
      projection_type_string = self.projection_type.upper()
      starting_axial_position = self.source_starting_position_mm
      axial_movement_per_view = self.source_translation_per_view_mm
      SDD = self.source_to_piercing_point_distance_mm
      SID = self.source_to_isocenter_distance_mm
      
      header = fio.create_header_dict_projection_basic(nview,nrow,nchannel,
                              angular_space_per_view,drow,dchannel,
                              starting_angular_position,orow,ochannel,
                              projection_data_type_numpy,
                              projection_type_string,
                              starting_axial_position,
                              axial_movement_per_view,
                              SDD,
                              SID)
      
      os.makedirs(self.raw_data_folder, exist_ok=True)
      fnProjection = phantom_name + '_beamavg'+str(nups_channel)+'x'+str(nups_row) + '_' + str(nchannel) + 'x' + str(nrow) + 'x' + str(nview) + '.prj'
      fnProjection = os.path.join(self.raw_data_folder, fnProjection)
      fio.write_header(fnProjection,header)
 
      # Generate projection data
      fig = plt.figure()
      ax = fig.add_subplot()
      for k in range(nview):
        CBGeo = self.generate_CBFPD_geometry_single_view(k)
        # Upsampling
        nrow = CBGeo.nrow
        nchannel = CBGeo.nchannel 
        CBGeo.nrow = nups_row*CBGeo.nrow
        CBGeo.drow = CBGeo.drow/nups_row
        CBGeo.orow = nups_row*orow + 0.5*(nups_row-1)
        CBGeo.nchannel = nups_channel*CBGeo.nchannel
        CBGeo.dchannel = CBGeo.dchannel/nups_channel
        CBGeo.ochannel = nups_channel*ochannel + 0.5*(nups_channel-1)
        gk = phantom.generate_line_integral_single_projection_FPD(CBGeo)        
        # Downsampling (averaging)
        gk = gk.reshape(nrow,nups_row,nchannel,nups_channel).mean(axis=(1,3)) # has to reassign it back to gk, otherwise, gk does not change
        # Save data
        fio.append_slices(fnProjection,gk,projection_data_type_numpy)

        if np.mod(k,10) == 0:
          print(f'Completed projection {k} of {nview}')
          ax.clear()  # clear previous content
          ax.set_title(f'View #: {k}')
          ax.imshow(gk, cmap='gray')
          plt.tight_layout()
          plt.axis('image')
          plt.pause(0.2)
      plt.show()
      
      # Save scan protocol XML file
      fn_scan_protocol_source = self.fn_scan_protocol
      fld_scan_protocol_target = self.raw_data_folder
      shutil.copy(fn_scan_protocol_source, fld_scan_protocol_target)

    '''------------------------------------------------------------------------
    [F] - generate_CBFPD_sinogram()
    [I] - save_path: path to save the sinogram
        - save_name: name of the sinogram
        - nups_channel, nups_row: Average of upsampled rays
          -- Both shall be odd numbers, default is 1
          -- nunew = nups_channel*nu
          -- dunew = du/nunew
          -- uoffnew = nups_channel*nu + (nu-1)/2
    [N] - Generate cone beam sinogram and store data
    [H] - created by JZhang prior to 20260404 
    ------------------------------------------------------------------------'''
    def generate_CBFPD_sinogram(self, save_path, save_name, nups_channel=1, nups_row=1):
        # Generate phantom
        phantom_name = self.phantom_name
        phantom = ap.StaticLACPhantom(phantom_name)

        # Generate projection data
        nview = self.views_per_rotation
        orow = self.detector_row_offset
        nchannel = self.detector_channel_number
        ochannel = self.detector_channel_offset
        sinogram = np.zeros([nchannel, nview])
        for k in range(nview):
            CBGeo = self.generate_CBFPD_geometry_single_view(k)
            # Upsampling
            nrow = CBGeo.nrow
            nchannel = CBGeo.nchannel
            CBGeo.nrow = nups_row * CBGeo.nrow
            CBGeo.drow = CBGeo.drow / nups_row
            CBGeo.orow = nups_row * orow + 0.5 * (nups_row - 1)
            CBGeo.nchannel = nups_channel * CBGeo.nchannel
            CBGeo.dchannel = CBGeo.dchannel / nups_channel
            CBGeo.ochannel = nups_channel * ochannel + 0.5 * (nups_channel - 1)
            gk = phantom.generate_line_integral_single_projection(CBGeo)
            # Downsampling (averaging)
            gk = gk.reshape(nrow, nups_row, nchannel, nups_channel).mean(axis=(1, 3))
            sinogram[:, k] = gk[int(orow)]
            if np.mod(k, 10) == 0:
                print(f'Completed projection {k} of {nview}')

        print(f'Sinogram 2D shape: {sinogram.shape}')
        plt.imshow(sinogram, cmap='gray')
        plt.title("Sinogram")
        plt.show()

        os.makedirs(save_path, exist_ok=True)
        file_name = f"{save_name}_sinogram_{nview}view_{nchannel}channel.npy"
        np.save(os.path.join(save_path, file_name), sinogram)


'''============================================================================
Class       
  CTCBCurveDHelicalLAC()
Parent
  n.a.
Attributes
  fn_scan_protocol: XML file containing scan protocol
  device_name: name of the scanner
  xml_version: version of the XML file
  raw_data_folder: folder to store raw data
  phantom_type: type of phantom used
  phantom_name: name of the phantom
  coordinate_system_type: type of coordinate system used
  trajectory_type: type of trajectory used
  geometry_type: type of geometry used
  focal_spot_model: model of focal spot
  detector_pixel_model: model of detector pixel
  gantry_tilt_angle_deg: gantry tilt angle in degrees
  source_starting_angle_deg: starting angle of source in degrees
  source_starting_position_mm: starting position of source in mm
  source_rotation_direction: direction of source rotation (RHR or LHR)
  total_number_of_projections: total number of projections
  views_per_rotation: number of views per rotation
  source_translation_per_rotation_mm: translation distance per rotation in mm
  source_to_isocenter_distance_mm: distance from source to isocenter in mm
  source_to_piercing_point_distance_mm: distance from source to piercing point in mm
  detector_layout: layout of the detector (U1V1, U1V0, U0V1, U0V0)
  detector_channel_space_mm: space between detector channels in mm
  detector_channel_number: number of detector channels
  detector_channel_offset: offset for detector channels in mm
  detector_row_space_mm: space between detector rows in mm
  detector_row_number: number of detector rows
  detector_row_offset: offset for detector rows in mm
  detector_fan_angle_degree: interval between fan angles in degrees
  detector_rotation_around_waxis_deg: rotation around w-axis in degrees
  detector_rotation_around_vaxis_deg: rotation around v-axis in degrees
  detector_rotation_around_uaxis_deg: rotation around u-axis in degrees
  detector_rotation_deployment_order: order of deployment for rotations (wvu)                  
Methods
  load_scan_protocol_xml(fn_scan_protocol): Load scan protocol from XML file
  generate_CBCurveD_geometry_angle_zero(): Generate CBCurveD geometry at angle zero
  generate_CBCurveD_geometry_single_view(idx_view): Generate CBCurveD geometry for a single view
  display_source_trajectory(disp_angular_interval_deg): Display source trajectory
  generate_CBCurveD_projections(): Generate cone beam projections and store data
  generate_CBCurveD_projections_beam_avarage(nups_channel=1, nups_row=1): Generate projections with beam averaging             
  generate_CBCurveD_sinogram(nups_channel=1, nups_row=1): Generate sinogram and store data
Notes
    
History: created by JZhang prior to 20250922
==============================================================================='''
class CTCBCurveDHelicalLAC():
    def __init__(self, fn_scan_protocol):
        self.fn_scan_protocol = fn_scan_protocol
        self.load_scan_protocol_xml(fn_scan_protocol)
        self.generate_CBCurveD_geometry_angle_zero()
        # self.generate_source_trajectory(fn_scan_protocol)
        # self.generate_scan_geometry(fn_scan_protocol)
        # self.generate_phantom(fn_scan_protocol)
        # self.generate_projection_data()

    '''------------------------------------------------------------------------
    [F] - load_scan_protocol_xml from XML file
    [I] - fn_scan_protocol
    [O] - self.attributes including info of 
          General info, phantom, geometry, and imaging chain
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20250922
    ------------------------------------------------------------------------'''

    def load_scan_protocol_xml(self, fn_scan_protocol=None):
        if fn_scan_protocol is None:
            fn_scan_protocol = self.fn_scan_protocol

        root = ET.parse(fn_scan_protocol)
        XSS = root.getroot()

        # ------ general
        self.device_name = XSS.find('device_name').text
        self.xml_version = XSS.find('xml_version').text
        self.raw_data_folder = XSS.find('raw_data_folder').text

        # ------ phantom
        phantom = XSS.find('phantom')
        self.phantom_type = phantom.find('phantom_type').text
        self.phantom_name = phantom.find('phantom_name').text

        # ------ Geometry
        geometry = XSS.find('scan_geometry')
        self.coordinate_system_type = geometry.find('coordinate_system_type').text
        self.trajectory_type = geometry.find('trajectory_type').text
        self.geometry_type = geometry.find('geometry_type').text
        self.focal_spot_model = geometry.find('focal_spot_model').text
        self.detector_pixel_model = geometry.find('detector_pixel_model').text
        self.gantry_tilt_angle_deg = np.float32(geometry.find('gantry_tilt_angle_deg').text)
        self.source_starting_angle_deg = np.float32(geometry.find('source_starting_angle_deg').text)
        self.source_starting_position_mm = np.float32(geometry.find('source_starting_position_mm').text)
        self.source_rotation_direction = geometry.find('source_rotation_direction').text
        self.total_number_of_projections = np.int32(geometry.find('total_number_of_projections').text)
        self.views_per_rotation = np.int32(geometry.find('views_per_rotation').text)
        self.source_translation_per_rotation_mm = np.float32(geometry.find('source_translation_per_rotation_mm').text)
        self.source_to_isocenter_distance_mm = np.float32(geometry.find('source_to_isocenter_distance_mm').text)
        self.source_to_piercing_point_distance_mm = np.float32(
            geometry.find('source_to_piercing_point_distance_mm').text)
        self.detector_layout = geometry.find('detector_layout').text.upper()
        self.detector_channel_space_mm = np.float32(geometry.find('detector_channel_space_mm').text)
        self.detector_channel_number = np.int32(geometry.find('detector_channel_number').text)
        self.detector_channel_offset = np.float32(geometry.find('detector_channel_offset').text)
        self.detector_row_space_mm = np.float32(geometry.find('detector_row_space_mm').text)
        self.detector_row_number = np.int32(geometry.find('detector_row_number').text)
        self.detector_row_offset = np.float32(geometry.find('detector_row_offset').text)
        self.detector_fan_angle_degree = np.float32(geometry.find('detector_fan_angle_degree').text)
        self.detector_rotation_around_waxis_deg = np.float32(geometry.find('detector_rotation_around_waxis_deg').text)
        self.detector_rotation_around_vaxis_deg = np.float32(geometry.find('detector_rotation_around_vaxis_deg').text)
        self.detector_rotation_around_uaxis_deg = np.float32(geometry.find('detector_rotation_around_uaxis_deg').text)
        self.detector_rotation_deployment_order = geometry.find('detector_rotation_deployment_order').text

        # ------ Imaging Chain
        imaging_chain = XSS.find('imaging_chain')
        self.imaging_chain_type = imaging_chain.find('imaging_chain_type').text
        self.forward_projection_method = imaging_chain.find('forward_projection_method').text
        self.projection_type = imaging_chain.find('projection_type').text.upper()
        self.projection_data_type = imaging_chain.find('projection_data_type').text.upper()
        self.energy_kev = np.float32(imaging_chain.find('energy_kev').text)
        self.photon_counts_per_ray = np.float32(imaging_chain.find('photon_counts_per_ray').text)
        self.electronic_noise_std = np.float32(imaging_chain.find('electronic_noise_std').text)
        self.poisson_noise = spt.str_to_bool(imaging_chain.find('poisson_noise').text)

        # ------ Derive key parameters
        self.gantry_tilt_angle_rad = np.deg2rad(self.gantry_tilt_angle_deg)
        self.source_starting_angle_rad = np.deg2rad(self.source_starting_angle_deg)
        self.detector_width_mm = self.detector_channel_space_mm * self.detector_channel_number
        self.detector_height_mm = self.detector_row_space_mm * self.detector_row_number
        self.magnification_factor = self.source_to_piercing_point_distance_mm / self.source_to_isocenter_distance_mm
        self.nominal_helical_pitch = self.source_translation_per_rotation_mm * self.magnification_factor / self.detector_width_mm
        self.source_translation_per_view_mm = self.source_translation_per_rotation_mm / self.views_per_rotation

        # angular step size in rad
        angular_space_rad = 2 * np.pi / self.views_per_rotation
        try:
            self.source_angular_space_rad = angular_space_rad if self.source_rotation_direction.upper() == em.RotationDirection.RHR.name else -angular_space_rad
        except KeyError:
            print('Invalid rotation direction!!')

            # gantry title and rotation axis
        etax = np.deg2rad(self.gantry_tilt_angle_deg)
        self.ve_rotation_axis = np.array([0, -np.sin(etax), np.cos(etax)])
        self.ve_translation_axis = np.array([1, 0, 0])

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_geometry_angle_zero()
    [I] - self
    [N] - Angle zero: source on the x-axis of the CT scan coordinate system
        - Generate CBCurveD object at view angle zero. Because gantry tilt is
          around x-axis, it has no impact on detector basis or source vector
        - Supported configuratoins
            detector_layout: 'U1V1','U1V0','U0V1','U0V0'
            detector orientation rotation order (nominal->general): 'wvu'
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20250922
    ------------------------------------------------------------------------'''

    def generate_CBCurveD_geometry_angle_zero(self):
        # Define basis vectors for the nominal detector coordiante system
        ve_normal = np.array([1, 0, 0])  # detector normal vector (vew)
        match self.detector_layout:
            case em.DetectorLayout.U1V1.name:
                ve_channel = np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = np.array([0, 0, 1])  # basis vector for row index (vev)
            case em.DetectorLayout.U1V0.name:
                ve_channel = np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = -np.array([0, 0, 1])  # basis vector for row index (vev)
            case em.DetectorLayout.U0V1.name:
                ve_channel = -np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = np.array([0, 0, 1])  # basis vector for row index (vev)
            case em.DetectorLayout.U0V0.name:
                ve_channel = -np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = -np.array([0, 0, 1])  # basis vector for row index (vev)
            case _:
                print('Undefined detector layout!')

        nchannel = self.detector_channel_number  # number of channels
        dchannel = self.detector_channel_space_mm  # channel space (def in mm)
        ochannel = self.detector_channel_offset  # channel offset
        nrow = self.detector_row_number  # number of rows
        drow = self.detector_row_space_mm  # row space (default in mm)
        orow = self.detector_row_offset  # row offset
        dgamma = self.detector_fan_angle_degree # angle interval (def in degree)
        SID = self.source_to_isocenter_distance_mm  # source to isocenter distance
        SPD = self.source_to_piercing_point_distance_mm  # source to piercing point distance
        vsource = SID * ve_normal  # assume in mm
        hs = self.source_starting_position_mm
        dh = self.source_translation_per_view_mm
        dlam = self.source_angular_space_rad
        lams = self.source_starting_angle_rad
        h = hs - lams * dh / dlam  # source position at lambda = 0
        vsource = vsource + h * self.ve_translation_axis

        SDD = SPD
        etau = self.detector_rotation_around_uaxis_deg
        etav = self.detector_rotation_around_vaxis_deg
        etaw = self.detector_rotation_around_waxis_deg

        # Detector orientation adjustment with rotation:
        #   - The rotation around different det axes are applied one after
        #     another, rather than applied relative to the old ones
        CBGeo = geo.CBCurveD(ve_channel, ve_row, ve_normal,
                          nchannel, dchannel, ochannel,
                          nrow, drow, orow, dgamma,
                          SDD, vsource)
        if self.detector_rotation_deployment_order.lower() == 'wvu':
            CBGeo.rotate_detector_around_normal_axis_at_piercing_RH(etaw)
            CBGeo.rotate_detector_around_row_axis_at_piercing_RH(etav)
            CBGeo.rotate_detector_around_channel_axis_at_piercing_RH(etau)
        else:
            print('Undefined detector rotation deployment order!')

        self.cone_beam_geometry_angle_zero = CBGeo

        return CBGeo

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_geometry_single_view(idx_view)
    [N] - one application of this function is to generate the CB projection 
          geometry for each view angle around the rotation axis
        - Supported configuratoins
            coordinatey system type: 'CT'    
    [H] - created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def generate_CBCurveD_geometry_single_view(self, idx_view):
        if hasattr(self, 'cone_beam_geometry_angle_zero') == False:
            self.generate_CBCurveD_geometry_angle_zero()

        lams = self.source_starting_angle_rad
        dlam = self.source_angular_space_rad
        lam = lams + idx_view * dlam
        lam_deg = np.rad2deg(lam)

        hs = self.source_starting_position_mm
        dh = self.source_translation_per_view_mm
        h = hs + idx_view * dh

        CBGeo = copy.deepcopy(self.cone_beam_geometry_angle_zero)
        # detector basis vectors
        CBGeo.ve_channel = geo.rotate_around_axis_RH(lam_deg, CBGeo.ve_channel, self.ve_rotation_axis)
        CBGeo.ve_row = geo.rotate_around_axis_RH(lam_deg, CBGeo.ve_row, self.ve_rotation_axis)
        CBGeo.ve_normal = geo.rotate_around_axis_RH(lam_deg, CBGeo.ve_normal, self.ve_rotation_axis)

        # x-ray source position rotation (first) + translation (second)
        vsource = CBGeo.vsource
        vsource[2] = 0
        vsource = geo.rotate_around_axis_RH(lam_deg, vsource, self.ve_rotation_axis)
        CBGeo.vsource = vsource + h * self.ve_translation_axis

        return CBGeo

    '''------------------------------------------------------------------------
    [F] - display_source_trajectory()
    [I] - disp_angular_interval_deg: angular step size used for display
    [N] - displays scan geometry every 10 degrees
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20250922
    ------------------------------------------------------------------------'''

    def display_source_trajectory(self, disp_angular_interval_deg=None):
        if disp_angular_interval_deg == None:
            disp_angular_interval_deg = 10

        # CB projection geo at view angle 0
        CBGeo = copy.deepcopy(self.generate_CBCurveD_geometry_angle_zero())
        CBGeo.vsource[2] = 0

        # gantry rotation
        ve_rotation_axis = self.ve_rotation_axis
        lams = self.source_starting_angle_rad
        dlam = self.source_angular_space_rad
        nlam = self.total_number_of_projections
        lam = lams + np.arange(nlam) * dlam
        lam_deg = np.rad2deg(lam)

        # source translation
        ve_translation_axis = self.ve_translation_axis
        hs = self.source_starting_position_mm
        dh = self.source_translation_per_view_mm
        nh = nlam
        h = hs + np.arange(nh) * dh

        # source positions: first rotation then translation
        # the order matters when the rotation and translation axes are diff
        vsource = np.outer(np.ones(nlam), CBGeo.vsource) # [nview,3]
        vsource = geo.rotate_around_axis_RH(lam_deg, vsource, ve_rotation_axis)
        vsource = vsource + np.outer(h, ve_translation_axis)

        # plot: source trajectory, detector boundary, piercing point
        nlam_skip = np.int32(disp_angular_interval_deg / np.rad2deg(dlam))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for k in np.arange(0, nlam, nlam_skip):
            ax.clear()
            # source trajectory
            ax.plot3D(vsource[:, 0], vsource[:, 1], vsource[:, 2], label='Source Trajectory')

            # CB geometry at view k
            CBGeok = self.generate_CBCurveD_geometry_single_view(k)
            # ic = CBGeok.isocenter
            vak = CBGeok.vsource
            uoff = CBGeok.ochannel
            nu = CBGeok.nchannel
            du = CBGeok.dchannel
            voff = CBGeok.orow
            nv = CBGeok.nrow
            dv = CBGeok.drow
            goff = CBGeok.ochannel
            ng = CBGeok.nchannel
            dg = CBGeok.dgamma

            uoffp = CBGeok.ochannel_piercing
            voffp = CBGeok.orow_piercing
            veu = CBGeok.ve_channel
            vev = CBGeok.ve_row
            vew = CBGeok.ve_normal
            D = CBGeok.SDD

            umin = -uoff * du
            umax = (nu - uoff - 1) * du
            vmin = -voff * dv
            vmax = (nv - voff - 1) * dv
            gmin = -goff * dg
            gmax = (ng - goff - 1) * dg
            upiercing = umin + uoffp * du
            vpiercing = vmin + voffp * dv

            # detector origin (ortho projection of source)
            vod = vak - D * vew
            v00 = vak - D * np.cos(np.deg2rad(gmin)) * vew + D * np.sin(np.deg2rad(gmin)) * veu + vmin * vev
            v01 = vak - D * np.cos(np.deg2rad(gmin)) * vew + D * np.sin(np.deg2rad(gmin)) * veu + vmax * vev
            v10 = vak - D * np.cos(np.deg2rad(gmax)) * vew + D * np.sin(np.deg2rad(gmax)) * veu + vmin * vev
            v11 = vak - D * np.cos(np.deg2rad(gmax)) * vew + D * np.sin(np.deg2rad(gmax)) * veu + vmax * vev
            v_bottom = np.zeros([ng, 3])
            v_top = np.zeros([ng, 3])
            for i in range(ng):
                gi = gmin + dg*i
                v_bottom[i] = vak - D * np.cos(np.deg2rad(gi)) * vew + D * np.sin(np.deg2rad(gi)) * veu + vmin * vev
                v_top[i] = vak - D * np.cos(np.deg2rad(gi)) * vew + D * np.sin(np.deg2rad(gi)) * veu + vmax * vev
            # piercing point: intersection between det and line connect src and iso
            vdp = vod + upiercing * veu + vpiercing * vev

            # source point
            ax.scatter3D(vak[0], vak[1], vak[2], color='red')
            # source to piercing
            vtmp = np.vstack((vak, vdp))
            ax.plot3D(vtmp[:, 0], vtmp[:, 1], vtmp[:, 2])
            # detector
            vtmp = np.vstack((v00, v01))
            ax.plot3D(vtmp[:, 0], vtmp[:, 1], vtmp[:, 2], color='green')
            vtmp = np.vstack((v10, v11))
            ax.plot3D(vtmp[:, 0], vtmp[:, 1], vtmp[:, 2], color='green')
            ax.plot3D(v_bottom[:, 0], v_bottom[:, 1], v_bottom[:, 2], color='green')
            ax.plot3D(v_top[:, 0], v_top[:, 1], v_top[:, 2], color='green')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(-1000, 1000)
            ax.set_ylim(-1000, 1000)
            ax.set_zlim(-1000, 1000)
            ax.set_box_aspect([1, 1, 1])
            ttl = str(np.int32(lam_deg[k])) + 'deg'
            ax.set_title(ttl)
            plt.tight_layout()
            # plt.axis('image') does not work for 3D plot
            plt.legend()
            plt.pause(0.2)

        plt.show()

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_projections()
    [N] - Generate cone beam projections and store data
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20260404 
          {"header" needs to be revised}
    ------------------------------------------------------------------------'''

    def generate_CBCurveD_projections(self):
        # Generate phantom
        phantom_name = self.phantom_name
        phantom = ap.StaticLACPhantom(phantom_name)

        # Generate projection data header
        nview = self.total_number_of_projections
        nrow = self.detector_row_number
        nchannel = self.detector_channel_number
        angular_space_per_view = self.source_angular_space_rad
        drow = self.detector_row_space_mm
        dchannel = self.detector_channel_space_mm
        dgamma = self.detector_fan_angle_degree
        starting_angular_position = self.source_starting_angle_rad
        orow = self.detector_row_offset
        ochannel = self.detector_channel_offset
        projection_data_type_numpy = em.DataTypeDtype[self.projection_data_type].value
        projection_type_string = self.projection_type.upper()
        starting_axial_position = self.source_starting_position_mm
        axial_movement_per_view = self.source_translation_per_view_mm
        SDD = self.source_to_piercing_point_distance_mm
        SID = self.source_to_isocenter_distance_mm

        header = fio.create_header_dict_projection_basic(nview, nrow, nchannel,
                                                         angular_space_per_view, drow, dchannel,
                                                         starting_angular_position, orow, ochannel,
                                                         dgamma,
                                                         projection_data_type_numpy,
                                                         projection_type_string,
                                                         starting_axial_position,
                                                         axial_movement_per_view,
                                                         SDD,
                                                         SID)

        os.makedirs(self.raw_data_folder, exist_ok=True)
        fnProjection = phantom_name + '_' + str(nchannel) + 'x' + str(nrow) + 'x' + str(nview) + '.prj'
        fnProjection = os.path.join(self.raw_data_folder, fnProjection)
        fio.write_header(fnProjection, header)

        # Generate projection data
        fig = plt.figure()
        ax = fig.add_subplot()
        for k in range(nview):
            CBGeo = self.generate_CBCurveD_geometry_single_view(k)
            gk = phantom.generate_line_integral_single_projection(CBGeo)
            fio.append_slices(fnProjection, gk, projection_data_type_numpy)

            if np.mod(k, 10) == 0:
                print(f'Completed projection {k} of {nview}')
                ax.clear()  # clear previous content
                ax.set_title(f'View #: {k}')
                ax.imshow(gk, cmap='gray')
                plt.tight_layout()
                plt.axis('image')
                plt.pause(0.2)
        plt.show()

        # Save scan protocol XML file
        fn_scan_protocol_source = self.fn_scan_protocol
        fld_scan_protocol_target = self.raw_data_folder
        shutil.copy(fn_scan_protocol_source, fld_scan_protocol_target)

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_projections_beam_average()
    [I] - nups_channel, nups_row: Average of upsampled rays
          -- Both shall be odd numbers, default is 1
          -- nunew = nups_channel*nu
          -- dunew = du/nunew
          -- uoffnew = nups_channel*nu + (nu-1)/2
    [N] - Generate cone beam projections and store data
    [H] - created by ZYu prior to 20250510
          revised by JZhang prior to 20260404
          {"header" needs to be revised}
    ------------------------------------------------------------------------'''

    def generate_CBCurveD_projections_beam_average(self, nups_channel=1, nups_row=1):
        # Generate phantom
        phantom_name = self.phantom_name
        phantom = ap.StaticLACPhantom(phantom_name)

        # Generate projection data header
        nview = self.total_number_of_projections
        nrow = self.detector_row_number
        nchannel = self.detector_channel_number
        angular_space_per_view = self.source_angular_space_rad
        drow = self.detector_row_space_mm
        dchannel = self.detector_channel_space_mm
        starting_angular_position = self.source_starting_angle_rad
        dgamma = self.detector_fan_angle_degree
        orow = self.detector_row_offset
        ochannel = self.detector_channel_offset
        projection_data_type_numpy = em.DataTypeDtype[self.projection_data_type].value
        projection_type_string = self.projection_type.upper()
        starting_axial_position = self.source_starting_position_mm
        axial_movement_per_view = self.source_translation_per_view_mm
        SDD = self.source_to_piercing_point_distance_mm
        SID = self.source_to_isocenter_distance_mm

        header = fio.create_header_dict_projection_basic(nview, nrow, nchannel,
                                                         angular_space_per_view, drow, dchannel,
                                                         starting_angular_position, orow, ochannel,
                                                         dgamma,
                                                         projection_data_type_numpy,
                                                         projection_type_string,
                                                         starting_axial_position,
                                                         axial_movement_per_view,
                                                         SDD,
                                                         SID)

        os.makedirs(self.raw_data_folder, exist_ok=True)
        fnProjection = phantom_name + '_beamavg' + str(nups_channel) + 'x' + str(nups_row) + '_' + str(
            nchannel) + 'x' + str(nrow) + 'x' + str(nview) + '.prj'
        fnProjection = os.path.join(self.raw_data_folder, fnProjection)
        fio.write_header(fnProjection, header)

        # Generate projection data
        fig = plt.figure()
        ax = fig.add_subplot()
        for k in range(nview):
            CBGeo = self.generate_CBCurveD_geometry_single_view(k)
            # Upsampling
            nrow = CBGeo.nrow
            nchannel = CBGeo.nchannel
            CBGeo.nrow = nups_row * CBGeo.nrow
            CBGeo.drow = CBGeo.drow / nups_row
            CBGeo.orow = nups_row * orow + 0.5 * (nups_row - 1)
            CBGeo.nchannel = nups_channel * CBGeo.nchannel
            CBGeo.dchannel = CBGeo.dchannel / nups_channel
            CBGeo.ochannel = nups_channel * ochannel + 0.5 * (nups_channel - 1)
            CBGeo.dgamma = CBGeo.dgamma / nups_channel
            gk = phantom.generate_line_integral_single_projection(CBGeo)
            # Downsampling (averaging)
            gk = gk.reshape(nrow, nups_row, nchannel, nups_channel).mean(
                axis=(1, 3))  # has to reassign it back to gk, otherwise, gk does not change
            # Save data
            fio.append_slices(fnProjection, gk, projection_data_type_numpy)

            if np.mod(k, 10) == 0:
                print(f'Completed projection {k} of {nview}')
                ax.clear()  # clear previous content
                ax.set_title(f'View #: {k}')
                ax.imshow(gk, cmap='gray')
                plt.tight_layout()
                plt.axis('image')
                plt.pause(0.2)
        plt.show()

        # Save scan protocol XML file
        fn_scan_protocol_source = self.fn_scan_protocol
        fld_scan_protocol_target = self.raw_data_folder
        shutil.copy(fn_scan_protocol_source, fld_scan_protocol_target)

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_sinogram()
    [I] - save_path: path to save the sinogram
        - save_name: name of the sinogram
        - nups_channel, nups_row: Average of upsampled rays
          -- Both shall be odd numbers, default is 1
          -- nunew = nups_channel*nu
          -- dunew = du/nunew
          -- uoffnew = nups_channel*nu + (nu-1)/2
    [N] - Generate cone beam sinogram and store data
    [H] - created by JZhang prior to 20260404 
    ------------------------------------------------------------------------'''

    def generate_CBCurveD_sinogram(self, save_path, save_name, nups_channel=1, nups_row=1):
        # Generate phantom
        phantom_name = self.phantom_name
        phantom = ap.StaticLACPhantom(phantom_name)

        # Generate projection data
        nview = self.views_per_rotation
        orow = self.detector_row_offset
        ochannel = self.detector_channel_offset
        nchannel = self.detector_channel_number
        sinogram = np.zeros([nchannel, nview])
        for k in range(nview):
            CBGeo = self.generate_CBCurveD_geometry_single_view(k)
            # Upsampling
            nrow = CBGeo.nrow
            nchannel = CBGeo.nchannel
            CBGeo.nrow = nups_row * CBGeo.nrow
            CBGeo.drow = CBGeo.drow / nups_row
            CBGeo.orow = nups_row * orow + 0.5 * (nups_row - 1)
            CBGeo.nchannel = nups_channel * CBGeo.nchannel
            CBGeo.dchannel = CBGeo.dchannel / nups_channel
            CBGeo.ochannel = nups_channel * ochannel + 0.5 * (nups_channel - 1)
            CBGeo.dgamma = CBGeo.dgamma / nups_channel
            gk = phantom.generate_line_integral_single_projection(CBGeo)
            # Downsampling (averaging)
            gk = gk.reshape(nrow, nups_row, nchannel, nups_channel).mean(axis=(1, 3))
            sinogram[:, k] = gk[int(orow)]
            if np.mod(k, 10) == 0:
                print(f'Completed projection {k} of {nview}')

        print(f'Sinogram 2D shape: {sinogram.shape}')
        plt.imshow(sinogram, cmap='gray')
        plt.title("Sinogram")
        plt.show()

        os.makedirs(save_path, exist_ok=True)
        file_name = f"{save_name}_sinogram_{nview}view_{nchannel}gamma.npy"
        np.save(os.path.join(save_path, file_name), sinogram)


'''============================================================================
Class       
  CTCBFPDRolicalLAC()
Parent
  n.a.
Attributes
  fn_scan_protocol: XML file containing scan protocol
  device_name: name of the scanner
  xml_version: version of the XML file
  raw_data_folder: folder to store raw data
  phantom_type: type of phantom used
  phantom_name: name of the phantom
  coordinate_system_type: type of coordinate system used
  trajectory_type: type of trajectory used
  geometry_type: type of geometry used
  focal_spot_model: model of focal spot
  detector_pixel_model: model of detector pixel
  gantry_tilt_angle_deg: gantry tilt angle in degrees
  source_starting_angle_deg: starting angle of source in degrees
  source_rotation_direction: direction of source rotation (RHR or LHR)
  total_number_of_projections: total number of projections
  views_per_rotation: number of views per rotation
  source_center_rotation_angle_deg: total angle of center rotation in degrees 
  source_center_starting angle_deg: starting angle of center in degrees
  source_center_isocenter_distance_mm: distance from center to isocenter in mm
  source_to_isocenter_distance_mm: distance from source to isocenter in mm
  source_to_piercing_point_distance_mm: distance from source to piercing point in mm
  detector_layout: layout of the detector (U1V1, U1V0, U0V1, U0V0)
  detector_channel_space_mm: space between detector channels in mm
  detector_channel_number: number of detector channels
  detector_channel_offset: offset for detector channels in mm
  detector_row_space_mm: space between detector rows in mm
  detector_row_number: number of detector rows
  detector_row_offset: offset for detector rows in mm
  detector_fan_angle_degree: interval between fan angles in degrees
  detector_rotation_around_waxis_deg: rotation around w-axis in degrees
  detector_rotation_around_vaxis_deg: rotation around v-axis in degrees
  detector_rotation_around_uaxis_deg: rotation around u-axis in degrees
  detector_rotation_deployment_order: order of deployment for rotations (wvu)                  
Methods
  load_scan_protocol_xml(fn_scan_protocol): Load scan protocol from XML file
  generate_CBFPD_geometry_angle_zero(): Generate CBFPD geometry at angle zero
  generate_CBFPD_geometry_single_view(idx_view): Generate CBFPD geometry for a single view
  display_source_trajectory(disp_angular_interval_deg): Display source trajectory
  generate_CBFPD_projections(): Generate cone beam projections and store data
  generate_CBFPD_projections_beam_avarage(nups_channel=1, nups_row=1): Generate projections with beam averaging             
  generate_CBFPD_sinogram(nups_channel=1, nups_row=1): Generate sinogram and store data
Notes
    
History: created by JZhang prior to 20260404
==============================================================================='''

class CTCBFPDRolicalLAC():
    def __init__(self, fn_scan_protocol):
        self.fn_scan_protocol = fn_scan_protocol
        self.load_scan_protocol_xml(fn_scan_protocol)
        self.generate_CBFPD_geometry_angle_zero()
        # self.generate_source_trajectory(fn_scan_protocol)
        # self.generate_scan_geometry(fn_scan_protocol)
        # self.generate_phantom(fn_scan_protocol)
        # self.generate_projection_data()

    '''------------------------------------------------------------------------
    [F] - load_scan_protocol_xml from XML file
    [I] - fn_scan_protocol
    [O] - self.attributes including info of 
          General info, phantom, geometry, and imaging chain
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20260404
    ------------------------------------------------------------------------'''

    def load_scan_protocol_xml(self, fn_scan_protocol=None):
        if fn_scan_protocol is None:
            fn_scan_protocol = self.fn_scan_protocol

        root = ET.parse(fn_scan_protocol)
        XSS = root.getroot()

        # ------ general
        self.device_name = XSS.find('device_name').text
        self.xml_version = XSS.find('xml_version').text
        self.raw_data_folder = XSS.find('raw_data_folder').text

        # ------ phantom
        phantom = XSS.find('phantom')
        self.phantom_type = phantom.find('phantom_type').text
        self.phantom_name = phantom.find('phantom_name').text

        # ------ Geometry
        geometry = XSS.find('scan_geometry')
        self.coordinate_system_type = geometry.find('coordinate_system_type').text
        self.trajectory_type = geometry.find('trajectory_type').text
        self.geometry_type = geometry.find('geometry_type').text
        self.focal_spot_model = geometry.find('focal_spot_model').text
        self.detector_pixel_model = geometry.find('detector_pixel_model').text
        self.gantry_tilt_angle_deg = np.float32(geometry.find('gantry_tilt_angle_deg').text)
        self.source_starting_angle_deg = np.float32(geometry.find('source_starting_angle_deg').text)
        self.source_rotation_direction = geometry.find('source_rotation_direction').text
        self.total_number_of_projections = np.int32(geometry.find('total_number_of_projections').text)
        self.views_per_rotation = np.int32(geometry.find('views_per_rotation').text)
        self.source_center_rotation_angle_deg = np.int32(geometry.find('source_center_rotation_angle_deg').text)
        self.source_center_starting_angle_deg = np.int32(geometry.find('source_center_starting_angle_deg').text)
        self.source_center_isocenter_distance_mm = np.float32(geometry.find('source_center_isocenter_distance_mm').text)
        self.source_to_isocenter_distance_mm = np.float32(geometry.find('source_to_isocenter_distance_mm').text)
        self.source_to_piercing_point_distance_mm = np.float32(
            geometry.find('source_to_piercing_point_distance_mm').text)
        self.detector_layout = geometry.find('detector_layout').text.upper()
        self.detector_channel_space_mm = np.float32(geometry.find('detector_channel_space_mm').text)
        self.detector_channel_number = np.int32(geometry.find('detector_channel_number').text)
        self.detector_channel_offset = np.float32(geometry.find('detector_channel_offset').text)
        self.detector_row_space_mm = np.float32(geometry.find('detector_row_space_mm').text)
        self.detector_row_number = np.int32(geometry.find('detector_row_number').text)
        self.detector_row_offset = np.float32(geometry.find('detector_row_offset').text)
        self.detector_rotation_around_waxis_deg = np.float32(geometry.find('detector_rotation_around_waxis_deg').text)
        self.detector_rotation_around_vaxis_deg = np.float32(geometry.find('detector_rotation_around_vaxis_deg').text)
        self.detector_rotation_around_uaxis_deg = np.float32(geometry.find('detector_rotation_around_uaxis_deg').text)
        self.detector_rotation_deployment_order = geometry.find('detector_rotation_deployment_order').text

        # ------ Imaging Chain
        imaging_chain = XSS.find('imaging_chain')
        self.imaging_chain_type = imaging_chain.find('imaging_chain_type').text
        self.forward_projection_method = imaging_chain.find('forward_projection_method').text
        self.projection_type = imaging_chain.find('projection_type').text.upper()
        self.projection_data_type = imaging_chain.find('projection_data_type').text.upper()
        self.energy_kev = np.float32(imaging_chain.find('energy_kev').text)
        self.photon_counts_per_ray = np.float32(imaging_chain.find('photon_counts_per_ray').text)
        self.electronic_noise_std = np.float32(imaging_chain.find('electronic_noise_std').text)
        self.poisson_noise = spt.str_to_bool(imaging_chain.find('poisson_noise').text)

        # ------ Derive key parameters
        self.gantry_tilt_angle_rad = np.deg2rad(self.gantry_tilt_angle_deg)
        self.source_starting_angle_rad = np.deg2rad(self.source_starting_angle_deg)
        self.source_rotation_number = np.int32(self.total_number_of_projections / self.views_per_rotation)
        self.detector_width_mm = self.detector_channel_space_mm * self.detector_channel_number
        self.detector_height_mm = self.detector_row_space_mm * self.detector_row_number
        self.magnification_factor = self.source_to_piercing_point_distance_mm / self.source_to_isocenter_distance_mm
        # pitch define
        # self.nominal_helical_pitch = self.source_translation_per_rotation_mm * self.magnification_factor / self.detector_width_mm


        # angular step size in rad
        # angular_space_rad = 2 * np.pi / self.views_per_rotation
        angular_space_rad = 2 * np.pi / self.views_per_rotation
        try:
            self.source_angular_space_rad = angular_space_rad if self.source_rotation_direction.upper() == em.RotationDirection.RHR.name else -angular_space_rad
        except KeyError:
            print('Invalid rotation direction!!')


        # gantry title and rotation axis
        etax = np.deg2rad(self.gantry_tilt_angle_deg)
        self.ve_rotation_axis = np.array([0, -np.sin(etax), np.cos(etax)])

        ea = np.array([1, 0, 0])
        eb = np.array([0, 1, 0])
        ec = np.array([0, 0, 1])
        self.ve_rotate_plane = np.vstack((ea, eb, ec))

        self.ve_rotation_axis = eb

    '''------------------------------------------------------------------------
    [F] - generate_CBFPD_geometry_angle_zero()
    [I] - self
    [N] - Angle zero: source on the x-axis of the CT scan coordinate system
        - Generate CBFPD object at view angle zero. Because gantry tilt is
          around x-axis, it has no impact on detector basis or source vector
        - Supported configuratoins
            detector_layout: 'U1V1','U1V0','U0V1','U0V0'
            detector orientation rotation order (nominal->general): 'wvu'
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20260404
          {updated axes in the rotation way}
    ------------------------------------------------------------------------'''

    def generate_CBFPD_geometry_angle_zero(self):
        # Define basis vectors for the nominal detector coordiante system
        ve_normal = np.array([1, 0, 0])  # detector normal vector (vew)
        match self.detector_layout:
            case em.DetectorLayout.U1V1.name:
                ve_channel = np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = np.array([0, 0, 1])  # basis vector for row index (vev)
            case em.DetectorLayout.U1V0.name:
                ve_channel = np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = -np.array([0, 0, 1])  # basis vector for row index (vev)
            case em.DetectorLayout.U0V1.name:
                ve_channel = -np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = np.array([0, 0, 1])  # basis vector for row index (vev)
            case em.DetectorLayout.U0V0.name:
                ve_channel = -np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = -np.array([0, 0, 1])  # basis vector for row index (vev)
            case _:
                print('Undefined detector layout!')

        nchannel = self.detector_channel_number  # number of channels
        dchannel = self.detector_channel_space_mm  # channel space (def in mm)
        ochannel = self.detector_channel_offset  # channel offset
        nrow = self.detector_row_number  # number of rows
        drow = self.detector_row_space_mm  # row space (default in mm)
        orow = self.detector_row_offset  # row offset
        SID = self.source_to_isocenter_distance_mm  # source to isocenter distance
        SPD = self.source_to_piercing_point_distance_mm  # source to piercing point distance
        CID = self.source_center_isocenter_distance_mm
        # vsource = SID * ve_normal  # assume in mm
        thetas = np.deg2rad(self.source_center_starting_angle_deg)
        dtheta = np.deg2rad(self.source_center_rotation_angle_deg) / self.total_number_of_projections
        dlam = self.source_angular_space_rad
        lams = self.source_starting_angle_rad
        source_rotate_matrix = np.array([(CID + SID * np.cos(lams)) * np.cos(thetas),
                                         (CID + SID * np.cos(lams)) * np.sin(thetas),
                                         SID * np.sin(lams)])
        vsource = self.ve_rotate_plane.T @ source_rotate_matrix
        channel_rotate_matrix = np.array([- np.sin(lams) * np.cos(thetas),
                                          - np.sin(lams) * np.sin(thetas),
                                          np.cos(lams)])
        ve_channel = self.ve_rotate_plane.T @ channel_rotate_matrix
        row_rotate_matrix = np.array([np.sin(thetas),
                                      - np.cos(thetas),
                                      np.zeros_like(thetas)])
        ve_row = self.ve_rotate_plane.T @ row_rotate_matrix
        normal_rotate_matrix = np.array([np.cos(lams) * np.cos(thetas),
                                         np.cos(lams) * np.sin(thetas),
                                         np.sin(lams)])
        ve_normal = self.ve_rotate_plane.T @ normal_rotate_matrix

        SDD = SPD
        etau = self.detector_rotation_around_uaxis_deg
        etav = self.detector_rotation_around_vaxis_deg
        etaw = self.detector_rotation_around_waxis_deg

        # Detector orientation adjustment with rotation:
        #   - The rotation around different det axes are applied one after
        #     another, rather than applied relative to the old ones
        CBGeo = geo.CBFPD(ve_channel, ve_row, ve_normal,
                          nchannel, dchannel, ochannel,
                          nrow, drow, orow,
                          SDD, vsource)
        if self.detector_rotation_deployment_order.lower() == 'wvu':
            CBGeo.rotate_detector_around_normal_axis_at_piercing_RH(etaw)
            CBGeo.rotate_detector_around_row_axis_at_piercing_RH(etav)
            CBGeo.rotate_detector_around_channel_axis_at_piercing_RH(etau)
        else:
            print('Undefined detector rotation deployment order!')

        self.cone_beam_geometry_angle_zero = CBGeo

        return CBGeo

    '''------------------------------------------------------------------------
    [F] - generate_single_view_CBFPD_geometry(idx_view)
    [N] - one application of this function is to generate the CB projection 
          geometry for each view angle around the rotation axis
        - Supported configuratoins
            coordinatey system type: 'CT'    
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20260404
          {update axes in the rotation way}
    ------------------------------------------------------------------------'''

    def generate_CBFPD_geometry_single_view(self, idx_view):
        if hasattr(self, 'cone_beam_geometry_angle_zero') == False:
            self.generate_CBFPD_geometry_angle_zero()

        lams = self.source_starting_angle_rad
        dlam = self.source_angular_space_rad
        lam = lams + (idx_view % self.views_per_rotation) * dlam

        thetas = np.deg2rad(self.source_center_starting_angle_deg)
        dtheta = np.deg2rad(self.source_center_rotation_angle_deg) / self.total_number_of_projections
        theta = thetas + idx_view * dtheta
        CID = self.source_center_isocenter_distance_mm
        SID = self.source_to_isocenter_distance_mm
        self.ve_rotate_axis = self.ve_rotate_plane.T @ np.array([-np.sin(theta), np.cos(theta), 0])

        CBGeo = copy.deepcopy(self.cone_beam_geometry_angle_zero)

        # x-ray source position rotation (first) + translation (second)
        vsource = CBGeo.vsource
        vsource[2] = 0
        source_rotate_matrix = np.array([(CID + SID * np.cos(lam)) * np.cos(theta),
                                         (CID + SID * np.cos(lam)) * np.sin(theta),
                                         SID * np.sin(lam)])
        CBGeo.vsource = self.ve_rotate_plane.T @ source_rotate_matrix
        channel_rotate_matrix = np.array([- np.sin(lam) * np.cos(theta),
                                          - np.sin(lam) * np.sin(theta),
                                          np.cos(lam)])
        CBGeo.ve_channel = self.ve_rotate_plane.T @ channel_rotate_matrix
        row_rotate_matrix = np.array([np.sin(theta),
                                      - np.cos(theta),
                                      np.zeros_like(theta)])
        CBGeo.ve_row = self.ve_rotate_plane.T @ row_rotate_matrix
        normal_rotate_matrix = np.array([np.cos(lam) * np.cos(theta),
                                         np.cos(lam) * np.sin(theta),
                                         np.sin(lam)])
        CBGeo.ve_normal = self.ve_rotate_plane.T @ normal_rotate_matrix

        return CBGeo

    '''------------------------------------------------------------------------
    [F] - display_source_trajectory()
    [I] - disp_angular_interval_deg: angular step size used for display
    [N] - displays scan geometry every 10 degrees
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20260404
    ------------------------------------------------------------------------'''

    def display_source_trajectory(self, disp_angular_interval_deg=None):
        if disp_angular_interval_deg == None:
            disp_angular_interval_deg = 10

        # CB projection geo at view angle 0
        CBGeo = copy.deepcopy(self.generate_CBFPD_geometry_angle_zero())
        CBGeo.vsource[2] = 0

        # gantry rotation
        ve_rotation_axis = self.ve_rotation_axis
        lams = self.source_starting_angle_rad
        dlam = self.source_angular_space_rad
        nlam = self.views_per_rotation
        lam = lams + np.tile(np.arange(nlam), self.source_rotation_number) * dlam
        lam_deg = np.rad2deg(lam)

        # source rotation
        thetas = np.deg2rad(self.source_center_starting_angle_deg)
        dtheta = np.deg2rad(self.source_center_rotation_angle_deg) / self.total_number_of_projections
        ntheta = self.total_number_of_projections
        theta = thetas + np.arange(ntheta) * dtheta
        CID = self.source_center_isocenter_distance_mm
        SID = self.source_to_isocenter_distance_mm
        rotate_matrix = np.array([(CID + SID * np.cos(lam)) * np.cos(theta),
                                  (CID + SID * np.cos(lam)) * np.sin(theta),
                                  SID * np.sin(lam)])

        vsource = (self.ve_rotate_plane.T @ rotate_matrix).T

        # test trajectory of detector center
        vdet = np.zeros((ntheta, 3), dtype=np.float64)
        for k in range(ntheta):
            CBGeok = self.generate_CBFPD_geometry_single_view(k)
            vak = CBGeok.vsource
            veu = CBGeok.ve_channel
            vev = CBGeok.ve_row
            vew = CBGeok.ve_normal
            D = CBGeok.SDD

            uoff = CBGeok.ochannel
            nu = CBGeok.nchannel
            du = CBGeok.dchannel
            voff = CBGeok.orow
            nv = CBGeok.nrow
            dv = CBGeok.drow
            umin = -uoff * du
            umax = (nu - uoff - 1) * du
            vmin = -voff * dv
            vmax = (nv - voff - 1) * dv

            vod = vak - D * vew
            v00 = vod + umin * veu + vmin * vev

            vdet[k, :] = vod
            # vdet[k, :] = v00

        # plot: source trajectory, detector boundary, piercing point
        if dtheta == 0:
            ntheta_skip = np.int32(disp_angular_interval_deg / np.rad2deg(dlam))
        else:
            ntheta_skip = np.int32(disp_angular_interval_deg / np.rad2deg(dtheta))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for k in np.arange(0, ntheta, ntheta_skip):
            ax.clear()
            # source trajectory
            ax.plot3D(vsource[:, 0], vsource[:, 1], vsource[:, 2], label='Source Trajectory')
            # detector trajectory
            ax.plot3D(vdet[:, 0], vdet[:, 1], vdet[:, 2], label='Detector Trajectory')

            # CB geometry at view k
            CBGeok = self.generate_CBFPD_geometry_single_view(k)
            vak = CBGeok.vsource
            uoff = CBGeok.ochannel
            nu = CBGeok.nchannel
            du = CBGeok.dchannel
            voff = CBGeok.orow
            nv = CBGeok.nrow
            dv = CBGeok.drow
            uoffp = CBGeok.ochannel_piercing
            voffp = CBGeok.orow_piercing
            veu = CBGeok.ve_channel
            vev = CBGeok.ve_row
            vew = CBGeok.ve_normal
            D = CBGeok.SDD

            umin = -uoff * du
            umax = (nu - uoff - 1) * du
            vmin = -voff * dv
            vmax = (nv - voff - 1) * dv
            upiercing = umin + uoffp * du
            vpiercing = vmin + voffp * dv

            # detector origin (ortho projection of source)
            vod = vak - D * vew
            v00 = vod + umin * veu + vmin * vev
            v01 = vod + umin * veu + vmax * vev
            v10 = vod + umax * veu + vmin * vev
            v11 = vod + umax * veu + vmax * vev
            # piercing point: intersection between det and line connect src and iso
            vdp = vod + upiercing * veu + vpiercing * vev

            # source point
            ax.scatter3D(vak[0], vak[1], vak[2], color='red')
            # source to piercing
            vtmp = np.vstack((vak, vdp))
            ax.plot3D(vtmp[:, 0], vtmp[:, 1], vtmp[:, 2])
            # detector
            vtmp = np.vstack((v00, v01, v11, v10, v00))
            ax.plot3D(vtmp[:, 0], vtmp[:, 1], vtmp[:, 2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(-1000, 1000)
            ax.set_ylim(-1000, 1000)
            ax.set_zlim(-1000, 1000)
            ax.set_box_aspect([1, 1, 1])
            # ax.set_aspect('auto')
            ttl = str(np.int32(lam_deg[k])) + 'deg'
            ax.set_title(ttl)
            plt.tight_layout()
            # plt.axis('image') does not work for 3D plot
            plt.legend()
            plt.pause(0.2)

        plt.show()

    '''------------------------------------------------------------------------
    [F] - generate_CBFPD_projections()
    [N] - Generate cone beam projections and store data
    [H] - created by ZYu prior to 20250614
    ------------------------------------------------------------------------'''

    def generate_CBFPD_projections(self):
        # Generate phantom
        phantom_name = self.phantom_name
        phantom = ap.StaticLACPhantom(phantom_name)

        # Generate projection data header
        nview = self.total_number_of_projections
        nrow = self.detector_row_number
        nchannel = self.detector_channel_number
        angular_space_per_view = self.source_angular_space_rad
        drow = self.detector_row_space_mm
        dchannel = self.detector_channel_space_mm
        starting_angular_position = self.source_starting_angle_rad
        orow = self.detector_row_offset
        ochannel = self.detector_channel_offset
        projection_data_type_numpy = em.DataTypeDtype[self.projection_data_type].value
        projection_type_string = self.projection_type.upper()
        source_rotation_number = self.source_rotation_number
        SDD = self.source_to_piercing_point_distance_mm
        SID = self.source_to_isocenter_distance_mm

        header = fio.create_header_dict_projection_basic(nview, nrow, nchannel,
                                                         angular_space_per_view, drow, dchannel,
                                                         starting_angular_position, orow, ochannel,
                                                         projection_data_type_numpy,
                                                         projection_type_string,
                                                         source_rotation_number,
                                                         SDD,
                                                         SID)

        os.makedirs(self.raw_data_folder, exist_ok=True)
        fnProjection = phantom_name + '_' + str(nchannel) + 'x' + str(nrow) + 'x' + str(nview) + '.prj'
        fnProjection = os.path.join(self.raw_data_folder, fnProjection)
        fio.write_header(fnProjection, header)

        # Generate projection data
        fig = plt.figure()
        ax = fig.add_subplot()
        for k in range(nview):
            CBGeo = self.generate_CBFPD_geometry_single_view(k)
            gk = phantom.generate_line_integral_single_projection(CBGeo)
            fio.append_slices(fnProjection, gk, projection_data_type_numpy)

            if np.mod(k, 10) == 0:
                print(f'Completed projection {k} of {nview}')
                ax.clear()  # clear previous content
                ax.set_title(f'View #: {k}')
                ax.imshow(gk, cmap='gray')
                plt.tight_layout()
                plt.axis('image')
                plt.pause(0.2)
        plt.show()

        # Save scan protocol XML file
        fn_scan_protocol_source = self.fn_scan_protocol
        fld_scan_protocol_target = self.raw_data_folder
        shutil.copy(fn_scan_protocol_source, fld_scan_protocol_target)

    '''------------------------------------------------------------------------
    [F] - generate_CBFPD_projections_beam_average()
    [I] - nups_channel, nups_row: Average of upsampled rays
          -- Both shall be odd numbers, default is 1
          -- nunew = nups_channel*nu
          -- dunew = du/nunew
          -- uoffnew = nups_channel*nu + (nu-1)/2
    [N] - Generate cone beam projections and store data
    [H] - 20250510, created by ZYu
    ------------------------------------------------------------------------'''

    def generate_CBFPD_projections_beam_average(self, nups_channel=1, nups_row=1):
        # Generate phantom
        phantom_name = self.phantom_name
        phantom = ap.StaticLACPhantom(phantom_name)

        # Generate projection data header
        nview = self.total_number_of_projections
        nrow = self.detector_row_number
        nchannel = self.detector_channel_number
        angular_space_per_view = self.source_angular_space_rad
        drow = self.detector_row_space_mm
        dchannel = self.detector_channel_space_mm
        starting_angular_position = self.source_starting_angle_rad
        orow = self.detector_row_offset
        ochannel = self.detector_channel_offset
        projection_data_type_numpy = em.DataTypeDtype[self.projection_data_type].value
        projection_type_string = self.projection_type.upper()
        source_rotation_number = self.source_rotation_number
        SDD = self.source_to_piercing_point_distance_mm
        SID = self.source_to_isocenter_distance_mm

        header = fio.create_header_dict_projection_basic(nview, nrow, nchannel,
                                                         angular_space_per_view, drow, dchannel,
                                                         starting_angular_position, orow, ochannel,
                                                         projection_data_type_numpy,
                                                         projection_type_string,
                                                         source_rotation_number,
                                                         SDD,
                                                         SID)

        os.makedirs(self.raw_data_folder, exist_ok=True)
        fnProjection = phantom_name + '_beamavg' + str(nups_channel) + 'x' + str(nups_row) + '_' + str(
            nchannel) + 'x' + str(nrow) + 'x' + str(nview) + '.prj'
        fnProjection = os.path.join(self.raw_data_folder, fnProjection)
        fio.write_header(fnProjection, header)

        # Generate projection data
        fig = plt.figure()
        ax = fig.add_subplot()
        for k in range(nview):
            CBGeo = self.generate_CBFPD_geometry_single_view(k)
            # Upsampling
            nrow = CBGeo.nrow
            nchannel = CBGeo.nchannel
            CBGeo.nrow = nups_row * CBGeo.nrow
            CBGeo.drow = CBGeo.drow / nups_row
            CBGeo.orow = nups_row * orow + 0.5 * (nups_row - 1)
            CBGeo.nchannel = nups_channel * CBGeo.nchannel
            CBGeo.dchannel = CBGeo.dchannel / nups_channel
            CBGeo.ochannel = nups_channel * ochannel + 0.5 * (nups_channel - 1)
            gk = phantom.generate_line_integral_single_projection(CBGeo)
            # Downsampling (averaging)
            gk = gk.reshape(nrow, nups_row, nchannel, nups_channel).mean(
                axis=(1, 3))  # has to reassign it back to gk, otherwise, gk does not change
            # Save data
            fio.append_slices(fnProjection, gk, projection_data_type_numpy)

            if np.mod(k, 10) == 0:
                print(f'Completed projection {k} of {nview}')
                ax.clear()  # clear previous content
                ax.set_title(f'View #: {k}')
                ax.imshow(gk, cmap='gray')
                plt.tight_layout()
                plt.axis('image')
                plt.pause(0.2)
        plt.show()

        # Save scan protocol XML file
        fn_scan_protocol_source = self.fn_scan_protocol
        fld_scan_protocol_target = self.raw_data_folder
        shutil.copy(fn_scan_protocol_source, fld_scan_protocol_target)

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_sinogram()
    [I] - save_path: path to save the sinogram
        - save_name: name of the sinogram
        - nups_channel, nups_row: Average of upsampled rays
          -- Both shall be odd numbers, default is 1
          -- nunew = nups_channel*nu
          -- dunew = du/nunew
          -- uoffnew = nups_channel*nu + (nu-1)/2
    [N] - Generate cone beam sinogram and store data
    [H] - created by JZhang prior to 20260404 
    ------------------------------------------------------------------------'''

    def generate_CBFPD_sinogram(self, save_path, save_name, nups_channel, nups_row):
        # Generate phantom
        phantom_name = self.phantom_name
        phantom = ap.StaticLACPhantom(phantom_name)

        # Generate projection data
        nview = self.views_per_rotation
        orow = self.detector_row_offset
        nchannel = self.detector_channel_number
        ochannel = self.detector_channel_offset
        sinogram = np.zeros([nchannel, nview])
        for k in range(nview):
            CBGeo = self.generate_CBFPD_geometry_single_view(k)
            # Upsampling
            nrow = CBGeo.nrow
            nchannel = CBGeo.nchannel
            CBGeo.nrow = nups_row * CBGeo.nrow
            CBGeo.drow = CBGeo.drow / nups_row
            CBGeo.orow = nups_row * orow + 0.5 * (nups_row - 1)
            CBGeo.nchannel = nups_channel * CBGeo.nchannel
            CBGeo.dchannel = CBGeo.dchannel / nups_channel
            CBGeo.ochannel = nups_channel * ochannel + 0.5 * (nups_channel - 1)
            gk = phantom.generate_line_integral_single_projection(CBGeo)
            # Downsampling (averaging)
            gk = gk.reshape(nrow, nups_row, nchannel, nups_channel).mean(axis=(1, 3))
            sinogram[:, k] = gk[int(orow)]
            if np.mod(k, 10) == 0:
                print(f'Completed projection {k} of {nview}')

        print(f'Sinogram 2D shape: {sinogram.shape}')
        plt.imshow(sinogram, cmap='gray')
        plt.title("Sinogram")
        plt.show()

        os.makedirs(save_path, exist_ok=True)
        file_name = f"{save_name}_sinogram_{nview}view_{nchannel}gamma.npy"
        np.save(os.path.join(save_path, file_name), sinogram)


'''============================================================================
Class       
  CTCBFPDRolicalLAC()
Parent
  n.a.
Attributes
  fn_scan_protocol: XML file containing scan protocol
  device_name: name of the scanner
  xml_version: version of the XML file
  raw_data_folder: folder to store raw data
  phantom_type: type of phantom used
  phantom_name: name of the phantom
  coordinate_system_type: type of coordinate system used
  trajectory_type: type of trajectory used
  geometry_type: type of geometry used
  focal_spot_model: model of focal spot
  detector_pixel_model: model of detector pixel
  gantry_tilt_angle_deg: gantry tilt angle in degrees
  source_starting_angle_deg: starting angle of source in degrees
  source_rotation_direction: direction of source rotation (RHR or LHR)
  total_number_of_projections: total number of projections
  views_per_rotation: number of views per rotation
  source_center_rotation_angle_deg: total angle of center rotation in degrees 
  source_center_starting angle_deg: starting angle of center in degrees
  source_center_isocenter_distance_mm: distance from center to isocenter in mm
  source_to_isocenter_distance_mm: distance from source to isocenter in mm
  source_to_piercing_point_distance_mm: distance from source to piercing point in mm
  detector_layout: layout of the detector (U1V1, U1V0, U0V1, U0V0)
  detector_channel_space_mm: space between detector channels in mm
  detector_channel_number: number of detector channels
  detector_channel_offset: offset for detector channels in mm
  detector_row_space_mm: space between detector rows in mm
  detector_row_number: number of detector rows
  detector_row_offset: offset for detector rows in mm
  detector_fan_angle_degree: interval between fan angles in degrees
  detector_rotation_around_waxis_deg: rotation around w-axis in degrees
  detector_rotation_around_vaxis_deg: rotation around v-axis in degrees
  detector_rotation_around_uaxis_deg: rotation around u-axis in degrees
  detector_rotation_deployment_order: order of deployment for rotations (wvu)                
Methods
  load_scan_protocol_xml(fn_scan_protocol): Load scan protocol from XML file
  generate_CBCurveD_geometry_angle_zero(): Generate CBCurveD geometry at angle zero
  generate_CBCurveD_geometry_single_view(idx_view): Generate CBCurveD geometry for a single view
  display_source_trajectory(disp_angular_interval_deg): Display source trajectory
  generate_CBCurveD_projections(): Generate cone beam projections and store data
  generate_CBCurveD_projections_beam_avarage(nups_channel=1, nups_row=1): Generate projections with beam averaging             
  generate_CBCurveD_sinogram(nups_channel=1, nups_row=1): Generate sinogram and store data
Notes

History: created by JZhang prior to 20260404
==============================================================================='''
class CTCBCurveDRolicalLAC():
    def __init__(self, fn_scan_protocol):
        self.fn_scan_protocol = fn_scan_protocol
        self.load_scan_protocol_xml(fn_scan_protocol)
        self.generate_CBCurveD_geometry_angle_zero()
        # self.generate_source_trajectory(fn_scan_protocol)
        # self.generate_scan_geometry(fn_scan_protocol)
        # self.generate_phantom(fn_scan_protocol)
        # self.generate_projection_data()

    '''------------------------------------------------------------------------
    [F] - load_scan_protocol_xml from XML file
    [I] - fn_scan_protocol
    [O] - self.attributes including info of 
          General info, phantom, geometry, and imaging chain
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20260404
          {update axes in the rotation way}
    ------------------------------------------------------------------------'''

    def load_scan_protocol_xml(self, fn_scan_protocol=None):
        if fn_scan_protocol is None:
            fn_scan_protocol = self.fn_scan_protocol

        root = ET.parse(fn_scan_protocol)
        XSS = root.getroot()

        # ------ general
        self.device_name = XSS.find('device_name').text
        self.xml_version = XSS.find('xml_version').text
        self.raw_data_folder = XSS.find('raw_data_folder').text

        # ------ phantom
        phantom = XSS.find('phantom')
        self.phantom_type = phantom.find('phantom_type').text
        self.phantom_name = phantom.find('phantom_name').text

        # ------ Geometry
        geometry = XSS.find('scan_geometry')
        self.coordinate_system_type = geometry.find('coordinate_system_type').text
        self.trajectory_type = geometry.find('trajectory_type').text
        self.geometry_type = geometry.find('geometry_type').text
        self.focal_spot_model = geometry.find('focal_spot_model').text
        self.detector_pixel_model = geometry.find('detector_pixel_model').text
        self.gantry_tilt_angle_deg = np.float32(geometry.find('gantry_tilt_angle_deg').text)
        self.source_starting_angle_deg = np.float32(geometry.find('source_starting_angle_deg').text)
        self.source_rotation_direction = geometry.find('source_rotation_direction').text
        self.total_number_of_projections = np.int32(geometry.find('total_number_of_projections').text)
        self.views_per_rotation = np.int32(geometry.find('views_per_rotation').text)
        self.source_center_rotation_angle_deg = np.int32(geometry.find('source_center_rotation_angle_deg').text)
        self.source_center_starting_angle_deg = np.int32(geometry.find('source_center_starting_angle_deg').text)
        self.source_center_isocenter_distance_mm = np.float32(geometry.find('source_center_isocenter_distance_mm').text)
        self.source_to_isocenter_distance_mm = np.float32(geometry.find('source_to_isocenter_distance_mm').text)
        self.source_to_piercing_point_distance_mm = np.float32(
            geometry.find('source_to_piercing_point_distance_mm').text)
        self.detector_layout = geometry.find('detector_layout').text.upper()
        self.detector_channel_space_mm = np.float32(geometry.find('detector_channel_space_mm').text)
        self.detector_channel_number = np.int32(geometry.find('detector_channel_number').text)
        self.detector_channel_offset = np.float32(geometry.find('detector_channel_offset').text)
        self.detector_row_space_mm = np.float32(geometry.find('detector_row_space_mm').text)
        self.detector_row_number = np.int32(geometry.find('detector_row_number').text)
        self.detector_row_offset = np.float32(geometry.find('detector_row_offset').text)
        self.detector_fan_angle_degree = np.float32(geometry.find('detector_fan_angle_degree').text)
        self.detector_rotation_around_waxis_deg = np.float32(geometry.find('detector_rotation_around_waxis_deg').text)
        self.detector_rotation_around_vaxis_deg = np.float32(geometry.find('detector_rotation_around_vaxis_deg').text)
        self.detector_rotation_around_uaxis_deg = np.float32(geometry.find('detector_rotation_around_uaxis_deg').text)
        self.detector_rotation_deployment_order = geometry.find('detector_rotation_deployment_order').text

        # ------ Imaging Chain
        imaging_chain = XSS.find('imaging_chain')
        self.imaging_chain_type = imaging_chain.find('imaging_chain_type').text
        self.forward_projection_method = imaging_chain.find('forward_projection_method').text
        self.projection_type = imaging_chain.find('projection_type').text.upper()
        self.projection_data_type = imaging_chain.find('projection_data_type').text.upper()
        self.energy_kev = np.float32(imaging_chain.find('energy_kev').text)
        self.photon_counts_per_ray = np.float32(imaging_chain.find('photon_counts_per_ray').text)
        self.electronic_noise_std = np.float32(imaging_chain.find('electronic_noise_std').text)
        self.poisson_noise = spt.str_to_bool(imaging_chain.find('poisson_noise').text)

        # ------ Derive key parameters
        self.gantry_tilt_angle_rad = np.deg2rad(self.gantry_tilt_angle_deg)
        self.source_starting_angle_rad = np.deg2rad(self.source_starting_angle_deg)
        self.source_rotation_number = np.int32(self.total_number_of_projections / self.views_per_rotation)
        self.detector_width_mm = self.detector_channel_space_mm * self.detector_channel_number
        self.detector_height_mm = self.detector_row_space_mm * self.detector_row_number
        self.magnification_factor = self.source_to_piercing_point_distance_mm / self.source_to_isocenter_distance_mm
        # pitch define
        # self.nominal_helical_pitch = self.source_translation_per_rotation_mm * self.magnification_factor / self.detector_width_mm


        # angular step size in rad
        # angular_space_rad = 2 * np.pi / self.views_per_rotation
        angular_space_rad = 2 * np.pi / self.views_per_rotation
        try:
            self.source_angular_space_rad = angular_space_rad if self.source_rotation_direction.upper() == em.RotationDirection.RHR.name else -angular_space_rad
        except KeyError:
            print('Invalid rotation direction!!')


        # gantry title and rotation axis
        etax = np.deg2rad(self.gantry_tilt_angle_deg)
        self.ve_rotation_axis = np.array([0, -np.sin(etax), np.cos(etax)])

        ea = np.array([1, 0, 0])
        eb = np.array([0, 1, 0])
        ec = np.array([0, 0, 1])
        self.ve_rotate_plane = np.vstack((ea, eb, ec))

        self.ve_rotation_axis = eb

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_geometry_angle_zero()
    [I] - self
    [N] - Angle zero: source on the x-axis of the CT scan coordinate system
        - Generate CBCurveD object at view angle zero. Because gantry tilt is
          around x-axis, it has no impact on detector basis or source vector
        - Supported configuratoins
            detector_layout: 'U1V1','U1V0','U0V1','U0V0'
            detector orientation rotation order (nominal->general): 'wvu'
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20260404
          {update axes in the rotation way}
    ------------------------------------------------------------------------'''

    def generate_CBCurveD_geometry_angle_zero(self):
        # Define basis vectors for the nominal detector coordiante system
        ve_normal = np.array([1, 0, 0])  # detector normal vector (vew)
        match self.detector_layout:
            case em.DetectorLayout.U1V1.name:
                ve_channel = np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = np.array([0, 0, 1])  # basis vector for row index (vev)
            case em.DetectorLayout.U1V0.name:
                ve_channel = np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = -np.array([0, 0, 1])  # basis vector for row index (vev)
            case em.DetectorLayout.U0V1.name:
                ve_channel = -np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = np.array([0, 0, 1])  # basis vector for row index (vev)
            case em.DetectorLayout.U0V0.name:
                ve_channel = -np.array([0, 1, 0])  # basis vector for channel index (veu)
                ve_row = -np.array([0, 0, 1])  # basis vector for row index (vev)
            case _:
                print('Undefined detector layout!')

        nchannel = self.detector_channel_number  # number of channels
        dchannel = self.detector_channel_space_mm  # channel space (def in mm)
        ochannel = self.detector_channel_offset  # channel offset
        nrow = self.detector_row_number  # number of rows
        drow = self.detector_row_space_mm  # row space (default in mm)
        orow = self.detector_row_offset  # row offset
        dgamma = self.detector_fan_angle_degree # angle interval (def in degree)
        SID = self.source_to_isocenter_distance_mm  # source to isocenter distance
        SPD = self.source_to_piercing_point_distance_mm  # source to piercing point distance
        CID = self.source_center_isocenter_distance_mm
        # vsource = SID * ve_normal  # assume in mm
        thetas = np.deg2rad(self.source_center_starting_angle_deg)
        dtheta = np.deg2rad(self.source_center_rotation_angle_deg) / self.total_number_of_projections
        dlam = self.source_angular_space_rad
        lams = self.source_starting_angle_rad
        source_rotate_matrix = np.array([(CID + SID * np.cos(lams)) * np.cos(thetas),
                                         (CID + SID * np.cos(lams)) * np.sin(thetas),
                                         SID * np.sin(lams)])
        vsource = self.ve_rotate_plane.T @ source_rotate_matrix
        channel_rotate_matrix = np.array([- np.sin(lams) * np.cos(thetas),
                                          - np.sin(lams) * np.sin(thetas),
                                          np.cos(lams)])
        ve_channel = self.ve_rotate_plane.T @ channel_rotate_matrix
        row_rotate_matrix = np.array([np.sin(thetas),
                                      - np.cos(thetas),
                                      np.zeros_like(thetas)])
        ve_row = self.ve_rotate_plane.T @ row_rotate_matrix
        normal_rotate_matrix = np.array([np.cos(lams) * np.cos(thetas),
                                         np.cos(lams) * np.sin(thetas),
                                         np.sin(lams)])
        ve_normal = self.ve_rotate_plane.T @ normal_rotate_matrix

        SDD = SPD
        etau = self.detector_rotation_around_uaxis_deg
        etav = self.detector_rotation_around_vaxis_deg
        etaw = self.detector_rotation_around_waxis_deg

        # Detector orientation adjustment with rotation:
        #   - The rotation around different det axes are applied one after
        #     another, rather than applied relative to the old ones
        CBGeo = geo.CBCurveD(ve_channel, ve_row, ve_normal,
                          nchannel, dchannel, ochannel,
                          nrow, drow, orow, dgamma,
                          SDD, vsource)
        if self.detector_rotation_deployment_order.lower() == 'wvu':
            CBGeo.rotate_detector_around_normal_axis_at_piercing_RH(etaw)
            CBGeo.rotate_detector_around_row_axis_at_piercing_RH(etav)
            CBGeo.rotate_detector_around_channel_axis_at_piercing_RH(etau)
        else:
            print('Undefined detector rotation deployment order!')

        self.cone_beam_geometry_angle_zero = CBGeo

        return CBGeo

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_geometry_single_view(idx_view)
    [N] - one application of this function is to generate the CB projection 
          geometry for each view angle around the rotation axis
        - Supported configuratoins
            coordinatey system type: 'CT'    
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20260404
          {update axes in the rotation way}
    ------------------------------------------------------------------------'''

    def generate_CBCurveD_geometry_single_view(self, idx_view):
        if hasattr(self, 'cone_beam_geometry_angle_zero') == False:
            self.generate_CBCurveD_geometry_angle_zero()

        lams = self.source_starting_angle_rad
        dlam = self.source_angular_space_rad
        lam = lams + (idx_view % self.views_per_rotation) * dlam

        thetas = np.deg2rad(self.source_center_starting_angle_deg)
        dtheta = np.deg2rad(self.source_center_rotation_angle_deg) / self.total_number_of_projections
        theta = thetas + idx_view * dtheta
        CID = self.source_center_isocenter_distance_mm
        SID = self.source_to_isocenter_distance_mm
        self.ve_rotate_axis = self.ve_rotate_plane.T @ np.array([-np.sin(theta), np.cos(theta), 0])

        CBGeo = copy.deepcopy(self.cone_beam_geometry_angle_zero)

        # x-ray source position rotation (first) + translation (second)
        vsource = CBGeo.vsource
        vsource[2] = 0
        source_rotate_matrix = np.array([(CID + SID * np.cos(lam)) * np.cos(theta),
                                         (CID + SID * np.cos(lam)) * np.sin(theta),
                                         SID * np.sin(lam)])
        CBGeo.vsource = self.ve_rotate_plane.T @ source_rotate_matrix
        channel_rotate_matrix = np.array([- np.sin(lam) * np.cos(theta),
                                          - np.sin(lam) * np.sin(theta),
                                          np.cos(lam)])
        CBGeo.ve_channel = self.ve_rotate_plane.T @ channel_rotate_matrix
        row_rotate_matrix = np.array([np.sin(theta),
                                      - np.cos(theta),
                                      np.zeros_like(theta)])
        CBGeo.ve_row = self.ve_rotate_plane.T @ row_rotate_matrix
        normal_rotate_matrix = np.array([np.cos(lam) * np.cos(theta),
                                         np.cos(lam) * np.sin(theta),
                                         np.sin(lam)])
        CBGeo.ve_normal = self.ve_rotate_plane.T @ normal_rotate_matrix

        return CBGeo

    '''------------------------------------------------------------------------
    [F] - display_source_trajectory()
    [I] - disp_angular_interval_deg: angular step size used for display
    [N] - displays scan geometry every 10 degrees
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20260404
    ------------------------------------------------------------------------'''

    def display_source_trajectory(self, disp_angular_interval_deg=None):
        if disp_angular_interval_deg == None:
            disp_angular_interval_deg = 10

        # CB projection geo at view angle 0
        CBGeo = copy.deepcopy(self.generate_CBCurveD_geometry_angle_zero())
        CBGeo.vsource[2] = 0

        # gantry rotation
        ve_rotation_axis = self.ve_rotation_axis
        lams = self.source_starting_angle_rad
        dlam = self.source_angular_space_rad
        nlam = self.views_per_rotation
        lam = lams + np.tile(np.arange(nlam), self.source_rotation_number) * dlam
        lam_deg = np.rad2deg(lam)

        # source rotation
        thetas = np.deg2rad(self.source_center_starting_angle_deg)
        dtheta = np.deg2rad(self.source_center_rotation_angle_deg) / self.total_number_of_projections
        ntheta = self.total_number_of_projections
        theta = thetas + np.arange(ntheta) * dtheta
        CID = self.source_center_isocenter_distance_mm
        SID = self.source_to_isocenter_distance_mm
        rotate_matrix = np.array([(CID + SID * np.cos(lam)) * np.cos(theta),
                                  (CID + SID * np.cos(lam)) * np.sin(theta),
                                  SID * np.sin(lam)])

        vsource = (self.ve_rotate_plane.T @ rotate_matrix).T

        # test trajectory of detector center
        vdet = np.zeros((ntheta, 3), dtype=np.float64)
        for k in range(ntheta):
            CBGeok = self.generate_CBCurveD_geometry_single_view(k)
            vak = CBGeok.vsource
            veu = CBGeok.ve_channel
            vev = CBGeok.ve_row
            vew = CBGeok.ve_normal
            D = CBGeok.SDD

            uoff = CBGeok.ochannel
            nu = CBGeok.nchannel
            du = CBGeok.dchannel
            voff = CBGeok.orow
            nv = CBGeok.nrow
            dv = CBGeok.drow
            goff = CBGeok.ochannel
            ng = CBGeok.nchannel
            dg = CBGeok.dgamma
            umin = -uoff * du
            umax = (nu - uoff - 1) * du
            vmin = -voff * dv
            vmax = (nv - voff - 1) * dv

            vod = vak - D * vew

            vdet[k, :] = vod

        # plot: source trajectory, detector boundary, piercing point
        if dtheta == 0:
            ntheta_skip = np.int32(disp_angular_interval_deg / np.rad2deg(dlam))
        else:
            ntheta_skip = np.int32(disp_angular_interval_deg / np.rad2deg(dtheta))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for k in np.arange(0, ntheta, ntheta_skip):
            ax.clear()
            # source trajectory
            ax.plot3D(vsource[:, 0], vsource[:, 1], vsource[:, 2], label='Source Trajectory')
            # detector trajectory
            ax.plot3D(vdet[:, 0], vdet[:, 1], vdet[:, 2], label='Detector Trajectory')

            # CB geometry at view k
            CBGeok = self.generate_CBCurveD_geometry_single_view(k)
            vak = CBGeok.vsource
            uoff = CBGeok.ochannel
            nu = CBGeok.nchannel
            du = CBGeok.dchannel
            voff = CBGeok.orow
            nv = CBGeok.nrow
            dv = CBGeok.drow
            goff = CBGeok.ochannel
            ng = CBGeok.nchannel
            dg = CBGeok.dgamma
            uoffp = CBGeok.ochannel_piercing
            voffp = CBGeok.orow_piercing
            veu = CBGeok.ve_channel
            vev = CBGeok.ve_row
            vew = CBGeok.ve_normal
            D = CBGeok.SDD

            umin = -uoff * du
            umax = (nu - uoff - 1) * du
            vmin = -voff * dv
            vmax = (nv - voff - 1) * dv
            gmin = -goff * dg
            gmax = (ng - goff - 1) * dg
            upiercing = umin + uoffp * du
            vpiercing = vmin + voffp * dv

            # detector origin (ortho projection of source)
            vod = vak - D * vew
            v00 = vak - D * np.cos(np.deg2rad(gmin)) * vew + D * np.sin(np.deg2rad(gmin)) * veu + vmin * vev
            v01 = vak - D * np.cos(np.deg2rad(gmin)) * vew + D * np.sin(np.deg2rad(gmin)) * veu + vmax * vev
            v10 = vak - D * np.cos(np.deg2rad(gmax)) * vew + D * np.sin(np.deg2rad(gmax)) * veu + vmin * vev
            v11 = vak - D * np.cos(np.deg2rad(gmax)) * vew + D * np.sin(np.deg2rad(gmax)) * veu + vmax * vev
            v_bottom = np.zeros([ng, 3])
            v_top = np.zeros([ng, 3])
            for i in range(ng):
                gi = gmin + dg * i
                v_bottom[i] = vak - D * np.cos(np.deg2rad(gi)) * vew + D * np.sin(np.deg2rad(gi)) * veu + vmin * vev
                v_top[i] = vak - D * np.cos(np.deg2rad(gi)) * vew + D * np.sin(np.deg2rad(gi)) * veu + vmax * vev
            # piercing point: intersection between det and line connect src and iso
            vdp = vod + upiercing * veu + vpiercing * vev

            # source point
            ax.scatter3D(vak[0], vak[1], vak[2], color='red')
            # source to piercing
            vtmp = np.vstack((vak, vdp))
            ax.plot3D(vtmp[:, 0], vtmp[:, 1], vtmp[:, 2])
            # detector
            vtmp = np.vstack((v00, v01))
            ax.plot3D(vtmp[:, 0], vtmp[:, 1], vtmp[:, 2], color='green')
            vtmp = np.vstack((v10, v11))
            ax.plot3D(vtmp[:, 0], vtmp[:, 1], vtmp[:, 2], color='green')
            ax.plot3D(v_bottom[:, 0], v_bottom[:, 1], v_bottom[:, 2], color='green')
            ax.plot3D(v_top[:, 0], v_top[:, 1], v_top[:, 2], color='green')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(-1000, 1000)
            ax.set_ylim(-1000, 1000)
            ax.set_zlim(-1000, 1000)
            ax.set_box_aspect([1, 1, 1])
            # ax.set_aspect('auto')
            ttl = str(np.int32(lam_deg[k])) + 'deg'
            ax.set_title(ttl)
            plt.tight_layout()
            # plt.axis('image') does not work for 3D plot
            plt.legend()
            plt.pause(0.2)

        plt.show()

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_projections()
    [N] - Generate cone beam projections and store data
    [H] - created by ZYu prior to 20250614
          revised by JZhang prior to 20260404 
          {"header" needs to be revised}
    ------------------------------------------------------------------------'''

    def generate_CBCurveD_projections(self):
        # Generate phantom
        phantom_name = self.phantom_name
        phantom = ap.StaticLACPhantom(phantom_name)

        # Generate projection data header
        nview = self.total_number_of_projections
        nrow = self.detector_row_number
        nchannel = self.detector_channel_number
        angular_space_per_view = self.source_angular_space_rad
        drow = self.detector_row_space_mm
        dchannel = self.detector_channel_space_mm
        dgamma = self.detector_fan_angle_degree
        starting_angular_position = self.source_starting_angle_rad
        orow = self.detector_row_offset
        ochannel = self.detector_channel_offset
        projection_data_type_numpy = em.DataTypeDtype[self.projection_data_type].value
        projection_type_string = self.projection_type.upper()
        source_rotation_number = self.source_rotation_number
        SDD = self.source_to_piercing_point_distance_mm
        SID = self.source_to_isocenter_distance_mm

        header = fio.create_header_dict_projection_basic(nview, nrow, nchannel,
                                                         angular_space_per_view, drow, dchannel,
                                                         starting_angular_position, orow, ochannel,
                                                         dgamma,
                                                         projection_data_type_numpy,
                                                         projection_type_string,
                                                         source_rotation_number,
                                                         SDD,
                                                         SID)

        os.makedirs(self.raw_data_folder, exist_ok=True)
        fnProjection = phantom_name + '_' + str(nchannel) + 'x' + str(nrow) + 'x' + str(nview) + '.prj'
        fnProjection = os.path.join(self.raw_data_folder, fnProjection)
        fio.write_header(fnProjection, header)

        # Generate projection data
        fig = plt.figure()
        ax = fig.add_subplot()
        for k in range(nview):
            CBGeo = self.generate_CBCurveD_geometry_single_view(k)
            gk = phantom.generate_line_integral_single_projection(CBGeo)
            fio.append_slices(fnProjection, gk, projection_data_type_numpy)

            if np.mod(k, 10) == 0:
                print(f'Completed projection {k} of {nview}')
                ax.clear()  # clear previous content
                ax.set_title(f'View #: {k}')
                ax.imshow(gk, cmap='gray')
                plt.tight_layout()
                plt.axis('image')
                plt.pause(0.2)
        plt.show()

        # Save scan protocol XML file
        fn_scan_protocol_source = self.fn_scan_protocol
        fld_scan_protocol_target = self.raw_data_folder
        shutil.copy(fn_scan_protocol_source, fld_scan_protocol_target)

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_projections_beam_average()
    [I] - nups_channel, nups_row: Average of upsampled rays
          -- Both shall be odd numbers, default is 1
          -- nunew = nups_channel*nu
          -- dunew = du/nunew
          -- uoffnew = nups_channel*nu + (nu-1)/2
    [N] - Generate cone beam projections and store data
    [H] - created by ZYu prior to 20250510
          revised by JZhang prior to 20260404
          {"header" needs to be revised}
    ------------------------------------------------------------------------'''

    def generate_CBCurveD_projections_beam_average(self, nups_channel=1, nups_row=1):
        # Generate phantom
        phantom_name = self.phantom_name
        phantom = ap.StaticLACPhantom(phantom_name)

        # Generate projection data header
        nview = self.total_number_of_projections
        nrow = self.detector_row_number
        nchannel = self.detector_channel_number
        angular_space_per_view = self.source_angular_space_rad
        drow = self.detector_row_space_mm
        dchannel = self.detector_channel_space_mm
        starting_angular_position = self.source_starting_angle_rad
        dgamma = self.detector_fan_angle_degree
        orow = self.detector_row_offset
        ochannel = self.detector_channel_offset
        projection_data_type_numpy = em.DataTypeDtype[self.projection_data_type].value
        projection_type_string = self.projection_type.upper()
        source_rotation_number = self.source_rotation_number
        SDD = self.source_to_piercing_point_distance_mm
        SID = self.source_to_isocenter_distance_mm

        header = fio.create_header_dict_projection_basic(nview, nrow, nchannel,
                                                         angular_space_per_view, drow, dchannel,
                                                         starting_angular_position, orow, ochannel,
                                                         dgamma,
                                                         projection_data_type_numpy,
                                                         projection_type_string,
                                                         source_rotation_number,
                                                         SDD,
                                                         SID)

        os.makedirs(self.raw_data_folder, exist_ok=True)
        fnProjection = phantom_name + '_beamavg' + str(nups_channel) + 'x' + str(nups_row) + '_' + str(
            nchannel) + 'x' + str(nrow) + 'x' + str(nview) + '.prj'
        fnProjection = os.path.join(self.raw_data_folder, fnProjection)
        fio.write_header(fnProjection, header)

        # Generate projection data
        fig = plt.figure()
        ax = fig.add_subplot()
        for k in range(nview):
            CBGeo = self.generate_CBCurveD_geometry_single_view(k)
            # Upsampling
            nrow = CBGeo.nrow
            nchannel = CBGeo.nchannel
            CBGeo.nrow = nups_row * CBGeo.nrow
            CBGeo.drow = CBGeo.drow / nups_row
            CBGeo.orow = nups_row * orow + 0.5 * (nups_row - 1)
            CBGeo.nchannel = nups_channel * CBGeo.nchannel
            CBGeo.dchannel = CBGeo.dchannel / nups_channel
            CBGeo.ochannel = nups_channel * ochannel + 0.5 * (nups_channel - 1)
            CBGeo.dgamma = CBGeo.dgamma / nups_channel
            gk = phantom.generate_line_integral_single_projection(CBGeo)
            # Downsampling (averaging)
            gk = gk.reshape(nrow, nups_row, nchannel, nups_channel).mean(
                axis=(1, 3))  # has to reassign it back to gk, otherwise, gk does not change
            # Save data
            fio.append_slices(fnProjection, gk, projection_data_type_numpy)

            if np.mod(k, 10) == 0:
                print(f'Completed projection {k} of {nview}')
                ax.clear()  # clear previous content
                ax.set_title(f'View #: {k}')
                ax.imshow(gk, cmap='gray')
                plt.tight_layout()
                plt.axis('image')
                plt.pause(0.2)
        plt.show()

        # Save scan protocol XML file
        fn_scan_protocol_source = self.fn_scan_protocol
        fld_scan_protocol_target = self.raw_data_folder
        shutil.copy(fn_scan_protocol_source, fld_scan_protocol_target)

    '''------------------------------------------------------------------------
    [F] - generate_CBCurveD_sinogram()
    [I] - save_path: path to save the sinogram
        - save_name: name of the sinogram
        - nups_channel, nups_row: Average of upsampled rays
          -- Both shall be odd numbers, default is 1
          -- nunew = nups_channel*nu
          -- dunew = du/nunew
          -- uoffnew = nups_channel*nu + (nu-1)/2
    [N] - Generate cone beam sinogram and store data
    [H] - created by JZhang prior to 20260404 
    ------------------------------------------------------------------------'''

    def generate_CBCurveD_sinogram(self, save_path, save_name, nups_channel, nups_row):
        # Generate phantom
        phantom_name = self.phantom_name
        phantom = ap.StaticLACPhantom(phantom_name)

        # Generate projection data
        nview = self.views_per_rotation
        orow = self.detector_row_offset
        nchannel = self.detector_channel_number
        ochannel = self.detector_channel_offset
        sinogram = np.zeros([nchannel, nview])
        for k in range(nview):
            CBGeo = self.generate_CBCurveD_geometry_single_view(k)
            # Upsampling
            nrow = CBGeo.nrow
            nchannel = CBGeo.nchannel
            CBGeo.nrow = nups_row * CBGeo.nrow
            CBGeo.drow = CBGeo.drow / nups_row
            CBGeo.orow = nups_row * orow + 0.5 * (nups_row - 1)
            CBGeo.nchannel = nups_channel * CBGeo.nchannel
            CBGeo.dchannel = CBGeo.dchannel / nups_channel
            CBGeo.ochannel = nups_channel * ochannel + 0.5 * (nups_channel - 1)
            CBGeo.dgamma = CBGeo.dgamma / nups_channel
            gk = phantom.generate_line_integral_single_projection(CBGeo)
            # Downsampling (averaging)
            gk = gk.reshape(nrow, nups_row, nchannel, nups_channel).mean(axis=(1, 3))
            sinogram[:, k] = gk[int(orow)]
            if np.mod(k, 10) == 0:
                print(f'Completed projection {k} of {nview}')

        print(f'Sinogram 2D shape: {sinogram.shape}')
        plt.imshow(sinogram, cmap='gray')
        plt.title("Sinogram")
        plt.show()

        os.makedirs(save_path, exist_ok=True)
        file_name = f"{save_name}_sinogram_{nview}view_{nchannel}channel.npy"
        np.save(os.path.join(save_path, file_name), sinogram)