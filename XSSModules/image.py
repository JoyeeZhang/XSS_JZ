import numpy as np
import matplotlib.pyplot as plt
import XSSModules.geometry as geo
import XSSModules.file_io as fio
import XSSModules.enum_list as enm

'''============================================================================
Class       
    Volume3D
Parent
    GeometryVolume3D
Attributes
    ---- Mandatory attributes for header
    ncol             np.uint16   number of colums [inherited]
    nrow             np.uint16   number of rows [inherited]
    nslice           np.uint16   number of slices [inherited]
    dcol             np.float32  column space [inherited]
    drow             np.float32  row space [inherited]
    dslice           np.float32  slice interval [inherited]
    ocol             np.float32  column offset [inherited]
    orow             np.float32  row offset [inherited]
    oslice           np.float32  slice offset [inherited]
    slice_thickness  np.float32  slice thickness (default: tslice = dslice)
    slope            np.float32
    intercept        np.float32
    header_type_int  np.uint16 
    data_type        np dtype       volume data type 
    ---- Other attributes   
    ve_col           np.float32  unit vector for column index (e.g., vex)
    ve_row           np.float32  unit vector for row index (e.g., vey)     
    ve_slice         np.float32  unit vector for slice index (e.g., vez)       
    vo               np.float32  origin relative to the parental [inherited]
    parent_crd       string [default - 'World'] [inherited]
    ---- Data
    volume           np.ndarray([nslice,nrow,ncol])
Methods
    write_header(self,fn,header_type = None, header_dict = None)
    write_volume(self,fn,volume = None)
    import_fromfile(self,fn)
Notes
    - Volume definition: volume[nslice,nrow,ncol]
    - Grid definition: col[k] = (k-ocol)*dcol for k = 0,1,2,3,...,ncol-1
History: Created by ZYu prior to 20250614
============================================================================'''
class Volume3D(geo.Volume3D):
    def __init__(self, ncol,nrow,nslice,dcol,drow,dslice,ocol,orow,oslice,
                 slice_thickness = None,slope = 1, intercept = 0, 
                 data_type_numpy = np.float64,
                 header_type_string = 'VOLUME_BASIC',
                 ve_col = np.array([1,0,0]),
                 ve_row = np.array([0,1,0]),
                 ve_slice = np.array([0,0,1]),
                 vo  = np.array([0,0,0]), 
                 parent_crd = 'world'):
                     
        # ---- Geometry
        super().__init__(ncol,nrow,nslice,
                         dcol,drow,dslice,
                         ocol,orow,oslice,
                         ve_col,ve_row,ve_slice,
                         vo, parent_crd)

        # ---- All attributs below comes with a default settings
        # Slice thickness: Default to slice interval is not provided
        if slice_thickness is None:
            self.slice_thickness = dslice
        else:
            self.slice_thickness = slice_thickness
        # Affine transform
        # Output volume = slope*Input Volume + intercept
        self.slope = slope
        self.intercept = intercept
        # Header type
        self.header_type_string = header_type_string
        self.header_type_int = enm.HeaderType[header_type_string.upper()].value
        # Volume data type
        self.data_type_numpy = data_type_numpy
        # Volume data initialized with 0
        self.volume = np.zeros([nslice,nrow,ncol],dtype = data_type_numpy)
    
    '''------------------------------------------------------------------------
    [F] update_slope_intercept_LAC2HU(self,LAC_water,LAC_air):
    [I] LAC_water: linear attenuation of water
    [I] LAC_air: linear attenuation of air
    [O] self.slope and intercept are updated such that they can convert 
        self.volume in LAC to HU such that
        (air,water) = (-1000 HU, 0 HU)
    [N] Assume the current self.volume is in LAC
    - ----------------------------------------------------------------------'''    
    def update_slope_intercept_LAC2HU(self,LAC_water,LAC_air):
        recip_dLAC = LAC_water - LAC_air
        self.slope = 1000*recip_dLAC
        self.intercept = -1000*LAC_water*recip_dLAC
    
    '''------------------------------------------------------------------------
    [F] update_slope_intercept_HU2LAC(self,LAC_water,LAC_air):
    [I] LAC_water: linear attenuation of water
    [I] LAC_air: linear attenuation of air
    [O] self.slope and intercept are updated such that they can convert 
        self.volume in HU to LAC 
    [N] Assume the current self.volume is in HU such that
        (air,water) = (-1000 HU, 0 HU)
    - ----------------------------------------------------------------------'''    
    def update_slope_intercept_HU2LAC(self,LAC_water,LAC_air):
        self.slope = 0.001*(LAC_water - LAC_air)
        self.intercept = LAC_water        
 
    '''------------------------------------------------------------------------
    [F] - write_header(fn,header_type = None,header_dict = None)
    [I] - fn: filename to store the volume data
    [I] - header_type [Optional] (np.uint16, 2 bytes)
    [I] - header_dict [Optional] 
    [O] - header_dict (dictionary)
    ------------------------------------------------------------------------'''    
    def write_header(self,fn):
        header_type_int = self.header_type_int
        
        match header_type_int:    
            # ---- Header type 0: Basic Header
            case enm.HeaderType.VOLUME_BASIC.value:              
                header_dict = fio.create_header_dict_volume_basic(
                            self.nslice,self.nrow,self.ncol,
                            self.dslice,self.drow,self.dcol,
                            self.oslice,self.orow,self.ocol,
                            self.slice_thickness,
                            self.slope,self.intercept,
                            self.data_type_numpy)
            case _:
                print('Invalid header type!!')
                
        # Write header
        fio.write_header(fn,header_dict)
        
        return header_dict

    '''------------------------------------------------------------------------
    [F] - write_volume(fn)
    [I] - fn: filename to store the volume data
    [I] - tag_conversion [optional]: 1 apply (slope,intercept), 0 not
    [N] - header should be already written before this function
    ------------------------------------------------------------------------'''     
    def write_volume(self,fn,tag_conversion = False): 
        if tag_conversion:
            volume = self.slope*self.volume + self.intercept
        else:
            volume = self.volume
            
        fio.write_volume(fn,volume)
    
    '''------------------------------------------------------------------------
    [F] - read_volume(self,fn)
    [I] - fn: filename to store the volume data
    [O] - update self with both the meta and data stored in fn
    [O] - header_dict
    ------------------------------------------------------------------------'''     
    def read_volume(self,fn): 
        # Import meta info from header
        header_dict = fio.read_header(fn)
        header_type_int = header_dict['header_type_int']
        
        if header_type_int == enm.HeaderType['VOLUME_BASIC'].value:
            self.header_type_int = header_dict['header_type_int']
            self.data_type_numpy = header_dict['data_type_numpy'] 
            self.ncol = header_dict['ncol']
            self.nrow = header_dict['nrow']
            self.nslice = header_dict['nslice']
            self.dcol = header_dict['dcol']
            self.drow = header_dict['drow']
            self.dslice = header_dict['dslice']
            self.ocol = header_dict['ocol']
            self.orow = header_dict['orow']
            self.oslice = header_dict['oslice']
            self.slice_thickness = header_dict['slice_thickness']
            self.slope = header_dict['slope']
            self.intercept = header_dict['intercept']
            
            # Import data from fn
            volume = fio.read_volume(fn)
            self.volume = volume
            
            return header_dict
        else:
            raise Exception('Wrong header type!!')

'''============================================================================
Class       
    Slice2D
Parent
    GeometryVolume3D
Attributes
    ---- Mandatory attributes for header
    ncol             np.uint16   number of colums [inherited]
    nrow             np.uint16   number of rows [inherited]
    dcol             np.float32  column space [inherited]
    drow             np.float32  row space [inherited]
    ocol             np.float32  column offset [inherited]
    orow             np.float32  row offset [inherited]
    slice_thickness  np.float32  slice thickness (default: -1 not used)
    slope            np.float32
    intercept        np.float32
    header_type_int  np.uint16 
    data_type        np dtype       volume data type 
    ---- Other attributes   
    ve_col           np.float32  unit vector for column index (e.g., vex)
    ve_row           np.float32  unit vector for row index (e.g., vey)     
    vo               np.float32  origin relative to the parental [inherited]
    parent_crd       string [default - 'World'] [inherited]
    ---- Data
    Slice           np.ndarray([nrow,ncol])
Methods
    write_header(self,fn,header_type = None, header_dict = None)
    write_slice(self,fn,volume = None)
    read_slice(self,fn)
Notes
    - Slice definition: slice[nrow,ncol]
    - Grid definition: col[k] = (k-ocol)*dcol for k = 0,1,2,3,...,ncol-1
History: 20250429 Created by ZYu
============================================================================'''
class Slice2D(geo.Slice2D):
    def __init__(self, ncol,nrow,dcol,drow,ocol,orow,
                 slice_thickness = -1, slope = 1,intercept = 0, 
                 data_type_numpy = np.float32,
                 header_type_string = 'VOLUME_BASIC',
                 ve_col = np.array([1,0]),
                 ve_row = np.array([0,1]),
                 vo  = np.array([0,0]), 
                 parent_crd = 'world'):
                     
        # ---- Geometry
        super().__init__(ncol,nrow,
                         dcol,drow,
                         ocol,orow,
                         ve_col,ve_row,
                         vo, parent_crd)

        # ---- All attributs below comes with a default settings
        # Slice thickness: Default to -1 if not used
        self.slice_thickness = slice_thickness
        # Affine transform
        # Output volume = slope*Input Volume + intercept
        self.slope = slope
        self.intercept = intercept
        # Header type
        self.header_type_string = header_type_string
        self.header_type_int = enm.HeaderType[header_type_string.upper()].value
        # Slice data type
        self.data_type_numpy = data_type_numpy
        # Slice data initialized with 0
        self.slice = np.zeros([nrow,ncol],dtype = data_type_numpy)
    
    '''------------------------------------------------------------------------
    [F] - update_slope_intercept_LAC2HU(self,LAC_water,LAC_air):
    [I] - LAC_water: linear attenuation of water
    [I] - LAC_air: linear attenuation of air
    [O] - self.slope and intercept are updated such that they can convert 
          self.volume in LAC to HU such that
          (air,water) = (-1000 HU, 0 HU)
    [N] - Assume the current self.volume is in LAC
    [H] - 20250429 created by ZYu
    - ----------------------------------------------------------------------'''    
    def update_slope_intercept_LAC2HU(self,LAC_water,LAC_air):
        recip_dLAC = LAC_water - LAC_air
        self.slope = 1000*recip_dLAC
        self.intercept = -1000*LAC_water*recip_dLAC
    
    '''------------------------------------------------------------------------
    [F] - update_slope_intercept_HU2LAC(self,LAC_water,LAC_air):
    [I] - LAC_water: linear attenuation of water
    [I] - LAC_air: linear attenuation of air
    [O] - self.slope and intercept are updated such that they can convert 
          self.volume in HU to LAC 
    [N] - Assume the current self.volume is in HU such that
          (air,water) = (-1000 HU, 0 HU)
    [H] - 20250429 created by ZYu
    - ----------------------------------------------------------------------'''    
    def update_slope_intercept_HU2LAC(self,LAC_water,LAC_air):
        self.slope = 0.001*(LAC_water - LAC_air)
        self.intercept = LAC_water        
 
    '''------------------------------------------------------------------------
    [F] - write_header(fn,header_type = None,header_dict = None)
    [I] - fn: filename to store the volume data
    [I] - header_type [Optional] (np.uint16, 2 bytes)
    [I] - header_dict [Optional] 
    [O] - header_dict (dictionary)
    [H] - 20250429 created by ZYu
    ------------------------------------------------------------------------'''    
    def write_header(self,fn):
        header_type_int = self.header_type_int
        
        match header_type_int:    
            # ---- Header type 0: Basic Header
            case enm.HeaderType.VOLUME_BASIC.value:              
                nslice = 1
                dslice = 1
                oslice = 0
                header_dict = fio.create_header_dict_volume_basic(
                            nslice,self.nrow,self.ncol,
                            dslice,self.drow,self.dcol,
                            oslice,self.orow,self.ocol,
                            self.slice_thickness,
                            self.slope,self.intercept,
                            self.data_type_numpy)
            case _:
                print('Invalid header type!!')
                
        # Write header
        fio.write_header(fn,header_dict)
        
        return header_dict

    '''------------------------------------------------------------------------
    [F] - write_slice(fn)
    [I] - fn: filename to store the volume data
    [I] - tag_conversion [optional]: 1 apply (slope,intercept), 0 not
    [N] - header should be already written before this function
    [H] - 20250429 created by ZYu
    ------------------------------------------------------------------------'''     
    def write_slice(self,fn,tag_conversion = False): 
        if tag_conversion:
            slice = self.slope*self.slice + self.intercept
        else:
            slice = self.slice
            
        fio.write_volume(fn,slice)
    
    '''------------------------------------------------------------------------
    [F] - read_slice(self,fn)
    [I] - fn: filename to store the volume data
    [O] - update self with both the meta and data stored in fn
    [O] - header_dict
    [H] - 20250429 created by ZYu
    ------------------------------------------------------------------------'''     
    def read_slice(self,fn): 
        # Import meta info from header
        header_dict = fio.read_header(fn)
        header_type_int = header_dict['header_type_int']
        
        if header_type_int == enm.HeaderType['VOLUME_BASIC'].value:
            self.header_type_int = header_dict['header_type_int']
            self.data_type_numpy = header_dict['data_type_numpy'] 
            self.ncol = header_dict['ncol']
            self.nrow = header_dict['nrow']
            self.nslice = header_dict['nslice']
            self.dcol = header_dict['dcol']
            self.drow = header_dict['drow']
            self.dslice = header_dict['dslice']
            self.ocol = header_dict['ocol']
            self.orow = header_dict['orow']
            self.oslice = header_dict['oslice']
            self.slice_thickness = header_dict['slice_thickness']
            self.slope = header_dict['slope']
            self.intercept = header_dict['intercept']
            
            # Import data from fn
            slice = fio.read_volume(fn)
            self.slice = slice.squeeze()
            
            return header_dict
        else:
            raise Exception('Wrong header type!!')

        
    
        

