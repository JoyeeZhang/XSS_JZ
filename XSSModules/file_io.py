import numpy as np
import os
import XSSModules.enum_list as em

'''============================================================================
List of functions
    create_header_dict_volume_basic(nslice,nrow,ncol,
                                  dslice,drow,dcol,
                                  oslice,orow,ocol,
                                  slice_thickness,
                                  slope,intercept,data_type_numpy = np.float32)
    create_header_dict_projection_basic(nview,nrow,nchannel,
                                   angular_space_per_view,drow,dchannel,
                                   starting_angular_position,orow,ochannel,
                                   data_type_numpy = np.float32,
                                   projection_type_string = 'count',
                                   starting_axial_position = 0,
                                   axial_movement_per_view = 0,
                                   SDD = 150,
                                   SID = 100,
                                   tube_current_mA = 80,
                                   tube_potential_kVp = 120,
                                   pulse_width_ms = 5)
    write_header(fn,header_dict)
    write_volume(fn,volume)
    append_slices(fn,slices)
    read_header(fn)
    read_volume(fn)
    read_slices(fn,slice_index_start = 0,slice_index_end = -1)   
============================================================================'''    

'''----------------------------------------------------------------------------
[F] - create_header_dict_volume_basic(header_type,data_type_dtype,
                         nslice,nrow,ncol,
                         dslice,drow,dcol,
                         oslice,orow,ocol,
                         slice_thickness,
                         slope,intercept)
[I] - data_type_dtype (np.dtype): type of data to be stored after header
[I] - nslice, nrow, ncol: size of the data to be stored
      the corresponding volume is of size [nslice, nrow, ncol]
[I] - dslice,drow, dcol: step size of the data grid in each dimension
[I] - oslice, orow, ocol: grid offsets in slice, row and col
      Example: slice position = (k - oslice)*dslice for k = 0,...,nslice-1
[I] - slice_thickness
[I] - slope,intercept: volume_scaled = slope*volume + intercept
[O] - header_dict: header dictionary
[N] - all headers shall contain the basic members
[H] - Created by ZYu prior to 20250614
----------------------------------------------------------------------------'''
def create_header_dict_volume_basic(nslice,nrow,ncol,
                              dslice,drow,dcol,
                              oslice,orow,ocol,
                              slice_thickness = -1,
                              slope=1,intercept=0,data_type_numpy = np.float32):   
    
    # ---- Header type 0: Volume Basic   
    header_type_string = 'VOLUME_BASIC'
    header_type_int = em.HeaderType[header_type_string.upper()].value
    # including 6 uint16 + 9 float32
    header_size_bytes = 48
    data_type_string = em.DataTypeDtype(data_type_numpy).name
    data_type_int = em.DataTypeInt[data_type_string.upper()].value
    
    header_dict = {
        'header_type_string': header_type_string,
        'header_type_int': np.uint16(header_type_int),
        'header_size_bytes': np.uint16(header_size_bytes),
        'data_type_int': np.uint16(data_type_int),
        'data_type_numpy': data_type_numpy,
        'nslice': np.uint16(nslice),
        'nrow': np.uint16(nrow),
        'ncol': np.uint16(ncol),
        'dslice': np.float32(dslice),
        'drow': np.float32(drow),
        'dcol': np.float32(dcol),
        'oslice': np.float32(oslice),
        'orow': np.float32(orow),
        'ocol': np.float32(ocol),
        'slice_thickness': np.float32(slice_thickness),
        'slope': np.float32(slope),
        'intercept': np.float32(intercept)}
    
    return header_dict

'''----------------------------------------------------------------------------
[F] - create_header_dict_projection_basic(nview,nrow,nchannel,
                               angular_space_per_view,drow,dchannel,
                               starting_angular_position,orow,ochannel,
                               data_type_dtype = np.dtype('float32'),
                               projection_type_string = 'count',
                               starting_axial_position = 0,
                               axial_movement_per_view = 0,
                               SDD = 150,
                               SID = 100,
                               tube_current_mA = 80,
                               tube_potential_kVp = 120,
                               pulse_width_ms = 5)
[I] - data_type_dtype (np.dtype): type of data to be stored after header
[I] - projection_type_string: type of projection data includes
        'photon_counts', 'ADU', 'attenuation' and 'ADU'
[I] - nview, nrow, nchannel: size of the data to be stored
      the corresponding projection data is of size [nview, nrow, nchannel]
[I] - angular_space_per_view,drow, dchannel: step size of the data grid in each dimension
[I] - starting_angular_positoin: first view angle
        Example: view = sview + k*dview
[I] - orow, ochannel: grid offsets in row and channel
        Example: slice position = (k - orow)*drow for k = 0,...,nrow-1
[I] - starting_axial_position: hs - starting source position in axial direction
[I] - axial_movement_per_view: dh
[I] - SDD: source-to-detector distance
[I] - SID: source-to-isocenter distance
[I] - tube_current_mA
[I] - tube_potential_kVp
[I] - pulse_width_ms
[H] - Created by ZYu prior to 20250614
----------------------------------------------------------------------------'''
def create_header_dict_projection_basic(nview,nrow,nchannel,
                               angular_space_per_view,drow,dchannel,
                               starting_angular_position,orow,ochannel,
                               data_type_numpy = np.float32,
                               projection_type_string = 'count',
                               starting_axial_position = 0,
                               axial_movement_per_view = 0,
                               SDD = 150,
                               SID = 100,
                               tube_current_mA = -1, # -1 means not used, typically 80
                               tube_potential_kVp = -1, # -1 means not used, typically 120
                               pulse_width_ms = -1): # -1 means not used, typically 5
    
    # ---- Header type 10: Projection Basic 
    header_type_string = 'PROJECTION_BASIC'
    header_type_int = em.HeaderType[header_type_string].value
    # including 7 uint16 (2bytes) + 13 float32 (4 bytes)
    header_size_bytes = 66
    data_type_string = em.DataTypeDtype(data_type_numpy).name
    data_type_int = em.DataTypeInt[data_type_string.upper()].value
    projection_type_int = em.ProjectionType[projection_type_string.upper()].value
    
    header_dict = {
        'header_type_string': header_type_string,
        'header_type_int': np.uint16(header_type_int),
        'header_size_bytes': np.uint16(header_size_bytes),
        'data_type_int': np.uint16(data_type_int),
        'data_type_numpy': data_type_numpy,
        'projection_type_int': np.uint16(projection_type_int),
        'projection_type_string': projection_type_string,
        'nview': np.uint16(nview),
        'nrow': np.uint16(nrow),
        'nchannel': np.uint16(nchannel),
        'angular_space_per_view': np.float32(angular_space_per_view),
        'drow': np.float32(drow),
        'dchannel': np.float32(dchannel),
        'starting_angular_position': np.float32(starting_angular_position),
        'orow': np.float32(orow),
        'ochannel': np.float32(ochannel),
        'starting_axial_position': np.float32(starting_axial_position),
        'axial_movement_per_view': np.float32(axial_movement_per_view),
        'SDD': np.float32(SDD),
        'SID': np.float32(SID),
        'tube_current_mA': np.float32(tube_current_mA),
        'tube_potential_kVp': np.float32(tube_potential_kVp),
        'pulse_width_ms': np.float32(pulse_width_ms),
        }
    
    return header_dict
              
'''------------------------------------------------------------------------
[F] - write_header(fn,header_type = None,header_dict = None)
[I] - fn: filename to store the volume/projection data
[I] - header_type [Optional](np.uint16, 2 bytes): default is 0
[I] - header_dict [Optional]: list 
[O] - header_dict (dictionary)
[H] - Created by ZYu prior to 20250614
------------------------------------------------------------------------'''    
def write_header(fn,header_dict):
    header_type_int = header_dict['header_type_int']
    fdir = os.path.dirname(fn)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
        
    # Write header
    with open(fn,"wb") as fid:
        fid.write(header_dict['header_type_int'])
        
    # ---- Header type 0: Volume Basic
    if header_type_int == em.HeaderType['VOLUME_BASIC'].value:                  
        with open(fn,"ab") as fid:
            # 5 np.uint16 (6 in total with header_type)
            fid.write(header_dict['header_size_bytes'])
            fid.write(header_dict['data_type_int'])
            fid.write(header_dict['nslice'])
            fid.write(header_dict['nrow'])
            fid.write(header_dict['ncol'])
            # 9 np.float32
            fid.write(header_dict['dslice'])
            fid.write(header_dict['drow'])
            fid.write(header_dict['dcol'])
            fid.write(header_dict['oslice'])
            fid.write(header_dict['orow'])
            fid.write(header_dict['ocol'])
            fid.write(header_dict['slice_thickness'])
            fid.write(header_dict['slope'])
            fid.write(header_dict['intercept'])
    # ---- Header type 10: Projection Basic
    elif header_type_int == em.HeaderType['PROJECTION_BASIC'].value:                  
        with open(fn,"ab") as fid:
            # 6 np.uint16 (7 in total with header_type) 
            fid.write(header_dict['header_size_bytes'])
            fid.write(header_dict['data_type_int'])
            fid.write(header_dict['projection_type_int'])
            fid.write(header_dict['nview'])
            fid.write(header_dict['nrow'])
            fid.write(header_dict['nchannel'])
            # 13 np.float32
            fid.write(header_dict['angular_space_per_view'])
            fid.write(header_dict['drow'])
            fid.write(header_dict['dchannel'])
            fid.write(header_dict['starting_angular_position'])
            fid.write(header_dict['orow'])
            fid.write(header_dict['ochannel'])
            fid.write(header_dict['starting_axial_position'])
            fid.write(header_dict['axial_movement_per_view'])
            fid.write(header_dict['SDD'])
            fid.write(header_dict['SID'])
            fid.write(header_dict['tube_current_mA'])
            fid.write(header_dict['tube_potential_kVp'])
            fid.write(header_dict['pulse_width_ms'])
    else:
        raise Exception('Invalid header type!')
        
    return header_dict

'''------------------------------------------------------------------------
[F] - write_volume(fn,volume/projection)
[I] - fn: filename to store the volume/projection data
[I] - volume/projection[optional]: np.ndarray([nslice,nrow,ncol])
[O] - volume/projection written to fn on disk
[N] - header should be already written before this function
[H] - Created by ZYu prior to 20250614
------------------------------------------------------------------------'''     
def write_volume(fn,volume,dtype = np.float32):         
    volume = volume.astype(dtype)   
    with open(fn,"ab") as fid:
        fid.write(volume)
            
'''------------------------------------------------------------------------
[F] - append_slices(fn)
[I] - fn: filename to store the volume data
[I] - slices: np.ndarray of size [nslice,nrow,ncol] w/ nslice = 1 mostly
[N] - header should be already written before this function
[H] - Created by ZYu prior to 20250614
------------------------------------------------------------------------'''
def append_slices(fn,slices,dtype = np.float32):
    slices = slices.astype(dtype)
    with open(fn,"ab") as fid:
        fid.write(slices)
            
'''------------------------------------------------------------------------
[F] - read_header(fn)
[I] - fn: filename to store the volume/projection data
[O] - header_dict (dictionary)
[H] - Created by ZYu prior to 20250614
------------------------------------------------------------------------'''    
def read_header(fn):
    with open(fn) as fid:
        header_type_int = np.fromfile(fid,dtype=np.uint16,count = 1)
        header_type_string = em.HeaderType(header_type_int).name
        
        # ---- Header type 0: Volume Basic
        if header_type_int == em.HeaderType['VOLUME_BASIC'].value:
            n_uint16 = 5
            n_float32 = 9
            array_uint16 = np.fromfile(fid,dtype=np.uint16,count=n_uint16)
            array_float32 = np.fromfile(fid,dtype=np.float32,count=n_float32)
            data_type_int = array_uint16[1]
            data_type_string = em.DataTypeInt(data_type_int).name
            data_type_numpy = em.DataTypeDtype[data_type_string.upper()].value
            header_dict = {
                'header_type_int': np.uint16(header_type_int),
                'header_type_string': header_type_string,
                'header_size_bytes': np.uint16(array_uint16[0]),
                'data_type_int': np.uint16(data_type_int),
                'data_type_numpy': data_type_numpy,
                'nslice': np.uint16(array_uint16[2]),
                'nrow': np.uint16(array_uint16[3]),
                'ncol': np.uint16(array_uint16[4]), # uint16 above
                'dslice': np.float32(array_float32[0]),
                'drow': np.float32(array_float32[1]),
                'dcol': np.float32(array_float32[2]),
                'oslice': np.float32(array_float32[3]),
                'orow': np.float32(array_float32[4]),
                'ocol': np.float32(array_float32[5]),
                'slice_thickness': np.float32(array_float32[6]),
                'slope': np.float32(array_float32[7]),
                'intercept': np.float32(array_float32[8])}
        # ---- Header type 10: Projection Basic
        elif header_type_int == em.HeaderType['PROJECTION_BASIC'].value:
            n_uint16 = 6
            n_float32 = 13
            array_uint16 = np.fromfile(fid,dtype=np.uint16,count=n_uint16)
            array_float32 = np.fromfile(fid,dtype=np.float32,count=n_float32)
            data_type_int = array_uint16[1]
            data_type_string = em.DataTypeInt(data_type_int).name
            data_type_numpy = em.DataTypeDtype[data_type_string.upper()].value
            projection_type_int = array_uint16[2]
            projection_type_string = em.ProjectionType(projection_type_int).name
            header_dict = {
                'header_type_int': np.uint16(header_type_int),
                'header_type_string': header_type_string,
                'header_size_bytes': np.uint16(array_uint16[0]),
                'data_type_int': np.uint16(data_type_int),
                'data_type_numpy': data_type_numpy,
                'projection_type_int': np.uint16(projection_type_int),
                'projection_type_string': projection_type_string,
                'nview': np.uint16(array_uint16[3]),
                'nrow': np.uint16(array_uint16[4]),
                'nchannel': np.uint16(array_uint16[5]), # uint16 above
                'angular_space_per_view': np.float32(array_float32[0]),
                'drow': np.float32(array_float32[1]),
                'dchannel': np.float32(array_float32[2]),
                'starting_angular_position': np.float32(array_float32[3]),
                'orow': np.float32(array_float32[4]),
                'ochannel': np.float32(array_float32[5]),
                'starting_axial_position': np.float32(array_float32[6]),
                'axial_movement_per_view': np.float32(array_float32[7]),
                'SDD': np.float32(array_float32[8]),
                'SID': np.float32(array_float32[9]),
                'tube_current_mA': np.float32(array_float32[10]),
                'tube_potential_kVp': np.float32(array_float32[11]),
                'pulse_width_ms': np.float32(array_float32[12])}
        else:
            raise Exception('Invalid header type!')
        
    return header_dict

'''------------------------------------------------------------------------
[F] - read_volume(fn)
[I] - fn: filename to store the volume data
[O] - volume: np.ndarray[nslice,nrow,ncol] or
      projection: np.ndarray[nview,nrow,nchannel]
[H] - Created by ZYu prior to 20250614
------------------------------------------------------------------------'''    
def read_volume(fn):
    header = read_header(fn)
    header_type_int = header['header_type_int']
    
    # Volume Basic
    if header_type_int == em.HeaderType['VOLUME_BASIC'].value:
        header_size_bytes = header['header_size_bytes']
        data_type_numpy = header['data_type_numpy']
        nslice = header['nslice']
        nrow = header['nrow']
        ncol = header['ncol']
            
        with open(fn,"rb") as fid:
            fid.seek(header_size_bytes)
            volume = np.fromfile(fid,dtype=data_type_numpy)
            volume = volume.reshape([nslice,nrow,ncol])        
        return volume
    # Basic projection header
    elif header_type_int == em.HeaderType['PROJECTION_BASIC'].value:
        header_size_bytes = header['header_size_bytes']
        data_type_numpy = header['data_type_numpy']
        nview = header['nview']
        nrow = header['nrow']
        nchannel = header['nchannel']
            
        with open(fn,"rb") as fid:
            fid.seek(header_size_bytes)
            volume = np.fromfile(fid,dtype=data_type_numpy)
            volume = volume.reshape([nview,nrow,nchannel])        
        return volume
    
    else:
        raise Exception('Invalid header type!')
             
'''------------------------------------------------------------------------
[F] - read_slices(fn,slice_index_start,slice_index_end)
[I] - fn: filename to store the volume/projection data
[I] - slice_index_start [optional]: default = 0
[I] - slice_index_end [optional]: default = -1
[O] - slices: np.ndarray[nslice_output,nrow,ncol] where
      nslice_output = abs( mod(slice_index_start,nslice) 
                         - mod(slice_index_end,nslice ) ) + 1
      slice_index_start = min( mod(slice_index_start,nslice), 
                               mod(slice_index_end,nslice ) )
      slices = volume(slice_index_start + [0:nslice_output-1]), or
      slices = projection(slice_index_start + [0:nslice_output-1])
      
[N] - read_slices(fn) is equivalent to read_volume(fn), but read_volume is 
      recommended, as it is more direct without calculating intermediates
[H] - Created by ZYu prior to 20250614
------------------------------------------------------------------------'''     
def read_slices(fn,slice_index_start = 0,slice_index_end = -1):
    header = read_header(fn)
    header_type_int = header['header_type_int']
    
    # Basic 3D Volume Header
    if header_type_int == em.HeaderType['VOLUME_BASIC'].value:
        header_size_bytes = header['header_size_bytes']
        data_type_numpy = header['data_type_numpy']
        nslice = header['nslice']
        nrow = header['nrow']
        ncol = header['ncol']
    
        slice_index_start = np.mod(slice_index_start,nslice)
        slice_index_end = np.mod(slice_index_end,nslice)
        nslice_skip = np.min([slice_index_start,slice_index_end])
        nslice_output = np.abs(slice_index_start - slice_index_end) + 1
        bytes_per_slice = np.dtype(data_type_numpy).itemsize*nrow*ncol
        slice_skip_bytes = bytes_per_slice*nslice_skip
            
        with open(fn,'rb') as fid:
            fid.seek(header_size_bytes + slice_skip_bytes)
            slices = np.fromfile(fid,data_type_numpy,count = nslice_output*nrow*ncol)
            slices = slices.reshape([nslice_output,nrow,ncol])
            
        return slices
    # Basic projection Header
    elif header_type_int == em.HeaderType['PROJECTION_BASIC'].value:
        header_size_bytes = header['header_size_bytes']
        data_type_numpy = header['data_type_numpy']
        nview = header['nview']
        nrow = header['nrow']
        nchannel = header['nchannel']
    
        slice_index_start = np.mod(slice_index_start,nview)
        slice_index_end = np.mod(slice_index_end,nview)
        nslice_skip = np.min([slice_index_start,slice_index_end])
        nslice_output = np.abs(slice_index_start - slice_index_end) + 1
        bytes_per_slice = np.dtype(data_type_numpy).itemsize*nrow*nchannel
        slice_skip_bytes = bytes_per_slice*nslice_skip
            
        with open(fn,'rb') as fid:
            fid.seek(header_size_bytes + slice_skip_bytes)
            slices = np.fromfile(fid,data_type_numpy,count = nslice_output*nrow*nchannel)
            slices = slices.reshape([nslice_output,nrow,nchannel])
            
        return slices
    else:
        raise Exception('Invalid header type!')
               

        
        
    
        

