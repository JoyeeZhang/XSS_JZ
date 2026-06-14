import numpy as np
from XSSModules import analytical_phantom as ap
from XSSModules import image as img
from XSSModules import support as spt
from XSSModules import geometry as geo
import matplotlib.pyplot as plt
from XSSModules import scanner as sc

Detector_option = 'CurveD' # 'FPD' or 'CurveD'
Scan_option = 'Rolical' # 'Helical' or 'Rolical'
Dis_traj_option = True
Dis_prj3D_option = False
Save_sino_option = True
save_path = './Data/Sinograms/'
file_name = 'Tire_phantom'

if Scan_option == 'Helical' and Detector_option == 'FPD':
    fn_scan_protocol = './Templates/scan_protocol_CT_CBFPD_Helical_LAC_template.xml'
    S = sc.CTCBFPDHelicalLAC(fn_scan_protocol)
    if Dis_traj_option:
        S.display_source_trajectory()
    if Dis_prj3D_option:
        S.generate_CBFPD_projections()
        # generate projection data with up sampling
        # nups_channel = 3
        # S.generate_CBFPD_projections_beam_average(nups_channel)
    if Save_sino_option:
        nups_channel = 3
        nups_row = 1
        save_name = file_name + '_' + Scan_option + '_' + Detector_option
        S.generate_CBFPD_sinogram(save_path, save_name, nups_channel, nups_row)

if Scan_option == 'Helical' and Detector_option == 'CurveD':
    fn_scan_protocol = './Templates/scan_protocol_CT_CBCurveD_Helical_LAC_template.xml'
    S = sc.CTCBCurveDHelicalLAC(fn_scan_protocol)
    if Dis_traj_option:
        S.display_source_trajectory()
    if Dis_prj3D_option:
        S.generate_CBCurveD_projections()
        # generate projection data with up sampling
        # nups_channel = 3
        # S.generate_CBCurveD_projections_beam_average(nups_channel)
    if Save_sino_option:
        nups_channel = 3
        nups_row = 1
        save_name = file_name + '_' + Scan_option + '_' + Detector_option
        S.generate_CBCurveD_sinogram(save_path, save_name, nups_channel, nups_row)

if Scan_option == 'Rolical' and Detector_option == 'FPD':
    fn_scan_protocol = './Templates/scan_protocol_CT_CBFPD_Rolical_LAC_template.xml'
    S = sc.CTCBFPDRolicalLAC(fn_scan_protocol)
    if Dis_traj_option:
        S.display_source_trajectory(10)
    if Dis_prj3D_option:
        S.generate_CBFPD_projections()
        # generate projection data with up sampling
        # nups_channel = 3
        # S.generate_CBFPD_projections_beam_average(nups_channel)
    if Save_sino_option:
        nups_channel = 3
        nups_row = 1
        save_name = file_name + '_' + Scan_option + '_' + Detector_option
        S.generate_CBFPD_sinogram(save_path, save_name, nups_channel, nups_row)

if Scan_option == 'Rolical' and Detector_option == 'CurveD':
    fn_scan_protocol = './Templates/scan_protocol_CT_CBCurveD_Rolical_LAC_template.xml'
    S = sc.CTCBCurveDRolicalLAC(fn_scan_protocol)
    if Dis_traj_option:
        S.display_source_trajectory(10)
    if Dis_prj3D_option:
        S.generate_CBCurveD_projections()
        # generate projection data with up sampling
        # nups_channel = 3
        # S.generate_CBCurveD_projections_beam_average(nups_channel)
    if Save_sino_option:
        nups_channel = 3
        nups_row = 1
        save_name = file_name + '_' + Scan_option + '_' + Detector_option
        S.generate_CBCurveD_sinogram(save_path, save_name, nups_channel, nups_row)








