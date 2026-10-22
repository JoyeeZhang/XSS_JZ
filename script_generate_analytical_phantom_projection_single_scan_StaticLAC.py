import os
import numpy as np
from XSSModules import analytical_phantom as ap
from XSSModules import file_io as fio
from XSSModules import image as img
from XSSModules import support as spt
from XSSModules import geometry as geo
import matplotlib.pyplot as plt
from XSSModules import scanner as sc

Detector_option = 'FPD' # 'FPD' or 'CurveD'
Scan_option = 'Helical' # 'Helical' or 'Rolical'
Dis_traj_option = True
Save_prj_option = True
Dis_sino_option = True

def display_middle_row_sinogram(scanner):
    nview = scanner.total_number_of_projections
    nrow = scanner.detector_row_number
    nchannel = scanner.detector_channel_number
    projection_name = (
        f'{scanner.phantom_name}_{nchannel}x{nrow}x{nview}.prj'
    )
    projection_path = os.path.join(scanner.raw_data_folder, projection_name)
    if not os.path.isfile(projection_path):
        raise FileNotFoundError(
            f'Saved projection data not found: {projection_path}'
        )

    projection = fio.read_volume(projection_path)

    middle_row = nrow // 2
    sinogram = projection[:, middle_row, :].T

    plt.figure(figsize=(10, 6))
    plt.imshow(sinogram, cmap='gray', aspect='auto', origin='lower')
    plt.title(f'Middle detector row sinogram (row {middle_row})')
    plt.xlabel('View number')
    plt.ylabel('Detector channel')
    plt.colorbar(label='Line integral')
    plt.tight_layout()
    plt.show()

if Scan_option == 'Helical' and Detector_option == 'FPD':
    fn_scan_protocol = './Templates/scan_protocol_CT_CBFPD_Helical_LAC_template.xml'
    S = sc.CTCBFPDHelicalLAC(fn_scan_protocol)
    if Dis_traj_option:
        S.display_source_trajectory()
    if Save_prj_option:
        S.generate_CBFPD_projections()
        # generate projection data with up sampling
        # nups_channel = 3
        # S.generate_CBFPD_projections_beam_average(nups_channel)
    if Dis_sino_option:
        display_middle_row_sinogram(S)


if Scan_option == 'Helical' and Detector_option == 'CurveD':
    fn_scan_protocol = './Templates/scan_protocol_CT_CBCurveD_Helical_LAC_template.xml'
    S = sc.CTCBCurveDHelicalLAC(fn_scan_protocol)
    if Dis_traj_option:
        S.display_source_trajectory()
    if Save_prj_option:
        S.generate_CBCurveD_projections()
        # generate projection data with up sampling
        # nups_channel = 3
        # S.generate_CBCurveD_projections_beam_average(nups_channel)
    if Dis_sino_option:
        display_middle_row_sinogram(S)


if Scan_option == 'Rolical' and Detector_option == 'FPD':
    fn_scan_protocol = './Templates/scan_protocol_CT_CBFPD_Rolical_LAC_template.xml'
    S = sc.CTCBFPDRolicalLAC(fn_scan_protocol)
    if Dis_traj_option:
        S.display_source_trajectory(10)
    if Save_prj_option:
        S.generate_CBFPD_projections()
        # generate projection data with up sampling
        # nups_channel = 3
        # S.generate_CBFPD_projections_beam_average(nups_channel)
    if Dis_sino_option:
        display_middle_row_sinogram(S)


if Scan_option == 'Rolical' and Detector_option == 'CurveD':
    fn_scan_protocol = './Templates/scan_protocol_CT_CBCurveD_Rolical_LAC_template.xml'
    S = sc.CTCBCurveDRolicalLAC(fn_scan_protocol)
    if Dis_traj_option:
        S.display_source_trajectory(10)
    if Save_prj_option:
        S.generate_CBCurveD_projections()
        # generate projection data with up sampling
        # nups_channel = 3
        # S.generate_CBCurveD_projections_beam_average(nups_channel)
    if Dis_sino_option:
        display_middle_row_sinogram(S)








