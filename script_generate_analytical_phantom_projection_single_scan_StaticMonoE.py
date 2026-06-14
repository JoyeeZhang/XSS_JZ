import numpy as np
from XSSModules import analytical_phantom as ap
from XSSModules import image as img
from XSSModules import support as spt
from XSSModules import geometry as geo
import matplotlib.pyplot as plt
from XSSModules import scanner as sc

# General information
fn_scan_protocol = 'C:/Users/jzhan685/CT/Codes/XSS/Templates/scan_protocol_CT_CBFPD_Helical_MonoE_template.xml'

S = sc.CTCBFPDHelicalLAC(fn_scan_protocol)

S.display_source_trajectory()

S.generate_CBFPD_projections()

nups_channel = 3
S.generate_CBFPD_projections_beam_avarage(nups_channel)
