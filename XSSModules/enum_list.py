from enum import Enum
import numpy as np

'''============================================================================
Examplary Usage
  TrajectoryType['NOMINAL'] == TrajectoryType.NOMINAL
  TrajectoryType.NOMINAL.name == 'NOMINAL'
  TrajectoryType.NOMINAL.value == 1
  TrajectoryType(1) == TrajectoryType['NOMINAL'] = TrajectoryType.NOMINAL
  TrajectoryType(1).name == 'NOMINAL'
  TrajectoryType(1).value == 1
============================================================================'''
#------ Geometry 
class CoordinateSystemType(Enum):
  CT = 1 # for diagnostic, z is the rotation axis
  RT = 2 # for radiation therapy, y is the rotation axis
  
class TrajectoryType(Enum):
  HELICAL = 1
  DYNAMIC = 2

class GeometryType(Enum):
  CBFPD = 1
  CBCurved = 2
  PB = 3

class PSFModelling(Enum):
  IDEAL = 1
  MULTIRAY = 2
  ADVANCED = 3
  
class ProjectorType(Enum):
  ANALYTICAL = 1
  
#------ Imaging Chain
class ImagingChainType(Enum):
  IDEAL = 1
  SPECTRAL = 2
  ADVANCED = 3

class ProjectionType(Enum):
  LINE_INTEGRAL = 1
  ATTENUATION = 2
  COUNTS = 3

class RotationDirection(Enum):
  RHR = 1 # right hand rule
  LHR = 2 # left hand rule
  
class DetectorLayout(Enum):
  U1V1 = 1 # reference
  U1V0 = 2
  U0V1 = 3
  U0V0 = 4
  V1U1 = 5
  V1U0 = 6
  V0U1 = 7
  V0U0 = 8

class QuadricShape(Enum):
  SPHERE = 0
  ELLIPSOID = 1
  CYLINDER = 2
  CONE = 3
  HYPERBOLOID1 = 4
  HYPERBOLOID2 = 5
  TORUS = 6
  
#------ Phantom
class PhantomType(Enum):
  ANALYTICAL = 1
  DIGITAL = 2

#------ Data
class DataTypeInt(Enum):
  UINT16 = 20
  INT16 = 21
  UINT32 = 40
  INT32 = 41
  FLOAT32 = 42
  FLOAT64 = 82

class DataTypeDtype(Enum):
  UINT16 = np.uint16
  INT16 = np.int16
  UINT32 = np.uint32
  INT32 = np.int32
  FLOAT32 = np.float32
  FLOAT64 = np.float64
  
#------ Header
class HeaderType(Enum):
  VOLUME_BASIC = 0
  PROJECTION_BASIC = 10
  
class CoordinateSystem(Enum):
  WORLD_CORDINATE = 1
  SOURCE_CORDINATE = 2
  DETECTOR_CORDINATE = 3

  
  
