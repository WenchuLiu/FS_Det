# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataloader, build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .fair import FairDataset
from .fair1m import FAIR1M2CocoDataset, FAIR1M2Dataset, FAIR1MtrisourceDataset
from .dota_1_5 import DOTADataset15
from .samplers import MultiSourceSampler
from .dota_ import Dota_Dataset
from .sardet import SARDetDataset
from .sardet_hbb import SARDet_hbb
from .sar_aircraft import SARAircraftDataset
from .sardet_hbb_trisource import SARDet_hbb_trisource
from .sardet_dota_ifred import SARDetDotaIFRedDataset
from .dronevehicle import DroneVehicle_Dataset
from .dior import DIORDataset, DIORBaseDataset, DIORNovelDataset

from .sardet_dota_ifred import SARDetDotaIFRedNovelDataset
from .sardet_hbb_trisource import SARDet_hbb_novel_trisource

from .sardet_hbb_trisource import SARDet_hbb_novel_fine_aircraft_trisource
from .sardet_dota_ifred import SARDetDotaIFRedNovelFineAircraftDataset

from .sardet_hbb_trisource import SARDet_hbb_novel_srsdd_trisource
from .sardet_dota_ifred import SARDetDotaIFRedNovelSRSDDDataset

from .proj_data import Proj_aircraft_trisource
from .proj_data import Proj_ship_trisource
from .proj_data import Proj_vehicle_trisource
from .proj_data import SARDetDotaIFRedProjAircraft
from .proj_data import SARDetDotaIFRedProjShip
from .proj_data import SARDetDotaIFRedProjVehicle

from .sardet_hbb_trisource import SARDet_base, SARDet_base_split2, SARDet_ALL



__all__ = ['SARDataset', 'DOTADataset', 'build_dataset', 'HRSCDataset', 
           'FairDataset', 'FAIR1M2Dataset', 'FAIR1M2CocoDataset', 'FAIR1MtrisourceDataset', 'DOTADataset15', 'MultiSourceSampler', 'build_dataloader',
           'Dota_Dataset', 'SARDetDataset', 'SARDet_hbb', 'SARAircraftDataset', 'DroneVehicle_Dataset', 
           'DIORDataset', 'DIORBaseDataset', 'DIORNovelDataset',
           'SARDetDotaIFRedDataset', 'SARDet_hbb_trisource', 'SARDetDotaIFRedNovelDataset', 
           'SARDet_hbb_novel_trisource','SARDet_hbb_novel_fine_aircraft_trisource',
           'SARDetDotaIFRedNovelFineAircraftDataset', 
           'SARDet_hbb_novel_srsdd_trisource', 'SARDetDotaIFRedNovelSRSDDDataset',
           'Proj_aircraft_trisource', 'Proj_ship_trisource', 'Proj_vehicle_trisource',
           'SARDetDotaIFRedProjAircraft', 'SARDetDotaIFRedProjShip', 'SARDetDotaIFRedProjVehicle',
           'SARDet_base', 'SARDet_base_split2', 'SARDet_ALL',
            # 'SARDetDotaIFRedProj_RGB_Aircraft', 'SARDetDotaIFRedProj_RGB_Ship',
            # 'SARDetDotaIFRedProj_RGB_Vehicle', 'Proj_rgb_aircraft_trisource',
            # 'Proj_rgb_ship_trisource', 'Proj_rgb_vehicle_trisource'
           ]
