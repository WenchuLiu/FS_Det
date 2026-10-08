from .dota import *
from .builder import ROTATED_DATASETS
from .sardet_hbb_trisource import SARDet_hbb_novel_trisource
from .sardet_dota_ifred import SARDetDotaIFRedDataset

@ROTATED_DATASETS.register_module()
class SARDetDotaIFRedProjAircraft(SARDetDotaIFRedDataset):
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
         'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
         'sar_Bomber', 'sar_Other', 'sar_ART', 'sar_Fighter')
        
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100),
               (0, 51, 153), (220, 220, 60),]

@ROTATED_DATASETS.register_module()
class SARDetDotaIFRedProjShip(SARDetDotaIFRedDataset):
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
         'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
         'sar_Civ', 'sar_Fri', 'sar_Auc', 'sar_Des', 'sar_Cru', 'sar_Amp',  'sar_Air')
        
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193), (0, 51, 153), (0, 51, 153), (0, 51, 153),
               (0, 51, 153), (0, 51, 153), (0, 51, 153), (0, 51, 153)]

@ROTATED_DATASETS.register_module()
class SARDetDotaIFRedProjVehicle(SARDetDotaIFRedDataset):
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
         'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
         'sar_Minibus', 'sar_Car', 'sar_Minivan', 'sar_ORV', 'sar_M1', 
         'sar_HM', 'sar_MV', 'sar_TH_RV', 'sar_A3', 'sar_Minican2', 
         'sar_TR_T')
        
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), ]

@ROTATED_DATASETS.register_module()
class Proj_aircraft_trisource(SARDet_hbb_novel_trisource):
	"""SAR Detection dataset with horizontal bounding boxes from three sources.

	This dataset supports custom evaluation metrics including detection rate,
	recognition rate, and false alarm rate.
	"""
	CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
		'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
		'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
		'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
		'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
		'sar_Bomber', 'sar_Other', 'sar_ART', 'sar_Fighter')
		
	PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
			(220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
			(0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
			(0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
			(220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
			(220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
			(0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100),
			(0, 51, 153), (220, 220, 60),]

	# ====================================================================
	# TODO: 请根据你的实验需求，正确地将上面的类别名称填充到这两个列表中
	# 注意: 这里的类别名称必须与 CLASSES 元组中的完全一致
	NOVEL_CLASSES = ('sar_Bomber', 'sar_Other', 'sar_ART', 'sar_Fighter')
	BASE_CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', )
	# ====================================================================

@ROTATED_DATASETS.register_module()
class Proj_ship_trisource(SARDet_hbb_novel_trisource):
	"""SAR Detection dataset with horizontal bounding boxes from three sources.
	
	This dataset supports custom evaluation metrics including detection rate,
	recognition rate, and false alarm rate.
	"""
	CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
		'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
		'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
		'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
		'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
		'sar_Civ', 'sar_Fri', 'sar_Auc', 'sar_Des', 'sar_Cru', 'sar_Amp',  'sar_Air')
		
	PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
				(220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
				(0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
				(0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
				(220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
				(0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
				(255, 193, 193), (0, 51, 153), (0, 51, 153), (0, 51, 153),
				(0, 51, 153), (0, 51, 153), (0, 51, 153), (0, 51, 153)]

	# ====================================================================
	# TODO: 请根据你的实验需求，正确地将上面的类别名称填充到这两个列表中
	# 注意: 这里的类别名称必须与 CLASSES 元组中的完全一致
	NOVEL_CLASSES = ('sar_Civ', 'sar_Fri', 'sar_Auc', 'sar_Des', 'sar_Cru', 'sar_Amp',  'sar_Air')
	BASE_CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', )
	# ====================================================================

@ROTATED_DATASETS.register_module()
class Proj_vehicle_trisource(SARDet_hbb_novel_trisource):
	"""SAR Detection dataset with horizontal bounding boxes from three sources.
	
	This dataset supports custom evaluation metrics including detection rate,
	recognition rate, and false alarm rate.
	"""
	CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
		'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
		'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
		'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
		'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
		'sar_Minibus', 'sar_Car', 'sar_Minivan', 'sar_ORV', 'sar_M1', 
		'sar_HM', 'sar_MV', 'sar_TH_RV', 'sar_A3', 'sar_Minican2', 
		'sar_TR_T')
		
	PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
				(220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
				(0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
				(0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
				(220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
				(0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
				(0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
				(220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
				(0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), ]

	# ====================================================================
	# TODO: 请根据你的实验需求，正确地将上面的类别名称填充到这两个列表中
	# 注意: 这里的类别名称必须与 CLASSES 元组中的完全一致
	NOVEL_CLASSES = ('sar_Minibus', 'sar_Car', 'sar_Minivan', 'sar_ORV', 'sar_M1', 
		'sar_HM', 'sar_MV', 'sar_TH_RV', 'sar_A3', 'sar_Minican2', 
		'sar_TR_T')
	BASE_CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', )
	# ====================================================================
