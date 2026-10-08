"""SAR/RGB base datasets, using the shared schema."""
import fsdet_configs.schema as _schema
from configs._base_.datasets.fsdet_data import build_base_data as _build_base_data

data = _build_base_data()
total_images = _schema.sar_base_num_images + _schema.rgb_base_num_images
