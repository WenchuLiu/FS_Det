"""Compatibility entrypoint for the separate SAR/RGB datasets."""
import fsdet_configs.schema as _schema

_base_ = ['fsdet_18shot_' + _schema.ft_modality + '.py']
