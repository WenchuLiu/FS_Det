"""Compatibility entrypoint for the separate SAR/RGB schedules."""
import fsdet_configs.schema as _schema

_base_ = ['fsdet_finetune_18_' + _schema.ft_modality + '.py']
