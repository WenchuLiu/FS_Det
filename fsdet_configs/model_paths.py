"""Resolve bundled model asset paths from the imported source packages."""

import os
from pathlib import Path


def get_sar_clip_model_dir():
    """Return the SAR-CLIP asset directory, with an optional local override."""
    override = os.environ.get('FSDET_SARCLIP_DIR')
    if override:
        return str(Path(override).expanduser().resolve())

    import mmrotate.models.sar_clip as sar_clip

    return str(
        Path(sar_clip.__file__).resolve().parent /
        'model_configs' / 'ViT-B-16')


def get_method3_text_encoder_config():
    """Load method3 RSCLIP exclusively from a local model directory."""
    model_path = os.environ.get('FSDET_METHOD3_CLIP_MODEL_PATH')
    if not model_path:
        raise ValueError(
            'Set FSDET_METHOD3_CLIP_MODEL_PATH to a local Hugging Face '
            'format RSCLIP model directory. Method3 never downloads weights.')
    processor_path = (
        os.environ.get('FSDET_METHOD3_CLIP_PROCESSOR_PATH') or model_path)
    model_dir = Path(model_path).expanduser().resolve()
    processor_dir = Path(processor_path).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f'Method3 RSCLIP model directory not found: {model_dir}')
    if not processor_dir.is_dir():
        raise FileNotFoundError(
            f'Method3 RSCLIP processor directory not found: {processor_dir}')
    return {
        'text_encoder_type': 'huggingface',
        'clip_model_name': str(model_dir),
        'clip_processor_name': str(processor_dir),
    }
