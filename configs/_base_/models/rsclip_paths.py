import os

clip_model_path = os.path.abspath(
    os.path.expanduser(os.environ.get('FSDET_METHOD3_CLIP_MODEL_PATH', '')))
clip_processor_path = os.path.abspath(
    os.path.expanduser(
        os.environ.get('FSDET_METHOD3_CLIP_PROCESSOR_PATH')
        or clip_model_path))
method3_text_encoder = os.environ.get(
    'FSDET_METHOD3_TEXT_ENCODER', 'huggingface').strip().lower()
if method3_text_encoder not in ('huggingface', 'hf', 'rgbclip', 'rgb_clip',
                                'sarclip', 'sar_clip'):
    raise ValueError('Unsupported FSDET_METHOD3_TEXT_ENCODER.')
if (method3_text_encoder not in ('sarclip', 'sar_clip') and
        not os.environ.get('FSDET_METHOD3_CLIP_MODEL_PATH')):
    raise ValueError(
        'Set FSDET_METHOD3_CLIP_MODEL_PATH to a local RSCLIP model directory.')
if method3_text_encoder not in ('sarclip', 'sar_clip') and (
        not os.path.isdir(clip_model_path) or
        not os.path.isdir(clip_processor_path)):
    raise FileNotFoundError('RSCLIP model/processor directory does not exist.')
