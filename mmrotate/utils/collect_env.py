# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_basic_env

import mmrotate
from .git import get_git_hash


def collect_env():
    """Collect environment information."""
    env_info = collect_basic_env()
    git_hash = get_git_hash(digits=7)
    env_info['MMRotate'] = mmrotate.__version__ + (
        '+' + git_hash if git_hash else '')
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
