# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache
from pathlib import Path
import subprocess


@lru_cache(maxsize=1)
def _get_full_git_hash():
    """Return this source tree's commit hash, or an empty string if absent."""
    project_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--verify', 'HEAD'],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            universal_newlines=True,
            timeout=2)
    except (OSError, subprocess.SubprocessError):
        return ''

    return result.stdout.strip()


def get_git_hash(digits=None):
    """Return the local Git hash without emitting errors outside a repo."""
    git_hash = _get_full_git_hash()
    return git_hash[:digits] if digits is not None else git_hash
