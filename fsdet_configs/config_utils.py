"""Apply local config overrides without MMConfig file inheritance."""
import copy


def merge_local(base, overrides):
    overrides = copy.deepcopy(overrides)
    if overrides.pop('_delete_', False):
        return overrides
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict):
            value = copy.deepcopy(value)
            replace = value.pop('_delete_', False)
            if not replace and isinstance(result.get(key), dict):
                value = merge_local(result[key], value)
        result[key] = value
    return result
