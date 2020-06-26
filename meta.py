from typing import Iterable, Union

VALID_META = set((
    'spine', 'leg', 'tail',
    'front', 'back',
    'left', 'right',
    'thigh', 'calf',
))

def valid_meta(meta: Union[str, Iterable]):
    if isinstance(meta, str):
        return meta in VALID_META
    else:
        return all(valid_meta(m) for m in meta)
