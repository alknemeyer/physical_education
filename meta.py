from typing import Iterable, Set, Union

VALID_META: Set[str] = set((
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
