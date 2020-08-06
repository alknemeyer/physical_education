from typing import Iterable, Set, Union

VALID_META: Set[str] = set((
    'spine', 'leg', 'tail',
    'front', 'back',
    'left', 'right',
    'thigh', 'calf',
))


def is_valid(meta: Union[str, Iterable[str]]) -> bool:
    if isinstance(meta, str):
        return meta in VALID_META
    else:
        return all(is_valid(m) for m in meta)
