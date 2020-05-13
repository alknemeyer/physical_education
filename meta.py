from typing import Iterable, Union

VALID_META = set((
    'spine', 'leg', 'tail',
    'front', 'back',
    'left', 'right',
))

def valid_meta(meta: Union[str, Iterable]):
    if not isinstance(meta, str):
        return all(valid_meta(m) for m in meta)
    
    if meta in VALID_META:
        return True
