"""A library for research into legged critters"""
__version__ = '0.1.2'

from . import (
    foot,
    leg,
    links,
    system,
    tasks,
    utils,
    models,
    drag,
    motor,
    visual,
    init_tools,
)

__all__ = [
    'foot',
    'leg',
    'links',
    'system',
    'tasks',
    'utils',
    'models',
    'drag',
    'motor',
    'visual',
    'init_tools'
]

# TODO: Rename library to something like rigid_body_traj_opt (import as rbt)
