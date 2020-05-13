from typing import Tuple
from .system import System3D
from . import utils
from pyomo.environ import Constraint

def periodic(robot: System3D, but_not: Tuple[str,...], but_not_vel: Tuple[str,...] = tuple()):
    """
    Make all position and velocity states in `robot` periodic, except for the
    positions (q) defined in `but_not` and the velocities (dq) in `but_not_vel`

    >>> periodic(robot, but_not=('x',))
    """
    nfe, ncp = len(robot.m.fe), len(robot.m.cp)

    if hasattr(robot.m, 'periodic_q') or hasattr(robot.m, 'periodic_dq'):
        utils.debug('Deleting previous periodic constraints')
        robot.m.del_component('periodic_q')
        robot.m.del_component('periodic_q_index')
        robot.m.del_component('periodic_dq')
        robot.m.del_component('periodic_dq_index')
    
    # periodic positions
    qs = [(link['q'][1,ncp,q], link['q'][nfe,ncp,q])
          for link in robot.links for q in link.pyomo_sets['q_set'] if q not in but_not
    ]
    robot.m.add_component('periodic_q',
                          Constraint(range(len(qs)), rule = lambda m,i: qs[i][0] == qs[i][1]))

    # periodic velocities
    dqs = [(link['dq'][1,ncp,q], link['dq'][nfe,ncp,q])
          for link in robot.links for q in link.pyomo_sets['q_set'] if q not in but_not_vel
    ]
    robot.m.add_component('periodic_dq',
                          Constraint(range(len(dqs)), rule = lambda m,i: dqs[i][0] == dqs[i][1]))

def feet_penalty(robot: System3D):
    return [link.foot.penalty_sum() for link in robot.links if link.has_foot()]
