from typing import Tuple, TYPE_CHECKING
from . import utils
from .argh import Constraint

if TYPE_CHECKING:
    from .system import System3D


def periodic(robot: 'System3D', but_not: Tuple[str, ...], but_not_vel: Tuple[str, ...] = tuple()):
    """
    Make all position and velocity states in `robot` periodic, except for the
    positions (q) defined in `but_not` and the velocities (dq) in `but_not_vel`
    ```
    >>> periodic(robot, but_not=('x',))
    >>> periodic(robot, but_not=('x', 'y'), but_not_vel=('x', 'y'))
    ```
    """
    nfe, ncp = len(robot.m.fe), len(robot.m.cp)

    utils.remove_constraint_if_exists(robot.m, 'periodic_q')
    utils.remove_constraint_if_exists(robot.m, 'periodic_dq')

    # periodic positions
    qs = [
        (link['q'][1, ncp, q], link['q'][nfe, ncp, q])
        for link in robot.links
        for q in link.pyomo_sets['q_set']
        if q not in but_not
    ]
    robot.m.add_component(
        'periodic_q', Constraint(
            range(len(qs)), rule=lambda m, i: qs[i][0] == qs[i][1])
    )

    # periodic velocities
    dqs = [
        (link['dq'][1, ncp, q], link['dq'][nfe, ncp, q])
        for link in robot.links
        for q in link.pyomo_sets['q_set']
        if q not in but_not_vel
    ]
    robot.m.add_component(
        'periodic_dq', Constraint(
            range(len(dqs)), rule=lambda m, i: dqs[i][0] == dqs[i][1])
    )
