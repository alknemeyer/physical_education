from typing import List, Tuple, TYPE_CHECKING
from . import utils
from pyomo.environ import Constraint, Var

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
    assert robot.m is not None,\
        'robot does not have a pyomo model defined on it'

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


def straight_leg(upper: Var, lower: Var, fes: List[int], state: str):
    """
    >>> straight_leg(robot['thigh']['q'], robot['calf']['q'], [4, 15], 'theta')
    which does the equivalent of,
    >>> m.straight_leg_thigh_calf = Constraint([4, 15],
    ...     rule=lambda m, fe: robot['thigh']['q'][fe,ncp,'theta'] == robot['thigh']['q'][fe,ncp,'theta'])

    or for all legs of a quadruped:

    >>> for (touchdown, liftoff), (upper, lower) in zip(foot_order_vals, (('UFL', 'LFL'), ('UFR', 'LFR'), ('UBL', 'LBL'), ('UBR', 'LBR'))):  # keep in sync with feet!
    ...     straight_leg(robot[upper]['q'], robot[lower]['q'], [touchdown, liftoff+1])
    """
    m = upper.model()
    ncp = len(m.cp)

    name = f'straight_leg_{upper.name}_{lower.name}'

    utils.remove_constraint_if_exists(m, name)

    setattr(m, name, Constraint(fes,
                                rule=lambda m, fe: lower[fe, ncp, state] == upper[fe, ncp, state]))
