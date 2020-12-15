# https://stackoverflow.com/questions/15853469/putting-current-class-as-return-type-annotation
# can't use this while supporting python 3.6 (PyPy 7.3), have to use strings for now
# from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, List, TYPE_CHECKING, cast
from typing_extensions import TypedDict
import sympy as sp
import numpy as np
import pyomo.environ as pyo
from sympy import Matrix as Mat
from . import utils, symdef, visual
from .template import Node
from .variable_list import VariableList

PlotConfig = TypedDict('PlotConfig', shape=str, linewidth=float, color=str)


class Link3D:
    def __init__(self, name: str, aligned_along: str, *,
                 start_I: Optional[Mat] = None,
                 base: bool = False,
                 meta: Iterable[str] = tuple(),
                 mass: Optional[float] = None,
                 length: Optional[float] = None,
                 radius: Optional[float] = None
                 ):
        """
        Define a 3D link. Assumes the link is vertical, ie aligned with the z-axis

        name:
            name of the link
        start_I:
            the location of the start of the link in inertial coordinates
        base:
            whether this link is the/a base link
        meta:
            information which qualitatively describes a link, like 'spine', 'leg', etc
        """
        # assert sum([top_I is not None, bottom_I is not None, base is True]) == 1,\
        #     'Only one of top_I, bottom_I or base can be specified'
        assert (start_I is not None) ^ (base is True), \
            'A link must either be given an offset, or be a base link'

        acceptable_args = ('+x', '+y', '+z', '-x', '-y', '-z')
        assert aligned_along in acceptable_args,\
            f'Align the link along the x, y or z axes. Got {aligned_along}, must be one of {acceptable_args}'

        self.mass_sym = sp.Symbol('m_{%s}' % name)
        self.length_sym = sp.Symbol('L_{%s}' % name)
        self.radius_sym = sp.Symbol('r_{%s}' % name)

        self.mass, self.length, self.radius = mass, length, radius

        self.name = name
        self.is_base = base

        self.q, self.dq, self.ddq = symdef.make_ang_syms(name)
        angles = self.q  # used for defining rotation matrix

        if base is True:
            xyz, dxyz, ddxyz = symdef.make_xyz_syms(name)
            self.q = Mat([*xyz,   *self.q])
            self.dq = Mat([*dxyz,  *self.dq])
            self.ddq = Mat([*ddxyz, *self.ddq])
        else:
            xyz = None

        # rotation matrix from body to inertial
        self.Rb_I = utils.euler_321(*angles).T

        I_len = (self.mass_sym * self.radius_sym**2)/2
        I_wid = (self.mass_sym * self.length_sym**2)/12
        if aligned_along.endswith('x'):
            self.inertia = sp.diag(I_len, I_wid, I_wid)
            offset_b = Mat([-self.length_sym/2, 0, 0])
        elif aligned_along.endswith('y'):
            self.inertia = sp.diag(I_wid, I_len, I_wid)
            offset_b = Mat([0, -self.length_sym/2, 0])
        else:
            self.inertia = sp.diag(I_wid, I_wid, I_len)
            offset_b = Mat([0, 0, -self.length_sym/2])

        if aligned_along.startswith('-'):
            offset_b = -offset_b

        # positions in body and inertial axes
        if self.is_base:
            assert xyz is not None  # a type hint for mypy
            self.Pb_I = xyz
            self.top_I = self.Pb_I + self.Rb_I @ offset_b
        else:
            assert start_I is not None  # a type hint for mypy
            self.top_I = start_I
            self.Pb_I = self.top_I - self.Rb_I @ offset_b

        self.bottom_I = self.Pb_I - self.Rb_I @ offset_b

        # defining other things for later use
        self.constraint_forces: List[sp.Symbol] = []
        self.angle_constraints: List['sp.Expression'] = []
        self._plot_config: PlotConfig = {
            'shape': 'line',
            'linewidth': 2.,
            'color': 'tab:orange',
        }

        self.aligned_along = aligned_along

        from typing import Set as _Set
        from . import meta as _meta
        self.meta: _Set[str] = set(meta)
        assert _meta.is_valid(self.meta)

        self.nodes: Dict[str, Node] = {}

    def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
        if len(self.angle_constraints) > 0:
            J_c = Mat(self.angle_constraints).jacobian(q)
            Fr = Mat(self.constraint_forces)
            Q = J_c.T @ Fr
        else:
            Q = sp.zeros(len(q), 1)  # type: ignore

        for node in self.nodes.values():
            Q += node.calc_eom(q, dq, ddq)

        return Q

    def add_hookes_joint(self, otherlink: 'Link3D', about: str) -> 'Link3D':
        """Add a hooke's joint about axis `about` of `self`
        >>> link_body.add_hookes_joint(link_UFL, about='xy')
        """
        assert all(ax in ('x', 'y', 'z') for ax in about) and len(about) == 2

        axes = Mat([1, 0, 0]), Mat([0, 1, 0]), Mat([0, 0, 1])
        vec1 = axes['xyz'.index(about[0])]
        vec2 = axes['xyz'.index(about[1])]

        ang_constr = (
            (self.Rb_I @ vec1).dot(otherlink.Rb_I @ vec2)
        )
        self.angle_constraints.append(ang_constr)

        self.constraint_forces.append(
            sp.Symbol('F_{r/%s/%s}' % (self.name, otherlink.name))
        )

        return self

    def add_revolute_joint(self, otherlink: 'Link3D', about: str) -> 'Link3D':
        """Adds a revolute connection about axis `about` of `self`

        >>> link_UFL.add_revolute_joint(link_LFL, about='y')
        """
        assert about in ('x', 'y', 'z')
        axes = Mat([1, 0, 0]), Mat([0, 1, 0]), Mat([0, 0, 1])
        constraint_axis = axes['xyz'.index(about)]
        other_axes = [mat for idx, mat in enumerate(axes)
                      if idx != 'xyz'.index(about)]

        self.angle_constraints.extend([
            (self.Rb_I @ constraint_axis).dot(otherlink.Rb_I @ other_axes[0]),
            (self.Rb_I @ constraint_axis).dot(otherlink.Rb_I @ other_axes[1]),
        ])

        self.constraint_forces.extend(
            sp.symbols('F_{r/%s/%s/:2}' % (self.name, otherlink.name))
        )

        return self

    def add_vars_to_pyomo_model(self, m: pyo.ConcreteModel) -> None:
        for attr in ('mass', 'length', 'radius'):
            assert isinstance(getattr(self, attr), float),\
                f'The {attr} for {self.name} must be set to a float'

        if self.is_base:
            q_set = pyo.Set(initialize=('x', 'y', 'z', 'phi', 'theta', 'psi'),
                            name='q_set', ordered=True)
        else:
            q_set = pyo.Set(initialize=('phi', 'theta', 'psi'),
                            name='q_set', ordered=True)

        Fr_set = pyo.Set(initialize=range(len(self.constraint_forces)),
                         name='Fr_set', ordered=True)

        q = pyo.Var(m.fe, m.cp, q_set, name='q', bounds=(-100, 100))
        dq = pyo.Var(m.fe, m.cp, q_set, name='dq', bounds=(-1000, 1000))
        # the bounds are more or less arbitrary!
        ddq = pyo.Var(m.fe, m.cp, q_set, name='ddq', bounds=(-10_000, 10_000))

        # constraint (reaction) forces (multiples of BW)
        Fr = pyo.Var(m.fe, m.cp, Fr_set, name='Fr', bounds=(-10, 10))

        mass = pyo.Param(initialize=self.mass, name='mass')
        length = pyo.Param(initialize=self.length, name='length')
        radius = pyo.Param(initialize=self.radius, name='radius')

        self.pyomo_params: Dict[str, pyo.Param] = {
            'mass': mass, 'length': length, 'radius': radius,
        }
        self.pyomo_sets: Dict[str, pyo.Set] = {
            'q_set': q_set, 'Fr_set': Fr_set,
        }
        self.pyomo_vars: Dict[str, pyo.Var] = {
            'q': q, 'dq': dq, 'ddq': ddq, 'Fr': Fr,
        }

        utils.add_to_pyomo_model(m, self.name, [
            self.pyomo_params.values(),
            self.pyomo_sets.values(),
            self.pyomo_vars.values(),
        ])

        if self.is_base:
            for fe in m.fe:
                for cp in m.cp:
                    q[fe, cp, 'z'].setlb(0)

        for node in self.nodes.values():
            node.add_vars_to_pyomo_model(m)

    def add_equations_to_pyomo_model(self,
                                     sp_variables: List[sp.Symbol],
                                     pyo_variables: VariableList,
                                     collocation: str):
        q = self.pyomo_vars['q']
        dq = self.pyomo_vars['dq']
        ddq = self.pyomo_vars['ddq']
        q_set = self.pyomo_sets['q_set']

        m = q.model()

        # add collocation
        from .collocation import get_collocation_func
        collocation_func = get_collocation_func(collocation)

        setattr(m, self.name + '_collocation_q',
                pyo.Constraint(m.fe, m.cp, q_set, rule=collocation_func(q,  dq)))
        setattr(m, self.name + '_collocation_dq',
                pyo.Constraint(m.fe, m.cp, q_set, rule=collocation_func(dq, ddq)))

        # while we're here, also make equations for the top and bottom of the link:
        self.lineanimation = visual.LineAnimation(
            self.top_I, self.bottom_I, sp_variables)

        for node in self.nodes.values():
            node.add_equations_to_pyomo_model(
                sp_variables, pyo_variables, collocation
            )

    def get_pyomo_vars(self, fe: int, cp: int) -> List:
        """fe, cp are one-based!"""
        # NB: keep in sync with get_sympy_vars()!!
        node_vars = []
        for node in self.nodes.values():
            node_vars.extend(node.get_pyomo_vars(fe, cp))

        v = self.pyomo_vars
        p = self.pyomo_params
        frset = self.pyomo_sets['Fr_set']
        return [
            *v['q'][fe, cp, :],
            *v['dq'][fe, cp, :],
            *v['ddq'][fe, cp, :],
            # using *v['Fr'][fe, cp, :] caused a bug in a newer pyomo version
            # maybe switch back at some point?
            *[v['Fr'][fe, cp, f] for f in frset],
            p['mass'], p['length'], p['radius'],
            *node_vars,
        ]

    def get_sympy_vars(self) -> List[sp.Symbol]:
        # NB: keep in sync with get_pyomo_vars()!!
        node_vars = []
        for node in self.nodes.values():
            node_vars.extend(node.get_sympy_vars())

        return [
            *self.q,
            *self.dq,
            *self.ddq,
            *self.constraint_forces,
            self.mass_sym, self.length_sym, self.radius_sym,
            *node_vars,
        ]

    def __getitem__(self, varname: str) -> pyo.Var:
        return self.pyomo_vars[varname]

    def save_data_to_dict(self) -> Dict[str, Any]:
        q_set = self.pyomo_sets['q_set']

        Fr_set = self.pyomo_sets['Fr_set']

        return {
            'name': self.name,
            'is_base': self.is_base,
            'meta': self.meta,
            'mass': self.mass,
            'length': self.length,
            'radius': self.radius,
            'q': utils.get_vals(self.pyomo_vars['q'], (q_set,)),
            'dq': utils.get_vals(self.pyomo_vars['dq'], (q_set,)),
            'ddq': utils.get_vals(self.pyomo_vars['ddq'], (q_set,)),
            'Fr': utils.get_vals(self.pyomo_vars['Fr'], (Fr_set,)) if len(Fr_set) > 0 else [],
            'nodes': [node.save_data_to_dict() for node in self.nodes.values()],
        }

    def init_from_dict_one_point(self, data: Dict[str, Any],
                                 fed: int, cpd: int,
                                 fes: Optional[int] = None, cps: Optional[int] = None,
                                 **kwargs
                                 ) -> None:
        if fes is None:
            fes = fed - 1
        if cps is None:
            cps = cpd - 1

        assert self.name == data['name'] and self.is_base == data['is_base']
        for attr in ('meta', 'mass', 'length', 'radius'):
            if getattr(self, attr) != data[attr]:
                utils.warn(
                    f'Attribute "{attr}" of link "{self.name}" is not the same as the data: {getattr(self, attr)} != {data[attr]}')

        for qstr in ('q', 'dq', 'ddq'):
            for idx, state in enumerate(self.pyomo_sets['q_set']):
                utils.maybe_set_var(
                    self.pyomo_vars[qstr][fed, cpd, state], data[qstr][fes, cps, idx], **kwargs)

        if len(data['Fr']) > 0:
            for idx, F in enumerate(self.pyomo_sets['Fr_set']):
                utils.maybe_set_var(
                    self.pyomo_vars['Fr'][fed, cpd, F], data['Fr'][fes, cps, idx], **kwargs)

        for node, nodedata in zip(self.nodes.values(), data['nodes']):
            node.init_from_dict_one_point(
                nodedata, fed, cpd, fes, cps, **kwargs)

    # Optional[Union[Literal['line'],Literal['box']]]
    def plot_config(self, *, color: Optional[str] = None,
                    shape: Optional[str] = None,
                    linewidth: Optional[float] = None) -> 'Link3D':
        """Configuration for how this link should be plotted"""
        # TODO: Still thinking about what's actually useful here. Not sure if this is a
        # wise move - we are essentially just overwriting defaults, to some extent?
        if color is not None:
            self._plot_config['color'] = color

        if shape is not None:
            assert shape in ('line', 'box')
            self._plot_config['shape'] = shape
            raise NotImplementedError('Get round to adding this!')

        if linewidth is not None:
            self._plot_config['linewidth'] = linewidth

        return self

    def animation_setup(self, fig, ax, data: List[List[float]]):
        self.lineanimation.animation_setup(fig, ax, data)

        line = self.lineanimation.line
        line.set_linewidth(self._plot_config['linewidth'])
        line.set_color(self._plot_config['color'])

        for node in self.nodes.values():
            node.animation_setup(fig, ax, data)

    def animation_update(self, fig, ax, fe: Optional[int] = None,
                         t: Optional[float] = None, t_arr: Optional[np.ndarray] = None,
                         track: bool = False):
        self.lineanimation.animation_update(fig, ax, fe, t, t_arr, track)

        for node in self.nodes.values():
            node.animation_update(fig, ax, fe=fe, t=t, t_arr=t_arr)

    def cleanup_animation(self, fig, ax):
        self.lineanimation.cleanup_animation(fig, ax)

        for node in self.nodes.values():
            node.cleanup_animation(fig, ax)

    def plot(self) -> None:
        m = self.pyomo_vars['q'].model()

        q_set = self.pyomo_sets['q_set']
        q = utils.get_vals(self.pyomo_vars['q'], (q_set,))
        dq = utils.get_vals(self.pyomo_vars['dq'], (q_set,))
        ddq = utils.get_vals(self.pyomo_vars['ddq'], (q_set,))

        import matplotlib.pyplot as plt

        def _plt(var, title: str):
            # plot using multiple axes, as in this SO link:
            # https://stackoverflow.com/a/45925049
            # or this: https://matplotlib.org/gallery/api/two_scales.html
            if self.is_base:  # plot x,y,z separately to angles
                # typing this as 'Any' because the method accesses following give
                # false warnings otherwise...
                fig = plt.figure()
                ax1 = plt.subplot()
                ax1.plot(var[:, :3])
                ax1.legend(list(q_set)[:3])
                ax1.set_ylabel('positions [m]')
                ax1.set_xlabel('Finite element')

                ax2 = ax1.twinx()
                colors = [next(ax1._get_lines.prop_cycler)['color']
                          for _ in range(3)]
                for i, color in enumerate(colors):
                    ax2.plot(var[:, 3+i], color=color)
                ax2.legend(list(q_set)[3:])
                ax2.set_ylabel('angles [rad]')

                plt.grid(True)
                plt.title(title)
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()
            else:
                plt.plot(var)
                plt.legend(q_set)
                plt.grid(True)
                plt.title(title)
                plt.xlabel('Finite element')
                plt.ylabel('angles [rad]')
                plt.show()

        ncp = len(m.cp)

        _plt(q[:, ncp-1, :], f'State positions in {self.name}')
        _plt(dq[:, ncp-1, :], f'State velocities in {self.name}')
        _plt(ddq[:, ncp-1, :], f'State accelerations in {self.name}')

        try:
            Fr_set = self.pyomo_sets['Fr_set']
            Fr = utils.get_vals(self.pyomo_vars['Fr'], (Fr_set,))
            plt.plot(Fr[:, ncp-1, :])
            plt.legend(Fr_set)
            plt.grid(True)
            plt.title('Constraint forces in link ' + self.name)
            plt.xlabel('Finite element')
            plt.ylabel('$F_c$ [N/body_weight]')
            plt.show()
        except StopIteration:
            pass

        for node in self.nodes.values():
            node.plot()

    def __repr__(self) -> str:
        nodes = ',\n    '.join(str(node) for node in self.nodes.values())
        nodes = f', nodes=[\n    {nodes}]' if len(nodes) > 0 else ''
        return f'Link3D(name="{self.name}", isbase={self.is_base}, mass={self.mass}kg, length={self.length}m, radius={self.radius}m{nodes})'


if TYPE_CHECKING:
    from pyomo.core.base.indexed_component_slice import _IndexedComponent_slice


def constrain_rel_angle(m: pyo.ConcreteModel, constr_name: str,
                        lowerbound: float, angle1: '_IndexedComponent_slice',
                        angle2: '_IndexedComponent_slice', upperbound: float):
    """
    >>> constrain_rel_angle(robot.m, 'knee',
    ...                     0, thigh[:, :, 'theta'], calf[:, :, 'theta'], π/2)
    """
    # TODO: maybe switch to something like:
    # >>> diffs = [ang1-ang2 for ang1,ang2 in zip(angle1, angle2)]
    # or use this:
    # https://pyomo.readthedocs.io/en/stable/library_reference/kernel/constraint.html

    # what follows is some _horribly_ hacky code...
    # one iterable each for upper and lower bounds
    ang_1_up = iter(angle1)
    ang_1_lo = iter(angle1)
    ang_2_up = iter(angle2)
    ang_2_lo = iter(angle2)

    # _fe and _cp are unused
    # TODO: switch to pyomo.environ.inequality!
    def func(m: pyo.ConcreteModel, _fe, _cp, bound: str):
        if bound == '+':
            ang1 = next(ang_1_up)
            ang2 = next(ang_2_up)
            # make sure fe/cp match up for both links
            assert ang1.index()[:2] == ang2.index()[:2]
            return ang1 - ang2 <= upperbound
        elif bound == '-':
            ang1 = next(ang_1_lo)
            ang2 = next(ang_2_lo)
            # make sure fe/cp match up for both links
            assert ang1.index()[:2] == ang2.index()[:2]
            return lowerbound <= ang1 - ang2

    name = f'rel_angle_{constr_name}'
    setattr(m, name, pyo.Constraint(m.fe, m.cp, ('+', '-'), rule=func))

    # def func(m: pyo.ConcreteModel, ang1, ang2, bound: str):
    #     # make sure fe/cp match up for both links
    #     assert ang1.index()[:2] == ang2.index()[:2]
    #     if bound == '+':
    #         return ang1 - ang2 <= upperbound
    #     elif bound == '-':
    #         return lowerbound <= ang1 - ang2

    # name = f'rel_angle_{constr_name}'
    # setattr(m, name, pyo.Constraint(angle1, angle2, ('+', '-'), rule=func))
