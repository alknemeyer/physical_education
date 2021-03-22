from typing import Any, Dict, List, Callable, Optional, Tuple, Union, Iterable, TYPE_CHECKING
from typing_extensions import Literal, TypedDict
import sympy as sp
import numpy as np

from pyomo.environ import (
    ConcreteModel, Param, Set, Var, Constraint,
)
from sympy import Matrix as Mat
from .system import System3D
from .links import Link3D
from . import utils
from . import visual


if TYPE_CHECKING:
    from .variable_list import VariableList

PlotConfig = TypedDict('PlotConfig', plot_forces=bool, force_scale=float)


class Foot3D:
    def __init__(self, name: str, Pb_I: Mat, nsides: Literal[4, 8],
                 GRFxy_max: float = 5.,
                 GRFz_max: float = 5.,
                 friction_coeff: Optional[float] = None):
        self.name = name
        self.nsides = nsides
        self.Pb_I = Pb_I
        self.friction_coeff = friction_coeff
        self.GRFxy_max = GRFxy_max
        self.GRFz_max = GRFz_max

        # the contact/friction stuff:
        self.D = friction_polygon(nsides)
        self.Lx = Mat(sp.symbols('L_{%s/x:%s}' % (name, nsides)))
        self.Lz = sp.Symbol('L_{%s/z}' % name)
        self.L = Mat(self.D).T @ self.Lx + Mat([0, 0, self.Lz])  # type: ignore

        self._plot_config: PlotConfig = {
            'plot_forces': True,
            'force_scale': 1,
        }

    def calc_eom(self, q: Mat, dq: Mat, ddq: Mat) -> Mat:
        self.Pb_I_vel = self.Pb_I.jacobian(q) @ dq

        jac_L = self.Pb_I.jacobian(q)
        return jac_L.T @ self.L

    def add_vars_to_pyomo_model(self, m: ConcreteModel):
        assert isinstance(self.friction_coeff, float), \
            f'The friction_coeff for {self.name} must be set to a float'

        # parameter and sets
        friction_coeff = Param(initialize=self.friction_coeff,
                               name='friction_coeff')
        xy_set = Set(initialize=('x', 'y'), name='xy_set', ordered=True)
        fric_set = Set(initialize=range(8), name='fric_set', ordered=True)

        GRFxy = Var(m.fe, m.cp, fric_set, name='GRFxy',
                    bounds=(0, self.GRFxy_max))
        GRFz = Var(m.fe, m.cp, name='GRFz',
                   bounds=(0, self.GRFz_max))

        # dummy vars equal to parts from EOM
        foot_height = Var(m.fe, m.cp, name='foot_height', bounds=(0, None))
        foot_xy_vel = Var(m.fe, m.cp, xy_set, name='foot_xyvel')
        gamma = Var(m.fe, m.cp, name='gamma (foot xy-velocity magnitude)',
                    bounds=(0, None))

        # penalty variables
        contact_penalty = Var(m.fe, name='contact_penalty', bounds=(0, 1))
        friction_penalty = Var(m.fe, name='friction_penalty', bounds=(0, 1))
        slip_penalty = Var(m.fe, fric_set, name='slip_penalty', bounds=(0, 1))

        self.pyomo_params: Dict[str, Param] = {
            'friction_coeff': friction_coeff
        }
        self.pyomo_sets: Dict[str, Set] = {
            'xy_set': xy_set,
            'fric_set': fric_set,
        }
        self.pyomo_vars: Dict[str, Var] = {
            'GRFxy': GRFxy, 'GRFz': GRFz,
            'foot_height': foot_height,
            'foot_xy_vel': foot_xy_vel,
            'gamma': gamma,
            'contact_penalty': contact_penalty,
            'friction_penalty': friction_penalty,
            'slip_penalty': slip_penalty,
        }

        utils.add_to_pyomo_model(m, self.name, [
            self.pyomo_params.values(),
            self.pyomo_sets.values(),
            self.pyomo_vars.values(),
        ])

    def get_pyomo_vars(self, fe: int, cp: int):
        """fe, cp are one-based!"""
        # NB: keep in sync with get_sympy_vars()!!
        v = self.pyomo_vars
        return [*v['GRFxy'][fe, cp, :], v['GRFz'][fe, cp]]

    def get_sympy_vars(self):
        # NB: keep in sync with get_pyomo_vars()!!
        return [*self.Lx, self.Lz]

    def save_data_to_dict(self) -> Dict[str, Any]:
        fric_set = self.pyomo_sets['fric_set']

        return {
            'name': self.name,
            'nsides': self.nsides,
            'friction_coeff': self.friction_coeff,
            'contact_penalty': utils.get_vals(self.pyomo_vars['contact_penalty']),
            'friction_penalty': utils.get_vals(self.pyomo_vars['friction_penalty']),
            'slip_penalty': utils.get_vals(self.pyomo_vars['slip_penalty'], (fric_set,)),
            'foot_height': utils.get_vals(self.pyomo_vars['foot_height'], tuple()),
            'GRFz': utils.get_vals(self.pyomo_vars['GRFz'], tuple()),
            'GRFxy': utils.get_vals(self.pyomo_vars['GRFxy'], (fric_set,)),
        }

    def init_from_dict_one_point(self, data: Dict[str, Any],
                                 fed: int, cpd: int,
                                 fes: Optional[int] = None, cps: Optional[int] = None,
                                 **kwargs) -> None:
        if fes is None:
            fes = fed - 1
        if cps is None:
            cps = cpd - 1

        assert self.name == data['name']
        for attr in ('nsides', 'friction_coeff'):
            if getattr(self, attr) != data[attr]:
                visual.warn(
                    f'Attribute "{attr}" of link "{self.name}" is not the same as the data: {getattr(self, attr)} != {data[attr]}')

        v = self.pyomo_vars
        utils.maybe_set_var(v['contact_penalty'][fed],
                            data['contact_penalty'][fes], **kwargs)
        utils.maybe_set_var(v['friction_penalty'][fed],
                            data['friction_penalty'][fes], **kwargs)

        utils.maybe_set_var(v['foot_height'][fed, cpd],
                            data['foot_height'][fes, cps], **kwargs)
        utils.maybe_set_var(v['GRFz'][fed, cpd],
                            data['GRFz'][fes, cps], **kwargs)

        for (idx, f) in enumerate(self.pyomo_sets['fric_set']):
            utils.maybe_set_var(v['slip_penalty'][fed, f],
                                data['slip_penalty'][fes, idx], **kwargs)
            utils.maybe_set_var(v['GRFxy'][fed, cpd, f],
                                data['GRFxy'][fes, cps, idx], **kwargs)

    def add_equations_to_pyomo_model(self,
                                     sp_variables: List[sp.Symbol],
                                     pyo_variables: 'VariableList',
                                     collocation: str):
        friction_coeff = self.pyomo_params['friction_coeff']
        m = friction_coeff.model()

        xy_set = self.pyomo_sets['xy_set']  # foot_xy_vel below
        fric_set = self.pyomo_sets['fric_set']

        GRFxy = self.pyomo_vars['GRFxy']
        GRFz = self.pyomo_vars['GRFz']

        foot_height = self.pyomo_vars['foot_height']
        foot_xy_vel = self.pyomo_vars['foot_xy_vel']
        gamma = self.pyomo_vars['gamma']

        contact_penalty = self.pyomo_vars['contact_penalty']
        friction_penalty = self.pyomo_vars['friction_penalty']
        slip_penalty = self.pyomo_vars['slip_penalty']

        self.foot_pos_func = utils.lambdify_EOM(self.Pb_I, sp_variables)
        self.foot_xy_vel_func = utils.lambdify_EOM(
            self.Pb_I_vel[:2], sp_variables)

        ncp = len(m.cp)

        def add_constraints(name: str, func: Callable, indexes: Iterable):
            setattr(m, self.name + '_' + name,
                    Constraint(*indexes, rule=func))

        def def_foot_height(m, fe, cp):    # foot height above z == 0 (xy-plane)
            if (fe == 1 and cp < ncp):
                return Constraint.Skip
            return foot_height[fe, cp] == self.foot_pos_func[2](pyo_variables[fe, cp])

        add_constraints('foot_height_constr', def_foot_height, (m.fe, m.cp))

        def def_foot_xy_vel(m, fe, cp, xy):  # foot velocity in xy-plane
            if (fe == 1 and cp < ncp):
                return Constraint.Skip
            i = 0 if xy == 'x' else 1
            return foot_xy_vel[fe, cp, xy] == self.foot_xy_vel_func[i](pyo_variables[fe, cp])

        add_constraints('foot_xy_vel_constr',
                        def_foot_xy_vel, (m.fe, m.cp, xy_set))

        def def_gamma(m, fe, cp, i):  # this sets gamma to the biggest of vx + vy
            if (fe == 1 and cp < ncp):
                return Constraint.Skip
            vx, vy = foot_xy_vel[fe, cp, 'x'], foot_xy_vel[fe, cp, 'y']
            return gamma[fe, cp] >= vx * self.D[i, 0] + vy * self.D[i, 1]

        add_constraints('gamma_constr', def_gamma, (m.fe, m.cp, fric_set))

        def def_friction_polyhedron(m, fe, cp):
            if (fe == 1 and cp < ncp):
                return Constraint.Skip
            return friction_coeff * GRFz[fe, cp] >= sum(GRFxy[fe, cp, :])

        add_constraints('friction_polyhedron_constr',
                        def_friction_polyhedron, (m.fe, m.cp))

        # complementarity equations
        # z[i+1]*GRFz[i] ≈ 0
        def def_contact_complementarity(m, fe):
            if fe < m.fe[-1]:
                α = sum(foot_height[fe+1, :])
                β = sum(GRFz[fe, :])
                return α * β <= contact_penalty[fe]
            else:
                return Constraint.Skip

        add_constraints('contact_complementarity_constr',
                        def_contact_complementarity, (m.fe,))

        # (μ * GRFz - Σ GRFxy) * γ ≈ 0
        def def_friction_complementarity(m, fe):
            α = friction_coeff * sum(GRFz[fe, :]) - sum(GRFxy[fe, :, :])
            β = sum(gamma[fe, :])
            return α * β <= friction_penalty[fe]

        add_constraints('friction_complementarity_constr',
                        def_friction_complementarity, (m.fe,))

        # GRFxy * (γ + dxyᵀ*Dᵢ) ≈ 0
        def def_slip_complementarity(m, fe, i):
            vx, vy = foot_xy_vel[fe, :, 'x'], foot_xy_vel[fe, :, 'y']
            α = sum(GRFxy[fe, :, i])
            β = sum(gamma[fe, :]) + sum(vx)*self.D[i, 0] + sum(vy)*self.D[i, 1]
            return α * β <= slip_penalty[fe, i]

        add_constraints('slip_complementarity_constr',
                        def_slip_complementarity, (m.fe, fric_set))

    def __getitem__(self, varname: str) -> Var:
        return self.pyomo_vars[varname]

    def penalty_sum(self):
        contact_penalty = self.pyomo_vars['contact_penalty']
        friction_penalty = self.pyomo_vars['friction_penalty']
        slip_penalty = self.pyomo_vars['slip_penalty']
        return (sum(contact_penalty[:])
                + 0.1*sum(friction_penalty[:])
                + 0.1*sum(slip_penalty[:, :]))

    def plot_config(self, *, plot_forces: Optional[bool] = None,
                    force_scale: Optional[float] = None) -> 'Foot3D':
        """Configuration for how this link should be plotted"""
        if plot_forces is not None:
            self._plot_config['plot_forces'] = plot_forces

        if force_scale is not None:
            self._plot_config['force_scale'] = force_scale

        return self

    def animation_setup(self, fig, ax, data: List[List[float]]):
        if self._plot_config['plot_forces'] is False:
            return

        self.has_line = False
        self.plot_data = np.empty((len(data), 6))
        cp = 1
        force_scale = self._plot_config['force_scale']

        for fe0, d in enumerate(data):  # fe0 = zero-based indexing
            fe = fe0 + 1                # fe = one-based indexed

            x, y, z = [f(d) for f in self.foot_pos_func]

            dx, dy = 0, 0
            for f in self.pyomo_sets['fric_set']:
                dx, dy = [dx, dy] + (self['GRFxy'][fe, cp, f].value
                                     * self.D[f, :2] * force_scale)

            dz = self.pyomo_vars['GRFz'][fe, cp].value * force_scale

            self.plot_data[fe0, :] = (x, y, z, dx, dy, dz)

    def animation_update(self, fig, ax,
                         fe: Optional[int] = None,
                         t: Optional[float] = None,
                         t_arr: Optional[np.ndarray] = None,
                         track: bool = False):
        if self._plot_config['plot_forces'] is False:
            return

        if self.has_line:
            self.line.remove()
            self.has_line = False

        if fe is not None:
            x, y, z, dx, dy, dz = self.plot_data[fe-1]
        else:
            assert t is not None and t_arr is not None
            x, y, z, dx, dy, dz = [
                np.interp(t, t_arr, self.plot_data[:, i]) for i in range(6)
            ]

        self.line = ax.quiver(
            x, y, z,    # <-- starting point of vector
            dx, dy, dz,  # <-- directions of vector
            arrow_length_ratio=0.15,
            color='red', alpha=.8, lw=1.5,
        )
        self.has_line = True

    def cleanup_animation(self, fig, ax):
        try:
            del self.line
        except:
            pass

    def plot(self, save_to: Optional[str] = None):
        m = self.pyomo_vars['GRFxy'].model()
        fe = list(range(len(m.fe)))

        # xy_set  = self.pyomo_sets['xy_set']  # foot_xy_vel below

        fric_set = self.pyomo_sets['fric_set']
        contact_penalty = utils.get_vals(self.pyomo_vars['contact_penalty'])
        friction_penalty = utils.get_vals(self.pyomo_vars['friction_penalty'])
        slip_penalty = utils.get_vals(self.pyomo_vars['slip_penalty'],
                                      (fric_set,))

        # TODO: plot GRFxy as an xy-thing? or individual line plots?
        # GRFxy = utils.get_vals(self.pyomo_vars['GRFxy'], (fric_set,))

        # foot_xy_vel = self.pyomo_vars['foot_xy_vel']
        # gamma = self.pyomo_vars['gamma']

        import matplotlib.pyplot as plt

        plt.plot(fe, contact_penalty,  label='contact')
        plt.plot(fe, friction_penalty, label='friction')
        for fric in fric_set:
            plt.plot(fe, slip_penalty[:, fric], label=f'slip_{fric}')

        plt.title(f'Penalties in foot {self.name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_to is not None:
            plt.gcf().savefig(f'{save_to}penalties-{self.name}.pdf')
        else:
            plt.show()
        plt.show()

        fig = plt.figure()
        ax1 = plt.subplot()
        plt.title('Foot height and ground reaction force in ' + self.name)
        plt.grid(True)

        foot_height = utils.get_vals(self.pyomo_vars['foot_height'], tuple())
        ax1.plot(fe, foot_height, label='Foot height')
        ax1.set_xlabel('Finite element')
        ax1.set_ylabel('Height [m]')

        ax2 = ax1.twinx()
        GRFz = utils.get_vals(self.pyomo_vars['GRFz'], tuple())
        # the color trick below is so that they don't both use the same color
        color = next(ax1._get_lines.prop_cycler)['color']
        ax2.plot(fe, GRFz, label='$GRFz$', color=color)
        ax2.set_ylabel('Force [Nm/body_weight]')

        fig.legend(loc='center')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.gcf().savefig('msc-plots/plt-'+self.name + '-footheight-force.pdf')
        plt.show()

    def __repr__(self) -> str:
        return f'Foot3D(name="{self.name}", nsides={self.nsides}, friction_coeff={self.friction_coeff})'


def add_foot(link, at: Literal['top', 'bottom'], name: Optional[str] = None, **kwargs):
    """
        `at` is usually `self.bottom_I`. Eg:
        >>> foot.add_foot(link, at='bottom')
        """
    name = utils.get_name(name, [link], 'foot')

    assert at in ('top', 'bottom'), \
       f"Can only add ground contacts at top or bottom of foot. Got: at={at}"

    Pb_I = link.bottom_I if at == 'bottom' else link.top_I
    foot = Foot3D(str(name), Pb_I, **kwargs)
    link.nodes[name] = foot
    return foot


def feet(robot_or_link: Union[System3D, Link3D]) -> List[Foot3D]:
    from typing import cast
    if isinstance(robot_or_link, System3D):
        robot = robot_or_link
        return [cast(Foot3D, node)
                for link in robot.links
                for node in link.nodes.values()
                if isinstance(node, Foot3D)]
    else:
        link = robot_or_link
        return [cast(Foot3D, node)
                for node in link.nodes.values()
                if isinstance(node, Foot3D)]


def feet_penalty(robot: 'System3D'):
    return sum(
        foot.penalty_sum()
        for foot in feet(robot)
    )


def friction_polygon(nsides: Literal[4, 8]) -> np.ndarray:
    if nsides == 4:
        # the four edges of a square - a very rough approximation to a friction cone
        return np.array([
            1, 0, 0,
            0, 1, 0,
            -1, 0, 0,
            0, -1, 0,
        ]).reshape(4, 3)

    elif nsides == 8:
        # for these numbers to make sense, think about moving around a square in the xy-plane,
        # going from an axis crossing, to an edge, to the next axis crossing, etc
        # this square gets normalized to approximate a unit circle
        D = np.array([
            1, 0, 0,
            1, 1, 0,
            0, 1, 0,
            -1, 1, 0,
            -1, 0, 0,
            -1, -1, 0,
            0, -1, 0,
            1, -1, 0,
        ]).reshape(8, 3)  # normalize by row
        return D / np.linalg.norm(D, axis=1).reshape(8, 1)  # type: ignore
    else:
        raise ValueError(
            'Only 4-sided and 8-sided friction polygons are implemented at the moment')


def interactively_set_timings(feet: Iterable[str], nfe: int, wait_until_all_set: bool = False, **kwargs) -> List:
    """
    Before using this, you'll likely need to enable `@jupyter-widgets/jupyterlab-manager`
    as a jupyterlab extension. See eg:
    https://stackoverflow.com/questions/36351109/ipython-notebook-ipywidgets-does-not-show

    >>> interactively_set_timings([link.foot.name for link in robot.links if link.has_foot()], nfe=50, time=0.35)
    """
    funcs = []
    for foot in feet:
        funcs.append(set_timing(description=foot, nfe=nfe, **kwargs))

    if wait_until_all_set is True:
        from time import sleep
        while not all(f()['done'] for f in funcs):
            sleep(0.5)
        return [f()['values'] for f in funcs]

    else:
        return funcs


def set_timing(nfe: int, *, time: Optional[float] = None, initial: Optional[Tuple] = None,
               description: str = 'Foot timing', width: str = '500px', when_done: str = 'disable'
               ) -> Callable[[], Dict[str, Union[bool, Tuple[int, int]]]]:
    """
    A simple example:
    >>> f = get_foot_timing(nfe=20, time=0.35)
    >>> from time import sleep
    >>> while not f()['done']: sleep(0.5)
    >>> print('The values are:', f()['values'])
    """
    from ipywidgets import SelectionRangeSlider, Button, HBox  # type: ignore

    assert when_done in ('close', 'disable')

    options = np.arange(nfe) + 1

    if time is not None:
        options = [(f'{o/nfe * time:.3}s', o) for o in options]
        # if initial is not None:
        #    initial = int(initial[0] * nfe/time), int(initial[1] * nfe/time)

    if initial is None:
        initial = (1, nfe)

    assert 1 <= initial[0] <= initial[1] <= nfe, f'Invalid initial range: {initial}'

    # for whatever reason, this is the only way
    # I've managed to stop PyLance from reporting
    # errors. Please fix!
    slider = SelectionRangeSlider(  # type: ignore
        value=initial,  # type: ignore
        options=options,  # type: ignore
        index=(0, len(options)-1),  # type: ignore
        description=description,  # type: ignore
        layout={'width': width},  # type: ignore
    )

    button = Button(
        description='Click when done',  # type: ignore
        disabled=False,  # type: ignore
        button_style='',  # type: ignore
        tooltip='Click me',  # type: ignore
        icon='check',  # type: ignore
    )

    done = False

    def finish_up(*args):
        nonlocal done
        done = True

        if when_done == 'close':
            slider.close()
            button.close()
        else:
            button.disabled = True
            slider.disabled = True

    button.on_click(finish_up)

    from IPython.display import display
    display(HBox([slider, button]))

    return lambda: {'done': done, 'values': slider.value}  # type: ignore


def prescribe_contact_order(feet: Iterable[Foot3D], ground_timings: Iterable[Tuple[int, int]],
                            min_foot_height: float = 0.005, min_GRFz: float = 0.001) -> None:
    """
    Prescribe `feet` to be on the ground using the ranges specified in `ground_timings`.
    `min_foot_height` sets the lower bound for feet during the flight phase, and `min_GRFz`
    does the same for the vertical ground reaction force during the stance phase.

    >>> feet = [link.foot for link in robot.links if link.has_foot()]
    >>> foot_order_vals = ((1, 9), (7, 15), (28, 37), (35, 43))  # corresponds to finite elements
    >>> list(zip(feet, foot_order_vals))
    [(Foot3D(name="LFL_foot", nsides=8), (1, 9)),
     (Foot3D(name="LFR_foot", nsides=8), (7, 15)),
     (Foot3D(name="LBL_foot", nsides=8), (28, 37)),
     (Foot3D(name="LBR_foot", nsides=8), (35, 43))]
    >>> prescribe_contact_order(feet, foot_order_vals)
    """
    def inclusive_range(start, stop): return range(start, stop+1)

    def foot_fix_util(foot: Foot3D, start: int, stop: int):
        m = foot['GRFz'].model()
        nfe = len(m.fe)
        GRFz = foot['GRFz']
        foot_height = foot['foot_height']

        # phase 1: flight
        for fe in inclusive_range(1, start-1):
            for cp in m.cp:
                foot_height[fe, cp].setlb(min_foot_height)
                GRFz[fe, cp].fix(0)

        # phase 2: stance
        for fe in inclusive_range(start+1, stop-1):
            for cp in m.cp:
                foot_height[fe, cp].fix(0)
                GRFz[fe, cp].setlb(min_GRFz)

        # phase 3: flight
        for fe in inclusive_range(stop+1, nfe):
            for cp in m.cp:
                foot_height[fe, cp].setlb(min_foot_height)
                GRFz[fe, cp].fix(0)

    for foot, (start, stop) in zip(feet, ground_timings):
        foot_fix_util(foot, start, stop)
