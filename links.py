# https://stackoverflow.com/questions/15853469/putting-current-class-as-return-type-annotation
# can't use this while supporting python 3.6 (PyPy 7.3), have to use strings for now
# from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, List, Tuple, TypeVar, Union
import sympy as sp, numpy as np, itertools
from sympy import Matrix as Mat
from pyomo.environ import (
    ConcreteModel, Set, Var, Constraint, Param
)
from . import utils, symdef, foot
from .variable_list import VariableList

# later:
# TODO: switch to **kwargs thingy from fast.ai for Foot3D: @utils.delegates(foot.Foot3D)
# TODO: add flags which show which stage the model is at:
#       creating_sympy_model, created_sympy_model, creating_pyomo_model, created_pyomo_model
#       optimization_has_run
# TODO: think of a different way to track stuff during animation

class Link3D():
    __slots__ = [
        'name', 'is_base',
        'mass_sym', 'length_sym', 'radius_sym', 'mass', 'length', 'radius',
        'q', 'dq', 'ddq', 'Rb_I', 'inertia', 'Pb_I', 'top_I', 'bottom_I',
        'input_torques', 'torques_on_body', 'constraint_forces', 'angle_constraints',
        '_plot_config', 'meta', 'foot',
        'pyomo_sets', 'pyomo_vars', 'pyomo_params',
        'top_I_func', 'bottom_I_func', 'line', 'plot_data',
    ]
    def __init__(self, name: str, aligned_along: str, *,
                 start_I: Optional[Mat] = None, base: bool = False, meta: Iterable[str] = tuple(),
                 mass: Optional[float] = None, length: Optional[float] = None, radius: Optional[float] = None):
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
        assert aligned_along in ('+x', '+y', '+z', '-x', '-y', '-z'),\
            'Align the link along the x, y or z axes'

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
            self.q   = Mat([  *xyz,   *self.q])
            self.dq  = Mat([ *dxyz,  *self.dq])
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
            self.Pb_I = xyz
            self.top_I = self.Pb_I + self.Rb_I @ offset_b
        else:
            self.top_I = start_I
            self.Pb_I = self.top_I - self.Rb_I @ offset_b

        self.bottom_I = self.Pb_I - self.Rb_I @ offset_b

        # defining other things for later use
        self.input_torques: List = []
        self.torques_on_body = Mat([0, 0, 0])

        self.constraint_forces: List = []
        self.angle_constraints: List = []
        self._plot_config: Dict[str] = {'shape': 'line',
                                        'linewidth': 2,
                                        'color': 'tab:orange'}

        self.meta: Set[str] = set(meta)

    def calc_eom(self, q, dq, ddq, Ek, Ep, M, C, G) -> None:
        if self.has_foot():
            self.foot.calc_eom(q, dq, ddq, Ek, Ep, M, C, G)
    
    def add_hookes_joint(self, otherlink: 'Link3D', about: str) -> 'Link3D':
        """Add a hooke's joint about axis `about` of `self`
        >>> link_body.add_hookes_joint(link_UFL, about='xy')
        """
        assert all(ax in ('x', 'y', 'z') for ax in about)
        ang_constr = (self.Rb_I @ Mat([1, 0, 0])).dot(otherlink.Rb_I @ Mat([0, 1, 0]))
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
        axes = Mat([1,0,0]), Mat([0,1,0]), Mat([0,0,1])
        constraint_axis = axes['xyz'.index(about)]
        other_axes = [mat for idx,mat in enumerate(axes) if idx != 'xyz'.index(about)]
        
        self.angle_constraints.extend([
            (self.Rb_I @ constraint_axis).dot(otherlink.Rb_I @ other_axes[0]),
            (self.Rb_I @ constraint_axis).dot(otherlink.Rb_I @ other_axes[1]),
        ])
        
        self.constraint_forces.extend(
            sp.symbols('F_{r/%s/%s/:2}' % (self.name, otherlink.name))
        )

        return self
    
    def add_input_torques_at(self, otherlink: 'Link3D', about: str) -> 'Link3D':
        """ Add input torques between two links, about axis `about` if `self`
        >>> link_body.add_input_torques_at(link_UFL, about='xy')
        TODO: THINK MORE ABOUT WHAT THIS REALLY MEANS! The second torque should really be rotated
        """
        assert all(ax in 'xyz' for ax in about)
        
        τ = Mat([*sp.symbols(r'\tau_{%s/%s/:%s}' % (self.name, otherlink.name, len(about)))])
        self.input_torques.extend(τ)
        for idx, ax in enumerate(about):
            constraint_axis = Mat([0,0,0])
            constraint_axis['xyz'.index(ax)] = 1
            torque = τ[idx] * self.Rb_I @ Mat(constraint_axis)
            self.torques_on_body += torque
            otherlink.torques_on_body -= torque
        return self

    def add_foot(self, at: str, **kwargs) -> 'Link3D':
        """
        `at` is usually `self.bottom_I`. Eg:
        >>> link.add_foot(at='bottom')
        """
        assert at in ('top', 'bottom'), \
            "Can only add ground contacts at top or bottom of foot"
        assert not hasattr(self, 'foot'), \
            "For now, a link can only have one ground contact point"

        Pb_I = self.bottom_I if at == 'bottom' else self.top_I
        self.foot = foot.Foot3D(self.name + '_foot', Pb_I, **kwargs)
        return self
    
    def has_foot(self) -> bool:
        return hasattr(self, 'foot')

    def add_vars_to_pyomo_model(self, m: ConcreteModel) -> None:
        for attr in ('mass', 'length', 'radius'):
            assert isinstance(getattr(self, attr), float),\
                f'The {attr} for {self.name} must be set to a float'

        if self.is_base:
            q_set = Set(initialize=('x', 'y', 'z', 'phi', 'theta', 'psi'), name='q_set', ordered=True)
        else:
            q_set = Set(initialize=('phi', 'theta', 'psi'), name='q_set', ordered=True)
        
        Tc_set = Set(initialize=range(len(self.input_torques)), name='Tc_set', ordered=True)
        Fr_set = Set(initialize=range(len(self.constraint_forces)), name='Fr_set', ordered=True)

        q = Var(m.fe, m.cp, q_set, name='q', bounds=(-100, 100))
        dq = Var(m.fe, m.cp, q_set, name='dq', bounds=(-1000, 1000))
        ddq = Var(m.fe, m.cp, q_set, name='ddq', bounds=(-5000, 5000)) # the bounds are more or less arbitrary!

        # input torque and constraint (reaction) forces
        # (multiples of BW)
        Tc = Var(m.fe, Tc_set,       name='Tc', bounds=(-1, 1))
        Fr = Var(m.fe, m.cp, Fr_set, name='Fr', bounds=(-5, 5))

        mass   = Param(initialize=self.mass, name='mass')
        length = Param(initialize=self.length, name='length')
        radius = Param(initialize=self.radius, name='radius')

        self.pyomo_params: Dict[str,Param] = {
            'mass': mass, 'length': length, 'radius': radius,
        }
        self.pyomo_sets: Dict[str,Set] = {
            'q_set': q_set, 'Tc_set': Tc_set, 'Fr_set': Fr_set,
        }
        self.pyomo_vars: Dict[str,Var] = {            
            'q': q, 'dq': dq, 'ddq': ddq, 'Tc': Tc, 'Fr': Fr,
        }

        for v in itertools.chain(self.pyomo_params.values(),
                                 self.pyomo_sets.values(),
                                 self.pyomo_vars.values()):
            newname = f'{self.name}_{v}'
            assert not hasattr(m, newname), f'The pyomo model already has a variable with the name "{newname}"'
            setattr(m, newname, v)
        
        if self.is_base:
            for fe in m.fe:
                for cp in m.cp:
                    q[fe,cp,'z'].setlb(0)
        
        if self.has_foot():
            self.foot.add_vars_to_pyomo_model(m)
    
    def add_equations_to_pyomo_model(self, sp_variables, pyo_variables: VariableList, collocation: str):
        q   = self.pyomo_vars['q']
        dq  = self.pyomo_vars['dq']
        ddq = self.pyomo_vars['ddq']
        q_set = self.pyomo_sets['q_set']

        m = q.model()

        # add collocation
        from .collocation import get_collocation_func
        collocation_func = get_collocation_func(collocation)
        
        setattr(m, self.name + '_collocation_q',
                Constraint(m.fe, m.cp, q_set, rule=collocation_func( q,  dq)))
        setattr(m, self.name + '_collocation_dq',
                Constraint(m.fe, m.cp, q_set, rule=collocation_func(dq, ddq)))

        # while we're here, also make equations for the top and bottom of the link:
        self.top_I_func = utils.lambdify_EOM(self.top_I, sp_variables)
        self.bottom_I_func = utils.lambdify_EOM(self.bottom_I, sp_variables)

        if self.has_foot():
            self.foot.add_equations_to_pyomo_model(sp_variables, pyo_variables, collocation)

    def get_pyomo_vars(self, fe: int, cp: int) -> List:
        """fe, cp are one-based!"""
        # NB: keep in sync with get_sympy_vars()!!
        foot_vars = self.foot.get_pyomo_vars(fe, cp) if self.has_foot() else []

        v = self.pyomo_vars
        p = self.pyomo_params
        return [
            *v['q'][fe,cp,:],
            *v['dq'][fe,cp,:],
            *v['ddq'][fe,cp,:],
            *v['Tc'][fe,:],
            *v['Fr'][fe,cp,:],
            p['mass'], p['length'], p['radius'],
            *foot_vars,
        ]
    
    def get_sympy_vars(self) -> List[sp.Symbol]:
        # NB: keep in sync with get_pyomo_vars()!!
        foot_vars = self.foot.get_sympy_vars() if self.has_foot() else []
        return [
            *self.q,
            *self.dq,
            *self.ddq,
            *self.input_torques,
            *self.constraint_forces,
            self.mass_sym, self.length_sym, self.radius_sym,
            *foot_vars,
        ]
    
    def __getitem__(self, varname: str) -> Var:
        return self.pyomo_vars[varname]
    
    def save_data_to_dict(self) -> Dict[str,Any]:
        q_set = self.pyomo_sets['q_set']
        Tc_set = self.pyomo_sets['Tc_set']
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
            'Tc': utils.get_vals(self.pyomo_vars['Tc'], (Tc_set,)) if len(Tc_set) > 0 else [],
            'Fr': utils.get_vals(self.pyomo_vars['Fr'], (Fr_set,)) if len(Fr_set) > 0 else [],
            'has_foot': self.has_foot(),
            'foot': self.foot.save_data_to_dict() if self.has_foot() else None,
        }
    
    def init_from_dict_one_point(self, data: Dict[str,Any], fed: int, cpd: int, fes: Optional[int] = None, cps: Optional[int] = None, **kwargs) -> None:
        if fes is None: fes = fed - 1
        if cps is None: cps = cpd - 1

        assert self.name == data['name'] and self.is_base == data['is_base']
        for attr in ('meta', 'mass', 'length', 'radius'):
            if getattr(self, attr) != data[attr]:
                utils.warn(f'Attribute "{attr}" of link "{self.name}" is not the same as the data: {getattr(self, attr)} != {data[attr]}') 
        
        for qstr in ('q', 'dq', 'ddq'):
            for idx, state in enumerate(self.pyomo_sets['q_set']):
                utils.maybe_set_var(self.pyomo_vars[qstr][fed,cpd,state], data[qstr][fes,cps,idx], **kwargs)
        
        if len(data['Tc']) > 0:
            for idx, T in enumerate(self.pyomo_sets['Tc_set']):
                utils.maybe_set_var(self.pyomo_vars['Tc'][fed,T], data['Tc'][fes,idx], **kwargs)

        if len(data['Fr']) > 0:
            for idx, F in enumerate(self.pyomo_sets['Fr_set']):
                utils.maybe_set_var(self.pyomo_vars['Fr'][fed,cpd,F], data['Fr'][fes,cps,idx], **kwargs)
        
        if self.has_foot():
            self.foot.init_from_dict_one_point(data['foot'], fed, cpd, fes, cps, **kwargs)

    # Optional[Union[Literal['line'],Literal['box']]]
    def plot_config(self, *, color: Optional[str] = None,
                             shape: Optional[str] = None,
                             linewidth: Optional[float] = None) -> 'Link3D':
        """Configuration for how this link should be plotted"""
        # TODO: Still thinking about what's actually useful here. Not sure if this is a
        # wise move - I essentially am just overwriting defaults, to some extent?
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
        self.line = ax.plot([], [], [],
                            linewidth=self._plot_config['linewidth'],
                            color=self._plot_config['color'])[0]

        self.plot_data = [np.empty((len(data), 3)),
                          np.empty((len(data), 3))]
        for idx, d in enumerate(data):  # xyz of top and bottom of the link
            self.plot_data[0][idx, :] = [f(*d) for f in self.top_I_func]
            self.plot_data[1][idx, :] = [f(*d) for f in self.bottom_I_func]

        if self.has_foot():
            self.foot.animation_setup(fig, ax, data)

    def animation_update(self, fig, ax, fe: Optional[int] = None,
                         t: Optional[float] = None, t_arr: Optional[np.ndarray] = None,
                         track: bool = False):
        if fe is not None:
            top_xyz    = self.plot_data[0][fe-1]
            bottom_xyz = self.plot_data[1][fe-1]
        else:
            top_xyz    = [np.interp(t, t_arr, self.plot_data[0][:,i]) for i in range(3)]
            bottom_xyz = [np.interp(t, t_arr, self.plot_data[1][:,i]) for i in range(3)]
        
        utils.update_3d_line(self.line, top_xyz, bottom_xyz)

        if track is True:
           lim = 1.0
           if top_xyz[2] < lim:
               top_xyz = [top_xyz[0], top_xyz[1], lim]
           utils.track_pt(ax, top_xyz, lim=lim)  # type: ignore
        
        if self.has_foot():
            self.foot.animation_update(fig, ax, fe=fe, t=t, t_arr=t_arr)
    
    def cleanup_animation(self, fig, ax):
        try:
            del self.line
        except:
            pass
        finally:
            if self.has_foot():
                self.foot.cleanup_animation(fig, ax)

    def plot(self) -> None:
        m = self.pyomo_vars['q'].model()

        q_set = self.pyomo_sets['q_set']
        q   = utils.get_vals(self.pyomo_vars['q'], (q_set,))
        dq  = utils.get_vals(self.pyomo_vars['dq'], (q_set,))
        ddq = utils.get_vals(self.pyomo_vars['ddq'], (q_set,))

        from matplotlib import pyplot as plt
        def _plt(var, title: str):
            # plot using multiple axes, as in this SO link:
            # https://stackoverflow.com/a/45925049
            # or this: https://matplotlib.org/gallery/api/two_scales.html
            if self.is_base:  # plot x,y,z separately to angles
                fig, ax1 = plt.subplots()
                ax1.plot(var[:,:3])
                ax1.legend(list(q_set)[:3])
                ax1.set_ylabel('positions [m]')
                ax1.set_xlabel('Finite element')
                
                ax2 = ax1.twinx()
                colors = [next(ax1._get_lines.prop_cycler)['color'] for _ in range(3)]
                for i,color in enumerate(colors):
                    ax2.plot(var[:,3+i], color=color)
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

        _plt(q[:,ncp-1,:], f'State positions in {self.name}')
        _plt(dq[:,ncp-1,:], f'State velocities in {self.name}')
        _plt(ddq[:,ncp-1,:], f'State accelerations in {self.name}')

        try:
            Tc_set = self.pyomo_sets['Tc_set']
            Tc = utils.get_vals(self.pyomo_vars['Tc'], (Tc_set,))
            plt.plot(Tc)
            plt.legend(Tc_set)
            plt.grid(True)
            plt.title('Input torques in link ' + self.name)
            plt.xlabel('Finite element')
            plt.ylabel('$T$ [Nm/body_weight]')
            plt.show()
        except StopIteration:
            pass
    
        try:
            Fr_set = self.pyomo_sets['Fr_set']
            Fr = utils.get_vals(self.pyomo_vars['Fr'], (Fr_set,))
            plt.plot(Fr[:,ncp-1,:])
            plt.legend(Fr_set)
            plt.grid(True)
            plt.title('Constraint forces in link ' + self.name)
            plt.xlabel('Finite element')
            plt.ylabel('$F_c$ [N/body_weight]')
            plt.show()
        except StopIteration:
            pass

        if self.has_foot():
            self.foot.plot()

    def __repr__(self) -> str:
        foot = f',\n    foot={self.foot})' if self.has_foot() else ')'
        return f'Link3D(name="{self.name}", isbase={self.is_base}, mass={self.mass}, length={self.length}, radius={self.radius}' + foot

class PrismaticLink3D(Link3D):
    def __init__(self, name: str, aligned_along: str, *, bounds: Tuple[float,float], **kwargs):
        utils.warn('Prismatic hasnt been tested yet! Remove this message if all seems okay!')

        assert 'length' in kwargs.keys()

        self.bounds = bounds
        self.Δlength = sp.Symbol(f'\\Delta\\ L_{name}')
        self.dΔlength = sp.Symbol(r'\dot{\Delta\ L_{%s}}' % name)
        self.ddΔlength = sp.Symbol(r'\ddot{\Delta\\ L_{%s}}' % name)
        length = self.Δlength + kwargs.pop('length')
        Link3D.__init__(self, name, aligned_along, **kwargs, length=length)

        self.q   = sp.Matrix([  *self.q,   self.Δlength])  # type: ignore
        self.dq  = sp.Matrix([ *self.dq,  self.dΔlength])  # type: ignore
        self.ddq = sp.Matrix([*self.ddq, self.ddΔlength])  # type: ignore
    
    def add_vars_to_pyomo_model(self, m: ConcreteModel) -> None:
        # NOTE: copied from Link3D above, but with the addition of 'ΔL' in q_set. No other differences
        raise NotImplementedError('Update this to keep in sync with Link3D!!')

        if self.is_base:
            q_set = Set(initialize=('x', 'y', 'z', 'phi', 'theta', 'psi', 'ΔL'), name='q_set', ordered=True)
        else:
            q_set = Set(initialize=('phi', 'theta', 'psi', 'ΔL'), name='q_set', ordered=True)
        
        Tc_set = Set(initialize=range(len(self.input_torques)), name='Tc_set', ordered=True)
        Fr_set = Set(initialize=range(len(self.constraint_forces)), name='Fr_set', ordered=True)

        q = Var(m.fe, m.cp, q_set, name='q', bounds=(-100, 100))
        dq = Var(m.fe, m.cp, q_set, name='dq', bounds=(-1000, 1000))
        ddq = Var(m.fe, m.cp, q_set, name='ddq', bounds=(-5000, 5000)) # the bounds are more or less arbitrary!

        # input torque and constraint (reaction) forces
        # (multiples of BW)
        Tc = Var(m.fe, Tc_set,       name='Tc', bounds=(-1, 1))
        Fr = Var(m.fe, m.cp, Fr_set, name='Fr', bounds=(-5, 5))

        self.pyomo_params: Dict[str,Param] = {}
        self.pyomo_sets: Dict[str,Set] = {
            'q_set': q_set, 'Tc_set': Tc_set, 'Fr_set': Fr_set
        }
        self.pyomo_vars: Dict[str,Var] = {            
            'q': q, 'dq': dq, 'ddq': ddq, 'Tc': Tc, 'Fr': Fr,
        }

        for v in itertools.chain(self.pyomo_params.values(),
                                 self.pyomo_sets.values(),
                                 self.pyomo_vars.values()):
            newname = f'{self.name}_{v}'
            assert not hasattr(m, newname), f'The pyomo model already has a variable with the name "{newname}"'
            setattr(m, newname, v)
        
        for fe in m.fe:
            for cp in m.cp:
                q[fe,cp,'ΔL'].setlb(self.bounds[0])
                q[fe,cp,'ΔL'].setub(self.bounds[1])
                if self.is_base:
                    q[fe,cp,'z'].setlb(0)
        
        if self.has_foot():
            self.foot.add_vars_to_pyomo_model(m)

    #def get_pyomo_vars(self, fe: int, cp: int) -> List:
        #return Link3D.get_pyomo_vars(self, fe, cp) + [self.pyomo_vars['ΔL'][fe,cp]]

    # def get_sympy_vars(self) -> List[sp.Symbol]:
    #     return Link3D.get_sympy_vars(self) + [self.Δlength, self.dΔlength, self.ddΔlength]

    def __repr__(self) -> str:
        return 'Prismatic' + Link3D.__repr__(self)



def constrain_rel_angle(m: ConcreteModel, constr_name: str,
                        lowerbound: float, angle1: Iterable,
                        angle2: Iterable, upperbound: float):
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
    def func(m: ConcreteModel, _fe, _cp, bound: str):
        if bound == '+':
            ang1 = next(ang_1_up)
            ang2 = next(ang_2_up)
            assert ang1.index()[:2] == ang2.index()[:2] # make sure fe/cp match up for both links
            return ang1 - ang2 <= upperbound
        elif bound == '-':
            ang1 = next(ang_1_lo)
            ang2 = next(ang_2_lo)
            assert ang1.index()[:2] == ang2.index()[:2] # make sure fe/cp match up for both links
            return lowerbound <= ang1 - ang2

    name = f'rel_angle_{constr_name}'
    setattr(m, name, Constraint(m.fe, m.cp, ('+', '-'), rule=func))

# # from quadruped
# import numpy as np
# def adjacent_corners(corners):
#     import itertools
#     idxs = []
#     for (c1, c2) in itertools.combinations(corners, 2):
#         cnt = 0
#         for i in range(3):
#             cnt += int(c1[i] == c2[i])
#         if cnt == 2:
#             idxs.append(True)
#         else:
#             idxs.append(False)
#     return idxs

# def get_points(i, q_data, q_sym, constants, P0_I, R0_I, PARAMS, legs):
#     import itertools
#     vals = {val: q_data[i,idx] for (idx,val) in enumerate(q_sym)}
#     vals.update(constants)
    
#     def _util(x, shape=(3,)):
#         return np.array(x.xreplace(vals)).reshape(*shape).astype(float)
    
#     xyz = _util(P0_I)
#     rot = _util(R0_I, shape=(3,3))
#     rxf, rxb, ry, rz = PARAMS['r_b_caud'], PARAMS['r_b_cran'],PARAMS['r_wh'], PARAMS['r_wh']
#     corners = list(itertools.product([-rxf, rxb], [-ry, ry], [-rz, rz]))
#     box_corners = [
#         xyz + rot @ np.array(corner) for corner in corners
#     ]
#     # only connect adjact links
#     idxs = np.array(adjacent_corners(corners))
#     edge_combinations = np.array(list(itertools.combinations(box_corners, 2)))[idxs]
    
#     # the head
#     from sympy import Matrix as Mat
#     head_position = P0_I + R0_I @ Mat([PARAMS['r_b_cran'] + 0.15, 0, 0.15])

#     return (
#          xyz, rot, edge_combinations,
#         [_util(leg['conn'][0]) for leg in legs], # hip connections
#         [_util(leg['conn'][1]) for leg in legs], # knee connections
#         [_util(leg['Pf_I'])    for leg in legs],
#          _util(head_position),
#     )

# # from hopper
# def get_points(i, R0_I, PARAMS, r, r0, r1, r2, conn_0, conn_1, Pf_I, _q, q):
#     import itertools
#     from math import isclose

#     vals = {val: _q[i,idx] for (idx,val) in enumerate(q)}
#     vals[r0] = PARAMS['r0']; vals[r1] = PARAMS['r1']; vals[r2] = PARAMS['r2']

#     def _util(x):
#         return np.array(x.xreplace(vals)).reshape(3).astype(float)

#     xyz = _util(conn_0)
#     rot = np.array(R0_I.xreplace(vals)).reshape(3,3).astype(float)
#     box_corners = [
#         xyz + rot @ np.array(corner) for corner in itertools.product([-r,r],[-r,r],[-r,r])
#     ]
#     # only connect adjacent links
#     edge_combinations = filter(lambda x: isclose(np.linalg.norm(x[0] - x[1]), 2*r),
#                                 itertools.combinations(box_corners, 2))

#     return _util(conn_0), _util(conn_1), _util(Pf_I), edge_combinations



### ATTIC
    # def remove_pyomo_stuff(self):
    #     del self.pyomo_params, self.pyomo_sets, self.pyomo_vars