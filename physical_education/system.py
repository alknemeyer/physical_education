from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Tuple, Union, TYPE_CHECKING
import sympy as sp
import numpy as np
from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, Constraint, ConstraintList,
)
from sympy import Matrix as Mat
from . import utils, variable_list, visual
from . import collocation as _collocation
from .utils import flatten

if TYPE_CHECKING:
    from .links import Link3D

T = TypeVar('T')


def getattrs(items: Iterable, attr: str) -> List:
    """
    Get the attribute `attr` from every element in `items`, which is usually a list of links. Eg:
    >>> getattrs(self.links, 'mass')
    """
    return [getattr(item, attr) for item in items]


class System3D:
    # m: Union[ConcreteModel, None] = None
    def __init__(self, name: str, links: List['Link3D']) -> None:
        self.name = name
        self.links = links
        self.m: Union[ConcreteModel, None] = None

    def add_link(self, link: 'Link3D') -> None:
        self.links.append(link)

    def get_state_vars(self) -> Tuple[Mat, Mat, Mat]:
        q = flatten(getattrs(self.links, 'q'))
        dq = flatten(getattrs(self.links, 'dq'))
        ddq = flatten(getattrs(self.links, 'ddq'))
        return Mat(q), Mat(dq), Mat(ddq)

    def calc_eom(self, *, simp_func: Callable[[T], T] = lambda x: x) -> None:
        q, dq, ddq = self.get_state_vars()

        Ek, Ep, _, _ = utils.calc_velocities_and_energies(
            getattrs(self.links, 'Pb_I'),
            getattrs(self.links, 'Rb_I'),
            getattrs(self.links, 'mass_sym'),
            getattrs(self.links, 'inertia'),
            q, dq, g=9.81
        )
        Ek = simp_func(Ek)
        Ep = simp_func(Ep)

        M, C, G = utils.manipulator_equation(Ek, Ep, q, dq)
        M = simp_func(M)
        C = simp_func(C)
        G = simp_func(G)

        # get angular constraints
        angle_constraints = simp_func(Mat(
            flatten(link.angle_constraints for link in self.links)))
        Fr = Mat(flatten(link.constraint_forces for link in self.links))

        # foot stuff
        from . import foot
        feet = foot.feet(self)

        Q = sp.zeros(*q.shape)
        for link in self.links:
            Q += link.calc_eom(q, dq, ddq)

        B = simp_func(Q)

        from . import motor
        self.force_scale = sp.Symbol('F_{scale}')
        to_sub = {force: force*self.force_scale for force in [  # TODO: this should be done by the links themselves!
            *flatten(torque.input_torques for torque in motor.torques(self)),
            *Fr,
            *flatten(foot.Lx for foot in feet),
            *[foot.Lz for foot in feet],
        ]}

        eom = M @ ddq + C + G - B
        eom = simp_func(Mat([*eom, *angle_constraints]).xreplace(to_sub))

        # eom_c = M @ ddq + G - B
        # eom_c = simp_func(Mat([*eom_c, *angle_constraints]).xreplace(to_sub))

        self.sp_variables: List[sp.Symbol] = flatten(
            link.get_sympy_vars() for link in self.links
        )

        utils.info(f'Number of operations in EOM is {sp.count_ops(eom)}')

        # TODO: the lambdifying step actually takes quite long -- any way to speed it up?
        from pyomo.environ import atan
        func_map = {
            'sqrt': lambda x: (x+1e-9)**(1/2),
            'atan': atan,
            'atan2': lambda y, x: 2 * atan(y/((x**2 + y**2 + 1e-9)**(1/2) + x))
        }
        self.eom_f = utils.lambdify_EOM(
            eom,
            self.sp_variables + [self.force_scale],
            func_map=func_map
        )

    def make_pyomo_model(self, nfe: int, collocation: str, total_time: float,
                         scale_forces_by: Optional[float] = None,
                         vary_timestep_within: Optional[Tuple[float, float]] = None,
                         presolve_no_C: bool = False,
                         include_dynamics: bool = True) -> None:
        """
        vary_timestep_within:
            the upper and lower bounds for how much the unscaled timestep can change.
            Eg: given a timestep of t=50ms and vary_timestep_within=(0.8, 1.2),
            the timestep can vary from 40ms to 60ms

        presolve_no_C:
            whether the model should be pre-solved without the centrifugal forces
        """
        _collocation.check_collocation_method(collocation)

        if presolve_no_C:
            raise NotImplementedError(
                'Both EOM and EOM no C are saved to the model, but EOM no C is not used')

        self.m = m = ConcreteModel(name=self.name)

        # time and collocation
        ncp = _collocation.get_ncp(collocation)
        m.fe = RangeSet(nfe)
        m.cp = RangeSet(ncp, name=collocation)
        m.hm0 = Param(initialize=total_time/nfe)

        if vary_timestep_within is not None:
            m.hm = Var(m.fe, initialize=1.0, bounds=vary_timestep_within)
        else:
            m.hm = Param(m.fe, initialize=1.0)

        # add constraints using pyomo_model.constraints.add( Constraint )
        m.constraints = ConstraintList()

        # 1: each link/body defines its own pyomo variables
        for link in self.links:
            link.add_vars_to_pyomo_model(m)

        # 2: the variables are all combined
        self.pyo_variables = variable_list.VariableList(m, self.links)

        # 3: the equations of motion are all added
        for link in self.links:
            link.add_equations_to_pyomo_model(
                self.sp_variables, self.pyo_variables, collocation)

        if include_dynamics is True:
            if scale_forces_by is None:
                total_mass = sum(link.mass for link in self.links)
                scale_forces_by = total_mass
            m.force_scale = Param(initialize=scale_forces_by)

            @m.Constraint(m.fe, m.cp, range(len(self.eom_f)))
            def EOM_constr(m, fe, cp, i):
                return self.eom_f[i]([*self.pyo_variables[fe, cp], m.force_scale]) == 0 \
                    if not (fe == 1 and cp < ncp) else Constraint.Skip
        else:
            utils.warn('Not including dynamics (EOM_constr) in model')

    def save_data_to_dict(self, description: str) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': description,
            'repr': self.__repr__(),
            'nfe': len(self.m.fe),
            'ncp': len(self.m.cp),
            'hm': utils.get_vals(self.m.hm),
            'hm0': float(self.m.hm0.value),
            'links': [link.save_data_to_dict() for link in self.links],
        }

    def save_data_to_file(self, filename: str, description: str, overwrite_existing: bool = False) -> None:
        import dill
        import os.path
        if os.path.isfile(filename) and overwrite_existing is False:
            raise FileExistsError

        with open(filename, 'wb') as f:
            dill.dump(self.save_data_to_dict(description), f, recurse=True)

    def init_from_dict_one_point(self, data: Dict[str, Any], fed: int, cpd: int, fes: Optional[int] = None, cps: Optional[int] = None, **kwargs):
        """fed and cpd are destination (ie pyomo, and 1-indexed), fes and cps are source (ie dictionary, and 0-indexed)"""
        assert self.m is not None

        if fes is None:
            fes = fed - 1
        if cps is None:
            cps = cpd - 1

        # TODO: ignore this warning when accounting for differing number of finite elements
        if not self.m.hm0.value == data['hm0']:
            utils.warn(
                f'init_from_dict_one_point: self.hm0 = {self.m.hm0.value} != {data["hm0"]} = data["hm0"]', once=True)

        if utils.has_variable_timestep(self.m):
            utils.maybe_set_var(self.m.hm[fed], data['hm'][fes], **kwargs)

        for link, linkdata in zip(self.links, data['links']):
            link.init_from_dict_one_point(
                linkdata, fed, cpd, fes, cps, **kwargs)

    def init_from_dict(self, data: Dict[str, Any], **kwargs):
        for fed, cpd in self.indices(one_based=True):
            self.init_from_dict_one_point(data, fed, cpd, **kwargs)

    def init_from_file(self, filename: str, **kwargs):
        import dill
        # this could just be:
        # >>> self.init_from_dict(dill.load(open(filename, 'rb')), **kwargs)
        with open(filename, 'rb') as f:
            data = dill.load(f)

        self.init_from_dict(data, **kwargs)

    def indices(self, *, one_based: bool, skipfirst: bool = True) -> List[Tuple[int, int]]:
        return utils.get_indexes(len(self.m.fe), len(self.m.cp), one_based=one_based, skipfirst=skipfirst)

    def __getitem__(self, linkname: str) -> 'Link3D':
        for link in self.links:
            if link.name == linkname:
                return link

        raise KeyError(
            f'No link with name "{linkname}". Available links are: {", ".join(link.name for link in self.links)}')

    def plot_keyframes(self,
                       keyframes: List[int],
                       view_along: Union[Tuple[float, float], str],
                       plot3d_config: Dict = {},
                       save_to: Optional[str] = None,
                       ground_color: str = 'lightgray',
                       lims: Optional[Tuple[Tuple, Tuple, Tuple]] = None):
        # need to import this to get 3D plots working, for some reason
        from mpl_toolkits import mplot3d

        # typing this as 'Any' because the method accesses following give
        # false warnings otherwise...
        fig, ax, add_ground = visual.plot3d_setup(
            scale_plot_size=False, **plot3d_config)
        if lims is not None:
            x, y, z = lims
            ax.set_xlim(*x)
            ax.set_ylim(*y)
            ax.set_zlim(*z)

        visual.set_view(ax, along=view_along)

        ncp = len(self.m.cp)

        cp = ncp
        data: List[List[float]] = [
            [v.value for v in self.pyo_variables[fe, cp]]
            for fe in self.m.fe
        ]

        try:
            add_ground((ax.get_xlim(), ax.get_ylim()), color=ground_color)
            for fe in keyframes:
                # iterate in reverse as matplotlib plots that last things on top, and we want
                # the body on top. Luckily, this also sorts out left vs right for the 3D
                # quadruped model, but it's a bad hack in general :/
                # a better approach would be to sort the order of the links such that those
                # furthest from the camera are plotted first
                # the best would obviously be if matplotlib's 3D plotting was improved, or if
                # we switch libraries
                for link in self.links[::-1]:
                    link.animation_setup(fig, ax, data)
                    link.animation_update(fig, ax, fe=fe, track=False)

            if save_to is not None:
                fig.savefig(save_to)

        except Exception as e:
            visual.error(f'Interrupted keyframes due to error: {e}')

        finally:
            for link in self.links:
                link.cleanup_animation(fig, ax)
            del fig, ax

    def animate(self,
                view_along: Union[Tuple[float, float], str],
                t_scale: float = 1.,
                camera: Optional[Union[Tuple[float, float],
                                       Tuple[float, float, float]]] = None,
                lim: Optional[float] = None,
                plot3d_config: Dict = {},
                lims: Optional[Tuple[Tuple, Tuple, Tuple]] = None,
                track: Optional[str] = None,
                dt: Optional[float] = None,
                save_to: Optional[str] = None):
        # need to import this to get 3D plots working, for some reason
        from mpl_toolkits import mplot3d
        import matplotlib.animation
        from matplotlib import pyplot as plt

        assert self.m is not None, \
            'robot does not have a pyomo model defined on it'

        # typing this as 'Any' because the method accesses following give
        # false warnings otherwise...
        fig, ax, add_ground = visual.plot3d_setup(
            scale_plot_size=False, **plot3d_config)

        if lims is not None:
            x, y, z = lims
            ax: Any
            ax.set_xlim(*x)
            ax.set_ylim(*y)
            ax.set_zlim(*z)

        visual.set_view(ax, along=view_along)
        ground = add_ground(((0, 0), (0, 0)))

        nfe = len(self.m.fe)
        ncp = len(self.m.cp)

        cp = ncp
        data: List[List[float]] = [
            [v.value for v in self.pyo_variables[fe, cp]]
            for fe in self.m.fe
        ]

        for link in self.links:
            link.animation_setup(fig, ax, data)

        if camera is not None and lim is not None:
            visual.track_pt(ax, camera, lim)

        def progress_bar(proportion: float, width: int = 80):
            import sys
            # a mini progress bar
            num_done = int(round(proportion * width))
            if num_done == width:
                sys.stdout.write(' '*width + '\r')
            else:
                sys.stdout.write('+'*num_done + '-'*(width-num_done) + '\r')

        if dt is None:
            def _animate(fe: int):  # fe is one-based
                progress_bar(((fe-1)*ncp + cp)/(nfe*ncp))

                for link in self.links:
                    link.animation_update(
                        fig, ax, fe=fe, track=(track == link.name))

                nonlocal ground
                ground.remove()
                ground = add_ground((ax.get_xlim(), ax.get_ylim()))
        else:
            if not utils.has_variable_timestep(self.m):
                t_arr = np.cumsum(
                    np.array([self.m.hm[fe] for fe in self.m.fe]))
            else:  # the t_arr below is variable step
                t_arr = np.cumsum(utils.get_vals(self.m.hm)) * self.m.hm0.value

            def _animate(t: float):  # t is a float from 0 to total_time
                progress_bar(t/frames[-1])

                for link in self.links:
                    link.animation_update(
                        fig, ax, t=t, t_arr=t_arr, track=(track == link.name))

                nonlocal ground
                ground.remove()
                ground = add_ground((ax.get_xlim(), ax.get_ylim()))

        # if you get impatient and cancel an animation while it's being made,
        # then try to clone a model, you get an error from matplotlib about
        # certain things not being cloneable. So -> try/catch/finally block
        try:
            if not utils.has_variable_timestep(self.m):
                t_sum = sum(self.m.hm[fe]
                            for fe in self.m.fe if fe != nfe) * self.m.hm0.value
            else:
                t_sum = sum(
                    self.m.hm[fe].value for fe in self.m.fe if fe != nfe) * self.m.hm0.value

            interval_ms = 1000*t_scale * \
                (t_sum / nfe / ncp if dt is None else dt)
            frames = [fe for fe in self.m.fe] if dt is None else np.arange(
                start=0, stop=t_sum+dt, step=dt)
            anim = matplotlib.animation.FuncAnimation(
                fig, _animate, frames=frames, interval=interval_ms
            )

            plt.close(anim._fig)

            if save_to is not None:
                anim.save(save_to)
            elif utils.in_ipython():
                from IPython.core.display import display, HTML
                display(HTML(anim.to_html5_video()))
            else:
                fig.show()

        except Exception as e:
            progress_bar(nfe, ncp)
            visual.error(f'Interrupted animation due to error: {e}')

        finally:
            for link in self.links:
                link.cleanup_animation(fig, ax)
            del fig, ax

    def plot(self, plot_links: bool = True) -> None:
        assert self.m is not None
        
        if utils.has_variable_timestep(self.m):
            from matplotlib import pyplot as plt
            plt.title('Timestep size vs finite element')
            data = 1000*utils.get_vals(self.m.hm) * self.m.hm0.value
            plt.plot(data)
            plt.ylabel('Timestep size [ms]')
            plt.ylim([0, max(data)*1.1])
            plt.show()

        if plot_links:
            for link in self.links:
                link.plot()

    def __repr__(self) -> str:
        child_links = '\n  '.join(str(link) + ',' for link in self.links)
        return f'System3D(name="{self.name}", [\n  {child_links}\n])'

    def post_solve(self, costs: Optional[Dict[str, Any]] = None, detailed: bool = False, tol: float = 1e-6):
        from .foot import feet_penalty
        from pyomo.environ import value as pyovalue
        print('Total cost:', pyovalue(self.m.cost))

        if costs is not None:
            for k, v in costs.items():
                print(f'-- {k}: {pyovalue(v)}')

        foot_pen = pyovalue(feet_penalty(self))
        if foot_pen > 1e-3:
            visual.error('Foot penalty seems to be unsolved')

        if detailed is True:
            from pyomo.util.infeasible import log_infeasible_constraints
            print('Infeasible constraints:')
            log_infeasible_constraints(self.m, tol=tol)

            from pyomo.util.infeasible import log_infeasible_bounds
            print('Infeasible bounds:')
            log_infeasible_bounds(self.m, tol=tol)

    # def init_from_robot(self, source: 'System3D', interpolation: str = 'linear'):
    #     """
    #     Initialize a model from a solved one, interpolating values if needed
    #     """
    #     # TODO: change this to be more like the dict approach above! Or just use that instead!
    #     utils.debug('Only use init_from_robot with two robots which are identical, but'
    #                 ' with possibly differing finite elements and collocation points')
    #     utils.warn('init_from_robot: haven\'t yet figured out what to do with `None`s at'
    #                ' the beginning of things')
    #     utils.warn('init_from_robot: some things (like dummy variables) aren\'t yet'
    #                ' copied across')
    #     # attempt to catch errors early on. TODO: think of more quick tests
    #     assert len(self.links) == len(source.links)
    #     assert interpolation == 'linear'

    #     import math

    #     def valid_num(num): return not (
    #         num is None or math.isnan(num) or math.isinf(num)
    #     )

    #     # TODO: refactor into a different func?
    #     # def supersample(new_x, old_x, data):
    #     #     return np.interp(
    #     #         np.linspace(0, 1, num=new_x),
    #     #         np.linspace(0, 1, num=old_x),
    #     #         data
    #     #     )

    #     # get data from source
    #     hm = utils.get_vals(source.m.hm)

    #     # interpolate
    #     hm2 = np.interp(
    #         np.linspace(0, 1, num=len(self.m.fe)),
    #         np.linspace(0, 1, num=len(source.m.fe)),
    #         hm)

    #     # add to destination model
    #     for fe in self.m.fe:
    #         self.m.hm[fe].value = hm2[fe-1] if valid_num(hm2[fe-1]) else 1.0

    #     for destlink, srclink in zip(self.links, source.links):
    #         # get data from source. using `astype` to get rid of None's
    #         srcdata = np.array([
    #             var.value for fe in source.m.fe
    #             for cp in source.m.cp
    #             for var in srclink.get_pyomo_vars(fe, cp)
    #         ]).reshape((len(source.m.fe) * len(source.m.cp), -1)).astype(float)

    #         # x interpolation points (replace with cumsum of hm!)
    #         x_orig = np.linspace(0, 1, num=len(source.m.fe) * len(source.m.cp))
    #         x_dest = np.linspace(0, 1, num=len(self.m.fe) * len(self.m.cp))

    #         destdata = [[destlink.get_pyomo_vars(fe, cp) for cp in self.m.cp]
    #                     for fe in self.m.fe]

    #         # interpolate
    #         interpolated_data = np.zeros(
    #             (x_dest.shape[0], srcdata.shape[1]))  # type: ignore
    #         for varidx in range(srcdata.shape[1]):
    #             interped = np.interp(x_dest, x_orig, srcdata[:, varidx])
    #             # replace the np.nan's with None, for pyomo
    #             interpolated_data[:, varidx] = np.where(
    #                 np.isnan(interped), None, interped)  # type: ignore

    #         # add to destination model
    #         ncp = len(self.m.cp)
    #         skipped_vars = []
    #         for fe, cp in self.indices(one_based=False):
    #             for varidx, var in enumerate(destdata[fe][cp]):
    #                 num = interpolated_data[fe*ncp + cp, varidx]

    #                 if var.is_fixed():
    #                     skipped_vars.append(var)
    #                     continue
    #                 if not valid_num(num):
    #                     print('skipping invalid number:', num,
    #                           'at index', varidx, 'for variable', var)
    #                     continue

    #                 var.value = num

    #         if len(skipped_vars) > 0:
    #             from textwrap import shorten
    #             utils.debug(
    #                 shorten(f'skipped variables because they are fixed: {skipped_vars}', width=100))


    # # Figure out how to handle constraints etc with this presolve approach. Eg min distance, etc
    # def presolve(self, collocation: str, nfe: int, setup_func: Callable[['System3D'], None], no_C: bool,
    #              make_pyomo_model_kwargs: dict = {}, default_solver_kwargs: dict = {}):
    #     """
    #     Create a new (simpler) model, solve it, then copy over the (interpolated) values
    #     to this model. Note -- this function is definitely worth reading through in detail, especially
    #     to know what defaults it chooses for you! If you're not sure, rather don't use this!

    #     Example:
    #     >>> robot.make_pyomo_model(nfe=50, collocation='radau', total_time=1.5)
    #     >>> def add_task(robot):
    #     ...     add_pyomo_constraints(robot)
    #     ...     high_speed_stop(robot, initial_vel=20.0)
    #     >>> add_task(robot)
    #     >>> solver_kwargs = {'OF_print_frequency_time': 10, 'OF_hessian_approximation': 'limited-memory', }
    #     >>> robot.presolve(nfe=10, collocation='euler', setup_func=add_task, no_C=True,
    #     ...                default_solver_kwargs=solver_kwargs)
    #     >>> ret = utils.default_solver(max_mins=30, solver='ma86', **solver_kwargs).solve(robot.m, tee=True)
    #     """
    #     import copy

    #     new_sys = copy.deepcopy(self)
    #     new_sys.make_pyomo_model(nfe=nfe, collocation=collocation, presolve_no_C=no_C,
    #                              total_time=float(
    #                                  self.m.hm0.value * len(self.m.fe)),
    #                              **make_pyomo_model_kwargs)
    #     setup_func(new_sys)

    #     results = utils.default_solver(max_mins=10, solver='ma86',
    #                                    OF_hessian_approximation='limited-memory',
    #                                    **default_solver_kwargs).solve(new_sys.m, tee=True)

    #     from pyomo.opt import TerminationCondition
    #     if results.solver.termination_condition == TerminationCondition.infeasible:  # type: ignore
    #         utils.warn('Presolving returned an infeasible result')

    #     self.init_from_robot(new_sys)
