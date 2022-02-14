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
    def __init__(self, name: str, links: List['Link3D']) -> None:
        self.name = name
        self.links = links
        self.m: Union[ConcreteModel, None] = None
        self.sp_variables: List[sp.Symbol] = flatten(
            link.get_sympy_vars() for link in self.links
        )

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
        visual.info(f'Number of operations in EOM is {sp.count_ops(eom)}')

        # the lambdifying step actually takes quite long
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
                'Both EOM and EOM no C are saved to the model, but EOM no C is not used'
            )

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
                scale_forces_by = total_mass * 9.81
            m.force_scale = Param(initialize=scale_forces_by)

            @m.Constraint(m.fe, m.cp, range(len(self.eom_f)))
            def EOM_constr(m, fe, cp, i):
                return self.eom_f[i]([*self.pyo_variables[fe, cp], m.force_scale]) == 0 \
                    if not (fe == 1 and cp < ncp) else Constraint.Skip
        else:
            visual.warn('Not including dynamics (EOM_constr) in model')

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
            raise FileExistsError(filename)

        with open(filename, 'wb') as f:
            dill.dump(self.save_data_to_dict(description), f, recurse=True)

    def init_from_dict_one_point(self, data: Dict[str, Any],
                                 fed: int, cpd: int,
                                 fes: Optional[int] = None, cps: Optional[int] = None,
                                 **kwargs):
        """
        fed and cpd are destination (ie pyomo, and 1-indexed), fes and cps
        are source (ie dictionary, and 0-indexed)
        """
        m = utils.get_pyomo_model_or_error(self)

        if fes is None:
            fes = fed - 1
        if cps is None:
            cps = cpd - 1

        # TODO: ignore this warning when accounting for differing number of finite elements
        if not m.hm0.value == data['hm0']:
            visual.warn(
                f'init_from_dict_one_point: self.hm0 = {m.hm0.value} != {data["hm0"]} = data["hm0"]', once=True
            )

        if utils.has_variable_timestep(m):
            utils.maybe_set_var(m.hm[fed], data['hm'][fes], **kwargs)

        for link, linkdata in zip(self.links, data['links']):
            link.init_from_dict_one_point(
                linkdata, fed, cpd, fes, cps, **kwargs
            )

    def init_from_dict(self, data: Dict[str, Any], **kwargs):
        for fed, cpd in self.indices(one_based=True):
            self.init_from_dict_one_point(data, fed, cpd, **kwargs)

    def init_from_file(self, filename: str, **kwargs):
        import dill

        with open(filename, 'rb') as f:
            data = dill.load(f)

        self.init_from_dict(data, **kwargs)

    def indices(self, *, one_based: bool, skipfirst: bool = True) -> List[Tuple[int, int]]:
        return utils.get_indexes(
            len(self.m.fe), len(self.m.cp), one_based=one_based, skipfirst=skipfirst
        )

    def __getitem__(self, linkname: str) -> 'Link3D':
        for link in self.links:
            if link.name == linkname:
                return link

        raise KeyError(
            f'No link with name "{linkname}". Available links are: {", ".join(link.name for link in self.links)}'
        )

    def plot_keyframes(self,
                       keyframes: List[int],
                       view_along: Union[Tuple[float, float], str],
                       plot3d_config: Dict = {},
                       save_to: Optional[str] = None,
                       ground_color: str = 'lightgray',
                       lims: Optional[Tuple[Tuple, Tuple, Tuple]] = None,
                       plotgrass: bool = False,
                       plotgrass_kwargs: Dict = {}):
        # need to import this to get 3D plots working, for some reason
        from mpl_toolkits import mplot3d

        # typing this as 'Any' because the method accesses following give
        # false warnings otherwise...
        fig, ax, add_ground = visual.plot3d_setup(
            scale_plot_size=False, **plot3d_config
        )

        if lims is not None:
            x, y, z = lims
            ax.set_xlim(*x)
            ax.set_ylim(*y)
            ax.set_zlim(*z)

            if plotgrass:
                visual.plotgrass(ax, *x, *y, **plotgrass_kwargs)

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
                save_to: Optional[str] = None,
                use_html5_video: bool = True):
        # need to import this to get 3D plots working, for some reason
        from mpl_toolkits import mplot3d
        import matplotlib.animation
        import matplotlib.pyplot as plt

        pyo_model = utils.get_pyomo_model_or_error(self)

        # typing this as 'Any' because the method accesses following give
        # false warnings otherwise...
        fig, ax, add_ground = visual.plot3d_setup(
            scale_plot_size=False, **plot3d_config
        )

        if lims is not None:
            x, y, z = lims
            ax.set_xlim(*x)
            ax.set_ylim(*y)
            ax.set_zlim(*z)

            visual.plotgrass(ax, *x, *y)

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
            if not utils.has_variable_timestep(pyo_model):
                t_arr = np.cumsum(
                    np.array([self.m.hm[fe] for fe in self.m.fe]))
            else:  # the t_arr below is variable step
                t_arr = np.cumsum(utils.get_vals(self.m.hm)) * self.m.hm0.value

            def _animate(t: float):  # t is a float from 0 to total_time
                progress_bar(t/frames[-1])

                for link in self.links:
                    link.animation_update(
                        fig, ax, t=t, t_arr=t_arr, track=(track == link.name)
                    )

                nonlocal ground
                ground.remove()
                ground = add_ground((ax.get_xlim(), ax.get_ylim()))

        # if you get impatient and cancel an animation while it's being made,
        # then try to clone a model, you get an error from matplotlib about
        # certain things not being cloneable. So -> try/catch/finally block
        try:
            if not utils.has_variable_timestep(pyo_model):
                t_sum = sum(self.m.hm[fe]
                            for fe in self.m.fe if fe != nfe) * self.m.hm0.value
            else:
                t_sum = sum(
                    self.m.hm[fe].value for fe in self.m.fe if fe != nfe) * self.m.hm0.value

            interval_ms = 1000*t_scale * \
                (t_sum / nfe / ncp if dt is None else dt)

            frames = [fe for fe in self.m.fe] if dt is None else \
                np.arange(start=0, stop=t_sum+dt, step=dt)

            anim = matplotlib.animation.FuncAnimation(
                fig, _animate, frames=frames, interval=interval_ms
            )

            plt.close(anim._fig)

            if save_to is not None:
                anim.save(save_to)
            elif utils.in_ipython():
                from IPython.core.display import display, HTML
                # If you are using Jupyter notebook in VS code, `anim.to_html5_video` does not work.
                # Refer to this link: https://github.com/microsoft/vscode-jupyter/issues/1912
                # Instead set this flag to false to use a compliant one.
                if use_html5_video:
                    display(HTML(anim.to_html5_video()))
                else:
                    display(HTML(anim.to_jshtml()))
            else:
                fig.show()

        except Exception as e:
            progress_bar(nfe, ncp)
            visual.error(f'Interrupted animation due to error: {e}')

        finally:
            for link in self.links:
                link.cleanup_animation(fig, ax)
            del fig, ax

    def plot(self, save_to: Optional[str] = None, plot_links: bool = True) -> None:
        pyo_model = utils.get_pyomo_model_or_error(self)

        if utils.has_variable_timestep(pyo_model):
            import matplotlib.pyplot as plt
            data = 1000*utils.get_vals(pyo_model.hm) * pyo_model.hm0.value
            plt.plot(data)
            plt.ylabel('Timestep size [ms]')
            plt.title('Timestep size vs finite element')
            plt.tight_layout()

            if save_to is not None:
                plt.gcf().savefig(f'{save_to}timestep-length-{self.name}.pdf')
            else:
                plt.show()

        if plot_links:
            for link in self.links:
                link.plot(save_to=save_to)

    def __repr__(self) -> str:
        child_links = '\n  '.join(str(link) + ',' for link in self.links)
        return f'System3D(name="{self.name}", [\n  {child_links}\n])'

    def post_solve(self, costs: Optional[Dict[str, Any]] = None, detailed: bool = False, tol: float = 1e-6):
        from .foot import feet_penalty
        from pyomo.environ import value as pyovalue

        pyo_model = utils.get_pyomo_model_or_error(self)
        print('Total cost:', pyovalue(pyo_model.cost))

        if costs is not None:
            for k, v in costs.items():
                print(f'-- {k}: {pyovalue(v)}')

        foot_pen = pyovalue(feet_penalty(self))
        if foot_pen > 1e-3:
            visual.error('Foot penalty seems to be unsolved')

        if detailed is True:
            from pyomo.util.infeasible import log_infeasible_constraints
            print('Infeasible constraints:')
            log_infeasible_constraints(pyo_model, tol=tol)

            from pyomo.util.infeasible import log_infeasible_bounds
            print('Infeasible bounds:')
            log_infeasible_bounds(pyo_model, tol=tol)
