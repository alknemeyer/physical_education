from typing import Any, Iterable, Callable, Optional, Tuple, List, TypeVar, Union
import sympy as sp
import numpy as np
from pyomo.environ import ConcreteModel, Var, Set, Param, RangeSet, Constraint
from sympy import Matrix as Mat
from .visual import info, debug, warn
sp.init_printing()

# derivatives ################################################################


def deriv(expr, q: Mat, dq: Mat):
    """Take the time derivative of an expression `expr` with respect to time,
    handling the chain rule semi-correctly"""
    return (sp.diff(expr, q).T * dq)[0]  # type: ignore


def full_deriv(var, q: Mat, dq: Mat, ddq: Mat):
    return deriv(var, q, dq) + deriv(var, dq, ddq)


# rotations ##################################################################
SymOrFloat = Union[sp.Symbol, float]


def rot(θ: SymOrFloat) -> Mat:
    return sp.Matrix([
        [sp.cos(θ), sp.sin(θ)],
        [-sp.sin(θ), sp.cos(θ)],
    ])


def rot_x(θ: SymOrFloat) -> Mat:
    return sp.Matrix([
        [1,         0,           0],
        [0,  sp.cos(θ),  sp.sin(θ)],
        [0, -sp.sin(θ),  sp.cos(θ)],
    ])


def rot_y(θ: SymOrFloat) -> Mat:
    return sp.Matrix([
        [sp.cos(θ), 0, -sp.sin(θ)],
        [0,         1,          0],
        [sp.sin(θ), 0,  sp.cos(θ)],
    ])


def rot_z(θ: SymOrFloat) -> Mat:
    return sp.Matrix([
        [sp.cos(θ),  sp.sin(θ), 0],
        [-sp.sin(θ), sp.cos(θ), 0],
        [0,                  0, 1],
    ])


def euler_321(phi: SymOrFloat, theta: SymOrFloat, psi: SymOrFloat) -> Mat:
    return rot_x(phi) @ rot_y(theta) @ rot_z(psi)

# utils to find equations of motion ##########################################


def skew_symmetric(Rx_I: Mat, q: Mat, dq: Mat) -> Mat:
    """Rx_I is the 3x3 rotation from body frame `x` to the inertial `I`
    `q` is a vector of symbolic variables
    `dq` is a vector of the derivates of `q`

    The return matrix can be simplified fairly quickly with `trigsimp` if
    it's made up of ~3 or fewer rotations
    """
    dRx_I = sp.zeros(3, 3)

    for i in range(3):
        dRx_I[:, i] = Rx_I[:, i].jacobian(q) * dq

    omega_Rx = Rx_I.T @ dRx_I
    return Mat([
        omega_Rx[2, 1],
        omega_Rx[0, 2],
        omega_Rx[1, 0]
    ])


def manipulator_equation(Ek: Mat, Ep: Mat, q: Mat, dq: Mat) -> Tuple[Mat, Mat, Mat]:
    """Ek and Ep are the kinetic and potential energy of the system.
    They must be passed in as sp.Matrix types (or at least have a
    `.jacobian` method defined on them)"""
    M = sp.hessian(Ek, dq)
    N = M.shape[0]

    dM = sp.zeros(N, N)
    for i in range(N):
        for j in range(N):
            dM[i, j] = Mat([M[i, j]]).jacobian(q) @ dq

    C = dM @ dq - Ek.jacobian(q).T

    G = Ep.jacobian(q).T

    return M, C, G


def calc_velocities_and_energies(
        positions: Iterable[Mat], rotations: Iterable[Mat],
        masses: Iterable[float], inertias: Iterable[Mat],
        q: Mat, dq: Mat, g: float = 9.81
) -> Tuple[Mat, Mat, List[Mat], List[Mat]]:
    """
        Calculate and return  the kinetic and potential energies of
        a system, given lists of:
        - positions of each body (in inertial)
        - rotations for each body (from body to inertial)
        - mass of each body
        - inertia of each body (given as a 3x3 matrix)
        - q (each state, such as q = [x, y, theta])
        - dq (derivative of each state, such as dq = [dx, dy, dtheta])
    """
    from functools import reduce

    dPs = [Mat(Px_I.jacobian(q) * dq) for Px_I in positions]
    ang_vels = [
        skew_symmetric(Rx_I, q, dq) for Rx_I in rotations
    ]
    # this should be sum(), but it inits the sum with the int 0, which can't
    # be added to matrices
    Ek = reduce(lambda a, b: a + b, [
        dPx_I.T * mx * dPx_I / 2. + dωx_I.T * Ix * dωx_I / 2.
        for (dPx_I, mx, dωx_I, Ix) in zip(dPs, masses, ang_vels, inertias)
    ])
    Ep = Mat([sum(m * Mat([0, 0, g]).dot(p)
                  for (m, p) in zip(masses, positions))])

    return Ek, Ep, dPs, ang_vels


def norm(vec: Mat, eps: float):
    assert vec.shape == (3, 1)
    assert eps >= 0

    x, y, z = vec
    return sp.sqrt(x**2 + y**2 + z**2 + eps)


def lambdify_EOM(EOM: Union[sp.Matrix, list], vars_in_EOM: List[sp.Symbol], *,
                 display_vars: bool = False, test_func: bool = True,
                 func_map: dict = {}) -> List[Callable[..., float]]:
    """ Returns a list of functions which, when called with arguments which match
    `vars_in_EOM`, will evaluate the equations of motion specified in `EOM`.
    `display_vars` specifies whether to print out `vars_in_EOM` as a sanity check.
    `test_func` specifies whether to test that the function returns a float when
    called with float arguments

        >>> lambdify_EOM(EOM, vars_in_EOM)
        [<function _lambdifygenerated(_Dummy_833, _Dummy_834, ...
         <function _lambdifygenerated(_Dummy_851, _Dummy_852, ... 
         <function _lambdifygenerated(_Dummy_923, _Dummy_924, ...]

        >>> lambdify_EOM(EOM, vars_in_EOM[:-3])  # not all vars in EOM
        AssertionError: The function didn't return a float - it's likely ...
        """
    import pyomo.environ
    import random
    import math

    if display_vars is True:
        try:
            from IPython.core.display import display
            display(vars_in_EOM)
        except:
            print(vars_in_EOM)

    func_map = {'sin': pyomo.environ.sin,
                'cos': pyomo.environ.cos,
                'pi': math.pi, **func_map}

    if isinstance(EOM, list):
        eom = sp.Matrix(EOM)
    elif EOM.is_Matrix is False:  # EOM is a sympy object, but not a Matrix
        eom = sp.Matrix([EOM])
    else:
        eom = EOM

    if not set(eom.free_symbols).issubset(set(vars_in_EOM)):
        raise ValueError('Some symbols in the eom aren\'t in `vars_in_EOM:'
                         + str(set(eom.free_symbols).difference(set(vars_in_EOM))))

    # Why pack the arguments into a vector? well... sympy lambdifies expressions
    # by writing out a function (key part!) and then calling `eval` on it. Unfortunately,
    # writing a python function with more than 255 arguments is a SyntaxError
    funcs = [sp.lambdify([vars_in_EOM], eqn, modules=[func_map])  # type: ignore
             for eqn in eom]

    # replace with set(EOM.free_symbols).difference(set(vars_in_EOM))?
    if test_func is True:
        vals = [random.random() for _ in range(len(vars_in_EOM))]
        for func in funcs:
            ret = func(vals)
            assert type(ret) == float, "The function didn't return a float - it's likely "\
                "because there are symbolic variables in the EOM "\
                "which weren't specified in `vars_in_EOM`. Got: " + \
                str(ret)

    return funcs

# simplification #################################################################


def parsimp_worker(_arg: Tuple[Any, Any], allow_recur: bool = True):
    expr, simp_func = _arg

    if expr.is_number or expr.is_Symbol:
        return expr

    args = []

    if hasattr(expr, 'args'):
        for arg in expr.args:
            if allow_recur is True and sp.count_ops(arg) > 200:
                arg = parsimp_worker((arg, simp_func), allow_recur=False)

            args.append(simp_func(arg))
    else:
        raise ValueError(f'Something is wrong here - got an object of type {type(expr)}'
                         f' with value: {expr}')

    return expr.func(*args)


def parsimp(mat: Mat, nprocs: int, f=sp.trigsimp) -> Mat:
    import multiprocessing
    with multiprocessing.Pool(processes=nprocs) as p:
        return sp.Matrix(
            p.map(parsimp_worker, [(v, f) for v in mat])
            # if something goes wrong and you need to debug, try this same function
            # but with the line below instead of the p.map line above
            # list(map(parsimp_worker, [(v, f) for v in mat]))
        ).reshape(*mat.shape)

#     import sys
#     outvals = []
#         for i, val in enumerate(p.imap_unordered(parsimp_worker,
#                                 [(v,f,i) for (i,v) in enumerate(mat)]), 1):
#             outvals.append(val)
#             if disp_progress is True:
#                 sys.stdout.write('\rSimplifying.... {0:%} done'.format(i/len(vec)))
#     return sp.Matrix(outvals).reshape(vec.shape)


T = TypeVar('T')


def flatten(ls: Iterable[Iterable[T]]) -> List[T]:
    """
    Flatten an iterable of iterables of items into a list of items
    """
    return [item for sublist in ls for item in sublist]

# interpolation ##############################################################


PyomoThing = Union[Var, Set, Param, RangeSet]


def add_to_pyomo_model(m: ConcreteModel, prefix: str, vals: Iterable[Iterable[PyomoThing]]):
    """
    Add the objects in `vals` to the pyomo model `m`, prefixed with `prefix`. Eg:

    ```python
    utils.add_to_pyomo_model(m, self.name, [
        self.pyomo_params.values(),
        self.pyomo_sets.values(),
        self.pyomo_vars.values()
    ])
    ```

    or,

    ```python
    utils.add_to_pyomo_model(m, 'shoulder', [[q, dq, ddq], [q_set]])
    ```

    where `q, dq, ddq` are pyomo `Var`'s and `q_set` is a pyomo `Set`
    """
    from itertools import chain
    for v in chain(*vals):
        newname = f'{prefix}_{v.name}'  # type: ignore
        assert not hasattr(m, newname), \
            f'The pyomo model already has an attribute with the name "{newname}"'
        setattr(m, newname, v)


def def_var(m: ConcreteModel, name: str, indexes: tuple, func: Callable, bounds: tuple = (None, None)):
    """Define a variable, equal to the result of a function. Eg:

    >>> def_var(m, 'foot_height', (m.fe,), bounds=(0, None),
                func=lambda m,fe: sum(m.foot_height[fe,:]))

    # which is equivalent to:
    >>> m.foot_height = Var(m.fe, bounds=(0, None))
    >>> @m.Constraint(m.fe)
        def foot_height_constr(m, fe):
            return m.foot_height_dummy[fe] == sum(m.foot_height[fe,:]))
    """
    # define the variable. eg:
    # >>> m.my_var_name = Var(m.fe, m.otherindex, bounds=(0, None))
    setattr(m, name, Var(*indexes, bounds=bounds))

    # define a function to set the variable. eg:
    # >>> var[m.fe, m.otherindex] == func(m, m.fe, m.otherindex)
    var = getattr(m, name)
    setterfunc = lambda *args: var.__getitem__(args[1:]) == func(*args)

    # set the variable equal to the expression. eg:
    # >>> m.my_var_name_constr = Constraint(m.fe, m.otherindex, rule=setterfunc)
    setattr(m, name + '_constr', Constraint(*indexes, rule=setterfunc))


def fix_within(lo: float, var: Var, up: float):
    """
    >>> fix_within(3., link['q'][:,:,'z'], 6.)
    """
    var.setlb(lo)
    var.setub(up)


def add_guiding_trajectory(var: Var, trajectory: Iterable[float], *, tol: float):
    """
    Guide a variable by fixing its values to be within `+-tol` of `trajectory`.

    >>> trajectory = 1 + 0.2 * np.sin(np.linspace(0, 2*np.pi, num=len(robot.m.fe)))
    >>> add_guiding_trajectory(robot['body']['q'][:,1,'z'], trajectory, tol=0.3)
    """
    warn('utils.add_guiding_trajectory has not been tested!', once=True)
    for _var, val in zip(var, trajectory):
        fix_within(val - tol, _var, val + tol)


def copy_state_init(src: Var, dst: Var):
    """
    >>> copy_state_init(robot['UFL']['q'], robot['UFR']['q'])
    """
    for idx, var in src.iteritems():
        dst[idx].value = var.value


def remove_constraint_if_exists(m: ConcreteModel, constraint: str):
    if hasattr(m, constraint):
        debug(f'Deleting existing constraint: {constraint}')
        m.del_component(constraint)

    constraint_index = constraint + '_index'
    if hasattr(m, constraint_index):
        debug(f'Deleting existing constraint index: {constraint_index}')
        m.del_component(constraint_index)


def maybe_set_var(var: Var, value: float, skip_if_fixed: bool, skip_if_not_None: bool, fix: bool) -> None:
    """
    >>> set_var(link['q'][fe,cp,'x'], 1.0, skip_if_fixed=True, skip_if_not_None=True, fix=False)
    """
    if var.fixed is True and skip_if_fixed is True:
        return
    elif var.value is not None and skip_if_not_None is True:
        return
    else:
        var.value = value
        if fix is True:
            var.fixed = True


def constrain_total_time(m: ConcreteModel, total_time: float):
    """
    >>> constrain_total_time(robot.m, total_time = (nfe-1)*robot.m.hm0.value)
    """
    remove_constraint_if_exists(m, 'total_time_constr')
    m.total_time_constr = Constraint(
        expr=sum(m.hm[fe] for fe in m.fe if fe != 1)*m.hm0 == total_time)

# other utils for pyomo ######################################################


IPOPT_PATH: Union[str, None] = None


def set_ipopt_path(ipopt_path: str):
    global IPOPT_PATH
    IPOPT_PATH = ipopt_path


def default_solver(*,
                   max_mins: int,
                   ipopt_path: Optional[str] = None,
                   solver: str = 'ma86',
                   max_iter: int = 50_000,
                   OF_print_frequency_time: int = 10,
                   output_file: str = './.ipopt-log.txt',
                   warm_start_init_point: bool = True,
                   **kwargs):
    """
    Tested linear solvers include 'mumps', 'ma77', 'ma97', 'ma86' and 'pardiso'. ma86 seems
    to perform the best for the types of systems modelled with this library

    Some solver options include,

    OF_print_frequency_iter: pos int, default 1
        Summarizing iteration output is printed every print_frequency_iter iterations,
        if at least print_frequency_time seconds have passed since last output

    OF_print_frequency_time: pos int, default 0
        Summarizing iteration output is printed if at least print_frequency_time seconds
        have passed since last output and the iteration number is a multiple of
        print_frequency_iter

    OF_warm_start_init_point: default 'no'
        Indicates whether this optimization should use a warm start initialization, where
        values of primal and dual variables are given (e.g., from a previous optimization
        of a related problem.)

    OF_hessian_approximation: default 'exact'
        Set to 'limited-memory' for L-GBFS. Default is 'exact'

    OF_accept_every_trial_step: default 'no'
        Setting this option to "yes" essentially disables the line search and makes the
        algorithm take aggressive steps, without global convergence guarantees

    print_info_string: default 'no'
        Enables printing of additional info string at end of iteration output. This string
        contains some insider information about the current iteration. For details, see
        diagnostic tags at bottom of https://coin-or.github.io/Ipopt/OUTPUT.html

    See https://coin-or.github.io/Ipopt/OPTIONS.html for more options. Note that options
    written to the options file should be prefixed with "OF_"

    Another useful page is https://coin-or.github.io/Ipopt/OUTPUT.html, which helps
    understand IPOPT's output
    """
    import os
    from datetime import datetime
    from pyomo.opt import SolverFactory
    from typing import Any

    if ipopt_path is None:
        global IPOPT_PATH
        if IPOPT_PATH is not None:
            ipopt_path = IPOPT_PATH
        # better to make this warning, or hope that IPOPT is on their path?
        else:
            warn('No path set for ipopt. Pass argument `ipopt_path`, '
                 'or call `utils.set_ipopt_path(str)`', once=True)

    opt: Any = SolverFactory('ipopt', executable=ipopt_path)
    opt.options['print_level'] = 5
    opt.options['max_cpu_time'] = max_mins * 60
    opt.options['max_iter'] = max_iter
    opt.options['Tol'] = 1e-6
    opt.options['OF_print_timing_statistics'] = 'yes'
    opt.options['halt_on_ampl_error'] = 'yes'
    opt.options['OF_print_frequency_time'] = OF_print_frequency_time
    opt.options['OF_acceptable_tol'] = 1e-3  # default: 1e-6
    opt.options['OF_warm_start_init_point'] = 'yes' if warm_start_init_point else 'no'
    opt.options['output_file'] = os.getcwd() + '/' + output_file

    opt.options['linear_solver'] = solver
    if solver == 'ma86':
        opt.options['OF_ma86_scaling'] = 'none'
    else:
        warn(f'Got solver {solver} but don\'t have any specific flags for it.')

    for key, val in kwargs.items():
        opt.options[key] = val

    tstart = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info(f'Optimization start time: {tstart}')
    return opt


def get_vals(var: Var, idxs: Optional[Tuple] = None) -> np.ndarray:
    m = var.model()
    assert isinstance(idxs, tuple) or idxs is None

    nfe = len(m.fe)
    ncp = len(m.cp)
    arr = np.array([var[idx].value for idx in var]).astype(float)

    if idxs is None:  # assume indexed with only fe
        assert var.dim() == 1, "This variable doesn't seem to be indexed with fe exclusively"
        return arr
    else:
        idxs = tuple(len(i) for i in idxs)

        # if m.cp wasn't given as an index
        idx_dims = next(iter(var))
        if len(idx_dims) != len(idxs) + 1:
            return arr.reshape(nfe, ncp, *idxs)
        else:
            return arr.reshape(nfe, *idxs)


# idxs: Tuple[Union[Set,RangeSet], ...]
def get_vals_v(var: Var, idxs: tuple) -> np.ndarray:
    """
    Verbose version that doesn't try to guess stuff for ya. Usage:

    >>> get_vals(m.q, (m.N, m.DOF))
    """
    m = var.model()
    arr = np.array([var[idx].value for idx in var]).astype(float)
    return arr.reshape(*(len(i) for i in idxs))


def get_indexes(nfe: int, ncp: int, *, one_based: bool, skipfirst: bool) -> List[Tuple[int, int]]:
    """ Get indices to index into variables, taking care of the funky first finite element

        >>> get_indexes(nfe=4, ncp=3, one_based=True)
        [(1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
        """
    offset = 1 if one_based else 0
    return [(i + offset, j + offset) for i in range(nfe) for j in range(ncp)
            if (i > 0 or j == ncp-1 or not skipfirst)]


class MarkovBinaryWalk():
    """Generate a Markov process random walk which alternates between two states:
    `a` and `b`. If in state `a`, return 1 then stay in state `a` with probability
    `a_prob`. Otherwise, go to state `b`. If in state `b`, return 0 and stay in
    state `b` with probability `b_prob`"""

    def __init__(self, a_prob: float, b_prob: float):
        self.a_prob = a_prob
        self.b_prob = b_prob
        # self.state: Union[Literal['a'],Literal['b']] = 'a'
        self.state = 'a'

    def step(self):
        import random
        if self.state == 'a':
            self.state = 'a' if random.random() < self.a_prob else 'b'
            return 1
        else:
            self.state = 'b' if random.random() < self.b_prob else 'a'
            return 0

    def walk(self, n):
        return np.array([self.step() for _ in range(n)], dtype=int)


# https://stackoverflow.com/a/22424821/1892669
def in_ipython() -> bool:
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # type: ignore
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def has_variable_timestep(m: ConcreteModel) -> bool:
    return m.hm.type() == Var  # .ctype


def get_name(name: Union[str, None], links: Iterable, suffix: str):
    if name is None:
        name = '_'.join(link.name for link in links) + f'_{suffix}'

    link = next(iter(links))

    assert name not in link.nodes,\
        f'Link {link.name} already has a node with the name {name}'

    return name
