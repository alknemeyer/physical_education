from typing import Tuple, List, Iterable, TYPE_CHECKING
from pyomo.environ import Var, ConcreteModel
from .utils import flatten


if TYPE_CHECKING:
    from .links import Link3D


class VariableList:
    """Access pyomo variables using `variable_list[fe,cp]`, with
    `fe = 1..nfe`, `cp = 1..ncp` (ie, one-based)
    """
    def __init__(self, m: ConcreteModel, sources_of_vars: Iterable['Link3D']):
        self.sources_of_vars = sources_of_vars

        self._var_list: List[List[List[Var]]] = [
            [self._get_vars(fe, cp) for cp in m.cp] for fe in m.fe
        ]

    def _get_vars(self, fe: int, cp: int) -> List[Var]:
        return flatten(src.get_pyomo_vars(fe, cp) for src in self.sources_of_vars)

    def __getitem__(self, idx: Tuple[int, int]) -> List[Var]:
        fe, cp = idx
        assert (1 <= fe <= len(self._var_list)
                and 1 <= cp <= len(self._var_list[0]))

        return self._var_list[fe-1][cp-1]
