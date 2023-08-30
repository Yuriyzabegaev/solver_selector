from copy import deepcopy
from itertools import count, product
from typing import Literal, Optional
from dataclasses import dataclass, field
from typing_extensions import Self
import numpy as np
from typing import Sequence, TypeAlias


number: TypeAlias = int | float


@dataclass(kw_only=True, slots=True, frozen=True)
class NumericalParameter:
    bounds: tuple[float, float]
    default: float
    scale: Literal["linear", "log10"] = "linear"
    dtype: Literal["float", "int"] = "float"
    is_optimized: bool = True


@dataclass(kw_only=True, slots=True, frozen=True)
class CategoryParameter:
    options: Sequence[str]
    default: str
    is_optimized: bool = True


@dataclass(kw_only=True, slots=True, frozen=True)
class ParametersSpace:
    """Represents numerical and categorical parameters of one algorithm node, e.g.
    paramets of ILU preconditioner."""

    numerical: dict[str, NumericalParameter] = field(default_factory=lambda: {})


@dataclass(kw_only=True, slots=True, frozen=True)
class Decision:
    """Represents a complete solver configuration decision."""

    subsolvers: dict[int, str]
    """Represent a set of chosen methods that are used inside the solver. 
    
    Key is a config node id.
    
    """

    parameters: dict[int, dict[str, number]]
    """Represent numerical parameters of the methods that are used inside the solver. 
    
    Key is a config node id.
    
    """


@dataclass(kw_only=True, slots=True, frozen=True)
class DecisionTemplate:
    """Represents one solver algorithm, but the specific parameters for it are not
    chosen yet. Their places are holded by the spaces of possible decisions.

    """

    subsolvers: dict[int, str]
    """Represent a set of chosen methods that are used inside the solver. 
    
    Key is a config node id.
    
    """

    parameters: dict[int, ParametersSpace]
    """Set of of the method parameters to be chosen. Key is a config node id."""

    def select_parameters(self, parameters: dict[int, dict[str, number]]) -> Decision:
        for node_id, param_space in self.parameters.items():
            assert node_id in parameters
            for param_name, param_value in parameters[node_id].items():
                assert param_name in param_space.numerical
                assert isinstance(param_value, number)
        return Decision(subsolvers=self.subsolvers, parameters=parameters)


class SolverConfigNode:
    """Represents an abstract solver configuration."""

    _counter = count(0)
    type: str
    """Type of solver algorithm."""

    def __repr__(self) -> str:
        return f"Solver config node {self._id}: {self.name}"

    def __init__(self, children: Sequence["SolverConfigNode"] = None, name: str = None):
        self._id = next(self._counter)
        self.children: Sequence[SolverConfigNode] = tuple(children or [])
        self.name: str = name or self.type
        self.parent: Optional[SolverConfigNode] = None
        for child in self.children:
            child.parent = self  # Reference cycle

    def copy(self) -> Self:
        return type(self)(
            children=[child.copy() for child in self.children], name=self.name
        )

    def get_all_solvers(self) -> Sequence[DecisionTemplate]:
        """Returns a list of all possible decisions, that the solver selector can select
        from. Later, the chosen decision can be translated into a config dictionary,
        which is readable by the solver assembler. For this, use
        `SolverConfigNode.config_from_decision`.

        """
        solver_templates = self._get_submethods()
        results = []
        for solver_template in solver_templates:
            submethods = {
                k: v for k, v in solver_template.items() if isinstance(v, str)
            }
            num_params = {
                k: v
                for k, v in solver_template.items()
                if isinstance(v, ParametersSpace)
            }
            res = DecisionTemplate(subsolvers=submethods, parameters=num_params)
            results.append(res)
        return results

    def _get_submethods(self) -> Sequence[dict[int, str | ParametersSpace]]:
        submethods = []
        params = []  # Of this method
        for child in self.children:
            if isinstance(child, ParametersNode):
                params.extend(child._get_submethods())
            else:
                submethods.extend(child._get_submethods())

        if len(submethods) > 0:
            for submethod in submethods:
                for param in params:
                    submethod.update(param)
        else:
            submethods = params
        return submethods

    @staticmethod
    def format_config(config) -> str:
        """Makes a human-readable string from the provided solver config."""
        if not isinstance(config, dict):
            return f"={config}"
        if len(config) == 0:
            return ""
        if len(config) == 1:
            key, value = next(iter(config.items()))
            child_config = SolverConfigNode.format_config(value)
            key = str(key)

            if key in ["none", "linear solver", "preconditioner", "parameters_node"]:
                return child_config

            key = key.replace(" ", "_")
            if len(child_config) > 0:
                if child_config.startswith("["):
                    return f"{key} {child_config}"
                if child_config.startswith("="):
                    return f"{key}{child_config}"
                return f"{key} - {child_config}"
            return str(key)

        results = []
        for key, value in config.items():
            results.append(SolverConfigNode.format_config({key: value}))
        return "[{:s}]".format(", ".join(results))

    def find_node_by_id(self, node_id: int) -> "SolverConfigNode":
        """Returns a child config node with a given id. The ids are unique."""
        if self._id == node_id:
            return self
        for child in self.children:
            try:
                return child.find_node_by_id(node_id)
            except ValueError:
                continue
        raise ValueError(f"Node with id={node_id} not found")

    def find_nodes_by_name(self, node_name: str) -> Sequence["SolverConfigNode"]:
        """Returns children config nodes with a given name. The names are not unique."""
        if self.name == node_name:
            return [self]
        nodes = []
        for child in self.children:
            nodes.extend(child.find_nodes_by_name(node_name))
        return nodes

    def config_from_decision(
        self, decision: Decision, optimized_only: bool = False
    ) -> dict:
        """Translates a solver selection decision into the solver config dictionary.
        `optimized_only=True` ignores config fields that we do not optimize."""
        if len(self.children) == 0:
            return self.name

        children_configs = [
            c.config_from_decision(decision, optimized_only=optimized_only)
            for c in self.children
        ]
        return {self.name: _merge_dicts(children_configs)}

    def decision_from_config(self, config: dict) -> Decision:
        """Translates a solver config dictionary into the solver selection decision."""
        subsolvers, parameters = self._parse_config_recursively(config)
        return Decision(subsolvers=subsolvers, parameters=parameters)

    def _parse_config_recursively(
        self, config: dict
    ) -> tuple[dict[int, str], dict[int, dict[str, number]]]:
        """Traverse config to find what subsolvers and what parameters were selected."""
        items = list(config.items())
        assert len(items) == 1, "Don't know what to do"
        node_name, subconfig = items[0]

        assert node_name == self.name

        if len(self.children) == 0:
            return {}, {}

        submethods = []
        subparams = []
        for child in self.children:
            child_name = child.name
            child_subconfig = subconfig[child_name]
            child_config = {child_name: child_subconfig}
            child_submethods, child_subparams = child._parse_config_recursively(
                child_config
            )
            submethods.append(child_submethods)
            subparams.append(child_subparams)
        return _merge_dicts(submethods), _merge_dicts(subparams)


class ConstantNode(SolverConfigNode):
    """Solver config node, which does not have any optimized parameters. However, you
    can provide its consant parameters with `params`.

    """

    type = "constant config node"

    def __init__(self, name: str, params: dict = None) -> None:
        super().__init__(children=None, name=name)
        self.params = params or {}

    def copy(self) -> Self:
        return type(self)(name=self.name, params=self.params)

    def config_from_decision(
        self, decision: Decision, optimized_only: bool = False
    ) -> dict:
        if optimized_only:
            return {self.name: {}}
        if len(self.params) == 0:
            return self.name
        return {self.name: self.params}


class ForkNode(SolverConfigNode):
    """Solver config node, which represent a variety of sub-solvers. The solver
    selection algorithm must select one of them. Thus, this is a categorical decision
    to make.

    """

    def __init__(
        self,
        options: Sequence[SolverConfigNode],
        name: str = None,
    ):
        self.options = {opt.name: opt for opt in options}
        super().__init__(children=options, name=name)

    def copy(self) -> Self:
        return type(self)(options=[option.copy() for option in self.options.values()])

    def _get_submethods(self) -> Sequence[dict[int, str | ParametersSpace]]:
        all_options = []
        for option_name, option in self.options.items():
            base_option = {self._id: option_name}
            solver_spaces_child = option._get_submethods()
            if len(solver_spaces_child) == 0:
                solver_spaces_child = [base_option]
            else:
                for solver_space in solver_spaces_child:
                    solver_space.update(base_option.copy())
            all_options.extend(solver_spaces_child)
        return all_options

    def config_from_decision(self, decision: Decision, optimized_only: bool = False):
        my_decision = decision.subsolvers[self._id]
        config_tree = self.options[my_decision].config_from_decision(
            decision, optimized_only=optimized_only
        )
        return {self.name: config_tree}

    def _parse_config_recursively(
        self, config: dict
    ) -> tuple[dict[int, str], dict[int, dict[str, number]]]:
        subconfig = config[self.name]
        if isinstance(subconfig, str):
            subconfig = {subconfig: {}}

        subconfig_items = list(subconfig.items())
        assert len(subconfig_items) == 1, "Don't know what to do"
        child_name, _ = subconfig_items[0]
        if child_name not in self.options:
            raise ValueError(
                f"Inconsistent solver space: Node {self._id}: {self.name}"
                f" doesn't have a child {child_name}"
            )

        child = self.options[child_name]
        child_decision, child_params = child._parse_config_recursively(subconfig)
        return {self._id: child_name} | child_decision, child_params


class ForkNodeNames:
    krylov_solver_picker = "linear solver"
    preconditioner_picker = "preconditioner"


class KrylovSolverDecisionNode(ForkNode):
    """Represents a variety of available Krylov solver configurations to select from."""

    type = ForkNodeNames.krylov_solver_picker


class PreconditionerDecisionNode(ForkNode):
    """Represents a variety of available preconditioner configurations to select from."""

    type = ForkNodeNames.preconditioner_picker


class KrylovSolverNode(SolverConfigNode):
    """Represents one specific Krylov solver algorithm configuration."""

    @classmethod
    def from_preconditioners_list(
        cls,
        preconditioners: Sequence[SolverConfigNode],
        other_children: Sequence[SolverConfigNode] = None,
        **kwargs,
    ):
        """Convenience method to assemble a space of a Krylov solver with various
        preconditioners.

        """
        preconditioners = [prec.copy() for prec in preconditioners]
        children = [PreconditionerDecisionNode(options=preconditioners)]
        if other_children is not None:
            children.extend(other_children)

        return cls(
            children=children,
            **kwargs,
        )


class SplittingNode(SolverConfigNode):
    """Represents one specific decoupling/splitting algorithm."""

    @classmethod
    def from_solvers_list(
        cls,
        solver_list: Sequence[SolverConfigNode] | dict[str, Sequence[SolverConfigNode]],
        **kwargs,
    ):
        """Convenience method to assemble a space of a splitting node with various
        sub-solvers.

        """
        if isinstance(solver_list, dict):
            solver_node = {
                key: KrylovSolverDecisionNode(solvers)
                for key, solvers in solver_list.items()
            }
        else:
            solver_node = KrylovSolverDecisionNode(solver_list)
        return cls(solver_node=solver_node, **kwargs)

    def __init__(
        self,
        solver_node: SolverConfigNode | dict[str:SolverConfigNode],
        name: str = None,
        other_children: Optional[Sequence[SolverConfigNode]] = None,
    ):
        # TODO: Something bad will happen here when it'll be more than two sub-solvers.
        if isinstance(solver_node, SolverConfigNode):
            solver_node = {"primary": solver_node, "secondary": solver_node}
        self.linear_solver_node_primary = solver_node["primary"].copy()
        self.linear_solver_node_secondary = solver_node["secondary"].copy()
        self.solver_nodes = solver_node
        self.linear_solver_node_primary.name = "primary"
        self.linear_solver_node_secondary.name = "secondary"
        if other_children is None:
            other_children = []
        self.other_children = other_children
        super().__init__(
            children=[
                self.linear_solver_node_primary,
                self.linear_solver_node_secondary,
            ]
            + other_children,
            name=name,
        )

    def _get_submethods(self) -> Sequence[dict[int, str | ParametersSpace]]:
        result = []
        for c in self.children:
            if len(arms := c._get_submethods()) > 0:
                result.append(arms)
        return [_merge_dicts(pair) for pair in tuple(product(*result))]

    def copy(self) -> Self:
        return type(self)(
            solver_node=self.solver_nodes,
            name=self.name,
            other_children=self.other_children,
        )


class ParametersNode(SolverConfigNode):
    """Represents numerical parametersof one algorithm to be optimized."""

    type = "parameters_node"

    def __init__(
        self,
        numerical: dict[str, NumericalParameter | number],
        name: str = None,
    ):
        # If the user passed numbers, make constant parameters from them.
        tmp = {}
        for param_name, param in numerical.items():
            if isinstance(param, number):
                param = NumericalParameter(
                    bounds=(param, param), default=param, is_optimized=False
                )
            tmp[param_name] = param
        numerical = tmp

        self.parameters: ParametersSpace = ParametersSpace(numerical=numerical)
        super().__init__(children=None, name=name)

    def copy(self):
        return type(self)(
            numerical=self.parameters.numerical,
            name=self.name,
        )

    def config_from_decision(
        self, decision: Decision, optimized_only: bool = False
    ) -> dict:
        actions = decision.parameters[self._id]

        action_dict = {}
        for action_name, action_action in actions.items():
            parameter_data = self.parameters.numerical[action_name]
            if optimized_only and not parameter_data.is_optimized:
                continue

            action_dict[action_name] = action_action

        return {self.name: action_dict}

    def _parse_config_recursively(
        self, config: dict
    ) -> tuple[dict[int, str], dict[int, dict[str, number]]]:
        config_items = list(config.items())
        assert len(config_items) == 1
        config_name, config_entries = config_items[0]
        assert config_name == self.name

        actions = {}
        for action_name in self.parameters.numerical.keys():
            value = config_entries[action_name]
            actions[action_name] = value

        return {}, {self._id: actions}

    def _get_submethods(self) -> Sequence[dict[int, str | ParametersSpace]]:
        return [{self._id: self.parameters}]


def _merge_dicts(list_of_dicts: Sequence[dict]) -> dict:
    result = {}
    for entry in list_of_dicts:
        result.update(entry)
    return result
