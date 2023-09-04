from collections import defaultdict
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

    def use_defaults(self) -> Decision:
        defaults = defaultdict(lambda: dict())
        for node_id, param_space in self.parameters.items():
            for param_name, param_value in param_space.numerical.items():
                defaults[node_id][param_name] = param_value.default
        return self.select_parameters(defaults)


class SolverConfigNode:
    """Represents an abstract solver configuration."""

    _counter = count(0)
    type: str
    """Type of solver algorithm. 
    
    Used for convenience in subclasses that represent one specific solver.
    It prevents passing a name in the constructor.
    
    """

    def __repr__(self) -> str:
        return f"Solver config node {self._id}: {self.name}"

    def __init__(self, children: Sequence["SolverConfigNode"] = None, name: str = None):
        self._id = next(self._counter)
        self.children: Sequence[SolverConfigNode] = tuple(children or [])
        self.name: str = name or self.type
        self.parent: Optional[SolverConfigNode] = None
        for child in self.children:
            child.parent = self  # Reference cycle
        self._ensure_unique_ids()

    def _ensure_unique_ids(self) -> set[int]:
        """If ids are not unique for some reason, many bad things can happen.

        Returns:
            Set of ids in the tree.

        """
        children_ids = [child._ensure_unique_ids() for child in self.children]
        total_ids = sum(len(x) for x in children_ids)
        merged_ids = {val for child_ids in children_ids for val in child_ids}
        if len(merged_ids) != total_ids:
            all_ids = [id for child_ids in children_ids for id in child_ids]
            presence, counts = np.unique(all_ids, return_counts=True)
            nonunique = presence[np.where(counts > 1)].tolist()
            raise ValueError(
                "There are more than one node with each of these ids: "
                f"{nonunique}. Probably, you should use .copy() somewhere."
            )

        return merged_ids | {self._id}

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
            return {self.name: {}}

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
            # child_name = child.name
            # child_subconfig = subconfig[child_name]
            # child_config = {child_name: child_subconfig}
            child_submethods, child_subparams = child._parse_config_recursively(
                subconfig
            )
            submethods.append(child_submethods)
            subparams.append(child_subparams)
        return _merge_dicts(submethods), _merge_dicts(subparams)


class ConstantNode(SolverConfigNode):
    """Solver config node, which does not have any optimized parameters. However, you
    can provide its consant parameters with `params`.

    """

    type = "constant config node"

    def __init__(self, name: str, params: dict | str = None) -> None:
        super().__init__(children=None, name=name)
        if params is None:
            params = {}
        elif isinstance(params, str):
            params = {params: {}}
        self.params = params or {}

    def copy(self) -> Self:
        return type(self)(name=self.name, params=self.params)

    def config_from_decision(
        self, decision: Decision, optimized_only: bool = False
    ) -> dict:
        if optimized_only:
            return {self.name: {}}
        return {self.name: self.params}


class DecisionNodeNames:
    fork_node = "linear solver"
    preconditioner = "preconditioner"
    parameters_picker = "parameter picker"


class ForkNode(SolverConfigNode):
    """Solver config node, which represent a variety of sub-solvers. The solver
    selection algorithm must select one of them. Thus, this is a categorical decision
    to make.

    """

    type = DecisionNodeNames.fork_node

    def __init__(
        self,
        options: Sequence[SolverConfigNode],
        name: str = None,
    ):
        self.options = {opt.name: opt.copy() for opt in options}
        super().__init__(children=list(self.options.values()), name=name)

    def copy(self) -> Self:
        return type(self)(options=list(self.options.values()), name=self.name)

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
        if self.name == DecisionNodeNames.fork_node:
            # If no specific name, just return the parameters
            return config_tree
        return {self.name: config_tree}

    def _parse_config_recursively(
        self, config: dict
    ) -> tuple[dict[int, str], dict[int, dict[str, number]]]:
        # if isinstance(config, str):
        #     subconfig = {subconfig: {}}
        if self.name in config:
            config = config[self.name]

        config_items = list(config.items())
        assert len(config_items) == 1, "Don't know what to do"
        child_name, _ = config_items[0]
        if child_name not in self.options:
            raise ValueError(
                f"Inconsistent solver space: Node {self._id}: {self.name}"
                f" doesn't have a child {child_name}"
            )

        child = self.options[child_name]
        child_decision, child_params = child._parse_config_recursively(config)
        return {self._id: child_name} | child_decision, child_params


class KrylovSolverDecisionNode(ForkNode):
    """Represents a variety of available Krylov solver configurations to select from."""

    @classmethod
    def from_preconditioners_list(
        self, krylov_solver_name: str, preconditioners: Sequence[SolverConfigNode]
    ) -> "KrylovSolverDecisionNode":
        """Use the given krylov solver and one of the given preconditioners"""
        return KrylovSolverDecisionNode(
            options=[
                KrylovSolverNode(name=krylov_solver_name, children=[prec.copy()])
                for prec in preconditioners
            ]
        )


class PreconditionerDecisionNode(ForkNode):
    """Represents a variety of available preconditioner configurations to select from."""

    type = DecisionNodeNames.preconditioner


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
        solver_nodes: dict[str, Sequence[SolverConfigNode]],
        **kwargs,
    ):
        """Convenience method to assemble a space of a splitting node with various
        sub-solvers.

        """
        solvers_list = [
            KrylovSolverDecisionNode(solvers, name=key)
            for key, solvers in solver_nodes.items()
        ]
        return cls(solver_nodes=solvers_list, **kwargs)

    def __init__(
        self,
        solver_nodes: Sequence[SolverConfigNode],
        name: str = None,
        other_children: Optional[Sequence[SolverConfigNode]] = None,
    ):
        self.solver_nodes = [node.copy() for node in solver_nodes]
        if other_children is None:
            other_children = []
        self.other_children = [child.copy() for child in other_children]
        super().__init__(
            children=solver_nodes + other_children,
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
            solver_nodes=self.solver_nodes,
            name=self.name,
            other_children=self.other_children,
        )


class ParametersNode(SolverConfigNode):
    """Represents numerical parametersof one algorithm to be optimized."""

    type = DecisionNodeNames.parameters_picker

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

        parameters_dict = {}
        for parameter_name, parameter in actions.items():
            parameter_data = self.parameters.numerical[parameter_name]
            if optimized_only and not parameter_data.is_optimized:
                continue
            parameters_dict[parameter_name] = parameter

        if self.name == DecisionNodeNames.parameters_picker:
            # If no specific name, just return the parameters
            return parameters_dict
        return {self.name: parameters_dict}

    def _parse_config_recursively(
        self, config: dict
    ) -> tuple[dict[int, str], dict[int, dict[str, number]]]:
        if self.name in config:
            config = config[self.name]

        actions = {}
        for action_name in self.parameters.numerical.keys():
            value = config[action_name]
            actions[action_name] = value

        return {}, {self._id: actions}

    def _get_submethods(self) -> Sequence[dict[int, str | ParametersSpace]]:
        return [{self._id: self.parameters}]


def _merge_dicts(list_of_dicts: Sequence[dict]) -> dict:
    result = {}
    for entry in list_of_dicts:
        result.update(entry)
    return result
