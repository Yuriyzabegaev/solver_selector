from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count, product
from typing import Literal, Optional, Sequence, TypeAlias

import numpy as np
from typing_extensions import Self

number: TypeAlias = int | float


@dataclass(kw_only=True, slots=True, frozen=True)
class NumericalParameter:
    bounds: tuple[number, number]
    default: number
    scale: Literal["linear", "log10"] = "linear"
    dtype: Literal["float", "int"] = "float"
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
        return Decision(subsolvers=self.subsolvers, parameters=parameters)

    def use_defaults(self) -> Decision:
        defaults: dict[int, dict[str, number]] = defaultdict(lambda: dict())
        for node_id, param_space in self.parameters.items():
            for param_name, param_value in param_space.numerical.items():
                defaults[node_id][param_name] = param_value.default
        return self.select_parameters(dict(defaults))


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

    def __init__(
        self,
        children: Optional[Sequence["SolverConfigNode"]] = None,
        name: Optional[str] = None,
    ):
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
        children_submethods_params: list[list[dict[int, str | ParametersSpace]]] = []
        for child in self.children:
            child_submethods_params = list(child._get_submethods())
            children_submethods_params.append(child_submethods_params)

        # Merging tuples of dicts into one dict for each decision.
        merged_results = []
        for tuple_of_decisions in list(product(*children_submethods_params)):
            merged_results.append(_merge_dicts(tuple_of_decisions))
        return merged_results

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

            if key in {
                "none",
                DecisionNodeNames.fork_node,
                DecisionNodeNames.preconditioner,
                DecisionNodeNames.parameters_picker,
            }:
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
        nodes: list[SolverConfigNode] = []
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
        children_config = _merge_dicts(children_configs)
        if optimized_only and len(children_config) == 0:
            return {}
        return {self.name: children_config}

    def decision_from_config(self, config: dict) -> Decision:
        """Translates a solver config dictionary into the solver selection decision."""
        subsolvers, parameters = self._parse_config_recursively(config)
        return Decision(subsolvers=subsolvers, parameters=parameters)

    def _parse_config_recursively(
        self, config: dict
    ) -> tuple[dict[int, str], dict[int, dict[str, number]]]:
        """Traverse config to find what subsolvers and what parameters were selected."""
        if self.name in config:
            config = config[self.name]
        submethods = []
        subparams = []

        for child in self.children:
            if child.name in config:
                subconfig = config[child.name]
            else:
                subconfig = config
            child_submethods, child_subparams = child._parse_config_recursively(
                subconfig
            )
            submethods.append(child_submethods)
            subparams.append(child_subparams)
        return _merge_dicts(submethods), _merge_dicts(subparams)


class ConstantNode(SolverConfigNode):
    """Config node, which represents a solver with no optimized parameters.

    Constant parameters can be passed with `params`.

    """

    type = "constant config node"

    def __init__(self, name: str, params: Optional[dict | str] = None) -> None:
        super().__init__(children=None, name=name)
        if params is None:
            params = {}
        self.params = params or {}

    def copy(self) -> Self:
        return type(self)(name=self.name, params=self.params)

    def config_from_decision(
        self, decision: Decision, optimized_only: bool = False
    ) -> dict:
        if optimized_only:
            return {self.name: {}}
        return {self.name: self.params}


class CategoryParameterSelector(SolverConfigNode):
    """Config node, which represents a selection from a list of categorical parameters."""

    def __init__(self, name: str, options: Sequence[str]):
        super().__init__(name=name)
        self.options = options

    def copy(self):
        return CategoryParameterSelector(name=self.name, options=self.options)

    def config_from_decision(
        self, decision: Decision, optimized_only: bool = False
    ) -> dict:
        chosen = decision.subsolvers[self._id]
        assert chosen in self.options
        return {self.name: chosen}

    def _get_submethods(self) -> Sequence[dict[int, str | ParametersSpace]]:
        return [{self._id: option} for option in self.options]

    def _parse_config_recursively(
        self, config: dict
    ) -> tuple[dict[int, str], dict[int, dict[str, number]]]:
        decision = config
        assert decision in self.options
        return {self._id: decision}, {}


class DecisionNodeNames:
    fork_node = "linear solver"
    preconditioner = "preconditioner"
    parameters_picker = "parameter picker"
    splitting = "splitting"


class ForkNode(SolverConfigNode):
    """Solver config node, which represent a variety of sub-solvers. The solver
    selection algorithm must select one of them. Thus, this is a categorical decision
    to make.

    """

    type = DecisionNodeNames.fork_node

    def __init__(
        self,
        options: Sequence[SolverConfigNode],
        name: Optional[str] = None,
    ):
        self.options = {opt.name: opt.copy() for opt in options}
        assert len(self.options) == len(options), "Options must have unique names"
        super().__init__(children=list(self.options.values()), name=name)

    def copy(self) -> Self:
        return type(self)(options=list(self.options.values()), name=self.name)

    def _get_submethods(self) -> Sequence[dict[int, str | ParametersSpace]]:
        all_options: list[dict[int, str | ParametersSpace]] = []
        for option_name, option in self.options.items():
            base_option = {self._id: option_name}
            solver_spaces_child: list[dict[int, str | ParametersSpace]] = list(
                option._get_submethods()
            )
            if len(solver_spaces_child) == 0:
                solver_spaces_child = [base_option]  # type: ignore[list-item]
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
        if self.name in config:
            config = config[self.name]

        chosen_child_name = None
        for child_name, child in self.options.items():
            if child_name in config:
                chosen_child_name = child_name
                config = config[child_name]
                break

        # An exotic case when a child is also the decision node without a name.
        if chosen_child_name is None:
            child_name = DecisionNodeNames.fork_node
            child = self.options[child_name]
            if isinstance(child, ForkNode):
                chosen_child_name = child_name

        if chosen_child_name is None:
            raise ValueError(
                f"Inconsistent solver space: Node {self._id}: {self.name}"
                f" with options: {[name for name in self.options.keys()]}"
                f" and config passed: {config}"
            )

        child_decision, child_params = child._parse_config_recursively(config)
        return {self._id: child_name} | child_decision, child_params


class PreconditionerDecisionNode(ForkNode):
    """Represents a variety of available preconditioner configurations to select from."""

    type = DecisionNodeNames.preconditioner


class KrylovSolverNode(SolverConfigNode):
    """Represents one specific Krylov solver algorithm configuration."""

    @classmethod
    def from_preconditioners_list(
        cls,
        preconditioners: Sequence[SolverConfigNode],
        other_children: Optional[Sequence[SolverConfigNode]] = None,
        **kwargs,
    ):
        """Convenience method to assemble a space of a Krylov solver with various
        preconditioners.

        """
        preconditioners = [prec.copy() for prec in preconditioners]
        children: list[SolverConfigNode] = [
            PreconditionerDecisionNode(options=preconditioners)
        ]
        if other_children is not None:
            children.extend(other_children)

        return cls(
            children=children,
            **kwargs,
        )


class SplittingNode(SolverConfigNode):
    """Represents one specific decoupling/splitting algorithm."""

    type = DecisionNodeNames.splitting

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
            ForkNode(solvers, name=key) for key, solvers in solver_nodes.items()
        ]
        return cls(solver_nodes=solvers_list, **kwargs)

    def __init__(
        self,
        solver_nodes: Sequence[SolverConfigNode],
        name: Optional[str] = None,
        other_children: Optional[Sequence[SolverConfigNode]] = None,
    ):
        self.solver_nodes = [node.copy() for node in solver_nodes]
        if other_children is None:
            other_children = []
        self.other_children = [child.copy() for child in other_children]
        super().__init__(
            children=self.solver_nodes + self.other_children,
            name=name,
        )

    def _get_submethods(self) -> Sequence[dict[int, str | ParametersSpace]]:
        result = []
        for c in self.children:
            if len(arms := c._get_submethods()) > 0:
                result.append(arms)
        return [_merge_dicts(pair) for pair in tuple(product(*result))]

    def config_from_decision(
        self, decision: Decision, optimized_only: bool = False
    ) -> dict:
        config = super().config_from_decision(decision, optimized_only)
        if self.name == DecisionNodeNames.splitting:
            return config[self.name]
        return config

    def _parse_config_recursively(
        self, config: dict
    ) -> tuple[dict[int, str], dict[int, dict[str, number]]]:
        if self.name in config:
            config = config[self.name]
        return super()._parse_config_recursively(config)

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
        name: Optional[str] = None,
    ):
        # If the user passed numbers, make constant parameters from them.
        tmp: dict[str, NumericalParameter] = {}
        for param_name, param in numerical.items():
            if isinstance(param, int | float):
                param = NumericalParameter(
                    bounds=(param, param),
                    default=param,
                    is_optimized=False,
                )
            tmp[param_name] = param

        self.parameters: ParametersSpace = ParametersSpace(numerical=tmp)
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
    expected_len = sum(len(d) for d in list_of_dicts)
    for entry in list_of_dicts:
        result.update(entry)
    assert len(result) == expected_len, "Ids are duplicating."
    return result
