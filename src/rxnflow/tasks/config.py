from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class MOOTaskConfig:
    """Common Config for the MOOTasks

    Attributes
    ----------
    objectives : list[str]
        The objectives to use for the multi-objective optimization.
    n_valid : int
        The number of valid cond_info tensors to sample.
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors.
    online_pareto_front : bool
        Whether to calculate the pareto front online.
    """

    objectives: list[str] = field(default_factory=lambda: [])
    n_valid: int = 15
    n_valid_repeats: int = 128
    log_topk: bool = False
    online_pareto_front: bool = True


@dataclass
class DockingTaskConfig:
    """Config for DockingTask

    Attributes
    ----------
    protein_path: str (path)
        Protein path
    ref_ligand_path: str (path)
        Reference ligand path
    """

    protein_path: str = MISSING
    ref_ligand_path: str = MISSING


@dataclass
class SemlaFlowConfig:
    """Config for SemlaFlow Module

    Attributes
    ----------
    ckpt_path: str (path)
        checkpoint path of cgflow
    ref_ligand_path: str (path)
        Reference ligand path
    """

    ckpt_path: str = MISSING


@dataclass
class ConstraintConfig:
    """Config for Filtering

    Attributes
    ----------
    rule: str (path)
        DrugFilter Rule
            - None
            - lipinski
            - veber
    """

    rule: str | None = None


@dataclass
class TasksConfig:
    moo: MOOTaskConfig = field(default_factory=MOOTaskConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    docking: DockingTaskConfig = field(default_factory=DockingTaskConfig)
