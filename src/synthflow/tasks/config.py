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
    redocking: bool
        if True, run docking; otherwise, run local opt
    ff_opt: bool
        if True when redocking is False, run UFF optimization before local opt
    """

    protein_path: str = MISSING
    ref_ligand_path: str = MISSING
    redocking: bool = True  # TODO: change to docking mode (score-only, local-opt, redock)
    ff_opt: bool = True


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
class PocketConditionalConfig:
    """Config for PocketConditional Training

    Attributes
    ----------
    proxy: tuple[str, str, str] (proxy_name, docking_program, train_dataset)
        Proxy Key from PharmacoNet
    """

    proxy: tuple[str, str,
                 str] = ("TacoGFN_Reward", "QVina", "CrossDocked2020")
    pocket_dir: str = "./data/CrossDocked2020/"
    train_key: str = "./data/CrossDocked2020/train_keys.csv"


@dataclass
class TasksConfig:
    moo: MOOTaskConfig = field(default_factory=MOOTaskConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    docking: DockingTaskConfig = field(default_factory=DockingTaskConfig)
    pocket_conditional: PocketConditionalConfig = field(
        default_factory=PocketConditionalConfig)
