from dataclasses import dataclass, field

from omegaconf import MISSING

from rxnflow import config
from rxnflow.config import AlgoConfig, ConditionalsConfig, ModelConfig, OptimizerConfig, ReplayConfig
from synthflow.tasks.config import TasksConfig


@dataclass
class SemlaFlowConfig:
    """Config for SemlaFlow Module

    Attributes
    ----------
    ckpt_path: str (path)
        checkpoint path of cgflow
    use_predicted_pose: bool
        if True, use \\hat{x}_1 instead of x_t
    num_inference_steps: int
        Number of inference steps
    """

    ckpt_path: str = MISSING
    use_predicted_pose: bool = True
    num_inference_steps: int = 100


@dataclass
class Config(config.Config):
    """Base configuration for training

    Attributes
    ----------
    desc : str
        A description of the experiment
    log_dir : str
        The directory where to store logs, checkpoints, and samples.
    device : str
        The device to use for training (either "cpu" or "cuda[:<device_id>]")
    seed : int
        The random seed
    validate_every : int
        The number of training steps after which to validate the model
    checkpoint_every : Optional[int]
        The number of training steps after which to checkpoint the model
    store_all_checkpoints : bool
        Whether to store all checkpoints or only the last one
    print_every : int
        The number of training steps after which to print the training loss
    start_at_step : int
        The training step to start at (default: 0)
    num_final_gen_steps : Optional[int]
        After training, the number of steps to generate graphs for
    num_training_steps : int
        The number of training steps
    num_workers : int
        The number of workers to use for creating minibatches (0 = no multiprocessing)
    hostname : Optional[str]
        The hostname of the machine on which the experiment is run
    pickle_mp_messages : bool
        Whether to pickle messages sent between processes (only relevant if num_workers > 0)
    git_hash : Optional[str]
        The git hash of the current commit
    overwrite_existing_exp : bool
        Whether to overwrite the contents of the log_dir if it already exists
    """

    desc: str = "noDesc"
    log_dir: str = MISSING
    device: str = "cuda"
    seed: int = 0
    validate_every: int = 1000
    checkpoint_every: int | None = None
    store_all_checkpoints: bool = False
    print_every: int = 100
    start_at_step: int = 0
    num_final_gen_steps: int | None = None
    num_validation_gen_steps: int | None = None
    num_training_steps: int = 10_000
    num_workers: int = 0
    num_workers_retrosynthesis: int = 4  # For retrosynthetic analysis
    hostname: str | None = None
    pickle_mp_messages: bool = False
    git_hash: str | None = None
    overwrite_existing_exp: bool = False
    env_dir: str = MISSING
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    opt: OptimizerConfig = field(default_factory=OptimizerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    cond: ConditionalsConfig = field(default_factory=ConditionalsConfig)
    cgflow: SemlaFlowConfig = field(default_factory=SemlaFlowConfig)
    task: TasksConfig = field(default_factory=TasksConfig)


def init_empty(cfg: Config) -> Config:
    """
    Initialize a dataclass instance with all fields set to MISSING,
    including nested dataclasses.

    This is meant to be used on the user side (tasks) to provide
    some configuration using the Config class while overwritting
    only the fields that have been set by the user.
    """
    return config.init_empty(cfg)
