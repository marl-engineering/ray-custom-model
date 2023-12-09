from dataclasses import dataclass


@dataclass
class AgentConfig:
    n_actions: int 
    input_dims: tuple
    gamma: float
    epsilon: float
    lr: float
    mem_size: int
    batch_size: float
    eps_min: float
    eps_dec: float
    replace: int

