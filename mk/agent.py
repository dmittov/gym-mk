import numpy as np
from abc import ABC


class Agent(ABC):
    def act(self, state: np.ndarray) -> np.ndarray:
        pass

    def load_model(self, model: str):
        pass


class RandomAgent(Agent):
    def __init__(self):
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        return np.random.choice([0, 1], 12).astype(np.int8)
