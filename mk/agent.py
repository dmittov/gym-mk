import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def load_model(self, model: str):
        pass


class RandomAgent(Agent):
    def __init__(self):
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        return np.random.choice([0, 1], 12).astype(np.int8)
