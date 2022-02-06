from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Info:
    enemy_health: int
    enemy_matches_won: int
    health: int
    matches_won: int
    wins: int


class Reward(ABC):
    """
    Abstract reward

    Barto & Sutton 3.2 Goals & Rewards
    We want the Agent use the final reward only, but it's hard to learn
    a meaningfull initial policy.

    The random initial policy will lose in most cases, so we use sub-goal
    rewards learn a good initial policy/states values.
    """

    @staticmethod
    @abstractmethod
    def compute_reward(reward: float, info: Info) -> float:
        pass


class TrueReward(Reward):
    @staticmethod
    def compute_reward(reward: float, info: Info) -> float:
        _ = info
        return reward


class MatchWinReward(Reward):
    @staticmethod
    def compute_reward(reward: float, info: Info) -> float:
        pass
