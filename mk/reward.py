from dataclasses import dataclass
import abc

@dataclass
class Info:
    enemy_health: int
    enemy_matches_won: int
    health: int
    matches_won: int
    wins: int

class Reward(abc.ABCMeta):
    def compute_reward(reward: float, info: Info) -> float:
        pass
 

class PretrainReward(Reward):
    def __init__(self):
        super().__init__()
        self.health = 166
        self.enemy_health = 166

    def compute_reward(reward: float, info: Info) -> float:
        reward = reward * 1000.
        reward -= self.health - info.health
        self.health = info.health
        reward += self.enemy_health - info.enemy_health
        self.enemy_health = info.enemy_health
        return reward


class FairReward(Reward):
    def compute_reward(reward: float, info: Info) -> float:
        return reward
