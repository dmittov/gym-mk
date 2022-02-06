from dataclasses import dataclass


@dataclass
class Info:
    enemy_health: int
    enemy_matches_won: int
    health: int
    matches_won: int
    wins: int


def compute_reward(reward: float, info: Info) -> float:
    return reward
