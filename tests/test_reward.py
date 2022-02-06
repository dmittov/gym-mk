from ast import In
import pytest
from mk.reward import TrueReward, Info
import numpy as np


class TestTrueReward:
    def test_compute_reward(self):
        info = Info(
            enemy_health=100,
            enemy_matches_won=1,
            health=100,
            matches_won=1,
            wins=1,
        )
        reward = TrueReward.compute_reward(1.0, info)
        assert np.isclose(reward, 1.0)
