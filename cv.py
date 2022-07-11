import click
import retro
from typing import Optional
from mk.wrappers import apply_warppers
from mk.agent import RandomAgent
from mk.reward import compute_reward, Info
import logging
import numpy as np
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--model", default=None, help="Path to the model checkpoint")
@click.option("--games", default=5, help="Number of games to play", type=int)
def play(model: Optional[str], games: int) -> None:
    env = retro.RetroEnv(game="MortalKombat3-Genesis", players=1, record=False)
    env = apply_warppers(env)
    agent = RandomAgent()
    logger.info("Agent: %s", agent)
    rewards = []
    if model:
        logger.info("Loading model from: %s", model)
        agent.load_model(model)
    for game_idx in range(games):
        state = env.reset()
        reward_computer = PretrainReward()
        is_done = False
        while not is_done:
            action = agent.act(state)
            next_state, reward, is_done, raw_info = env.step(action)
            state = next_state
            info = Info(**raw_info)
            if is_done:
                reward = reward_computer.compute_reward(reward, info)
                rewards.append(reward)
    logger.info("Avg reward in %d games: %f", games, np.mean(rewards))


if __name__ == "__main__":
    play()
