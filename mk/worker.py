from typing import Callable
import torch.multiprocessing as mp
from typing import Callable
import logging
import gym
import retro
from mk.reward import Info, Episode
from mk.agent import Agent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_env() -> gym.Env:
    env = retro.RetroEnv(game="MortalKombat3-Genesis", players=1, record=False)
    return env


class Worker(mp.Process):

    def __init__(self, 
                 exit_event: mp.Event, 
                 make_env: Callable, 
                 queue: mp.Queue,
                 agent: Agent
                 ) -> None:
        super().__init__()
        self.make_env = make_env
        self.exit_event = exit_event
        self.agent = agent
        self.queue = queue

    def run(self) -> None:
        env = self.make_env()
        logger.info(f"Process {self.name} started ...")
        while not self.exit_event.is_set():
            state = env.reset()
            is_done = False
            episode_idx = 0
            while not is_done:
                action = self.agent.act(state)
                next_state, reward, is_done, raw_info = env.step(action)
                if episode_idx % 10 == 0:
                    info = Info(**raw_info)
                    episode = Episode(
                        info=info,
                        reward=reward,
                        is_done=is_done,
                        state=state,
                        action=action,
                    )
                    self.queue.put(episode)
                state = next_state
                episode_idx += 1
                if is_done:
                    pass

