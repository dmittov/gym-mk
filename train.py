import click
import torch
from mk.wrappers import apply_warppers
from mk.agent import RandomAgent
from mk.reward import TrueReward, Info
import logging
import numpy as np
from mk.worker import Worker, make_env
import torch.multiprocessing as mp
from mk.model import View, HealthPredictor
from mk.reward import Info, Episode
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from typing import List
from dataclasses import dataclass
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeBatch:
    info: List[Info]
    reward: np.ndarray
    is_done: np.ndarray
    state: np.ndarray
    action: np.ndarray
    n_workers: int
    top_worker_ratio: float


class HealthLearner:
    def __init__(self, exit_event: mp.Event, q: mp.Queue, batch_size: int = 64) -> None:
        super().__init__()
        self.q = q
        self.exit_event = exit_event
        self.batch_size = batch_size

    def make_batch(self) -> EpisodeBatch:
        episodes: List[Episode] = list()
        for _ in range(self.batch_size):
            episodes.append(self.q.get())
        workers = Counter(ep.worker for ep in episodes)
        batch = EpisodeBatch(
            info=[ep.info for ep in episodes],
            reward=np.array([ep.reward for ep in episodes]),
            is_done=np.array([ep.is_done for ep in episodes]),
            state=np.stack([ep.state for ep in episodes], axis=0),
            action=np.stack([ep.action for ep in episodes], axis=0),
            n_workers=len(workers),
            top_worker_ratio=1.0
            * workers.most_common(1)[0][1]
            / sum(list(workers.values())),
        )
        return batch

    def run(self) -> None:
        warm_up = 10
        logger.info("Consumer started")
        predictor = HealthPredictor(View(frames=3))
        optimizer = torch.optim.Adam(predictor.parameters())
        writer = SummaryWriter()
        episode_idx = 0

        predictor.train()

        for episode_idx in tqdm(range(200)):
            batch: EpisodeBatch = self.make_batch()
            optimizer.zero_grad()
            t_input = torch.Tensor(batch.state)
            t_true_health = torch.Tensor(
                np.array([[info.health for info in batch.info]])
            )
            t_pred_health = predictor(t_input)
            t_loss = F.mse_loss(t_pred_health, t_true_health.reshape(t_pred_health.shape))
            t_loss.backward()
            optimizer.step()
            loss = float(t_loss.cpu().detach().numpy())
            if episode_idx > warm_up:  # and (episode_idx % 100 == 0):
                loss = float(t_loss.cpu().detach().numpy())
                writer.add_scalar("Loss", loss, episode_idx)
                gradients = predictor.actions.weight.grad.cpu().detach().numpy()[0]
                gradients = sum(np.abs(gradients) ** 2)
                writer.add_scalar("GradNorm", gradients, episode_idx)
                y_pred = np.array(t_pred_health.cpu().detach().numpy())
                y_true = t_true_health.cpu().detach().numpy().ravel()
                mae = mean_absolute_error(y_true, y_pred)
                writer.add_scalar("MAE", mae, episode_idx)
                writer.add_scalar("True", y_true[0], episode_idx)
                writer.add_scalar("N_workers", batch.n_workers, episode_idx)
                writer.add_scalar(
                    "Top worker ratio", batch.top_worker_ratio, episode_idx
                )
                # matplotlib.image.imsave(f"img/img_{episode_idx}_{y_pred[0]}_{y_true[0]}.png", episode.state)
        self.exit_event.set()


@click.command()
@click.option("--workers", default=5, help="Number of game processes", type=int)
def play(workers: int) -> None:
    exit_producer = mp.Event()
    exit_consumer = mp.Event()
    q = mp.Queue(maxsize=5_000)
    agent = RandomAgent()
    workers = [
        Worker(
            exit_event=exit_producer,
            make_env=make_env,
            q=q,
            agent=agent,
        )
        for _ in range(workers)
    ]
    consumer = HealthLearner(exit_event=exit_producer, q=q)
    [worker.start() for worker in workers]
    logger.info("Workers started")
    consumer.run()
    # while not exit_producer.is_set():
    #     # print(f"Queue size: {q.qsize()}")
    #     time.sleep(2)
    [worker.join() for worker in workers]
    logger.info("All process closed")


if __name__ == "__main__":
    play()
