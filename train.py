import click
import retro
import torch
from typing import Optional
from mk.wrappers import apply_warppers
from mk.agent import RandomAgent
from mk.reward import TrueReward, Info
import logging
import numpy as np
from mk.worker import Worker, make_env
import time
import torch.multiprocessing as mp
from mk.model import View, HealthPredictor
from mk.reward import Info, Episode
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import matplotlib.image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthLearner(mp.Process):
    def __init__(self, exit_event: mp.Event, queue: mp.Queue) -> None:
        super().__init__()
        self.queue = queue
        self.exit_event = exit_event

    def run(self) -> None:
        warm_up = 1000
        logger.info("Consumer started")
        predictor = HealthPredictor(View(frames=3))
        optimizer = torch.optim.Adam(predictor.parameters())
        writer = SummaryWriter()
        episode_idx = 0

        predictor.train()

        for episode_idx in tqdm(range(100_000)):
            episode: Episode = self.queue.get()
            optimizer.zero_grad()
            t_input = torch.Tensor(
                episode.state[np.newaxis, :]
            )
            t_true_health = torch.Tensor(
                np.array([[episode.info.health]])
            )
            t_pred_health = predictor(t_input)
            t_loss = F.mse_loss(t_pred_health, t_true_health)
            t_loss.backward()
            optimizer.step()
            loss = float(t_loss.cpu().detach().numpy())
            if (episode_idx > warm_up) and (episode_idx % 100 == 0):
                loss = float(t_loss.cpu().detach().numpy())
                writer.add_scalar("Loss", loss, episode_idx)
                y_pred = np.array([float(t_pred_health.cpu().detach().numpy())])
                y_true = t_true_health.cpu().detach().numpy().ravel()
                mae = mean_absolute_error(y_true, y_pred)
                writer.add_scalar("MAE", mae, episode_idx)
                writer.add_scalar("True", y_true[0], episode_idx)
                # matplotlib.image.imsave(f"img/img_{episode_idx}_{y_pred[0]}_{y_true[0]}.png", episode.state)
        self.exit_event.set()
        

@click.command()
@click.option("--workers", default=5, help="Number of game processes", type=int)
def play(workers: int) -> None:
    exit_producer = mp.Event()
    exit_consumer = mp.Event()
    queue = mp.Queue(maxsize=5_000)
    agent = RandomAgent()
    workers = [
        Worker(
            exit_event=exit_producer, 
            make_env=make_env, 
            queue=queue,
            agent=agent,
        ) 
        for _ in range(workers)
    ]
    consumer = HealthLearner(exit_event=exit_producer, queue=queue)
    [worker.start() for worker in workers]
    logger.info("Workers started")
    consumer.start()
    logger.info("Consumer started")
    # while not exit_producer.is_set():
    #     # print(f"Queue size: {queue.qsize()}")
    #     time.sleep(2)
    [worker.join() for worker in workers]
    consumer.join()
    logger.info("All process closed")


if __name__ == "__main__":
    play()
