from mk.model import View
import numpy as np
from torch import Tensor


class TestView:
    def test_output_size(self):
        n_batch = 128
        expected_input_shape = (n_batch, 3, 100, 128)
        view = View(frames=3)
        frames = np.random.randn(*expected_input_shape)
        t_frames = Tensor(frames)
        t_img_embedding = view.forward(t_frames)
        img_embedding = t_img_embedding.detach().numpy()
        assert img_embedding.shape == (n_batch, 6720)


# class TestActor:
