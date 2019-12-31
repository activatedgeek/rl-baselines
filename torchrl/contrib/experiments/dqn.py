from torchrl.experiments import BaseExperiment
from torchrl.contrib.controllers import DQNController
from torchrl.utils.storage import TransitionTupleDataset


class DQNExperiment(BaseExperiment):
  def __init__(self, double_dqn=False, gamma=.99, rollout_steps=100,
               batch_size=32, lr=1e-3, buffer_size=1000, eps_max=1.0,
               eps_min=1e-2, n_eps_anneal=500, n_update_interval=10, **kwargs):
    self._controller_args = dict(
        double_dqn=double_dqn,
        gamma=gamma,
        lr=lr,
        eps_max=eps_max,
        eps_min=eps_min,
        n_eps_anneal=n_eps_anneal,
        n_update_interval=n_update_interval,
    )

    self.buffer = TransitionTupleDataset(size=buffer_size)

    super().__init__(**kwargs)

  def store(self, transition_list):
    self.buffer.extend(transition_list)

  def build_controller(self):
    return DQNController(obs_size=self.rollout_env.observation_space.shape[0],
                         action_size=self.rollout_env.action_space.n,
                         **self._controller_args)
