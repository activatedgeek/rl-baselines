import gym

import torchrl.registry as registry
import torchrl.registry.hparams as hparams
from torchrl.registry.problems import DQNProblem
from torchrl.learners import BaseDQNLearner


class CartPoleDQNLearner(BaseDQNLearner):
  def learn(self, obs, action, reward, next_obs, done):
    for i, _ in enumerate(reward):
      if done[i] == 1:
        reward[i] = -1.0

    return super(CartPoleDQNLearner, self).learn(
        obs, action, reward, next_obs, done)


@registry.register_problem('dqn-cartpole-v1')
class CartPoleDQNProblem(DQNProblem):
  def make_env(self):
    return gym.make('CartPole-v1')

  def init_agent(self):
    hparams = self.hparams

    observation_space, action_space = self.get_gym_spaces()

    agent = CartPoleDQNLearner(
        observation_space,
        action_space,
        lr=hparams.actor_lr,
        gamma=hparams.gamma,
        target_update_interval=hparams.target_update_interval)

    return agent


@registry.register_hparam('dqn-cartpole')
def hparam_dqn_cartpole():
  params = hparams.base_dqn()

  params.rollout_steps = 1
  params.num_processes = 1
  params.actor_lr = 1e-3
  params.gamma = 0.8
  params.target_update_interval = 5
  params.eps_min = 0.15
  params.buffer_size = 5000
  params.batch_size = 64
  params.num_total_steps = 10000

  return params
