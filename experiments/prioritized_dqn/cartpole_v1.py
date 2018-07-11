import gym
import torchrl.registry as registry
import torchrl.utils as utils
from torchrl.problems import PrioritizedDQNProblem
from torchrl.agents import BaseDQNAgent

from ..dqn.cartpole_v1 import hparam_dqn_cartpole


@registry.register_problem('prioritized-dqn-cartpole-v1')
class PrioritizedCartPoleDQNProblem(PrioritizedDQNProblem):
  def make_env(self):
    return gym.make('CartPole-v1')

  def init_agent(self):
    observation_space, action_space = utils.get_gym_spaces(self.make_env)

    agent = BaseDQNAgent(
        observation_space,
        action_space,
        double_dqn=self.hparams.double_dqn,
        lr=self.hparams.actor_lr,
        gamma=self.hparams.gamma,
        num_eps_steps=self.hparams.num_eps_steps,
        target_update_interval=self.hparams.target_update_interval)

    return agent


@registry.register_hparam('per-dqn-cartpole')
def hparam_per_dqn_cartpole():
  params = hparam_dqn_cartpole()

  params.alpha = 0.6
  params.beta = 0.4
  params.beta_anneal_steps = 1000

  return params
