# pylint: disable=redefined-outer-name

import gym
import pytest
import torchrl.registry as registry
import torchrl.utils as utils
import torchrl.utils.cli as cli
import torchrl.problems.base_hparams as base_hparams
from torchrl.agents.gym_random_agent import GymRandomAgent


@pytest.fixture(scope='function')
def problem_argv(request):
  env_id = request.param
  args_dict = {
      'problem': 'random_gym_problem',
      'extra-hparams': 'num_total_steps=100',
  }
  argv = ['--{}={}'.format(key, value) for key, value in args_dict.items()]

  @registry.register_problem  # pylint: disable=unused-variable
  class RandomGymProblem(registry.Problem):
    def make_env(self):
      return gym.make(env_id)

    def init_agent(self):
      observation_space, action_space = utils.get_gym_spaces(self.make_env)

      return GymRandomAgent(observation_space, action_space)

    def train(self, history_list: list) -> dict:
      return {}

  @registry.register_hparam
  def random_hparams():  # pylint: disable=unused-variable
    return base_hparams.base()

  yield argv

  registry.remove_problem('random_gym_problem')
  registry.remove_hparam('random_hparams')


@pytest.mark.parametrize('problem_argv', [
    'Acrobot-v1',
    'CartPole-v1',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
], indirect=['problem_argv'])
def test_gym_agent(problem_argv):
  cli.main(problem_argv)
