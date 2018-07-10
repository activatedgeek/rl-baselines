import gym
import torchrl.registry as registry
from torchrl.agents.random_agent import RandomAgent


@registry.register_problem('random-cartpole')
class RandomProblem(registry.Problem):
  def make_env(self):
    return gym.make('CartPole-v1')

  def init_agent(self):
    observation_space, action_space = self.get_gym_spaces()

    return RandomAgent(observation_space, action_space)

  def train(self, history_list: list) -> dict:
    return {}
