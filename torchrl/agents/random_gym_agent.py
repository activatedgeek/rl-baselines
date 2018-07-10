from .base_agent import BaseAgent


class RandomGymAgent(BaseAgent):
  """Take random actions on a Gym environment.

  @NOTE: Work in Progress. Not supported yet.
  """
  @property
  def models(self) -> list:
    return []

  @property
  def checkpoint(self) -> object:
    return None

  def act(self, obs):
    return [[self.action_space.sample()] for _ in range(len(obs))]

  def learn(self, *args, **kwargs):
    return {}
