import time
import gym

from .base_runner import BaseRunner
from ..agents import BaseAgent
from ..utils.gym_utils import init_run_history
from ..utils.gym_utils import append_run_history


class GymRunner(BaseRunner):
  """Runner for OpenAI Gym Environments

  This class is a simple wrapper around
  OpenAI Gym environments with essential
  plug points into various steps of the
  rollout.
  """
  def __init__(self, env_id: str, seed: int = None):
    super(GymRunner, self).__init__()

    self.env_id = env_id
    self.env = self.make_env(seed)

    self.obs = None

  def make_env(self, seed: int = None) -> gym.Env:
    env = gym.make(self.env_id)
    env.seed(seed)
    return env

  def reset(self):
    self.obs = self.env.reset()

  def compute_action(self, agent: BaseAgent):
    """Compute Actions from the agent.

    Note the conversion to list. This is
    because the underlying act function handles
    batches and not single instances.
    """
    return agent.act([self.obs])[0][0]

  def process_transition(self, history,
                         transition: tuple) -> list:
    if history is None:
      history = init_run_history(self.env.observation_space,
                                 self.env.action_space)

    append_run_history(history, *transition[:-1])
    return history

  def rollout(self, agent, steps: int = None,
              render: bool = False, fps: int = 30):
    assert self.obs is not None, """state is not defined,
    please `reset()`
    """

    steps = steps or self.MAX_STEPS

    if render:
      self.env.render()
      time.sleep(1. / fps)

    history = None

    while steps:
      action = self.compute_action(agent)
      next_state, reward, done, info = self.env.step(action)

      history = self.process_transition(history,
                                        (self.obs, action,
                                         reward, next_state, done,
                                         info))

      if render:
        self.env.render()
        time.sleep(1. / fps)

      if done:
        break

      steps -= 1
      self.obs = next_state

    return history

  def close(self):
    """Close the environment."""
    self.env.close()
