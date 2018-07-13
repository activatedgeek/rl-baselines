import time
import gym

from .base_runner import BaseRunner
from ..agents import BaseAgent


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

    self.state = None

  def make_env(self, seed: int = None) -> gym.Env:
    """Create the environment, optionally with seed."""
    env = gym.make(self.env_id)
    env.seed(seed)
    return env

  def reset(self):
    """Reset the environment."""
    self.state = self.env.reset()

  def get_action(self, agent: BaseAgent):
    """Use the agent to get actions.

    This method should be overridden to
    make use of the agent.
    """
    return agent.act([self.state])[0][0]

  def process_transition(self, history: list,
                         transition: tuple) -> list:
    """Process the transition tuple and update history.

    This routine takes the transition tuple of
    (state, action, reward, next_state, done, info),
    applies transformation and appends to the history
    list. This method should be overridden for any
    non-trivial transformations needed, for instance
    conversion of boolean done to stack of ints.
    """
    history.append(transition)
    return history

  def rollout(self, agent, steps: int = None,
              render: bool = False, fps: int = 30):
    """Execute a rollout of the given environment.

    This is a simple utility and the main entrypoint to
    the runner. It allows flags for rendering and the
    maximum number of steps to execute in the current
    rollout.
    """
    assert self.state is not None, """state is not defined,
    please use `.reset()`
    """

    steps = steps or self.MAX_STEPS

    if render:
      self.env.render()
      time.sleep(1. / fps)

    history = []

    while steps:
      action = self.get_action(agent)
      next_state, reward, done, info = self.env.step(action)

      history = self.process_transition(history,
                                        (self.state, action,
                                         reward, next_state, done,
                                         info))

      if render:
        self.env.render()
        time.sleep(1. / fps)

      if done:
        break

      steps -= 1
      self.state = next_state

    return history

  def close(self):
    """Close the environment."""
    self.env.close()
