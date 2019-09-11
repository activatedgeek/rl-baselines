import gym
import numpy as np
from typing import Callable, Tuple


def get_gym_spaces(make_env_fn: Callable[..., gym.Env]) -> Tuple[gym.Space, gym.Space]:  # pylint: disable=line-too-long
  """
  A utility function to get observation and actions spaces of a
  Gym environment
  """
  env = make_env_fn()
  observation_space = env.observation_space
  action_space = env.action_space
  env.close()
  return observation_space, action_space


def init_run_history(observation_space: gym.Space,
                     action_space: gym.Space) -> list:
  """Initialize history for numpy-based Gym environments.

  The function returns numpy arrays in the following order
  * obs_history: T x observation_shape
  * action_history: T x action_shape
      action_shape is 1 for discrete environments
  * reward history: T x 1
  * next_obs_history: T x observation_shape
  * done_history: T x 1
  """
  is_discrete = action_space.__class__.__name__ == 'Discrete'

  obs_history = np.empty((0, *observation_space.shape), dtype=np.float)
  action_history = np.empty((0, *((1,) if is_discrete else
                                  action_space.shape)),
                            dtype=np.int if is_discrete else np.float)
  reward_history = np.empty((0, 1), dtype=np.float)
  next_obs_history = np.empty_like(obs_history)
  done_history = np.empty((0, 1), dtype=np.int)

  return [
      obs_history, action_history, reward_history,
      next_obs_history, done_history
  ]
