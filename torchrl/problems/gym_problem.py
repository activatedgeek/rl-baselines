from ..registry import Problem
from ..runners import GymRunner
from ..runners import BaseRunner


class GymProblem(Problem):
  def make_runner(self, n_envs=1, seed=None) -> BaseRunner:
    return GymRunner(self.env_id, n_envs=n_envs, seed=seed)
