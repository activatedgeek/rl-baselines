import abc


class BaseRunner(metaclass=abc.ABCMeta):
  """An abstract class for runner.

  This class provides the spec for what
  a runner should be.
  """
  MAX_STEPS = int(1e6)

  @abc.abstractmethod
  def make_env(self, seed: int = None):
    raise NotImplementedError

  @abc.abstractmethod
  def get_action(self, *args, **kwargs):
    raise NotImplementedError

  @abc.abstractmethod
  def process_transition(self, *args, **kwargs):
    raise NotImplementedError

  @abc.abstractmethod
  def rollout(self, agent, steps: int = None,
              render: bool = False, fps: int = 30):
    raise NotImplementedError
