import gym


class TransitionMonitor(gym.Wrapper):
  def __init__(self, env: gym.Env):
    super().__init__(env)

    # Episode Monitors
    self._ep_return = None
    self._ep_len = None
    self._ep_done = False
    self._ep_trans = []
    self._ep_info = {}
    self._obs = None

  @property
  def obs(self):
    return self._obs

  @property
  def is_done(self) -> bool:
    return self._ep_done

  @property
  def transitions(self) -> list:
    return self._ep_trans

  @property
  def info(self) -> dict:
    return {**self._ep_info,
            'return': self._ep_return,
            'len': self._ep_len}

  def reset(self, **kwargs):
    self._ep_return = 0.0
    self._ep_len = 0
    self._ep_done = False
    self._ep_trans = []
    self._ep_info = {}
    self._obs = self.env.reset(**kwargs)
    return self._obs

  def step(self, action):
    next_obs, reward, self._ep_done, self._ep_info = self.env.step(action)

    transition = (self.obs, action, reward, next_obs, self._ep_done)

    self._ep_return += reward
    self._ep_len += 1
    self._ep_trans.append(transition)
    self._obs = next_obs

    return next_obs, reward, self._ep_done, self._ep_info
