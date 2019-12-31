import pytest

from torchrl.envs import make_gym_env, TransitionMonitor


@pytest.mark.parametrize('spec_id', [
    'Acrobot-v1',
    'CartPole-v1',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
])
def test_transition_monitor(spec_id: str):
  env = TransitionMonitor(make_gym_env(spec_id))

  for _ in range(2):
    env.reset()

    info = env.info
    assert not env.is_done
    assert info.get('len') == 0
    assert info.get('return') == 0.0

    while not env.is_done:
      env.step(env.action_space.sample())

    info = env.info

    assert info.get('return') is not None
    assert info.get('len') > 0
    assert info.get('len') == len(env.transitions)

  env.close()
