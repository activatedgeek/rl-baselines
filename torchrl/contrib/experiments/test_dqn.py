import pytest
import torch
from torchrl.contrib.experiments.dqn import DQNExperiment


@pytest.mark.parametrize('spec_id, n_frames, buffer_size', [
    ('Acrobot-v1', 200, 500),
    ('CartPole-v1', 600, 500),
    ('MountainCar-v0', 800, 500),
])
def test_base_exp(spec_id: str, n_frames: str, buffer_size: int):
  exp = DQNExperiment(env_id=spec_id, n_frames=n_frames,
                      buffer_size=buffer_size)
  exp.run()

  assert len(exp.buffer) == min(buffer_size, n_frames)
  ids = torch.randperm(len(exp.buffer))[:len(exp.buffer) // 2]
  for v in exp.buffer[ids]:
    assert v.size(0) == ids.size(0)
  assert exp._cur_frames == n_frames
