from torchrl import registry
from torchrl import utils
from torchrl.problems import base_hparams, DDPGProblem
from torchrl.agents import BaseDDPGAgent

@registry.register_problem
class DDPGHalfCheetah(DDPGProblem):
  def __init__(self, *args, **kwargs):
    super(DDPGHalfCheetah, self).__init__(*args, **kwargs)

  def init_agent(self):
    observation_space, action_space = utils.get_gym_spaces(self.runner.make_env)
    

    agent = BaseDDPGAgent(
        observation_space,
        action_space,
        actor_lr=self.hparams.actor_lr,
        critic_lr=self.hparams.critic_lr,
        gamma=self.hparams.gamma,
        tau=self.hparams.tau)

    return agent
  @staticmethod
  def hparams_ddpg_half_cheetah():
    params.env_id = 'HalfCheetah-v2'
    params = base_hparams.base_ddpg()

    params.num_processes = 16

    params.rollout_steps = 1
    params.max_episode_steps = 500
    params.num_total_steps = int(2e6)

    params.gamma = 0.995
    params.buffer_size = int(1e6)

    params.batch_size = 128
    params.tau = 1e-2
    params.actor_lr = 1e-4
    params.critic_lr = 1e-3

    return params