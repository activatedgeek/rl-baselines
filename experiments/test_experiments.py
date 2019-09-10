# pylint: disable=redefined-outer-name

"""Test Experiments.

This test runs all problems and hyperparameter
pairs for 100 time steps. It only guarantees
correct API compatiblity and not the problem
performance metrics.
"""

import pytest
import argparse
from torchrl import registry

problem_hparams_tuples = []
for problem_id, hparams_list in registry.list_problem_hparams().items():
  for hparam_set_id in hparams_list:
    problem_hparams_tuples.append((problem_id, hparam_set_id))


@pytest.fixture(scope='function')
def problem_argv(request):
  problem_id, hparam_set_id = request.param
  args_dict = {
      'problem': problem_id,
      'hparam_set': hparam_set_id,
      'seed': None,
      'extra_hparams': {
          'num_total_steps': 100,
      },
      'log_interval': 50,
      'eval_interval': 50,
      'num_eval': 1,
  }

  yield args_dict


@pytest.mark.parametrize('problem_argv', problem_hparams_tuples,
                         indirect=['problem_argv'])
def test_problem(problem_argv):

  def wrap(problem, hparam_set, extra_hparams, **kwargs):
    problem_cls = registry.get_problem(problem)
    hparams = registry.get_hparam(hparam_set)()
    hparams.update(extra_hparams)

    problem = problem_cls(hparams, argparse.Namespace(**kwargs),
                          None, device='cpu')

    problem.run()

  wrap(**problem_argv)
