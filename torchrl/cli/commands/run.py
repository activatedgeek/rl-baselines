import argparse
import ast
import click
import torch


from ... import registry


class ExtraHParamType(click.ParamType):

  def convert(self, value, param, ctx):
    try:
      key, val = value.split('=', 1)
      try:
        val = ast.literal_eval(val)
      except ValueError:
        pass

      ctx.obj['extra_hparams'][key] = val

      return [key, val]
    except ValueError:
      self.fail('%s is not a input' % value, param, ctx)


@click.command()
@click.argument('problem',
                envvar='PROBLEM', metavar='<problem>',
                required=True)
@click.option('--hparam-set',
              help='Hyperparameter set name. If not provided, first associated used by default.',
              envvar='HPARAM_SET')
@click.option('--extra-hparams',
              help='Comma-separated list of extra key-value pairs.',
              envvar='EXTRA_HPARAMS', metavar='',
              type=ExtraHParamType(), multiple=True)
@click.option('--seed',
              help='Random Seed.',
              envvar='SEED', metavar='', type=int)
@click.option('--progress/--no-progress',
              help='Show/Hide epoch progress.',
              envvar='PROGRESS', metavar='', default=False)
@click.option('--cuda/--no-cuda',
              help='Enable/Disable CUDA.',
              metavar='', default=False)
@click.option('--device',
              help='Device selection.',
              envvar='DEVICE', metavar='', default='cpu')
@click.option('--log-dir',
              help='Directory to store logs.',
              envvar='LOG_DIR', metavar='',
              type=click.Path(file_okay=False,
                              writable=True,
                              resolve_path=True))
@click.option('--log-interval',
              help='Log interval w.r.t epochs.',
              metavar='', default=100, type=int)
@click.option('--eval-interval',
              help='Eval interval w.r.t epochs.',
              metavar='', default=1000, type=int)
@click.option('--num-eval',
              help='Number of evaluations.',
              metavar='', default=10, type=int)
@click.pass_context
def run(ctx, problem,
        hparam_set: str = None,
        extra_hparams: dict = None,
        progress: bool = False,
        cuda: bool = False,
        device: str = None,
        log_dir: str = None,
        **kwargs):
  """Run Experiments."""

  problem_cls = registry.get_problem(problem)
  if not hparam_set:
    hparam_set_list = registry.get_problem_hparam(problem)
    assert hparam_set_list
    hparam_set = hparam_set_list[0]

  hparams = registry.get_hparam(hparam_set)()
  if ctx:
    hparams.update(ctx.obj['extra_hparams'])
  elif extra_hparams:
    hparams.update(extra_hparams)

  cuda = cuda and torch.cuda.is_available()
  if not cuda:
    device = 'cpu'

  problem = problem_cls(hparams, argparse.Namespace(**kwargs),
                        log_dir,
                        device=device,
                        show_progress=progress)

  problem.run()
