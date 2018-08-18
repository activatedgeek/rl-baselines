import click

from ... import registry


@click.command()
@click.argument('problem',
                envvar='PROBLEM', metavar='<problem>',
                required=True)
@click.argument('load_dir',
                envvar='LOAD_DIR', metavar='<load_dir>',
                type=click.Path(file_okay=False,
                                writable=True,
                                resolve_path=True))
@click.option('--start-epoch',
              help='Epoch to start with after a load.',
              metavar='', type=int)
def resume(problem, load_dir, start_epoch):
  """Resume an experiment."""
  # TODO
  problem_cls = registry.get_problem(problem)
