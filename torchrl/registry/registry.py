__all__ = [
    'register_hparam',
    'list_hparams',
    'get_hparam',
    'remove_hparam',
    'register_problem',
    'list_problems',
    'get_problem',
    'remove_problem',
]


_HPARAMS = {}
_PROBLEMS = {}


def _common_decorator(func, registration_name, target_dict):
  assert registration_name not in target_dict, \
    'Attempt to re-register "{}"'.format(registration_name)
  target_dict[registration_name] = func
  return func


def _common_list(target_dict: dict):
  return list(target_dict.keys())


def _common_get(target_dict: dict, key: str):
  return target_dict[key]


def _common_remove(target_dict: dict, key: str):
  target_dict.pop(key)


def register_hparam(name=None):
  target_dict = _HPARAMS

  if callable(name):
    return _common_decorator(name, name.__name__, target_dict)

  return lambda func: _common_decorator(func, name, target_dict)


def list_hparams():
  return _common_list(_HPARAMS)


def get_hparam(key: str):
  return _common_get(_HPARAMS, key)


def remove_hparam(key: str):
  return _common_remove(_HPARAMS, key)


def register_problem(name=None):
  target_dict = _PROBLEMS

  if callable(name):
    return _common_decorator(name, name.__class__, target_dict)

  return lambda func: _common_decorator(func, name, target_dict)


def list_problems():
  return _common_list(_PROBLEMS)


def get_problem(key: str):
  return _common_get(_PROBLEMS, key)


def remove_problem(key: str):
  return _common_remove(_PROBLEMS, key)
