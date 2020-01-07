from torchrl.utils.misc import to_camel_case


def test_camel_case():
  assert to_camel_case('A2C4') == 'a2c4'
  assert to_camel_case('ABC') == 'abc'
  assert to_camel_case('DoesThisWork') == 'does_this_work'
  assert to_camel_case('doesThisAlsoWork') == 'does_this_also_work'
