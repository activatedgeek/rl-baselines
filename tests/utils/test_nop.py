import pytest
from torchrl.utils.nop import Nop


@pytest.fixture(scope='module')
def nop_class():
  return Nop()

def test_any_function(nop_class):
  nop_class.any_function()

def test_any_attribute(nop_class):
  val = nop_class.any_attribute
