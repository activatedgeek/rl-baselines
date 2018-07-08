# pylint: disable=redefined-outer-name

import pytest
import random
from torchrl.utils import SumTree


@pytest.fixture(scope='function')
def tree():
  yield SumTree(capacity=16)

def test_clear(tree: SumTree):
  for _ in range(tree.capacity):
    value = random.random()
    tree.add(value)

  assert tree.max_value > 0

  tree.clear()
  assert tree.max_value == 0.0
  assert tree.sum_value == 0.0

def test_sum(tree: SumTree):
  sum_value = 0.0
  for _ in range(tree.capacity):
    value = random.random()
    tree.add(value)

    sum_value += value
    assert tree.sum_value == sum_value

def test_max(tree: SumTree):
  max_value = 0.0
  for _ in range(tree.capacity):
    value = random.random()
    tree.add(value)

    max_value = max(value, max_value)
    assert tree.max_value == max_value

def test_overflow(tree: SumTree):
  for _ in range(tree.capacity):
    tree.add(random.random())

  max_value = 0.0
  for _ in range(tree.capacity):
    value = random.random()
    tree.add(value)

    max_value = max(value, max_value)

  assert tree.max_value == max_value
