class SumTree:
  """Implements a Sum Tree data structure.

  A Sum Tree data structure is a binary tree
  with leaf nodes containing data and internal
  nodes containing sum of the tree rooted at
  that node. The binary tree here is represented
  by an array.
  """
  def __init__(self, capacity: int = 8):
    self.capacity = capacity
    self.tree = None
    self._next_target_index = 0

    self.clear()

  def add(self, value):
    self.update(self._next_target_index, value)
    self._next_target_index = (self._next_target_index + 1) % self.capacity

  def update(self, index, value):
    tree_index = self.capacity - 1 + index
    delta = value - self.tree[tree_index]

    if delta:
      iter_index = tree_index
      while iter_index >= 0:
        self.tree[iter_index] += delta

        iter_index = (iter_index - 1) // 2

  def clear(self):
    self.tree = [0.0] * (2 * self.capacity - 1)

  def __len__(self):
    return self.capacity

  def __repr__(self):
    return ' '.join([str(t) for t in self.tree])

  @property
  def max_value(self):
    return max(self.tree[-self.capacity:])

  @property
  def sum_value(self):
    return self.tree[0]


if __name__ == '__main__':
  import random

  t = SumTree(capacity=8)

  sum = 0.0
  max_val = 0.0
  for _ in range(8):
    v = random.random()
    t.add(v)
    sum += v
    max_val = max(max_val, v)
    assert sum == t.sum_value
    assert max_val == t.max_value
    print(t)
