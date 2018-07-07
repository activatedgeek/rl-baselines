import random
from collections import deque


class ReplayBuffer:
  def __init__(self, size=int(1e6)):
    self.buffer = deque(maxlen=size)

  def push(self, item):
    self.buffer.append(item)

  def extend(self, *items):
    self.buffer.extend(*items)

  def clear(self):
    self.buffer.clear()

  def sample(self, batch_size):
    assert batch_size <= self.__len__(), \
      'Unable to sample {} items, current buffer size {}'.format(
          batch_size, self.__len__())

    return random.sample(self.buffer, batch_size)

  def __len__(self):
    return len(self.buffer)
