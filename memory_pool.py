import numpy as np


class MemoryPool(object):
    def __init__(self, max_batch, batch_size):
        self.batch_size = batch_size
        self.max_size = max_batch * batch_size
        self.current_size = 0
        self.memory_pool = [None] * self.max_size

    def add_memory(self, s, a, r, _s):
        if len(self.memory_pool) == self.max_size:
            index = self.current_size % self.max_size
        else:
            index = self.current_size
        self.memory_pool[index] = {"s": s, "a": a, "r": r, "_s": _s}
        self.current_size += 1

    def pick_memory(self, batch=0):
        index = self.current_size % self.max_size
        if (batch == 0):
            return self.memory_pool[index]
        elif (batch == 1):
            res = list()
            indice = np.random.choice(self.max_size, self.batch_size)
            for i in range(len(indice)):
                res.append(self.memory_pool[indice[i]])
            return res
        else:
            return self.memory_pool


aa = MemoryPool(5, 5)
for i in range(5 * 5):
    aa.add_memory(1, 1, 1, 1)
print(aa.pick_memory(1))
