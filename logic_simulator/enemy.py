from logic_simulator.entity import Entity
from logic_simulator.pos import Pos
import numpy as np
import random
class Enemy(Entity):
    def __init__(self, id, pos: Pos, priority):
        super().__init__(id,pos)
        self._priority = priority

    @property
    def priority(self):
        return self._priority

    def step(self):
        offset_axis = [np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0])]
        offset_dir = [1,-1]
        max_offset = 1.0
        offset = max_offset * random.random() * random.choice(offset_dir) * random.choice(offset_axis)
        assert not offset is None
        self._pos.add(offset)

    def state(self):
         return [[self.pos.X, self.pos.Y, self.pos.Z], self.health, self.priority]

    def clone(self):
        e = Enemy(self.id, self.pos)
        return e
