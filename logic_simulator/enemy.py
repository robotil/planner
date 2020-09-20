from logic_simulator.entity import Entity
from logic_simulator.pos import Pos
import numpy as np
import random
import copy
class Enemy(Entity):

    NUM_OF_PRIORITIES = 4
    MAX_OFFSET = 3.0

    def __init__(self, id, pos: Pos, priority):
        super().__init__(id,pos)
        assert isinstance(priority, int) and priority >= 0 and priority < Enemy.NUM_OF_PRIORITIES, '{} is not a valid priority'.format(priority)
        self._priority = priority

    @property
    def priority(self):
        return self._priority

    def step(self):
        offset_axis = [np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0])]
        offset_dir = [1,-1]
        max_offset = Enemy.MAX_OFFSET
        offset = max_offset * random.random() * random.choice(offset_dir) * random.choice(offset_axis)
        assert not offset is None
        self._pos = copy.copy(self._startpos)
        self._pos.add(offset)
        
    @property
    def state(self):
         return [[self.pos.x, self.pos.y, self.pos.z], self.health, self.priority]

    def clone(self):
        e = Enemy(self.id, self.pos, self.priority)
        return e
