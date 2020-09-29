from logic_simulator.entity import Entity
from logic_simulator.pos import Pos
import numpy as np
import random
import copy


class Enemy(Entity):
    NUM_OF_PRIORITIES = 4
    MAX_OFFSET = 3.0

    def __init__(self, id, pos: Pos, priority, cep=None, tclass=None):
        super().__init__(id, pos)
        assert isinstance(priority, int) and 0 <= priority < Enemy.NUM_OF_PRIORITIES, \
            '{} is not a valid priority'.format(priority)
        self._priority = priority
        self._cep = cep
        self._tclass = tclass
        self._is_alive = True

    @property
    def cep(self):
        return self._cep

    @property
    def tclass(self):
        return self._tclass

    @property
    def is_alive(self):
        return self.health > 0.0

    @property
    def priority(self):
        return self._priority

    def step(self):
        offset_axis = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        offset_dir = [1, -1]
        max_offset = Enemy.MAX_OFFSET
        offset = max_offset * random.random() * random.choice(offset_dir) * random.choice(offset_axis)
        assert offset is not None
        self._pos = copy.copy(self._startpos)
        self._pos.add(offset)

    @property
    def state(self):
        return [[self.pos.x, self.pos.y, self.pos.z], self.health, self.priority]

    def clone(self):
        e = Enemy(self.id, self.pos, self.priority, self._cep, self._tclass)
        return e
