from logic_simulator.drone import Drone
from logic_simulator.pos import Pos
import logging


class SuicideDrone(Drone):
    FIELD_OF_VIEW = 0.1745  # radians.   approx. 10 deg
    EPSILON = 0.1

    def __init__(self, id, pos: Pos):
        super().__init__(id, pos)
        self._fov = SuicideDrone.FIELD_OF_VIEW
        self._attacking = False

    def update(self):
        if self._attacking:
            self.attack(self._attack_pos, self._enemies_in_danger)
        else:
            super().update()

    def attack(self, pos, enemies_in_danger):
        logging.info("SuicideDrone Attack on {} {} {}".format(pos.X, pos.Y, pos.Z))
        self._attacking = True
        self._attack_pos = pos
        self._enemies_in_danger = enemies_in_danger
        if pos.distance_to(self.pos) > SuicideDrone.EPSILON:
            self.go_to(pos)
        else:
            self._attacking = False
            self._attack_pos = None
            self._enemies_in_danger = None
            for e in enemies_in_danger:
                e.health = 0.0
            self.health = 0.0
