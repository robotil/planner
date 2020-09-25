from logic_simulator.drone import Drone
from logic_simulator.pos import Pos
import logging


class SuicideDrone(Drone):
    FIELD_OF_VIEW = 0.1745  # radians.   approx. 10 deg
    EPSILON = 0.1

    def __init__(self, id, pos: Pos):
        super().__init__(id, pos)
        self._fov = SuicideDrone.FIELD_OF_VIEW

    def attack(self, pos, enemies_in_danger):
        logging.info("SuicideDrone Attack on {} {} {}".format(pos.X, pos.Y, pos.Z))
        if pos.distance_to(self.pos) > SuicideDrone.EPSILON:
            self.go_to(pos)
        else:
            for e in enemies_in_danger:
                e.health = 0.0
            self.health = 0.0
