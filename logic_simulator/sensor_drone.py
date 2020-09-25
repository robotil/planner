from logic_simulator.drone import Drone
from logic_simulator.pos import Pos
import numpy as np

class SensorDrone(Drone):
    FIELD_OF_VIEW = np.pi / 4.0  # radians.   45 deg.


    def __init__(self,id, pos: Pos):
        super().__init__(id,pos)
        self._fov = SensorDrone.FIELD_OF_VIEW

    