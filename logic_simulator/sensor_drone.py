from logic_simulator.drone import Drone
from logic_simulator.pos import Pos


class SensorDrone(Drone):
    def __init__(self,id, pos: Pos):
        super().__init__(id,pos)

    