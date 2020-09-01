from logic_simulator.drone import Drone
from logic_simulator.pos import Pos
import logging

class SuicideDrone(Drone):
    def __init__(self,id, pos: Pos):  
        super().__init__(id,pos)

    def attack(self, pos):
        logging.info("SuicideDrone Attack on {} {} {}".format(pos.X, pos.Y, pos.Z))

