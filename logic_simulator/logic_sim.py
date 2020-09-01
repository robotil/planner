import numpy as np
# import time, thread, random
import random
from logic_simulator.drone import Drone
from logic_simulator.ugv import Ugv
from logic_simulator.sensor_drone import SensorDrone
from logic_simulator.suicide_drone import SuicideDrone
from logic_simulator.pos import Pos
from logic_simulator.enemy import Enemy

class LogicSim:
    
    ACTIONS_TO_METHODS = {
        'MOVE_TO':{SuicideDrone: Drone.go_to, SensorDrone: Drone.go_to, Ugv: Ugv.go_to},
        'LOOK_AT':{SuicideDrone: Drone.look_at, SensorDrone: Drone.look_at, Ugv: Ugv.look_at},
        'ATTACK':{SuicideDrone: SuicideDrone.attack, Ugv: Ugv.attack},
        'TAKE_PATH':{Ugv: Ugv.go_to}
    }

    def __init__(self, entities: dict, enemies: list):    
        self._entities = entities
        self._enemies = enemies

    def step(self, actions):
        '''
        actions - list of key value pair where key is the entity and value is the action
        '''
        
        # execute entities actions
        for action_name, ent_params_list in actions.items():
            assert action_name in LogicSim.ACTIONS_TO_METHODS.keys(),\
                'LogicSim.ACTIONS_TO_METHODS does not have key {}'.format(action_name)
            for ent_params in ent_params_list:
                for entity_id, params in ent_params.items():
                    assert entity_id in self._entities.keys(),\
                        'self._entities does not have key {}'.format(str(entity_id)) 
                    entity = self._entities[entity_id]
                    method = LogicSim.ACTIONS_TO_METHODS[action_name][entity.__class__]
                    if isinstance(params, tuple):
                        method(entity, *params)
                    else:
                        method(entity, params)

        # update enemies
        for e in self._enemies:
            e.step()
                
            
        
    def clone(self):
        entities = {}
        for k, v in self._entities.items():
            entities[k] =  v.clone()
        enemies = []
        for i,e in enumerate(self._enemies):
            enemies[i]=e.clone()
        return LogicSim(entities, enemies)

    @property
    def state(self):
        return np.array([e.state for e in self._entities.values()])

    def __str__(self):
        s = 'LogicSim state \n'
        for e in self._entities.values():
            s += str(e)
            s += '\n'
        return str(s)



    

# def myfunction(string, sleeptime, lock, *args):
#     while True:
#         lock.acquire()
#         time.sleep(sleeptime)
#         lock.release()
#         time.sleep(sleeptime)





    
    # lock = thread.allocate_lock()
    # thread.start_new_thread(myfunction, ("Thread #: 1", 2, lock))
    # thread.start_new_thread(myfunction, ("Thread #: 2", 2, lock))