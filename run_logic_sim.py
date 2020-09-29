
import numpy as np
import random
import copy
import logging
import sys
from logic_simulator.suicide_drone import SuicideDrone
from logic_simulator.sensor_drone import SensorDrone
from logic_simulator.drone import Drone
from logic_simulator.ugv import Ugv
from logic_simulator.pos import Pos
from logic_simulator.logic_sim import LogicSim
from logic_simulator.enemy import Enemy

LOGGER_LEVEL = logging.INFO


Ugv.paths = {
        'Path1': [Pos(-47.0, -359.0, 1.00792499),
                  Pos(-49.0, -341.0, 1.04790355),
                  Pos(-29.0, -295.0, 0.40430533),
                  Pos(-17.0, -250.0, 1.06432373),
                  Pos(14.0, -180.0, 0.472875877),
                  Pos(22.0, -137.0, 1.80694756),
                  Pos(21.0, -98.0, 0.002950645),
                  Pos(19.0, -78.0, - 0.194334967),
                  Pos(17.0, -72.0, - 0.000997688),
                  Pos(19.0, -71.0, - 0.194334959)
                  ],
        'Path2': [Pos(19.0, -72.0, - 0.194336753),
                  Pos(26.0, -62.0, - 0.001001044),
                  Pos(26.0, -54.0, - 0.001001044),
                  Pos(27.0, -54.0, - 0.001000144)
                  ]
    }

    
UGV_START_POS = Pos(-47.0, -359.0, 1.00792499)
SENSOR_DRONE_START_POS = Pos(-47.0, -359.0, 30.0)
SUICIDE_DRONE_START_POS = Pos(-47.0, -359.0, 15)
NORTH_WEST_SUICIDE = Pos(49.0, -13.0, 19.8076557)
NORTH_EAST_SUICIDE = Pos(81.0, -20.0, 20.5166231)
NORTH_EAST_OBSERVER = Pos(106.0, -5.0, 23.7457948)
# SOUTH_WEST = Pos(400.0, 200.0, 30.0)
SOUTH_EAST = Pos(120.0, -100.0, 25.4169388)
WEST_WINDOW_POS = Pos(43.0, -56.0, 3.95291735)
NORTH_WINDOW_POS = Pos(47.0, -47.0, 3.4749414)
SOUTH_WINDOW_POS = Pos(48.0, -58.0, 3.47494173)
EAST_WINDOW_POS = Pos(51.0, -56.0, 10.0)
PATH_ID = 'Path1'
SOUTH_WEST_UGV_POS = Ugv.paths[PATH_ID][-1]
GATE_POS = Ugv.paths['Path2'][-1]
TIME_TO_STIMULATE_1 = LogicSim.MAX_STEPS / 4
TIME_TO_STIMULATE_2 = LogicSim.MAX_STEPS / 2
SUICIDE_WPS = [NORTH_WEST_SUICIDE, NORTH_EAST_SUICIDE]
OBSERVER_WPS = [NORTH_EAST_OBSERVER, SOUTH_EAST]
ENEMY_POS = Pos(48.0, -58.0, 3.47494173)

def add_action(actions, entity, action_name, params):
    if not action_name in actions.keys():
        actions[action_name]=[]
    actions[action_name].append({entity.id:params})


def is_entity_positioned(entity, pos):
    MINMUM_DISTANCE = 6.0
    return entity.pos.distance_to(pos) < MINMUM_DISTANCE

def order_drones_movement(actions, suicide_drone, sensor_drone, plan_index):
    assert len(SUICIDE_WPS) == len(OBSERVER_WPS)

    change_target = is_entity_positioned(suicide_drone, SUICIDE_WPS[plan_index]) and is_entity_positioned(sensor_drone, OBSERVER_WPS[plan_index])

    plan_index = plan_index if not change_target else (plan_index + 1) % len(OBSERVER_WPS)

    #   suicide.goto(SUICIDE_WPS[plan_index])
    add_action(actions, suicide_drone, 'MOVE_TO', (SUICIDE_WPS[plan_index],))
    #   observer.goto(OBSERVER_WPS[plan_index])
    add_action(actions, sensor_drone, 'MOVE_TO', (OBSERVER_WPS[plan_index],))
        
    return plan_index

def order_drones_look_at(actions, suicide_drone, sensor_drone):
    
    suicide_look_at = WEST_WINDOW_POS if suicide_drone.pos.y < WEST_WINDOW_POS.y else NORTH_WINDOW_POS

    sensor_drone_look_at = EAST_WINDOW_POS if sensor_drone.pos.x > SOUTH_WINDOW_POS.x else SOUTH_WINDOW_POS 

    sensor_drone_look_at = NORTH_WINDOW_POS if sensor_drone_look_at.equals(WEST_WINDOW_POS) else sensor_drone_look_at

    suicide_look_at = EAST_WINDOW_POS if sensor_drone_look_at.equals(SOUTH_WINDOW_POS) else suicide_look_at

    add_action(actions, sensor_drone, "LOOK_AT", (sensor_drone_look_at,))

    add_action(actions, suicide_drone, "LOOK_AT", (sensor_drone_look_at,))

def line_of_sight(ent, pos):
    return ent.is_line_of_sight_to(pos)

def line_of_sight_to_enemy(entities):
    return [ent for ent in entities if line_of_sight(ent, ENEMY_POS)]
  

def simple_building_ambush():
    logging.debug('start simple_building_ambush ...')
    sensor_drone = SensorDrone('SensorDrone', SENSOR_DRONE_START_POS)
    suicide_drone = SuicideDrone('Suicide', SUICIDE_DRONE_START_POS)
    ugv = Ugv('UGV', UGV_START_POS)
    enemy_positions = [ENEMY_POS]
    enemies = [Enemy("Enemy" + str(i), p, 1) for i,p in enumerate(enemy_positions)]
    ls = LogicSim({suicide_drone.id: suicide_drone, sensor_drone.id:sensor_drone, ugv.id:ugv}, enemies)
    ls.reset()
    step, start_ambush_step, stimulation_1_step, stimulation_2_step, plan_index = 0, 0, 0, 0, 0
    done, all_entities_positioned = False, False

    while step < LogicSim.MAX_STEPS and not done:
        step += 1
        entities_with_los_to_enemy = line_of_sight_to_enemy([suicide_drone, sensor_drone, ugv])

        actions = {}

        if len(entities_with_los_to_enemy) > 0:
            # ENEMY FOUND !!!
            if ugv in entities_with_los_to_enemy:
                #ugv.attack(ENEMY_POS)
                add_action(actions, ugv, 'ATTACK', (ENEMY_POS,))
            elif suicide_drone in entities_with_los_to_enemy:
                # suicide.attack(ENEMY_POS)
                add_action(actions, suicide_drone, 'ATTACK', (ENEMY_POS,))
            else:
                # suicide.goto(ENEMY_POS)
                add_action(actions, suicide_drone, 'MOVE_TO', (ENEMY_POS,))
        elif not all_entities_positioned:
            # MOVE TO INDICATION TARGET
            all_entities_positioned = is_entity_positioned(suicide_drone, NORTH_WEST_SUICIDE) and\
                                      is_entity_positioned(sensor_drone, NORTH_EAST_OBSERVER) and\
                                      is_entity_positioned(ugv, SOUTH_WEST_UGV_POS) 
            #suicide.goto(NORTH_WEST)
            add_action(actions, suicide_drone, 'MOVE_TO', (NORTH_WEST_SUICIDE,)) 
            # observer.goto(NORTH_EAST)
            add_action(actions, sensor_drone, 'MOVE_TO', (NORTH_EAST_OBSERVER,))
            # ugv.goto(PATH_ID, SOUTH_WEST)
            add_action(actions, ugv, 'TAKE_PATH', (PATH_ID, SOUTH_WEST_UGV_POS))
        else:
            if start_ambush_step == 0:
                start_ambush_step = step
                logging.info('step {} all entities positioned... start ambush phase'.format(step))
            # AMBUSH ON INDICATION TARGET
            if step > start_ambush_step + TIME_TO_STIMULATE_1 and step < start_ambush_step + TIME_TO_STIMULATE_2:
                # STIMULATION 1
                if stimulation_1_step == 0:
                    stimulation_1_step = step
                    logging.info('step {} stimulation 1 phase'.format(step))
                # ugv.attack(WEST_WINDOW_POS)
                add_action(actions, ugv, 'ATTACK', (WEST_WINDOW_POS,))
            elif step > start_ambush_step + TIME_TO_STIMULATE_2:
                # STIMULATION 2
                # ugv.goto(PATH_ID, GATE_POS)
                if stimulation_2_step == 0:
                    stimulation_2_step = step
                    logging.info('step {} stimulation 2 phase'.format(step))
                if is_entity_positioned(ugv, GATE_POS):
                    # ugv.attack(WEST_WINDOW_POS)
                    add_action(actions, ugv, 'ATTACK', (WEST_WINDOW_POS,))
                else:
                    add_action(actions, ugv, 'TAKE_PATH', ('Path2', GATE_POS))
            plan_index = order_drones_movement(actions, suicide_drone, sensor_drone, plan_index)
            order_drones_look_at(actions, suicide_drone, sensor_drone)
        ls.render()
        obs, reward, done, _ =  ls.step(actions)
        # print (obs)
        logging.debug('obs = {}, reward = {}, done = {}'.format(obs,reward,done))
    logging.info("step {} done {} reward {} enemy alive {}".format(step, done, reward, enemies[0]._health > 0.0))
def get_new_target(old_target):
    assert not old_target is None
    offset_axis = [np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0])]
    offset_dir = [1,-1]
    max_offset = 1000.0
    offset = max_offset * random.random() * random.choice(offset_dir) * random.choice(offset_axis)
    assert not offset is None
    new_target = copy.copy(old_target)
    new_target.add(offset)
    return new_target

def test_logic_sim():
    target_wp1 = Pos(1.1, 2.2, 3.3)
    target_wp2 = Pos(10.1, 20.2, 30.3)
    target_wp3 = Pos(20.1, 10.2, 0.0)

    sensor_drone = SensorDrone('SensorDrone', target_wp1)
    suicide_drone = SuicideDrone('Suicide', target_wp2)
    ugv = Ugv('UGV', target_wp3)
    ls = LogicSim({suicide_drone.id: suicide_drone, sensor_drone.id:sensor_drone, ugv.id:ugv} )
    

    for _ in range(1000):
        if random.random() > 0.9: 
            target_wp1 = get_new_target(target_wp1)
            target_wp2 = get_new_target(target_wp2)
            target_wp3 = get_new_target(target_wp3)

        assert not target_wp1 is None
        assert not target_wp2 is None
        assert not target_wp3 is None
        # actions = {11:target_wp1, 22:target_wp2}

        actions = {'MOVE_TO':[{'SensorDrone': (target_wp1)},{'Suicide': (target_wp2)}],
                    'LOOK_AT':[{'SensorDrone': (target_wp3)}],
                    'ATTACK':[],
                    'TAKE_PATH':[{'UGV':('Path1',target_wp3)}]}

        # actions = {11:target_wp1}
        ls.step(actions)
        
        str_actions = [str(k)+','+str(v) for k,v in actions.items()]
        print(str_actions) 
        print(ls)

def configure_logger():
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s() %(asctime)s %(levelname)s] %(message)s"
    formatter = logging.Formatter(FORMAT)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root

if __name__ == "__main__":
    # test_logic_sim()
    root = configure_logger()
    root.setLevel(LOGGER_LEVEL)
    simple_building_ambush()