#!/usr/bin/env python3
import argparse
import os
import random
from logging.handlers import SocketHandler

from stable_baselines.gail import ExpertDataset
from stable_baselines import TRPO, A2C, DDPG, PPO1, PPO2, SAC, ACER, ACKTR, GAIL, DQN, HER, TD3, logger
import gym
import logging, sys
import threading

import time, datetime
import numpy as np
import tensorflow as tf
from typing import Dict
from geometry_msgs.msg import PointStamped, PolygonStamped, Twist, TwistStamped, PoseStamped, Point
from planner.EntityState import UGVLocalMachine, SuicideLocalMachine, DroneLocalMachine
import math

from logic_simulator.logic_sim import LogicSim
from logic_simulator.enemy import Enemy as lg_enemy
from logic_simulator.entity import Entity as lg_entity
from logic_simulator.sensor_drone import SensorDrone as lg_scn_drone
from logic_simulator.suicide_drone import SuicideDrone as lg_scd_drone
from logic_simulator.ugv import Ugv as lg_ugv
from logic_simulator.pos import Pos
import copy
import datetime
from keras.models import load_model
from matplotlib import pyplot as plt
# from tensor_board_cb import TensorboardCallback
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.common.evaluation import evaluate_policy
import planner.envs

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.sac.policies import SACPolicy, gaussian_entropy, gaussian_likelihood, apply_squashing_func, mlp, \
    nature_cnn

# for custom callbacks stable-baselines should be upgraded using -
# pip3 install stable-baselines[mpi] --upgrade
from stable_baselines.common.callbacks import BaseCallback

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo1': PPO1,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3,
    'gail': GAIL
}
JOBS = ['train', 'record', 'BC_agent', 'play']

POLICIES = ['MlpPolicy', 'CnnPolicy', 'CnnMlpPolicy']

BEST_MODELS_NUM = 0

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20
DISCOUNT_FACTOR = 0.95
lg_ugv.paths = {

    'Path1': [Pos(29.9968816, 32.9994866, 1.75025599),
              Pos(29.9969181, 32.9994912, 2.30123867),
              Pos(29.9973316, 32.9996937, 1.02103409),
              Pos(29.9977419, 32.9998162, 2.34626527),
              Pos(29.9983693, 33.0001438, 1.14929717),
              Pos(29.9987616, 33.0002219, 3.81971129),
              Pos(29.9991068, 33.0002142, 0.213150453),
              Pos(29.9992864, 33.0001952, -0.182928821),
              Pos(29.9993341, 33.0001884, -0.180931998),
              Pos(29.9993667, 33.0002117, -0.18193179)
              ],
    'Path2': [Pos(29.9993825, 33.0002122, -0.390682418),
              Pos(29.9995118, 33.0002082, -0.390672229),
              Pos(29.9995114, 33.0002533, -0.390669329),
              Pos(29.999509, 33.0002783, -0.00499354924)
              ]
}

SUICIDE_FLIGHT_HEIGHT = 25.0
OBSERVER_FLIGHT_HEIGHT = 30.0

UGV_START_POS = Pos()
SENSOR_DRONE_START_POS = Pos()
SUICIDE_DRONE_START_POS = Pos()

NORTH_WEST_SUICIDE = Pos()
NORTH_WEST_OBSERVER = Pos()

NORTH_EAST_SUICIDE = Pos()
NORTH_EAST_OBSERVER = Pos()

SOUTH_EAST_SUICIDE = Pos()
SOUTH_EAST_OBSERVER = Pos()

SOUTH_WEST_SUICIDE = Pos()
SOUTH_WEST_OBSERVER = Pos()

WEST_WINDOW_POS = Pos()
NORTH_WINDOW_POS = Pos()
SOUTH_WINDOW_POS = Pos()
EAST_WINDOW_POS = Pos()

PATH_ID = 'Path1'
SOUTH_WEST_UGV_POS = lg_ugv.paths[PATH_ID][-1]
GATE_POS = lg_ugv.paths['Path2'][-1]
TIME_TO_STIMULATE_1 = LogicSim.MAX_STEPS / 4
TIME_TO_STIMULATE_2 = LogicSim.MAX_STEPS / 2
# SUICIDE_WPS = [NORTH_WEST_SUICIDE, NORTH_EAST_SUICIDE]
# OBSERVER_WPS = [NORTH_EAST_OBSERVER, SOUTH_EAST]
SUICIDE_WPS = []
OBSERVER_WPS = []


# ENEMY_POS = Pos(29.999796, 33.0004159, 0.0447149366)

def populate_positions(positions_dict):
    global UGV_START_POS, SENSOR_DRONE_START_POS, SUICIDE_DRONE_START_POS
    global SOUTH_WEST_UGV_POS
    global NORTH_WEST_SUICIDE, NORTH_WEST_OBSERVER
    global NORTH_EAST_SUICIDE, NORTH_EAST_OBSERVER
    global SOUTH_EAST_SUICIDE, SOUTH_EAST_OBSERVER
    global SOUTH_WEST_SUICIDE, SOUTH_WEST_OBSERVER
    global NORTH_WINDOW_POS, SOUTH_WINDOW_POS, EAST_WINDOW_POS, WEST_WINDOW_POS
    global SUICIDE_WPS, OBSERVER_WPS
    global SUICIDE_FLIGHT_HEIGHT, OBSERVER_FLIGHT_HEIGHT

    lg_ugv.paths['Path1'] = positions_dict['Path1']
    lg_ugv.paths['Path2'] = positions_dict['Path2']

    UGV_START_POS = copy.copy(lg_ugv.paths['Path1'][0])
    SENSOR_DRONE_START_POS = copy.copy(lg_ugv.paths['Path1'][0])
    SUICIDE_DRONE_START_POS = copy.copy(SENSOR_DRONE_START_POS)

    SENSOR_DRONE_START_POS.z = OBSERVER_FLIGHT_HEIGHT
    SUICIDE_DRONE_START_POS.z = SUICIDE_FLIGHT_HEIGHT

    SOUTH_WEST_UGV_POS = lg_ugv.paths['Path1'][-1]

    NORTH_WEST_SUICIDE = copy.copy(positions_dict['Waypoint 49'])
    NORTH_WEST_OBSERVER = copy.copy(positions_dict['Waypoint 49'])

    NORTH_EAST_SUICIDE = copy.copy(positions_dict['Waypoint 76'])
    NORTH_EAST_OBSERVER = copy.copy(positions_dict['Waypoint 76'])

    SOUTH_EAST_SUICIDE = copy.copy(positions_dict['Waypoint 79'])
    SOUTH_EAST_OBSERVER = copy.copy(positions_dict['Waypoint 79'])

    SOUTH_WEST_SUICIDE = copy.copy(positions_dict['Waypoint 52'])
    SOUTH_WEST_OBSERVER = copy.copy(positions_dict['Waypoint 52'])

    SOUTH_WEST_OBSERVER.z = OBSERVER_FLIGHT_HEIGHT
    SOUTH_EAST_OBSERVER.z = OBSERVER_FLIGHT_HEIGHT
    NORTH_EAST_OBSERVER.z = OBSERVER_FLIGHT_HEIGHT
    NORTH_WEST_OBSERVER.z = OBSERVER_FLIGHT_HEIGHT

    SOUTH_WEST_SUICIDE.z = SUICIDE_FLIGHT_HEIGHT
    SOUTH_EAST_SUICIDE.z = SUICIDE_FLIGHT_HEIGHT
    NORTH_EAST_SUICIDE.z = SUICIDE_FLIGHT_HEIGHT
    NORTH_WEST_SUICIDE.z = SUICIDE_FLIGHT_HEIGHT

    SUICIDE_WPS = [NORTH_WEST_SUICIDE, NORTH_EAST_SUICIDE, SOUTH_EAST_SUICIDE, SOUTH_WEST_SUICIDE]
    OBSERVER_WPS = [NORTH_EAST_OBSERVER, SOUTH_EAST_OBSERVER, SOUTH_WEST_OBSERVER, NORTH_WEST_OBSERVER]

    WEST_WINDOW_POS = positions_dict['Window1']
    NORTH_WINDOW_POS = positions_dict['House3']
    SOUTH_WINDOW_POS = positions_dict['House1']
    EAST_WINDOW_POS = positions_dict['House2']


def dist2d(one, two):
    return math.sqrt((one.x - two.x) ** 2.0 + (one.y - two.y) ** 2.0)


def dist3d(one, two):
    return math.sqrt((one.x - two.x) ** 2.0 + (one.y - two.y) ** 2.0 + (one.z - two.z) ** 2.0)


def add_action_logic(actions, entity, action_name, params):
    if action_name not in actions.keys():
        actions[action_name] = []
    actions[action_name].append({entity.id: params})


def add_action(action_list, action_type, entity_id, parameter):
    assert isinstance(parameter, tuple), "Parameter should be a tuple"
    if isinstance(action_type, str):
        todo = {entity_id: parameter}
        action_list[action_type].append(todo)
    else:
        add_action_logic(action_list, action_type, entity_id, parameter)


def compute_reward(rel_diff_step, num_of_dead, num_of_lost_devices):
    """

    """

    # For the first scenario, the reward depends only on"
    # Did we kill the enemy?
    # With which platform?
    # How long did it take

    total_num_of_enemies = 3
    total_num_of_devices = 3
    # if scenario_completed:
    #     reward = -10
    #     if num_of_lost_devices != 0:
    #         reward = reward - 10  # nothing accomplished and we lost a drone!
    #     return reward  # = 0
    reward = rel_diff_step * ((num_of_dead / total_num_of_enemies) - 0.1 * (num_of_lost_devices / total_num_of_devices))
    return reward


def sim_point(point):
    is_sim = True
    return Pos(point.x, point.y, point.z) if is_sim else point


def sim_ent(entity):
    # str_2_type = {'Suicide':SuicideDrone, 'SensorDrone':SensorDrone, 'UGV':Ugv}
    if entity.id == 'Suicide':
        return lg_scn_drone(entity.id, Pos(entity.gpoint.x, entity.gpoint.y, entity.gpoint.z))
    if entity.id == 'SensorDrone':
        return lg_scd_drone(entity.id, Pos(entity.gpoint.x, entity.gpoint.y, entity.gpoint.z))
    if entity.id == 'UGV':
        return lg_ugv(entity.id, Pos(entity.gpoint.x, entity.gpoint.y, entity.gpoint.z))

    # return str_2_type[entity.id].__class__(entity.id, Pos(entity.gpoint.x,entity.gpoint.y,entity.gpoint.z))


def sim_enemy(enemy):
    return lg_enemy(enemy.id, Pos(enemy.gpoint.x, enemy.gpoint.y, enemy.gpoint.z), enemy.priority)


def populate():
    time.sleep(0.5)
    os.system("scripts/populate-scen-0.bash ")


# global number_of_line_of_sight_true
# number_of_line_of_sight_true = 0

def line_of_sight(ent, pos):
    res = ent.is_line_of_sight_to(pos)
    return res


def line_of_sight_to_enemy(entities):
    return  [ent for ent in entities if len(ent.los_enemies) > 0]


def is_entity_positioned(entity, pos):
    MINMUM_DISTANCE = 10.0
    return entity.pos.distance_to(pos) < MINMUM_DISTANCE


def order_drones_movement(actions, suicide_drone, sensor_drone, plan_index, move_commanded):
    global SUICIDE_WPS, OBSERVER_WPS

    assert len(SUICIDE_WPS) == len(OBSERVER_WPS)


    change_target = is_entity_positioned(suicide_drone, SUICIDE_WPS[plan_index]) and \
                    is_entity_positioned(sensor_drone, OBSERVER_WPS[plan_index])

    if change_target:
        # Drones positioned in last plan command - ready for next command
        move_commanded = False

    if not move_commanded:
        plan_index = plan_index if not change_target else (plan_index + random.randint(1, len(OBSERVER_WPS) - 1)) % len(
            OBSERVER_WPS)
        logging.debug('action selected - plan index = {}'.format(plan_index))
        #   suicide.goto(SUICIDE_WPS[plan_index])
        add_action(actions, suicide_drone, 'MOVE_TO', (SUICIDE_WPS[plan_index],))
        #   observer.goto(OBSERVER_WPS[plan_index])
        add_action(actions, sensor_drone, 'MOVE_TO', (OBSERVER_WPS[plan_index],))
        # current plan index commanded
        move_commanded = True

    return plan_index, move_commanded


def order_drones_look_at(actions, suicide_drone, sensor_drone):
    suicide_look_at = WEST_WINDOW_POS if suicide_drone.pos.y < WEST_WINDOW_POS.y else NORTH_WINDOW_POS

    sensor_drone_look_at = EAST_WINDOW_POS if sensor_drone.pos.x > SOUTH_WINDOW_POS.x else SOUTH_WINDOW_POS

    sensor_drone_look_at = NORTH_WINDOW_POS if sensor_drone_look_at.equals(WEST_WINDOW_POS) else sensor_drone_look_at

    suicide_look_at = EAST_WINDOW_POS if sensor_drone_look_at.equals(SOUTH_WINDOW_POS) else suicide_look_at

    add_action(actions, sensor_drone, "LOOK_AT", (sensor_drone_look_at,))

    add_action(actions, suicide_drone, "LOOK_AT", (suicide_look_at,))


def run_logical_sim(env, is_logical):
    # Wait until there is some enemy
    # logging.debug('Wait for enemies...')
    # x = threading.Thread(target=populate, args=())
    # logging.info("Before running thread")
    # x.start()

    global UGV_START_POS, SENSOR_DRONE_START_POS, SUICIDE_DRONE_START_POS, NORTH_WEST_SUICIDE
    global NORTH_EAST_SUICIDE, NORTH_EAST_OBSERVER, SOUTH_EAST, WEST_WINDOW_POS
    global NORTH_WINDOW_POS, SOUTH_WINDOW_POS, EAST_WINDOW_POS

    obs = env.reset()

    while not bool(obs['enemies']):
        continue

    logging.info('enemies found! Start simple_building_ambush')

    # generic get sniper
    enemies = [enemy for enemy in env.enemies if enemy.id == 'Sniper']
    assert len(enemies) == 1
    sniper = enemies[0]

    # Since pre-defined scenario, let's get all the entities
    # Returns logic env and entities if _is_logical==True
    # planner_env and planner_env entities if _is_logical ==False
    sim_env, sensor_drone, suicide_drone, ugv = get_env_and_entities(env, is_logical)

    # redundant - sim_env.reset()

    step, start_ambush_step, stimulation_1_step, stimulation_2_step, plan_index, num_of_dead, num_of_lost_devices = \
        0, 0, 0, 0, 0, 0, 0
    done, all_entities_positioned, move_to_indication_target_commanded, gate_pos_commanded, \
    plan_phase_commanded, attack2_commanded = False, False, False, False, False, False
    reason = ""
    global number_of_line_of_sight_true
    number_of_line_of_sight_true = 0
    episode_experience = []

    while not done:
        step += 1

        # ACTION LOGIC
        # Reset Actions
        action_list = {'MOVE_TO': [], 'LOOK_AT': [], 'ATTACK': [], 'TAKE_PATH': []}
        # List the enemies in line of sight
        if step < 10:
            entities_with_los_to_enemy = []
        else:
            entities_with_los_to_enemy = line_of_sight_to_enemy([suicide_drone, sensor_drone, ugv])
        if len(entities_with_los_to_enemy) > 0:
            # ENEMY FOUND !!!
            attack_enemy(action_list, entities_with_los_to_enemy, suicide_drone, ugv, sensor_drone)
        elif not all_entities_positioned:
            # MOVE TO INDICATION TARGET
            if not move_to_indication_target_commanded:
                move_to_indication_target_commanded = True
                move_to_indication_target(action_list, all_entities_positioned, sensor_drone, suicide_drone, ugv)

            all_entities_positioned = is_entity_positioned(suicide_drone, NORTH_WEST_SUICIDE) and \
                                      is_entity_positioned(sensor_drone, NORTH_EAST_OBSERVER) and \
                                      is_entity_positioned(ugv, SOUTH_WEST_UGV_POS)


        else:
            # AMBUSH ON INDICATION TARGET
            plan_index, stimulation_1_step, stimulation_2_step, gate_pos_commanded, plan_phase_commanded, start_ambush_step, attack2_commanded \
                = ambush_on_indication_target(action_list, sensor_drone, suicide_drone, ugv, sniper, plan_index,
                                              start_ambush_step, step, stimulation_1_step, stimulation_2_step,
                                              gate_pos_commanded, plan_phase_commanded, attack2_commanded,
                                              episode_experience)

        # Execute Actions in simulation
        try:
            obs, reward, done, _ = sim_env.step(action_list)
            logging.debug('step {}: obs = {}, reward = {}, done = {}'.format(step, obs, reward, done))
            sim_env.render()
        except RuntimeError:
            logging.debug('step {}: LOS SERVER DOWN - Rerun the episode'.format(step))
            done = 1
            reason = 'LOS Exception'

        # DONE LOGIC
        if done or step >= sim_env.MAX_STEPS:
            if not reason == 'LOS Exception':
                reason = "Logical Simulation" if done else "Step is " + str(step)
                num_of_dead += len([enemy for enemy in sim_env.enemies if not enemy.is_alive])
                num_of_lost_devices += len([e for e in sim_env.entities if e.health['state '] != '0'])
                done = True
            else:
                #  'LOS Exception'  - Restart scenario
                return (0, 0, 0, 0)
    # Episode is done
    diff_step = sim_env.MAX_STEPS - step + 1
    diff_step = diff_step / sim_env.MAX_STEPS
    this_reward = compute_reward(diff_step, num_of_dead, num_of_lost_devices)

    if len(episode_experience) > 0:
        for i, experience in enumerate(episode_experience):
            experience[-1] = this_reward * DISCOUNT_FACTOR ** (len(episode_experience) - i)

        # add last next_state
        episode_experience[-1][5] = suicide_drone.pos
        episode_experience[-1][6] = sensor_drone.pos
        episode_experience[-1][7] = ugv.pos
        episode_experience[-1][8] = sniper.pos
    else:
        logging.warning('Episode Experience is empty')
    # save episode experience replay to file
    now = datetime.datetime.now()
    with open('/tmp/experience.txt', 'a') as f:
        f.write("----------"+now.strftime("%Y-%m-%d-%H:%M:%S")+"----------\n"
                                                               ".")
        for experience in episode_experience:
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                    .format(experience[0].x, experience[0].y, experience[0].z,
                            experience[1].x, experience[1].y, experience[1].z,
                            experience[2].x, experience[2].y, experience[2].z,
                            experience[3].x, experience[3].y, experience[3].z,
                            experience[4],
                            experience[5].x, experience[5].y, experience[5].z,
                            experience[6].x, experience[6].y, experience[6].z,
                            experience[7].x, experience[7].y, experience[7].z,
                            experience[8].x, experience[8].y, experience[8].z,
                            experience[9]))

    logging.info("Scenario completed: step " + ascii(step) + " reward " + ascii(this_reward) + " Done " + ascii(
        done) + " Reason " + reason)
    return this_reward, int(diff_step * sim_env.MAX_STEPS), num_of_dead, num_of_lost_devices


def get_env_and_entities(env, is_logical):
    """
    Source for gym env and entities
    Args:
        env: PlannerEnv
        is_logical: bool

    Returns:
        env: PlannerEnv
        drn: PlannerEnv.Entity if _is_logical else SensorDrone
        scd: PlannerEnv.Entity if _is_logical else SuicideDrone
        ugv: PlannerEnv.Entity if _is_logical else Ugv
    """
    global UGV_START_POS, SUICIDE_DRONE_START_POS, SENSOR_DRONE_START_POS

    ugv_entity = env.get_entity('UGV')
    scd_entity = env.get_entity('Suicide')
    drn_entity = env.get_entity('SensorDrone')
    while not bool(ugv_entity):
        ugv_entity = env.get_entity('UGV')
    ugv = lg_ugv('UGV', UGV_START_POS if is_logical else Pos(ugv_entity.gpoint.x, ugv_entity.gpoint.y,
                                                             ugv_entity.gpoint.z)) \
        if is_logical else ugv_entity
    while not bool(scd_entity):
        scd_entity = env.get_entity('Suicide')
    scd = lg_scd_drone('Suicide',
                       SUICIDE_DRONE_START_POS if is_logical else Pos(scd_entity.gpoint.x, scd_entity.gpoint.y,
                                                                      scd_entity.gpoint.z)) \
        if is_logical else scd_entity
    while not bool(drn_entity):
        drn_entity = env.get_entity('SensorDrone')
    drn = lg_scn_drone('SensorDrone',
                       SENSOR_DRONE_START_POS if is_logical else Pos(drn_entity.gpoint.x, drn_entity.gpoint.y,
                                                                     drn_entity.gpoint.z)) \
        if is_logical else drn_entity

    if is_logical:
        log_entities = {drn.id: drn, scd.id: scd, ugv.id: ugv}
        log_enemies = [sim_enemy(e) for e in env.enemies]
        env = LogicSim(log_entities, log_enemies)

    return env, drn, scd, ugv


def attack_enemy(action_list, entities_with_los_to_enemy, log_scd, log_ugv, log_sensor):
    attacked_enemies = []

    if log_ugv in entities_with_los_to_enemy:
        # ugv.attack(ENEMY_POS)
        for enemy in log_ugv.los_enemies:
            attacked_enemies.append(enemy)
            add_action(action_list, log_ugv, 'ATTACK', (enemy.pos,))
            print("UGV attack:" + enemy.pos.__str__())
    elif log_scd in entities_with_los_to_enemy:
        # suicide.attack(ENEMY_POS)
        for enemy in log_scd.los_enemies:
            if enemy not in attacked_enemies:
                attacked_enemies.append(enemy)
                add_action(action_list, log_scd, 'ATTACK', (enemy.pos,))
                print("SUICIDE attack:" + enemy.pos.__str__())
    else:
        for enemy in log_sensor.los_enemies:
            if enemy not in attacked_enemies:
                # suicide.goto(ENEMY_POS)
                add_action(action_list, log_scd, 'MOVE_TO', (Pos(enemy.gpoint.x, enemy.gpoint.y, log_scd.gpoint.z),))
                print("DRONE attack:" + enemy.pos.__str__())


def ambush_on_indication_target(action_list, log_drn, log_scd, log_ugv, sniper, plan_index, start_ambush_step, step,
                                stimulation_1_step, stimulation_2_step, gate_pos_commanded, move_commanded,
                                attack2_commanded, episode_experience):
    if start_ambush_step == 0:
        start_ambush_step = step
        logging.info('step {} all entities positioned... start ambush phase'.format(step))
    if start_ambush_step + TIME_TO_STIMULATE_1 < step < start_ambush_step + TIME_TO_STIMULATE_2:
        # STIMULATION 1
        if stimulation_1_step == 0:
            stimulation_1_step = step
            logging.info('step {} stimulation 1 phase'.format(step))
            # ugv.attack(WEST_WINDOW_POS)
            add_action(action_list, log_ugv, 'ATTACK', (WEST_WINDOW_POS,))
    elif step > start_ambush_step + TIME_TO_STIMULATE_2:
        # STIMULATION 2
        if stimulation_2_step == 0:
            stimulation_2_step = step
            logging.info('step {} stimulation 2 phase'.format(step))
        if not attack2_commanded and is_entity_positioned(log_ugv, GATE_POS):
            # ugv.attack(WEST_WINDOW_POS)
            add_action(action_list, log_ugv, 'ATTACK', (WEST_WINDOW_POS,))
            attack2_commanded = True
        else:
            if not gate_pos_commanded:
                add_action(action_list, log_ugv, 'TAKE_PATH', ('Path2', GATE_POS))
                gate_pos_commanded = True
    plan_index, move_commanded = order_drones_movement(action_list, log_scd, log_drn, plan_index, move_commanded)
    if move_commanded:
        if len(episode_experience) > 0:
            # save current state as next state to previous experience
            episode_experience[-1][5] = log_scd.pos
            episode_experience[-1][6] = log_drn.pos
            episode_experience[-1][7] = log_ugv.pos
            episode_experience[-1][8] = sniper.pos
        # save current state and action
        episode_experience.append([log_scd.pos, log_drn.pos, log_ugv.pos, sniper.pos, plan_index
                                      , Pos(), Pos(), Pos(), Pos(), 0])


    order_drones_look_at(action_list, log_scd, log_drn)
    return plan_index, stimulation_1_step, stimulation_2_step, gate_pos_commanded, move_commanded, start_ambush_step, attack2_commanded


def move_to_indication_target(action_list, all_entities_positioned, log_drn, log_scd, log_ugv):
    # suicide.goto(NORTH_WEST)
    add_action(action_list, log_scd, 'MOVE_TO', (NORTH_WEST_SUICIDE,))
    # observer.goto(NORTH_EAST)
    add_action(action_list, log_drn, 'MOVE_TO', (NORTH_EAST_OBSERVER,))
    # ugv.goto(PATH_ID, SOUTH_WEST)
    add_action(action_list, log_ugv, 'TAKE_PATH', (PATH_ID, SOUTH_WEST_UGV_POS))


def run_scenario(action_list, at_house1, at_house2, at_point1, at_point2, at_scanner1, at_scanner2, at_scanner3,
                 at_suicide1, at_suicide2, at_suicide3, at_window1, env, min_dist, start_time_x, start_time_y,
                 start_time_zz, timer_x_period, timer_y_period, timer_zz_period):
    obs = env.reset()
    # Wait until there is some enemy
    while not bool(obs['enemies']):
        continue
    # Since pre-defined scenario, let's get all the entities
    ugv_entity = env.get_entity('UGV')
    scd_entity = env.get_entity('Suicide')
    drn_entity = env.get_entity('SensorDrone')
    while not bool(ugv_entity):
        ugv_entity = env.get_entity('UGV')
    ugv_state = UGVLocalMachine()
    while not bool(scd_entity):
        scd_entity = env.get_entity('Suicide')
    scd_state = SuicideLocalMachine()
    while not bool(drn_entity):
        drn_entity = env.get_entity('SensorDrone')
    drn_state = DroneLocalMachine()
    # Start to move the entities
    add_action(action_list, 'TAKE_PATH', 'UGV', ('Path1', at_point1))  # 444
    add_action(action_list, 'MOVE_TO', 'SensorDrone', (at_scanner1,))
    add_action(action_list, 'LOOK_AT', 'SensorDrone', (at_house1,))  # 444
    add_action(action_list, 'MOVE_TO', 'Suicide', (at_suicide2,))  # 444
    # Set state to each entity
    ugv_state.phase1()
    drn_state.phase1()
    scd_state.phase_i_2()
    done = False
    step = 0
    num_of_dead = 0
    num_of_lost_devices = 0
    scenario_completed = False
    attacking_los = False
    start_time_x = time.time()

    while not done:
        obs, _ = env.step(action_list)
        #       obs, _, done, _ = env.step(action_list)
        # obs = env.get_obs()
        # Check if there is is dead
        list_actual_enemies = obs['enemies']
        for i in range(len(list_actual_enemies)):
            if not list_actual_enemies[i].is_alive:
                num_of_dead += 1
                # for j in range(len(obs['entities'])):
                #     if j.health something
                #         num_of_lost_devices +=1
                done = True

        # Check if there is an enemy in line of sight
        if bool(obs['los_mesh']):
            nm1 = obs['los_mesh']
            for key in nm1:
                enemy_name = key
                list_of_entities = nm1[key]
                enemy = env.get_enemy(enemy_name)
                # Choose to attack the first enemy in the list
                num_of_entities = len(list_of_entities)
                if num_of_entities > 1:
                    # Choose first to shoot if possible
                    found_ugv = False
                    found_suicide = False
                    found_scan = False
                    for i in range(num_of_entities):
                        ent0 = list_of_entities[i]
                        if env.get_entity(ent0).diagstatus.name == "UGV":
                            # We can shoot
                            add_action(action_list, 'ATTACK', ent0.id, (enemy.gpoint,))
                            attacking_los = True
                            found_ugv = True
                        elif env.get_entity(ent0).diagstatus.name == "Suicide":
                            # We can commit suicide
                            found_suicide = True
                            suicide_entity = ent0
                        else:
                            found_scan = True
                    if not found_ugv:
                        # We cannot shoot
                        if found_suicide:
                            add_action(action_list, 'ATTACK', suicide_entity.id, (enemy.gpoint,))
                            attacking_los = True
                        elif found_scan:
                            # Tell suicide to get close to the enemy so that it will be able
                            # to attack him eventually
                            last_goal_for_suicide = enemy.gpoint
                            add_action(action_list, 'MOVE_TO', 'Suicide', (enemy.gpoint,))
                            if scd_state.is_suicide2:
                                scd_state.phase2_ZZ()
                            elif scd_state.is_suicide3:
                                scd_state.phase3_ZZ()
                            else:
                                print("Strange state for Suicide")
                            start_time_zz = time.time()

        # Deal with scenario
        # Take care of UGV
        if not attacking_los:
            if ugv_state.is_point1:
                add_action(action_list, 'TAKE_PATH', 'UGV', ('Path1', at_point1))  # Repeat
                ugv_state.phase2()
                start_time_x = time.time()
            elif ugv_state.is_wait1:
                # Check time
                right_now = time.time()
                diff = right_now - start_time_x
                if diff >= timer_x_period:
                    add_action(action_list, 'ATTACK', 'UGV', (at_window1,))
                    ugv_state.phase3()
                    start_time_y = time.time()
            elif ugv_state.is_wait2:
                # Check time
                right_now = time.time()
                diff = right_now - start_time_y
                if diff >= timer_y_period:
                    add_action(action_list, 'TAKE_PATH', 'UGV', ('Path2', at_point2))
                    ugv_state.phase4()
            elif ugv_state.is_point2:
                if dist3d(ugv_entity.gpoint, at_point2) <= min_dist:
                    # Reach Point2
                    # What to do?
                    # done = True
                    print("Don't know what to do: ugv at point2")
                else:
                    add_action(action_list, 'TAKE_PATH', 'UGV', ('Path2', at_point2))

        # Take care of suicide
        if not attacking_los:
            if scd_state.is_suicide2:
                if dist3d(scd_entity.gpoint, at_suicide2) <= min_dist:
                    if dist3d(scd_entity.gpoint, at_scanner1) <= 2 * min_dist:
                        add_action(action_list, 'MOVE_TO', 'Suicide', (at_suicide3,))
                        scd_state.phase3()
                    else:
                        # Just stay in position and wait for scan drone
                        print(
                            "Step:" + step.__str__() + " Suicide in state:" + scd_state.current_state.__str__() + " is waiting for scan drone")
                else:
                    add_action(action_list, 'MOVE_TO', 'Suicide', (at_suicide2,))
            elif scd_state.is_suicide3:
                if dist3d(scd_entity.gpoint, at_suicide3) <= min_dist:
                    if dist3d(scd_entity.gpoint, at_scanner2) <= 2 * min_dist:
                        add_action(action_list, 'MOVE_TO', 'Suicide', (at_suicide2,))
                        scd_state.phase4()
                    else:
                        # Just stay in position and wait for scan drone
                        print(
                            "Step:" + step + " Suicide in state:" + scd_state.current_state + " is waiting for scan drone")
                else:
                    add_action(action_list, 'MOVE_TO', 'Suicide', (at_suicide3,))
            elif scd_state.is_suicideZZ:  ## Was asked to get close to enemy
                right_now = time.time()
                diff = right_now - start_time_zz
                ### After some timeout (timer_zz_period), return to usual itinerary
                if diff >= timer_zz_period:
                    add_action(action_list, 'MOVE_TO', 'Suicide', (at_suicide3,))
                    scd_state.phaseZZ_3()
                else:
                    if last_goal_for_suicide:
                        add_action(action_list, 'MOVE_TO', 'Suicide', (last_goal_for_suicide,))
                    else:
                        print(
                            "VERY WRONG: Step:" + step.__str__() + " Suicide in state:" + scd_state.current_state.__str__() + " without last_goal_for_suicide")
            elif scd_state.is_suicide1:  ## Shouldn't happen
                if dist3d(scd_entity.gpoint, at_suicide1) <= min_dist:
                    add_action(action_list, 'MOVE_TO', 'Suicide', (at_suicide2,))
                    scd_state.phase2()

        # Take care of drone
        if not attacking_los:
            if drn_state.is_scanner1:
                if dist3d(drn_entity.gpoint, at_scanner1) <= min_dist:
                    if dist3d(scd_entity.gpoint, at_suicide2) <= 2 * min_dist:
                        add_action(action_list, 'MOVE_TO', 'SensorDrone', (at_scanner2,))
                        add_action(action_list, 'LOOK_AT', 'SensorDrone', (at_house2,))
                        scd_state.phase2()
                    else:
                        print(
                            "Step:" + step.__str__() + " Scanner Drone in state:" + drn_state.current_state.__str__() + " is waiting for suicide drone")
                else:
                    add_action(action_list, 'MOVE_TO', 'SensorDrone', (at_scanner1,))
                    add_action(action_list, 'LOOK_AT', 'SensorDrone', (at_house1,))

            elif scd_state.is_scanner2:
                if dist3d(drn_entity.gpoint, at_scanner2) <= min_dist:
                    if dist3d(scd_entity.gpoint, at_suicide3) <= 2 * min_dist:
                        add_action(action_list, 'MOVE_TO', 'SensorDrone', (at_scanner1,))
                        add_action(action_list, 'LOOK_AT', 'SensorDrone', (at_house1,))
                        scd_state.phase3()
                    else:
                        print(
                            "Step:" + step.__str__() + " Scanner Drone in state:" + drn_state.current_state.__str__() + " is waiting for suicide drone")
                else:
                    add_action(action_list, 'MOVE_TO', 'SensorDrone', (at_scanner2))
                    add_action(action_list, 'LOOK_AT', 'SensorDrone', (at_house2))

            elif scd_state.is_scanner3:  ## Shouldn't happen
                if dist3d(drn_entity.gpoint, at_scanner3) <= min_dist:
                    add_action(action_list, 'MOVE_TO', 'SensorDrone', (at_scanner1,))
                    add_action(action_list, 'LOOK_AT', 'SensorDrone', (at_house1,))
                    scd_state.phase5()

        ### done is True
    ### Compute reward


def play(save_dir, env):
    # action_list = {'MOVE_TO': [{}], 'LOOK_AT': [{}], 'ATTACK': [{}], 'TAKE_PATH': [{}]}
    at_scanner1 = Pos(29.9994623, 33.0012379, 32.9879463)
    at_scanner2 = Pos(29.9999499, 33.0010976, 29.3279355)
    at_scanner3 = Pos(30.0001362, 33.0003552, 27.6133062)

    at_house1 = Pos(29.9994717, 33.0004931, 3.47207769)
    at_house2 = Pos(29.9995199, 33.0005282, 10.00616)
    at_house3 = Pos(29.9995706, 33.0004819, 9.53707147)

    at_suicide1 = Pos(29.9993852, 33.0008575, 20.5126056)
    at_suicide2 = Pos(29.999815, 33.0008424, 21.0201285)
    #    at_suicide3 = Pos(x=-0.000118638, y=0.000438844, z=19.8076561)

    at_point1 = Point(x=29.9993667, y=33.0002117, z=-0.18193179)
    at_point2 = Point(x=29.9995096, y=33.0002783, z=-0.00499354924)
    at_window1 = Point(x=29.9994918, y=33.0004458, z=10.015047)

    lg_scanner1 = Pos(120.0, -59.0, 25.4169388)
    lg_scanner2 = Pos(106.0, -5.0, 23.7457948)
    lg_scanner3 = Pos(34.0, 16.0, 23.2363824)

    lg_house1 = Pos(48.0, -58.0, 3.47494173)
    lg_house2 = Pos(51.0, -52.0, 3.94403049)
    lg_house3 = Pos(47.0, -47.0, 3.4749414)

    lg_suicide1 = Pos(83.0, -67.0, 20.2996388)
    lg_suicide2 = Pos(81.0, -20.0, 20.5166231)
    lg_suicide3 = Pos(49.0, -13.0, 19.8076557)

    lg_point1 = lg_ugv.paths['Path1'][-1]
    lg_point2 = lg_ugv.paths['Path2'][-1]
    lg_window1 = Pos(43.0, -56.0, 3.95291735)

    timer_x_period = 10.0  # First timer UGV
    timer_y_period = 10.0  # Second timer UGV
    timer_zz_period = 10.0  # Timer for suicide sent to seen enemy
    start_time_x = 0.0
    start_time_y = 0.0
    start_time_zz = 0.0
    min_dist = 1.9
    end_of_session = False
    session_num = 1
    root = configure_logger()
    root.setLevel(logging.DEBUG)

    while not end_of_session:
        action_list = {'MOVE_TO': [], 'LOOK_AT': [], 'ATTACK': [], 'TAKE_PATH': []}
        print("LALALALA - Starting session: session ", session_num)
        curr_reward, curr_step, curr_num_of_dead, curr_num_of_lost_devices = \
            run_logical_sim(env, is_logical=False)
        # run_scenario(action_list, at_house1, at_house2, at_point1, at_point2, at_scanner1, at_scanner2, at_scanner3,
        #         at_suicide1, at_suicide2, at_suicide3, at_window1, env, min_dist, start_time_x, start_time_y,
        #         start_time_zz, timer_x_period, timer_y_period, timer_zz_period)
        f = open("results.csv", "a")
        curr_string = datetime.datetime.now().__str__() + "," + curr_reward.__str__() + "," + curr_step.__str__() + "," + curr_num_of_dead.__str__() + \
                      "," + curr_num_of_lost_devices.__str__() + "," + "\n"
        f.write(curr_string)
        f.close()
        session_num += 1
        if session_num > 10000:
            end_of_session = True
        # Print reward
        print(curr_reward, curr_step, curr_num_of_dead, curr_num_of_lost_devices)


# For Logical Simulation
def configure_logger():
    root = logging.getLogger()
    formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)s() %(asctime)s %(levelname)s] %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)
    now = datetime.datetime.now()
    log_filename = '/tmp/log'+now.strftime("%Y-%m-%d-%H:%M:%S")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    socket_handler = SocketHandler('127.0.0.1', 19996)
    socket_handler.setLevel(logging.DEBUG)
    socket_handler.setFormatter(formatter)
    root.addHandler(socket_handler)

    return root


def CreateLogAndModelDirs(args):
    '''
    Create log and model directories according to algorithm, time and incremental index
    :param args:
    :return:
    '''

    #
    dir = args.dir_pref + args.mission
    model_dir = dir + args.model_dir + args.algo
    log_dir = dir + args.tensorboard_log + args.algo
    os.makedirs(model_dir, exist_ok=True)
    # create new folder
    try:
        tests = os.listdir(model_dir)
        indexes = []
        for item in tests:
            indexes.append(int(item.split('_')[1]))
        if not bool(indexes):
            k = 0
        else:
            k = max(indexes) + 1
    except FileNotFoundError:
        os.makedirs(log_dir)
        k = 0
    suffix = '/test_{}'.format(str(k))
    model_dir = os.getcwd() + '/' + model_dir + suffix
    log_dir += suffix
    logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    print('log directory created', log_dir)
    return dir, model_dir, log_dir


def main(args):
    # register_policy('CnnMlpPolicy',CnnMlpPolicy)
    env_name = args.mission + '-' + args.env_ver
    # ??    env = gym.make(env_name)  # .unwrapped  <= NEEDED?
    # ??    print('gym env created', env_name, env)
    if args.job != 'train':
        env = gym.make(env_name)
        print('gym env created', env_name, env)

    save_dir, model_dir, log_dir = CreateLogAndModelDirs(args)

    if args.job == 'train':
        # model_path = args.load_model
        # # If there is a path in load model, then load before training
        # if model_path != "" and os.path.exists(model_path):
        #     train_loaded(args.algo, args.policy, model_path, args.n_timesteps, log_dir, model_dir, env_name,
        #                  args.save_interval)
        # else:
        #     train(args.algo, args.policy, args.pretrain, args.n_timesteps, log_dir, model_dir, env_name,
        #           args.save_interval)
        raise NotImplementedError
    elif args.job == 'record':
        raise NotImplementedError
        # record(env)
    elif args.job == 'play':
        play(save_dir, env)
    elif args.job == 'BC_agent':
        raise NotImplementedError
    else:
        raise NotImplementedError(args.job + ' is not defined')


def add_arguments(parser):
    parser.add_argument('--mission', type=str, default="PlannerEnv", help="The agents' task")
    parser.add_argument('--env-ver', type=str, default="v0", help="The custom gym environment version")
    parser.add_argument('--dir-pref', type=str, default="stable_bl/", help="The log and model dir prefix")

    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='/log_dir/', type=str)
    parser.add_argument('-mdl', '--model-dir', help='model directory', default='/model_dir/', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='sac', type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument('--policy', help='Network topography', default='CnnMlpPolicy', type=str, required=False,
                        choices=POLICIES)

    parser.add_argument('--job', help='job to be done', default='play', type=str, required=False, choices=JOBS)
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=int(1e6), type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)
    parser.add_argument('--save-interval', help='Number of timestamps between model saves', default=2000, type=int)
    parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',
                        default=10000, type=int)
    parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation', default=5, type=int)
    parser.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)', default=-1,
                        type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)
    parser.add_argument('--pretrain', help='Evaluate pretrain phase', default=False, type=bool)
    parser.add_argument('--load-expert-dataset', help='Load Expert Dataset', default=False, type=bool)
    parser.add_argument('--load-model', help='Starting model to load', default="", type=str)
    # parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict,
    #                     help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    # parser.add_argument('-uuid', '--uuid', action='store_true', default=False,
    #                     help='Ensure that the run has a unique ID')
    # parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict,
    #                     help='Optional keyword argument to pass to the env constructor')


def string_positions_list(str_list):
    l = len(str_list)
    assert l % 3 == 0
    positions = []
    for i in range(int(l / 3)):
        positions.append(Pos(float(str_list[i * 3]), float(str_list[i * 3 + 1]), float(str_list[i * 3 + 2])))
    return positions


def read_positions():
    import csv

    positions_dict = {}
    irrelevant_keys = ['Ellipse1']
    paths_key = ['Path1', 'Path2']
    with open('planner/PlannerPositions.csv', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            key = row[0]
            if key not in irrelevant_keys:
                positions_dict[row[0]] = Pos(
                    float(row[5]), float(row[6]), float(row[7])) if key not in paths_key else string_positions_list(
                    [field for field in row[5:] if bool(field)])

    return positions_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    positions_dict = read_positions()
    populate_positions(positions_dict)
    main(args)
