#!/usr/bin/env python3
import argparse
import os
import random
from logging.handlers import SocketHandler

from stable_baselines.gail import ExpertDataset
from stable_baselines import TRPO, A2C, DDPG, PPO1, PPO2, SAC, ACER, ACKTR, GAIL, DQN, HER, TD3, logger
import gym
from gym import spaces
import logging, sys
import threading

import time, datetime
import numpy as np
import tensorflow as tf
from typing import Dict
from geometry_msgs.msg import PointStamped, PolygonStamped, Twist, TwistStamped, PoseStamped, Point
# from planner.EntityState import UGVLocalMachine, SuicideLocalMachine, DroneLocalMachine
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


def populate():
    # This should be used in case we don't have a simlation and we are using the DummyServer
    time.sleep(0.5)
    os.system("scripts/populate-scen-0.bash ")


def dist2d(one, two):
    return math.sqrt((one.x - two.x) ** 2.0 + (one.y - two.y) ** 2.0)


def dist3d(one, two):
    return math.sqrt((one.x - two.x) ** 2.0 + (one.y - two.y) ** 2.0 + (one.z - two.z) ** 2.0)


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


# ENEMY_POS = Pos(29.999796, 33.0004159, 0.0447149366)
class PlannerScenarioEnv(gym.Env):
    SUICIDE_FLIGHT_HEIGHT = 20.0
    OBSERVER_FLIGHT_HEIGHT = 25.0
    MAX_DISTANCE_TO_ENEMY = 100.0

    SUICIDE_ATTACKING_DISTANCE = 15.0

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

    SOUTH_WEST_UGV_POS = Pos()
    PATH_ID = 'Path1'

    GATE_POS = lg_ugv.paths['Path2'][-1]
    TIME_TO_STIMULATE_1 = LogicSim.MAX_STEPS / 4
    TIME_TO_STIMULATE_2 = LogicSim.MAX_STEPS / 2
    # SUICIDE_WPS = [NORTH_WEST_SUICIDE, NORTH_EAST_SUICIDE]
    # OBSERVER_WPS = [NORTH_EAST_OBSERVER, SOUTH_EAST]
    SUICIDE_WPS = []
    OBSERVER_WPS = []
    NUM_ACTIONS = 4

    def __init__(self):
        # register_policy('CnnMlpPolicy',CnnMlpPolicy)

        self.populate_positions(positions_dict=self.read_positions())
        inner_env_name = 'PlannerEnv-v0'
        self.plannerEnv = gym.make(inner_env_name)
        print('gym env created', inner_env_name, self.plannerEnv)
        self._is_logical = False
        self._plan_index, self._stimulation_1_step, self._stimulation_2_step, \
        self._gate_pos_commanded, self._plan_phase_commanded, self._start_ambush_step, \
        self._attack2_commanded = 0, 0, 0, 0, 0, 0, 0
        self._sensor_drone, self._suicide_drone, self._ugv, self._sniper = None, None, None, None
        self._step = 0
        self._start_ambush_step = 0
        self._stimulation_1_step = 0
        self._stimulation_2_step = 0
        self._plan_index = 0
        self._num_of_dead = 0
        self._num_of_lost_devices = 0
        self._gate_pos_commanded = False
        self._plan_phase_commanded = False
        self._attack2_commanded = False
        self._change_target = False
        self._episode_experience = []
        self.action_space = spaces.Discrete(PlannerScenarioEnv.NUM_ACTIONS)

        self.observation_space = spaces.Box(low=-5.0, high=40.0, shape=(4, 3), dtype=np.float16)

        self.rootlog = self.configure_logger()

    def string_positions_list(self, str_list):
        l = len(str_list)
        assert l % 3 == 0
        positions = []
        for i in range(int(l / 3)):
            positions.append(Pos(float(str_list[i * 3]), float(str_list[i * 3 + 1]), float(str_list[i * 3 + 2])))
        return positions

    def read_positions(self):
        import csv
        positions_dict = {}
        irrelevant_keys = ['Ellipse1']
        paths_key = ['Path1', 'Path2']
        with open('../../../PlannerPositions.csv', newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            next(reader)
            for row in reader:
                key = row[0]
                if key not in irrelevant_keys:
                    positions_dict[row[0]] = Pos(float(row[5]), float(row[6]), float(row[7])) \
                        if key not in paths_key \
                        else self.string_positions_list([field for field in row[5:] if bool(field)])

        return positions_dict

    def populate_positions(self, positions_dict):

        lg_ugv.paths['Path1'] = positions_dict['Path1']
        lg_ugv.paths['Path2'] = positions_dict['Path2']

        PlannerScenarioEnv.UGV_START_POS = copy.copy(lg_ugv.paths['Path1'][0])
        PlannerScenarioEnv.SENSOR_DRONE_START_POS = copy.copy(lg_ugv.paths['Path1'][0])
        PlannerScenarioEnv.SUICIDE_DRONE_START_POS = copy.copy(PlannerScenarioEnv.SENSOR_DRONE_START_POS)

        PlannerScenarioEnv.SENSOR_DRONE_START_POS.z = PlannerScenarioEnv.OBSERVER_FLIGHT_HEIGHT
        PlannerScenarioEnv.SUICIDE_DRONE_START_POS.z = PlannerScenarioEnv.SUICIDE_FLIGHT_HEIGHT

        PlannerScenarioEnv.SOUTH_WEST_UGV_POS = lg_ugv.paths['Path1'][-1]

        PlannerScenarioEnv.NORTH_WEST_SUICIDE = copy.copy(positions_dict['Waypoint 40'])
        PlannerScenarioEnv.NORTH_WEST_OBSERVER = copy.copy(positions_dict['Waypoint 29'])

        PlannerScenarioEnv.NORTH_EAST_SUICIDE = copy.copy(positions_dict['Waypoint 76'])
        PlannerScenarioEnv.NORTH_EAST_OBSERVER = copy.copy(positions_dict['Waypoint 102'])

        PlannerScenarioEnv.SOUTH_EAST_SUICIDE = copy.copy(positions_dict['Waypoint 79'])
        PlannerScenarioEnv.SOUTH_EAST_OBSERVER = copy.copy(positions_dict['Waypoint 90'])

        PlannerScenarioEnv.SOUTH_WEST_SUICIDE = copy.copy(positions_dict['Waypoint 43'])
        PlannerScenarioEnv.SOUTH_WEST_OBSERVER = copy.copy(positions_dict['Waypoint 16'])

        PlannerScenarioEnv.SOUTH_WEST_OBSERVER.z = PlannerScenarioEnv.OBSERVER_FLIGHT_HEIGHT
        PlannerScenarioEnv.SOUTH_EAST_OBSERVER.z = PlannerScenarioEnv.OBSERVER_FLIGHT_HEIGHT
        PlannerScenarioEnv.NORTH_EAST_OBSERVER.z = PlannerScenarioEnv.OBSERVER_FLIGHT_HEIGHT
        PlannerScenarioEnv.NORTH_WEST_OBSERVER.z = PlannerScenarioEnv.OBSERVER_FLIGHT_HEIGHT

        PlannerScenarioEnv.SOUTH_WEST_SUICIDE.z = PlannerScenarioEnv.SUICIDE_FLIGHT_HEIGHT
        PlannerScenarioEnv.SOUTH_EAST_SUICIDE.z = PlannerScenarioEnv.SUICIDE_FLIGHT_HEIGHT
        PlannerScenarioEnv.NORTH_EAST_SUICIDE.z = PlannerScenarioEnv.SUICIDE_FLIGHT_HEIGHT
        PlannerScenarioEnv.NORTH_WEST_SUICIDE.z = PlannerScenarioEnv.SUICIDE_FLIGHT_HEIGHT

        PlannerScenarioEnv.SUICIDE_WPS = [PlannerScenarioEnv.NORTH_WEST_SUICIDE,
                                          PlannerScenarioEnv.NORTH_EAST_SUICIDE,
                                          PlannerScenarioEnv.SOUTH_EAST_SUICIDE,
                                          PlannerScenarioEnv.SOUTH_WEST_SUICIDE]
        PlannerScenarioEnv.OBSERVER_WPS = [PlannerScenarioEnv.NORTH_EAST_OBSERVER,
                                           PlannerScenarioEnv.SOUTH_EAST_OBSERVER,
                                           PlannerScenarioEnv.SOUTH_WEST_OBSERVER,
                                           PlannerScenarioEnv.NORTH_WEST_OBSERVER]

        PlannerScenarioEnv.WEST_WINDOW_POS = positions_dict['Window1']
        PlannerScenarioEnv.NORTH_WINDOW_POS = positions_dict['House3']
        PlannerScenarioEnv.SOUTH_WINDOW_POS = positions_dict['House1']
        PlannerScenarioEnv.EAST_WINDOW_POS = positions_dict['House2']

    def add_action_logic(self, actions, entity, action_name, params):
        if action_name not in actions.keys():
            actions[action_name] = []
        actions[action_name].append({entity.id: params})

    def add_action(self, action_list, action_type, entity_id, parameter):
        assert isinstance(parameter, tuple), "Parameter should be a tuple"
        if isinstance(action_type, str):
            todo = {entity_id: parameter}
            action_list[action_type].append(todo)
        else:
            self.add_action_logic(action_list, action_type, entity_id, parameter)

    def compute_reward(self, rel_diff_step):
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
        reward = rel_diff_step * (
                (self._num_of_dead / total_num_of_enemies) - 0.1 * (self._num_of_lost_devices / total_num_of_devices))
        return reward

    # global number_of_line_of_sight_true
    # number_of_line_of_sight_true = 0

    def line_of_sight(self, ent, pos):
        res = ent.is_line_of_sight_to(pos)
        return res

    def line_of_sight_to_enemy(self, entities):
        return [ent for ent in entities if len(ent.los_enemies) > 0]

    def is_entity_positioned(self, entity, pos):
        MINMUM_DISTANCE = 10.0
        return entity.pos.distance_to(pos) < MINMUM_DISTANCE

    # In the step
    def order_drones_movement(self, actions):

        assert len(PlannerScenarioEnv.SUICIDE_WPS) == len(PlannerScenarioEnv.OBSERVER_WPS)

        actions_num = len(PlannerScenarioEnv.OBSERVER_WPS)

        self._change_target = self.is_entity_positioned(self._suicide_drone,
                                                        PlannerScenarioEnv.SUICIDE_WPS[self._plan_index]) and \
                              self.is_entity_positioned(self._sensor_drone,
                                                        PlannerScenarioEnv.OBSERVER_WPS[self._plan_index])
        if self._change_target:
            self.rootlog.debug('move commanded ? {} plan index {}'.format(self._move_commanded, self._plan_index))

        # if self._change_target:
        #     # Drones positioned in last plan command - ready for next command
        #     self._move_commanded = False

        if not self._move_commanded:
            self._plan_index = self._plan_index \
                if not self._change_target else (self._plan_index + random.randint(1, actions_num - 1)) % actions_num
            self.rootlog.warning('action selected - plan index = {}'.format(self._plan_index))
            #   suicide.goto(SUICIDE_WPS[plan_index])
            self.add_action(actions, self._suicide_drone, 'MOVE_TO',
                            (PlannerScenarioEnv.SUICIDE_WPS[self._plan_index],))
            #   observer.goto(OBSERVER_WPS[plan_index])
            self.add_action(actions, self._sensor_drone, 'MOVE_TO',
                            (PlannerScenarioEnv.OBSERVER_WPS[self._plan_index],))
            # current plan index commanded
            self._move_commanded = True

        # return plan_index, move_commanded

    def order_drones_look_at(self, actions):
        suicide_look_at = PlannerScenarioEnv.WEST_WINDOW_POS \
            if self._suicide_drone.pos.y < PlannerScenarioEnv.WEST_WINDOW_POS.y else PlannerScenarioEnv.NORTH_WINDOW_POS

        sensor_drone_look_at = PlannerScenarioEnv.EAST_WINDOW_POS \
            if self._sensor_drone.pos.x > PlannerScenarioEnv.SOUTH_WINDOW_POS.x else PlannerScenarioEnv.SOUTH_WINDOW_POS

        sensor_drone_look_at = PlannerScenarioEnv.NORTH_WINDOW_POS \
            if sensor_drone_look_at.equals(PlannerScenarioEnv.WEST_WINDOW_POS) else sensor_drone_look_at

        suicide_look_at = PlannerScenarioEnv.EAST_WINDOW_POS \
            if sensor_drone_look_at.equals(PlannerScenarioEnv.SOUTH_WINDOW_POS) else suicide_look_at

        self.add_action(actions, self._sensor_drone, "LOOK_AT", (sensor_drone_look_at,))

        self.add_action(actions, self._suicide_drone, "LOOK_AT", (suicide_look_at,))

    def get_env_and_entities(self, is_logical):
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

        ugv_entity = self.plannerEnv.get_entity('UGV')
        scd_entity = self.plannerEnv.get_entity('Suicide')
        drn_entity = self.plannerEnv.get_entity('SensorDrone')
        while not bool(ugv_entity):
            ugv_entity = self.plannerEnv.get_entity('UGV')
        ugv = lg_ugv('UGV',
                     PlannerScenarioEnv.UGV_START_POS
                     if is_logical else Pos(ugv_entity.gpoint.x, ugv_entity.gpoint.y, ugv_entity.gpoint.z)) \
            if is_logical else ugv_entity
        while not bool(scd_entity):
            scd_entity = self.plannerEnv.get_entity('Suicide')

        scd = lg_scd_drone('Suicide',
                           PlannerScenarioEnv.SUICIDE_DRONE_START_POS
                           if is_logical else Pos(scd_entity.gpoint.x, scd_entity.gpoint.y, scd_entity.gpoint.z)) \
            if is_logical else scd_entity
        while not bool(drn_entity):
            drn_entity = self.plannerEnv.get_entity('SensorDrone')
        drn = lg_scn_drone('SensorDrone',
                           PlannerScenarioEnv.SENSOR_DRONE_START_POS
                           if is_logical else Pos(drn_entity.gpoint.x, drn_entity.gpoint.y, drn_entity.gpoint.z)) \
            if is_logical else drn_entity

        if is_logical:
            log_entities = {drn.id: drn, scd.id: scd, ugv.id: ugv}
            log_enemies = [sim_enemy(e) for e in self.plannerEnv.enemies]
            self.plannerEnv = LogicSim(log_entities, log_enemies)

        return drn, scd, ugv

    def attack_enemy(self, action_list, entities_with_los_to_enemy):
        attacked_enemies = []

        if self._ugv in entities_with_los_to_enemy:
            # ugv.attack(ENEMY_POS)
            for enemy in self._ugv.los_enemies:
                attacked_enemies.append(enemy)
                self.add_action(action_list, self._ugv, 'ATTACK', (enemy.pos,))
                self.rootlog.debug("UGV attack:".format(enemy.pos.__str__()))
        elif self._suicide_drone in entities_with_los_to_enemy:
            # suicide.attack(ENEMY_POS)
            for enemy in self._suicide_drone.los_enemies:
                if enemy not in attacked_enemies:
                    attacked_enemies.append(enemy)
                    self.add_action(action_list, self._suicide_drone, 'ATTACK', (enemy.pos,))
                    self.rootlog.debug("SUICIDE attack:".format(enemy.pos.__str__()))
        else:
            for enemy in self._sensor_drone.los_enemies:
                if enemy not in attacked_enemies:
                    # stopping sensor drone - positive LOS to enemy
                    self.add_action(action_list, self._sensor_drone, 'MOVE_TO', (self._sensor_drone.pos,))
                    self.rootlog.warning("SENSOR stop - you have LOS!!!")
                    # suicide.goto(ENEMY_POS)
                    if self._suicide_drone.pos.distance_to(enemy.pos) > PlannerScenarioEnv.SUICIDE_ATTACKING_DISTANCE:
                        self.add_action(action_list, self._suicide_drone, 'MOVE_TO',
                                        (Pos(enemy.gpoint.x, enemy.gpoint.y, self._suicide_drone.gpoint.z),))
                        self.rootlog.warning("SUICIDE goto above enemy:".format(enemy.pos.__str__()))
                    else:
                        self.add_action(action_list, self._suicide_drone, 'ATTACK', (enemy.pos,))
                        self.rootlog.warning("SUICIDE attack:".format(enemy.pos.__str__()))

    def ambush_on_indication_target(self, action_list):
        if self._start_ambush_step == 0:
            self._start_ambush_step = self._step
            self.rootlog.info('step {} all entities positioned... start ambush phase'.format(self._step))

        if self._start_ambush_step + PlannerScenarioEnv.TIME_TO_STIMULATE_1 < \
                self._step < self._start_ambush_step + PlannerScenarioEnv.TIME_TO_STIMULATE_2:
            self.stimulation_1(action_list)
        elif self._step > self._start_ambush_step + PlannerScenarioEnv.TIME_TO_STIMULATE_2:
            self.stimulation_2(action_list)
        else:
            logging.debug('redundant else')

        self.order_drones_movement(action_list)

        if self._move_commanded:
            if len(self._episode_experience) > 0:
                # save current state as next state to previous experience
                self._episode_experience[-1][2] = self.get_obs()
            # save current state and action
            self._episode_experience.append(
                [self.get_obs(), self._plan_index, np.zeros(shape=(4, 3), dtype=np.float16), 0.0])
            # [self._suicide_drone.pos, self._sensor_drone.pos, self._ugv.pos, self._sniper.pos,
            #  self._plan_index, Pos(), Pos(), Pos(), Pos(), 0])

        # TODO uncomment when simulation can control sensors
        # self.order_drones_look_at(action_list)

    def stimulation_2(self, action_list):
        # STIMULATION 2
        if self._stimulation_2_step == 0:
            self._stimulation_2_step = self._step
            self.rootlog.info('step {} stimulation 2 phase'.format(self._step))
        if not self._attack2_commanded and self.is_entity_positioned(self._ugv, PlannerScenarioEnv.GATE_POS):
            # ugv.attack(WEST_WINDOW_POS)
            self.add_action(action_list, self._ugv, 'ATTACK', (PlannerScenarioEnv.WEST_WINDOW_POS,))
            self._attack2_commanded = True
        else:
            if not self._gate_pos_commanded:
                self.add_action(action_list, self._ugv, 'TAKE_PATH', ('Path2', PlannerScenarioEnv.GATE_POS))
                self._gate_pos_commanded = True

    def stimulation_1(self, action_list):
        # STIMULATION 1
        if self._stimulation_1_step == 0:
            self._stimulation_1_step = self._step
            self.rootlog.info('step {} stimulation 1 phase'.format(self._step))
            # ugv.attack(WEST_WINDOW_POS)
            self.add_action(action_list, self._ugv, 'ATTACK', (PlannerScenarioEnv.WEST_WINDOW_POS,))

    def move_to_indication_target(self, action_list):
        # suicide.goto(NORTH_WEST)
        self.add_action(action_list, self._suicide_drone, 'MOVE_TO', (PlannerScenarioEnv.NORTH_WEST_SUICIDE,))
        # observer.goto(NORTH_EAST)
        self.add_action(action_list, self._sensor_drone, 'MOVE_TO', (PlannerScenarioEnv.NORTH_EAST_OBSERVER,))
        # ugv.goto(PATH_ID, SOUTH_WEST)
        self.add_action(action_list, self._ugv, 'TAKE_PATH',
                        (PlannerScenarioEnv.PATH_ID, PlannerScenarioEnv.SOUTH_WEST_UGV_POS))

    # For Logical Simulation
    def configure_logger(self):
        root = logging.getLogger("GymPlanner")
        formatter = logging.Formatter(
            "[%(filename)s:%(lineno)s - %(funcName)s() %(asctime)s %(levelname)s] %(message)s")

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        root.addHandler(stdout_handler)
        now = datetime.datetime.now()
        log_filename = '/tmp/gymplanner-' + now.strftime("%Y-%m-%d-%H:%M:%S")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        socket_handler = SocketHandler('127.0.0.1', 19996)
        socket_handler.setLevel(logging.DEBUG)
        socket_handler.setFormatter(formatter)
        root.addHandler(socket_handler)

        return root

    def reset(self):

        self.rootlog.setLevel(logging.DEBUG)

        obs = self.plannerEnv.reset()

        while not bool(obs['enemies']):
            continue

        self.rootlog.info('enemies found! Start simple_building_ambush')

        # generic get sniper
        enemies = [enemy for enemy in self.plannerEnv.enemies if enemy.id == 'Sniper']
        assert len(enemies) == 1
        self._sniper = enemies[0]

        # Since pre-defined scenario, let's get all the entities
        # Returns logic env and entities if _is_logical==True
        # planner_env and planner_env entities if _is_logical ==False
        self._sensor_drone, self._suicide_drone, self._ugv = self.get_env_and_entities(self._is_logical)

        self._step = 0
        self._start_ambush_step = 0
        self._stimulation_1_step = 0
        self._stimulation_2_step = 0
        self._plan_index = 0
        self._num_of_dead = 0
        self._num_of_lost_devices = 0
        done, all_entities_positioned, move_to_indication_target_commanded = False, False, False
        self._gate_pos_commanded, self._plan_phase_commanded, self._attack2_commanded = False, False, False
        reason = ""
        global number_of_line_of_sight_true
        number_of_line_of_sight_true = 0
        self._episode_experience = []

        while not all_entities_positioned and not done:
            self._step += 1

            # ACTION LOGIC
            # Reset Actions
            action_list = {'MOVE_TO': [], 'LOOK_AT': [], 'ATTACK': [], 'TAKE_PATH': []}
            # List the enemies in line of sight
            entities_with_los_to_enemy = self.line_of_sight_to_enemy([self._suicide_drone,
                                                                      self._sensor_drone,
                                                                      self._ugv])
            if len(entities_with_los_to_enemy) > 0:
                # ENEMY FOUND !!!
                self._move_commanded = False
                self.attack_enemy(action_list, entities_with_los_to_enemy)
            else:  # elif not all_entities_positioned:
                # MOVE TO INDICATION TARGET
                if not move_to_indication_target_commanded:
                    move_to_indication_target_commanded = True
                    self.move_to_indication_target(action_list)

                all_entities_positioned = self.is_entity_positioned(self._suicide_drone,
                                                                    PlannerScenarioEnv.NORTH_WEST_SUICIDE) and \
                                          self.is_entity_positioned(self._sensor_drone,
                                                                    PlannerScenarioEnv.NORTH_EAST_OBSERVER) and \
                                          self.is_entity_positioned(self._ugv,
                                                                    PlannerScenarioEnv.SOUTH_WEST_UGV_POS)

            # Execute Actions in simulation
            try:
                obs, reward, done, _ = self.plannerEnv.step(action_list)
                self.rootlog.debug('step {}: obs = {}, reward = {}, done = {}'.format(self._step, obs, reward, done))
                self.plannerEnv.render()
            except RuntimeError:
                self.rootlog.debug('step {}: LOS SERVER DOWN - Rerun the episode'.format(self._step))
                done = 1
                reason = 'LOS Exception'

            # DONE LOGIC
            if done or self._step >= self.plannerEnv.MAX_STEPS:
                if not reason == 'LOS Exception':
                    reason = "Logical Simulation" if done else "Step is " + str(self._step)
                    self._num_of_dead += len([enemy for enemy in self.plannerEnv.enemies if not enemy.is_alive])
                    self._num_of_lost_devices += len(
                        [e for e in self.plannerEnv.entities if e.health['state '] != '0'])
                    done = True
                else:
                    #  'LOS Exception'  - Restart scenario
                    return (0, 0, 0, 0)
        if done:
            # Episode is done
            diff_step = self.plannerEnv.MAX_STEPS - self._step + 1
            diff_step = diff_step / self.plannerEnv.MAX_STEPS
            this_reward = self.compute_reward(diff_step)

            if len(self._episode_experience) > 0:
                for i, experience in enumerate(self._episode_experience):
                    experience[-1] = this_reward * DISCOUNT_FACTOR ** (len(self._episode_experience) - i)

                # add last next_state
                self._episode_experience[-1][2] = self.get_obs()

            else:
                self.rootlog.warning('Episode Experience is empty')
            # save episode experience replay to file
            now = datetime.datetime.now()
            with open('/tmp/experience.txt', 'a') as f:
                f.write("----------" + now.strftime("%Y-%m-%d-%H:%M:%S") + "----------\n"
                                                                           ".")
                for experience in self._episode_experience:
                    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                            .format(experience[0][0], experience[0][1], experience[0][2],
                                    experience[1][0], experience[1][1], experience[1][2],
                                    experience[2][0], experience[2][1], experience[2][2],
                                    experience[3][0], experience[3][1], experience[3][2],
                                    experience[4],
                                    experience[5][0], experience[5][1], experience[5][2],
                                    experience[6][0], experience[6][1], experience[6][2],
                                    experience[7][0], experience[7][1], experience[7][2],
                                    experience[8][0], experience[8][1], experience[8][2],
                                    experience[9]))

            self.rootlog.info(
                "Scenario completed: step " + ascii(self._step) + " reward " + ascii(this_reward) + " Done " + ascii(
                    done) + " Reason " + reason)
        # return this_reward, int(diff_step * self.plannerEnv.MAX_STEPS), self._num_of_dead, self._num_of_lost_devices

        return self.get_obs()

    def step(self, action):
        assert isinstance(action, int) and 0 <= action <= len(PlannerScenarioEnv.OBSERVER_WPS)
        reason = 'No reason'
        self._plan_index = action
        inner_step_start = self._step
        immediate_reward = 0.0
        self._change_target, done, self._move_commanded = False, False, False
        while not self._change_target and not done:
            self._step += 1
            # ACTION LOGIC
            # Reset Actions
            action_list = {'MOVE_TO': [], 'LOOK_AT': [], 'ATTACK': [], 'TAKE_PATH': []}
            # List the enemies in line of sight
            entities_with_los_to_enemy = self.line_of_sight_to_enemy(
                [self._suicide_drone, self._sensor_drone, self._ugv])
            if len(entities_with_los_to_enemy) > 0:
                # ENEMY FOUND !!!
                suicide_distance_to_enemy = self._suicide_drone.pos.distance_to(self._sniper.pos)

                immediate_reward += (PlannerScenarioEnv.MAX_DISTANCE_TO_ENEMY - suicide_distance_to_enemy) \
                                / PlannerScenarioEnv.MAX_DISTANCE_TO_ENEMY

                # TODO uncomment these if you are fascist
                # self._move_commanded = False
                # self.attack_enemy(action_list, entities_with_los_to_enemy)
            else:
                self.ambush_on_indication_target(action_list)

            # Execute Actions in simulation
            try:
                self.rootlog.debug(
                    'inner step {}: step with action list =  MOVE_TO {} ATTACK {} LOOK AT {} TAKE PATH {} '.format(
                        self._step, action_list['MOVE_TO'], action_list['ATTACK'], action_list['LOOK_AT'],
                        action_list['TAKE_PATH']))
                obs, reward, done, _ = self.plannerEnv.step(action_list)
                # self.rootlog.debug(
                #     'inner step {}: obs = entity {}-{}- entity {}-{} entity {}--{}, reward = {}, done = {}'.format(
                #         self._step, \
                #         obs['entities'][0].id, obs['entities'][0].gpoint, obs['entities'][2].id,
                #         obs['entities'][2].gpoint, obs['entities'][1].id, obs['entities'][1].gpoint, \
                #         reward, done))
                self.plannerEnv.render()
            except RuntimeError:
                self.rootlog.error('inner step {}: LOS SERVER DOWN - Rerun the episode'.format(self._step))
                done = 1
                reason = 'LOS Exception'
                return np.zeros(shape=(4, 3), dtype=np.float16), 0.0, True, {}
                #return self.get_obs(), reward, done, {}
            this_reward = 0.0

            # DONE LOGIC
            if done or self._step >= self.plannerEnv.MAX_STEPS:
                reason = "Logical Simulation" if done else "Step is " + str(self._step)
                self._num_of_dead += len([enemy for enemy in self.plannerEnv.enemies if not enemy.is_alive])
                self._num_of_lost_devices += len([e for e in self.plannerEnv.entities if e.health['state '] != '0'])
                done = True

                # Episode is done
                diff_step = self.plannerEnv.MAX_STEPS - self._step + 1
                diff_step = diff_step / self.plannerEnv.MAX_STEPS
                this_reward = self.compute_reward(diff_step)

                if len(self._episode_experience) > 0:
                    for i, experience in enumerate(self._episode_experience):
                        experience[-1] = this_reward * DISCOUNT_FACTOR ** (len(self._episode_experience) - i)

                    # add last next_state
                    self._episode_experience[-1][2] = self.get_obs()

                else:
                    self.rootlog.warning('Episode Experience is empty')
                # save episode experience replay to file
                now = datetime.datetime.now()
                with open('/tmp/experience.txt', 'a') as f:
                    f.write("----------" + now.strftime("%Y-%m-%d-%H:%M:%S") + "----------\n"
                                                                               ".")
                    for experience in self._episode_experience:
                        f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                                .format(experience[0][0][0], experience[0][0][1], experience[0][0][2],
                                        experience[0][1][0], experience[0][1][1], experience[0][1][2],
                                        experience[0][2][0], experience[0][2][1], experience[0][2][2],
                                        experience[0][3][0], experience[0][3][1], experience[0][3][2],
                                        experience[1],
                                        experience[2][0][0], experience[2][0][1], experience[2][0][2],
                                        experience[2][1][0], experience[2][1][1], experience[2][1][2],
                                        experience[2][2][0], experience[2][2][1], experience[2][2][2],
                                        experience[2][3][0], experience[2][3][1], experience[2][3][2],
                                        experience[3]))

                self.rootlog.info(
                    "Scenario completed: step " + ascii(self._step) + " reward " + ascii(this_reward) + " Done "
                    + ascii(done) + " Reason " + reason)

        reward = immediate_reward / (self._step - inner_step_start + 1)

        return self.get_obs(), reward, done, {}
        # return self._get_obs(), this_reward, done, {}

    def get_obs(self):
        ref_pos = PlannerScenarioEnv.SOUTH_WEST_UGV_POS
        return np.array([self._suicide_drone.pos.from_ref(ref_pos), self._sensor_drone.pos.from_ref(ref_pos),
                         self._ugv.pos.from_ref(ref_pos), self._sniper.pos.from_ref(ref_pos)])


def main(args=None):
    trainer_root = logging.getLogger(__name__)
    formatter = logging.Formatter(
        "[%(filename)s:%(lineno)s - %(funcName)s() %(asctime)s %(levelname)s] %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    trainer_root.addHandler(stdout_handler)
    now = datetime.datetime.now()
    log_filename = '/tmp/plannertrainer-' + now.strftime("%Y-%m-%d-%H:%M:%S")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    trainer_root.addHandler(file_handler)

    socket_handler = SocketHandler('127.0.0.1', 19996)
    socket_handler.setLevel(logging.DEBUG)
    socket_handler.setFormatter(formatter)
    trainer_root.addHandler(socket_handler)
    trainer_root.setLevel(logging.DEBUG)
    trainer_root.info("trainer coucou")
    logging.info("coucou")
    planner_scenario_env = PlannerScenarioEnv()
    end_of_session = False
    experience_buffer =[]
    session_num = 1

    while not end_of_session:
        num_step = 0
        action = 0
        done = False
        obs = planner_scenario_env.reset()
        trainer_root.info('SESSION: {} initial state suicide {} sensor {} ugv {} sniper {}'.format(session_num, obs[0], obs[1], obs[2], obs[3]))

        while not done:
            num_step += 1
            action = (action + random.randint(action,
                                              PlannerScenarioEnv.NUM_ACTIONS - 1)) % PlannerScenarioEnv.NUM_ACTIONS
            trainer_root.debug('action selected {}'.format(action))
            obs, reward, done, _ = planner_scenario_env.step(action)
            trainer_root.debug(
                'state OUTER STEP {} suicide {} sensor {} ugv {} sniper {}'.format(num_step, obs[0], obs[1], obs[2], obs[3]))
            trainer_root.critical('OUTER STEP {} reward {}'.format(num_step, reward))

            if len(experience_buffer) > 0:
                # save current state as next state to previous experience
                experience_buffer[-1][2] = obs
            # save current state and action
            experience_buffer.append([obs, action, np.zeros(shape=(4, 3), dtype=np.float16), reward])

        with open('/tmp/outer_experience.txt', 'a') as f:
            f.write("----------" + now.strftime("%Y-%m-%d-%H:%M:%S") + "----------\n"
                                                                       ".")
            for experience in experience_buffer:
                f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                        .format(experience[0][0][0], experience[0][0][1], experience[0][0][2],
                                experience[0][1][0], experience[0][1][1], experience[0][1][2],
                                experience[0][2][0], experience[0][2][1], experience[0][2][2],
                                experience[0][3][0], experience[0][3][1], experience[0][3][2],
                                experience[1],
                                experience[2][0][0], experience[2][0][1], experience[2][0][2],
                                experience[2][1][0], experience[2][1][1], experience[2][1][2],
                                experience[2][2][0], experience[2][2][1], experience[2][2][2],
                                experience[2][3][0], experience[2][3][1], experience[2][3][2],
                                experience[3]))

        trainer_root.info('episode done! reward {}'.format(reward))
        f = open("results.csv", "a")
        curr_string = datetime.datetime.now().__str__() + "," + reward.__str__() + "," + num_step.__str__() + "\n"
        f.write(curr_string)
        f.close()
        session_num += 1
        if session_num > 10000:
            end_of_session = True


if __name__ == '__main__':
    main()
