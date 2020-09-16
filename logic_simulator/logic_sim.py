import time

import numpy as np
import random
from logic_simulator.drone import Drone
from logic_simulator.ugv import Ugv
from logic_simulator.sensor_drone import SensorDrone
from logic_simulator.suicide_drone import SuicideDrone
from logic_simulator.pos import Pos
from logic_simulator.enemy import Enemy
import copy
from itertools import chain
import gym
import logging
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

class LogicSim(gym.Env):
    NUM_ACTIONS = 4
    MAX_STEPS = 1000
    NUM_OF_ENTITIES = 3
    NUM_OF_ENEMIES = 1
    EPSILON = 0.1
    ACTIONS_TO_METHODS = {
        'MOVE_TO': {SuicideDrone: Drone.go_to, SensorDrone: Drone.go_to},
        'LOOK_AT': {SuicideDrone: Drone.look_at, SensorDrone: Drone.look_at, Ugv: Ugv.look_at},
        'ATTACK': {SuicideDrone: SuicideDrone.attack, Ugv: Ugv.attack},
        'TAKE_PATH': {Ugv: Ugv.go_to}
    }

    # # Set this in SOME subclasses
    # metadata = {'render.modes': []}
    # reward_range = (-np.inf, np.inf)
    # spec = None

    # # Set these in ALL subclasses

    observation_space = gym.spaces.Tuple([
        gym.spaces.Tuple([  # Entities obs
            gym.spaces.Tuple([
                gym.spaces.Box(low=-1000, high=1000, shape=(3,), dtype=float),  # pos
                gym.spaces.Box(low=-1000, high=1000, shape=(3,), dtype=float),  # velocity
                gym.spaces.Box(low=-1000, high=1000, shape=(3,), dtype=float),  # look_at
                gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float)
            ])  # health
            for _ in range(NUM_OF_ENTITIES)
        ]),
        gym.spaces.Tuple([  # Enemies Obs
            gym.spaces.Tuple([
                gym.spaces.Box(low=-1000, high=1000, shape=(3,), dtype=float),  # pos
                gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float),  # health
                gym.spaces.Discrete(Enemy.NUM_OF_PRIORITIES)  # priority
            ])
            for _ in range(NUM_OF_ENEMIES)
        ]),
        gym.spaces.Tuple([gym.spaces.MultiBinary(1) for _ in range(NUM_OF_ENTITIES)])  # match_los
    ])
    action_space = gym.spaces.Tuple([
        gym.spaces.Tuple([gym.spaces.Discrete(4), gym.spaces.Box(low=-1000, high=1000, shape=(3,), dtype=float)]) for _
        in range(NUM_OF_ENTITIES)
    ])

    def __init__(self, entities: dict, enemies=[]):
        self._entities = entities
        self._enemies = enemies
        self._step = 0
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._scatter = None
        matplotlib.interactive(True)


        print(LogicSim.observation_space)

    def reset(self):
        self._step = 0
        for e in chain(self._entities.values(), self._enemies):
            e.reset()
        return self._get_obs()

    def render(self, mode='human'):
        # logging.error('Env cannot be rendered')
        # assert False, 'Env cannot be rendered'

        positions = [(e.pos.x, e.pos.y, e.pos.z, 'r' if isinstance(e, Enemy) else 'b') for e in
                     chain(self._entities.values(), self._enemies)]
        res = list(zip(*positions))
        if not self._scatter is None:
            # self._scatter.remove()
            self._ax.cla()
        self._scatter = self._ax.scatter(res[0], res[1], res[2], c=res[3], marker='o')

        self._ax.set_xlabel('X')
        self._ax.set_ylabel('Y')
        self._ax.set_zlabel('Z')

        plt.draw()
        plt.pause(0.02)

    def step(self, actions):
        """
        actions - dictionary of actions_id to entity_id-params pairs
                e.g. actions = {'MOVE_TO':[{'SensorDrone': (target_wp1)},{'Suicide': (target_wp2)}],
                    'LOOK_AT':[{'SensorDrone': (target_wp3)}],
                    'ATTACK':[],
                    'TAKE_PATH':[{'UGV':('Path1',target_wp3)}]}

        """
        self._step += 1

        self._execute_entities_actions(actions)

        self._update_enemies()

        return self._get_obs(), self.reward(), self.is_done(), {}

    def _get_obs(self):
        match_los = self._compute_all_los()
        entities_state = [e.state for e in self._entities.values()]
        enemies_state = [e.state for e in self._enemies]
        return entities_state, enemies_state, match_los

    def _update_enemies(self):
        for e in self._enemies:
            e.step()
        self._enemies = [enemy for enemy in self._enemies if enemy.health > 0]

    def _execute_entities_actions(self, actions):
        for action_name, ent_params_list in actions.items():
            assert action_name in LogicSim.ACTIONS_TO_METHODS.keys(), \
                'LogicSim.ACTIONS_TO_METHODS does not have key {}'.format(action_name)
            for ent_params in ent_params_list:
                for entity_id, params in ent_params.items():
                    assert entity_id in self._entities.keys(), \
                        'self._entities does not have key {}'.format(str(entity_id))
                    entity = self._entities[entity_id]
                    method = LogicSim.ACTIONS_TO_METHODS[action_name][entity.__class__]
                    params = self._parse_attack_params(params) if action_name == 'ATTACK' else params
                    method(entity, *params)

    def _parse_attack_params(self, params):
        assert isinstance(params, tuple), 'params should be tuple'
        assert isinstance(params[0], Pos), "ATTACK gets a Pos to attack"
        pos = params[0]
        enemies_in_danger = [e for e in self._enemies if e.pos.distance_to(pos) < LogicSim.EPSILON]
        param_list = list(params)
        param_list.append(enemies_in_danger)
        params = tuple(param_list)
        return params

    def reward(self):
        return 0.0

    def is_done(self):
        return self._step >= LogicSim.MAX_STEPS or len(self._enemies) == 0

    def _compute_all_los(self):
        match_los = {}
        for enemy in self._enemies:
            match_los[enemy.id] = [entity for entity in self._entities.values() if
                                   entity.is_line_of_sight_to(enemy.pos)]
            if len(match_los[enemy.id]) > 0:
                logging.info('{} in line of sight !!!'.format(enemy.id))
        return match_los

    def clone(self):
        entities = {}
        for k, v in self._entities.items():
            entities[k] = v.clone()
        enemies = []
        for i, e in enumerate(self._enemies):
            enemies[i] = e.clone()
        return LogicSim(entities, enemies)

    @property
    def state(self):
        return np.array([ent.state for ent in self._entities.values() + self._enemies])

    def __str__(self):
        s = 'LogicSim state \n'
        for e in self._entities.values():
            s += str(e)
            s += '\n'
        return str(s)
