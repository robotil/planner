# !/usr/bin/env python3
# building custom gym environment:
# # https://medium.com/analytics-vidhya/building-custom-gym-environments-for-reinforcement-learning-24fa7530cbb5

import gym
from gym import spaces
import numpy as np
import threading
import math
from math import pi as pi
from scipy.spatial.transform import Rotation as R
import sys, time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Header
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped, PolygonStamped, Twist, TwistStamped, PoseStamped, Point
from planner_msgs.msg import SDiagnosticStatus, SGlobalPose, SHealth, SImu, EnemyReport, OPath, SPath, SGoalAndPath, \
    STwist, EntityEnemyReport

from logic_simulator.pos import Pos
from planner.sim_admin import check_state_simulation, act_on_simulation
from planner.sim_services import check_line_of_sight, get_all_possible_ways
# from planner.EntityState import UGVLocalMachine, SuicideLocalMachine, DroneLocalMachine
from collections import deque
import logging
import random
import pathlib
import os

STOP = 0
START = 1
PAUSE = 2
RUN = 3

LIST_ACTIONS = ['MOVE_TO', 'LOOK_AT', 'ATTACK', 'TAKE_PATH']


def point_to_pos(point: Point) -> Pos:
    return Pos(point.x, point.y, point.z)


def pos_to_point(pos: Pos) -> Point:
    lon, lat, alt = pos.toLongLatAlt()
    return Point(x=lat, y=lon, z=alt)


class PlannerEnv(gym.Env):
    MAX_STEPS = 200
    STEP_REWARD = 1 / MAX_STEPS
    FINAL_REWARD = 1.0
    ENEMY_POS_2 = Point(x=29.999796, y=33.0004159, z=0.0447149366)
    TIMEOUT_SEC = 60.0

    class Enemy:
        def __init__(self, msg):
            self.cep = msg.cep
            # self.gpoint = msg.gpose # self.gpoint = Point(x=-0.000204155, y=0.00035984, z=0.044715006)
            self.gpoint = msg.gpose  # Point(x=40.0, y=-23.0, z=0.044715006)
            self.priority = msg.priority
            self.tclass = msg.tclass
            self.is_alive = msg.is_alive
            # self.is_alive = msg.is_alive
            self.id = msg.id
            self._pos = Pos()

        def update(self, n_enn):
            self.cep = n_enn.cep
            # 28/10/2020: Add 1.5 to the sniper, i.e all enemies
            self.gpoint = Point(x=n_enn.gpoint.x, y=n_enn.gpoint.y, z=n_enn.gpoint.z + 1.7)
            self._pos = point_to_pos(n_enn.gpoint)
            self.priority = n_enn.priority
            self.tclass = n_enn.tclass
            self.is_alive = n_enn.is_alive

        @property
        def pos(self):
            return self._pos

    class Entity:
        def __init__(self, msg):
            # def __init__(self, msg, state='zero'):
            self.id = msg.id
            self.diagstatus = msg.diagstatus
            self.gpoint = Point()
            self.imu = Imu()
            self.health = {}
            self.twist = Twist()
            self._pos = Pos()
            self._los_enemies = []
            self._discovered_enemies = {}
            self._disc_enemies_list = []

        @property
        def los_enemies(self):
            return self._los_enemies

        @property
        def disc_enemies_list(self):
            return self._disc_enemies_list

        @property
        def pos(self):
            return self._pos
            # return point_to_pos(self.gpoint)

        def update_desc(self, n_ent):
            self.diagstatus = n_ent.diagstatus

        def update_gpose(self, n_pose):
            if self.id == 'UGV':
                # 28/10/2020: Add 3.5 to UGV
                self.gpoint = Point(x=n_pose.x, y=n_pose.y, z=n_pose.z + 3.5)
            else:
                self.gpoint = Point(x=n_pose.x, y=n_pose.y, z=n_pose.z - 1.0)
            self._pos = point_to_pos(self.gpoint)

        def update_imu(self, n_imu):
            self.imu = n_imu

        def update_health(self, n_health):
            for a in n_health:
                self.health[a.key] = a.value

        def update_twist(self, n_twist):
            self.twist = n_twist

        def is_line_of_sight_to(self, pos):
            res = False
            for enm in self._los_enemies:
                if enm.pos.equals(pos):
                    res = True
                    break
            return res

        def is_los_enemy(self, enemy):
            res = False
            for enm in self._los_enemies:
                if enm.id == enemy.id:
                    res = True
                    break
            return res

        def update_discovered_enemy(self, enemy):
            right_now = time.time()
            if enemy.id in self._discovered_enemies:
                self._discovered_enemies[enemy.id]=right_now
                self._disc_enemies_list.append(enemy)
            else:
                self._discovered_enemies[enemy.id]=right_now
            for enmid in list[self._discovered_enemies]:
                if enmid != enemy.id:
                    if (right_now - self._discovered_enemies[enmid]) > self.TIMEOUT_SEC:
                        del self._discovered_enemies[enmid]
                        this_enemy = self.get_enemy(enmid)
                        if this_enemy is not None:
                            self._disc_enemies_list.remove[this_enemy]




    def get_entity(self, id):
        found = None
        for elem in self.entities:
            if elem.id == id:
                found = elem
                break
        return found

    def get_enemy(self, id):
        found = None
        for elem in self.enemies:
            if elem.id == id:
                found = elem
                break
        return found

    def global_pose_callback(self, msg):
        this_entity = self.get_entity(msg.id)
        if (this_entity == None):
            self.node.get_logger().info('This entity "%s" is not managed yet' % msg.id)
            return
        this_entity.update_gpose(msg.gpose.point)
        self.node.get_logger().debug('Received: "%s"' % msg)

    def entity_description_callback(self, msg):
        a = self.Entity(msg)
        res = False
        for elem in self.entities:
            if elem.id == a.id:
                elem.update_desc(a)
                res = True
                break
        if not res:
            self.entities.append(a)
            # if not a.id == 'UGV':
            self.entities_queue.append(a)

        self.node.get_logger().debug('Received: "%s"' % msg)

    def enemy_description_callback(self, msg):
        a = self.Enemy(msg)
        if a.id == 'Sniper':
            res = False
            for elem in self.enemies:
                if elem.id == a.id:
                    elem.update(a)
                    res = True
                break
            if not res:
                self.enemies.append(a)
                self.match_los[a.id] = []
        self.node.get_logger().debug('Received: "%s"' % msg)

    def entity_imu_callback(self, msg):
        this_entity = self.get_entity(msg.id)
        if (this_entity == None):
            self.node.get_logger().info('This entity "%s" is not managed yet' % msg.id)
            return
        this_entity.update_imu(msg.imu)
        self.node.get_logger().debug('Received: "%s"' % msg)

    def entity_overall_health_callback(self, msg):
        this_entity = self.get_entity(msg.id)
        if (this_entity == None):
            self.node.get_logger().info('This entity "%s" is not managed yet' % msg.id)
            return
        this_entity.update_health(msg.values)
        self.node.get_logger().debug('Received: "%s"' % msg)

    def entity_twist_callback(self, msg):
        this_entity = self.get_entity(msg.id)
        if (this_entity == None):
            self.node.get_logger().info('This entity "%s" is not managed yet' % msg.id)
            return
        this_entity.update_twist(msg.twist)
        self.node.get_logger().debug('Received: "%s"' % msg)

    def entity_discovered_enemy_callback(self, msg):
        this_entity = self.get_entity(msg.id)
        if this_entity is None:
            self.node.get_logger().info('This entity "%s" is not managed yet' % msg.id)
            return
        this_entity.update_discovered_enemy(msg.enemy)
        self.node.get_logger().debug('Received: "%s"' % msg)

    def move_entity_to_goal(self, entity_id, goal):
        self.node.get_logger().info('Move entity:' + entity_id + " to position:" + goal.__str__())
        msg = SGlobalPose()
        msg.gpose = goal
        msg.id = entity_id
        self.moveToPub.publish(msg)


    def look_at_goal(self, entity_id, goal):
        self.node.get_logger().info('Entity:' + entity_id + " should look at:" + goal.__str__())
        msg = SGlobalPose()
        msg.gpose = goal
        msg.id = entity_id
        self.lookPub.publish(msg)

    # def take_path(self, entity_id, path):
    #     self.node.get_logger().info('Entity:' + entity_id + " should take the path:" + path.name)
    #     msg = SPath()
    #     msg.path = path
    #     msg.id = entity_id
    #     self.takePathPub.publish(msg)

    def take_goal_path(self, entity_id, path, goal):
        self.node.get_logger().info(
            'Entity:' + entity_id + " should take the path:" + path.name + " to reach: " + goal.__str__())
        msg = SGoalAndPath()
        msg.goal = goal  # Point
        msg.path = path  # OPath
        msg.id = entity_id
        self.takeGoalPathPub.publish(msg)

    def attack_goal(self, entity_id, goal):
        self.node.get_logger().info('Entity:' + entity_id + " attack at:" + goal.__str__())
        msg = SGlobalPose()
        msg.gpose = goal
        msg.id = entity_id
        self.attackPub.publish(msg)

    def thread_ros(self):
        print("Starting thread of ros threads")
        # executor = MultiThreadedExecutor(num_threads=4)
        # executor.add_node(self.node)
        # executor.spin()
        # When we will be solid:
        try:
            executor = MultiThreadedExecutor(num_threads=4)
            executor.add_node(self.node)
            try:
                executor.spin()
            except KeyboardInterrupt:
                act_on_simulation(ascii(STOP))
                print('server stopped cleanly')
            finally:
                act_on_simulation(ascii(STOP))
                executor.shutdown()
                self.node.destroy_node()
        finally:
            print("nothing")

    # entity1 = {'Entity:Suicide'}
    # entity2 = {'Entity:Drone'}
    # match_los[enemy_id].append(entity1)
    # match_los[enemy_id].append(entity2)
    #   match_los = {enemy_id: [{}], enemy+id: [{}],}
    def compute_all_los(self):
        self.node.get_logger().set_level(rclpy.logging.LoggingSeverity.WARN)
        # match_los = {}
        for enemy in [ene for ene in self.enemies if ene.is_alive]:
            this_enemy = enemy
            one = this_enemy.gpoint
            # onePsik =  Point(x=one.y, y=one.x, z=one.z)
            # match_los[this_enemy.id] = []
            # for entity in self.entities:
            #     two = entity.gpoint
            # twoPsik = Point(x=two.y, y=two.x, z=two.z)
            for entity in self.entities_queue:
            # while self.entities_queue:
            #     entity = self.entities_queue.popleft()
                two = entity.gpoint
                try:
                    start = time.time()
                    answer = check_line_of_sight(one, two)
                    self.keep_los_recording(entity.pos.x, entity.pos.y, entity.pos.z, enemy.pos.x, enemy.pos.y, enemy.pos.z,
                                            answer)
                    if answer:  # check_line_of_sight(one, two):
                        ### moshe melachlech
                        if entity.id == 'Suicide':
                            moshe_melachlech_distance_parameter = 30
                        else:
                            moshe_melachlech_distance_parameter = 100
                        moshe_melachlech_epsilon_parameter = 0.005
                        dist = entity.pos.distance_to(enemy.pos)
                        # rand = random.random()
                        self.node.get_logger().warning(
                            'enemy:' + this_enemy.id + ' entity:' + entity.id + " dist:" + ascii(dist))
                        # self.node.get_logger().warning('enemy:' + this_enemy.id + ' entity:'+ entity.id + " dist:"+ ascii(dist)+ " rand:"+ascii(rand))
                        # if math.sqrt(((enemy.pos.x - entity.pos.x) ** 2) + ((enemy.pos.y - entity.pos.y) ** 2) + ((enemy.pos.z - entity.pos.z) ** 2)) < moshe_melachlech_distance_parameter and random.random()< moshe_melachlech_epsilon_parameter:
                        # if entity.pos.distance_to(enemy.pos) < moshe_melachlech_distance_parameter and rand < moshe_melachlech_epsilon_parameter:
                        if entity.pos.distance_to(enemy.pos) < moshe_melachlech_distance_parameter:
                            # nret = act_on_simulation(ascii(PAUSE))
                            # nret = act_on_simulation(ascii(RUN))
                            ### end of moshe melachlech
                            res = False
                            for x in self.match_los[this_enemy.id]:
                                if x == entity.id:
                                    res = True
                                    break
                            if res == False:
                                self.match_los[this_enemy.id].append(entity.id)
                            if not entity.is_los_enemy(this_enemy):
                                if this_enemy.is_alive:
                                    entity._los_enemies.append(this_enemy)
                    else:
                        for x in self.match_los[this_enemy.id]:
                            if x == entity.id:
                                self.match_los[this_enemy.id].remove(entity.id)
                        if entity.is_los_enemy(this_enemy):
                            entity._los_enemies.remove(this_enemy)
                    self.node.get_logger().debug('duration:' + ascii(time.time() - start))
                except RuntimeError:
                    self.node.get_logger().error('Problems with LOS Service... Do Restart Simulation and Planner')
                    self.node.get_logger().set_level(rclpy.logging.LoggingSeverity.UNSET)
                    raise
                except KeyboardInterrupt:
                    act_on_simulation(ascii(STOP))
                    self.node.get_logger().set_level(rclpy.logging.LoggingSeverity.UNSET)
            # self.entities_queue.append(entity)
            self.node.get_logger().set_level(rclpy.logging.LoggingSeverity.UNSET)
        return self.match_los

    def __init__(self):
        super(PlannerEnv, self).__init__()
        print('Planner environment created!')
        self.hist_size = 3
        self.simOn = False

        # For time step
        self.current_time = time.time()
        self.last_time = self.current_time
        self.time_step = []
        self.last_obs = np.array([])
        self.TIME_STEP = 0.05  # 10 mili-seconds
        self.steps = 0
        self.total_reward = 0
        self.done = False
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([-1] * 4), high=np.array([1] * 4), dtype=np.float16)
        # spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=512,
                                            shape=(43657,), dtype=np.uint8)
        # Observation: for now two lists: one of entities and one of enemies
        self._obs = []
        # As a first step, actions can be only of 3 types: move, look, attack
        # For each type of action, there should be a list of pairs of {entity, goal}
        self._actions = {'MOVE_TO': [], 'LOOK_AT': [], 'ATTACK': [], 'TAKE_PATH': []}

        # ROS2 Support
        self.entities = []
        self.enemies = []
        self.entities_queue = deque([])
        self.match_los = {}

        rclpy.init()
        self.node = rclpy.create_node("planner")

        # Subscribe to topics
        self.entityPoseSub = self.node.create_subscription(SGlobalPose, '/entity/global_pose',
                                                           self.global_pose_callback, 10)
        self.entityDescriptionSub = self.node.create_subscription(SDiagnosticStatus, '/entity/description',
                                                                  self.entity_description_callback, 10)
        self.enemyDescriptionSub = self.node.create_subscription(EnemyReport, '/enemy/description',
                                                                 self.enemy_description_callback, 10)
        self.entityImuSub = self.node.create_subscription(SImu, '/entity/imu', self.entity_imu_callback, 10)
        self.entityOverallHealthSub = self.node.create_subscription(SHealth, '/entity/overall_health',
                                                                    self.entity_overall_health_callback, 10)
        self.entityTwistSub = self.node.create_subscription(STwist, '/entity/twist',
                                                            self.entity_twist_callback, 10)
        self.entityEnemySub = self.node.create_subscription(EntityEnemyReport, '/entity/enemy',
                                                            self.entity_discovered_enemy_callback, 10)
        # Publish topics
        self.moveToPub = self.node.create_publisher(SGlobalPose, '/entity/moveto/goal', 10)
        self.attackPub = self.node.create_publisher(SGlobalPose, '/entity/attack/goal', 10)
        self.lookPub = self.node.create_publisher(SGlobalPose, '/entity/look/goal', 10)
        self.takePathPub = self.node.create_publisher(SPath, '/entity/takepath', 10)
        self.takeGoalPathPub = self.node.create_publisher(SGoalAndPath, '/entity/followpath/goal', 10)

        self.num_of_dead_enemies = 0
        if  os.environ.get('ROS_DOMAIN_ID') == None:
            ros_domain_id = 0
        else:
            ros_domain_id = os.environ.get('ROS_DOMAIN_ID')
        self.ros_domain_id = ros_domain_id
        self.recordlosfn = '/tmp/'+ros_domain_id
        path = pathlib.Path(self.recordlosfn)
        path.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.recordlosfn = '/tmp/'+ros_domain_id +'/record-los.csv'

        #       self.node.create_rate(10.0)
        #        rclpy.spin(self.node)
        #         executor = MultiThreadedExecutor(num_threads=4)
        #         executor.add_node(self.node)
        #         executor.spin()
        x = threading.Thread(target=self.thread_ros, args=())
        print("Before running thread")
        x.start()

    def render(self, mode='human'):
        pass

    def keep_los_recording(self, x1, y1, z1, x2, y2, z2, los):
        f = open(self.recordlosfn, "a")
        curr_string = '{},{},{},{},{},{},{}\n'.format(x1, y1, z1, x2, y2, z2, los)
        f.write(curr_string)
        f.close()

    def get_obs(self):
        obs = self.update_state()
        # check line of sight?
        # check paths?
        return obs

    def update_state(self):
        entities = self.entities
        enemies = self.enemies
        line_of_sight_mesh = self.compute_all_los()
        # Line of sight?
        # Different path
        obs = {'entities': entities, 'enemies': enemies, 'los_mesh': line_of_sight_mesh}
        return obs

    def init_env(self):
        if self.simOn:
            ret = check_state_simulation()
            # if ret != START or ret != PAUSE:
            #     print("Inconsistent state of the simulation = " + str(ret)) LO Ichpat ...
            nret = act_on_simulation(ascii(STOP))
            if nret != STOP:
                print("Couldn't stop the simulation")
            else:
                self.simOn = False

        # Restart simulation
        ret = act_on_simulation(ascii(START))
        if ret != START:
            print("Couldn't start the simulation")
        ret = act_on_simulation(ascii(RUN))
        self.simOn = True

    def reset(self):
        # what happens when episode is done

        # clear all
        self.steps = 0
        self.total_reward = 0
        self._obs = []

        # initial state depends on environment (mission)
        self.init_env()

        # wait for simulation to set up
        # Temporary comments
        while True:  # wait for all topics to arrive
            if bool(self.entities) and bool(self.enemies):  # if there is some data:
                break

        # wait for simulation to stabilize
        # time.sleep(5)

        self._obs = self.get_obs()
        return self._obs

    def time_stuff(self):
        self.current_time = time.time()
        time_step = self.current_time - self.last_time
        if time_step < self.TIME_STEP:
            time.sleep(self.TIME_STEP - time_step)
            self.current_time = time.time()
            time_step = self.current_time - self.last_time
        self.time_step.append(time_step)
        self.last_time = self.current_time

    def reward_func(self):
        previous = self.num_of_dead_enemies
        num_of_dead_enemies = 0
        for enemy in self.enemies:
            if not enemy.is_alive:
                num_of_dead_enemies = num_of_dead_enemies + 1
        self.num_of_dead_enemies = num_of_dead_enemies

        bonus = 0.1 if num_of_dead_enemies > previous else 0.0

        malus = (- bonus) * self.steps / PlannerEnv.MAX_STEPS

        return malus

    def step(self, action):
        # send action to simulation
        self.do_action(action)

        self.time_stuff()
        # get observation from simulation
        self._obs = self.update_state()

        # calc step reward and add to total
        r_t = self.reward_func()

        # check if done
        done, final_reward, reset = self.end_of_episode()

        step_reward = r_t + final_reward
        self.total_reward = self.total_reward + step_reward

        self.done = done
        # if done:
        #     self.enemies = {}
        #     self.entities = {}
        #     print('Done ')

        info = {"state": self._obs, "action": action, "reward": self.total_reward, "step": self.steps}

        return self._obs, self.total_reward, self.done, info

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0
        current_pos = self._obs
        # self._obs = {'Entities':self.entities, 'Enemies':self.ennemies
        #
        # threshold = 7.5
        threshold = 0.0
        suicide_has_suicide = False
        for entity in self.entities:
            if entity.health['state '] != '0':
                suicide_has_suicide = True
                done = True

        num_of_enemies = len(self.enemies)
        num_of_dead_enemies = 0
        for enemy in self.enemies:
            if not enemy.is_alive:
                num_of_dead_enemies = num_of_dead_enemies + 1

        # if num_of_dead_enemies / num_of_enemies > threshold:
        if num_of_dead_enemies > 0:
            done = True

        if done == True:
            reset = 'goal achieved'
            print('----------------', reset, '--------dead enemies:', ascii(num_of_dead_enemies),
                  '--------dead entities:', ascii(suicide_has_suicide))
            logging.info('----------------' + reset + '--------dead enemies:' + ascii(
                num_of_dead_enemies) + '--------dead entities:' + ascii(suicide_has_suicide))
            self.node.get_logger().info('----------------' + reset + '--------dead enemies:' + ascii(
                num_of_dead_enemies) + '--------dead entities:' + ascii(suicide_has_suicide))

            # nret = act_on_simulation(ascii(STOP))
            # if nret != STOP:
            #     print("Couldn't stop the simulation")
            # else:
            #     self.simOn = False
            # final_reward = PlannerEnv.FINAL_REWARD

        self.steps += 1

        return done, final_reward, reset

    #
    # ex1 = {'Entity:Suicide': (0.0, 0.0, 0.0)}
    # ex2 = {'Entity:Drone': (0.1, 0.1, 0.1)}
    # _actions['MOVE_TO'].append(ex1)
    # _actions['MOVE_TO'].append(ex2)
    #    self._actions = {'MOVE_TO': [{}], 'LOOK_AT': [{}], 'ATTACK': [{}], 'TAKE_PATH':[{}]}

    def do_action(self, agent_action):
        self._actions = agent_action
        for act in self._actions['MOVE_TO']:
            if len(act) > 0:
                for elm in act:
                    # entity_id = elm.popitem()[0]  # get key of dictionary
                    entity_id = elm
                    goal = PointStamped()
                    lon, lat, alt = act[entity_id][0].toLongLatAlt()
                    goal.point = Point(x=lat, y=lon, z=alt)
                    self.move_entity_to_goal(entity_id, goal)
                    # self._actions['MOVE_TO'].remove(elm)
        for act in self._actions['LOOK_AT']:
            if len(act) > 0:
                for elm in act:
                    # entity_id = elm.popitem()[0]  # get key of dictionary
                    entity_id = elm
                    goal = PointStamped()
                    lon, lat, alt = act[entity_id][0].toLongLatAlt()
                    goal.point = Point(x=lat, y=lon, z=alt)
                    self.look_at_goal(entity_id, goal)
                    # self._actions['LOOK_AT'].remove(elm)
        for act in self._actions['ATTACK']:
            if len(act) > 0:
                for elm in act:
                    # entity_id = elm.popitem()[0]  # get key of dictionary
                    entity_id = elm
                    self.node.get_logger().info(
                        'Entity:' + entity_id + " attack at:" + ascii(act[entity_id][0].x) + ", " + ascii(
                            act[entity_id][0].y) + ", " + ascii(act[entity_id][0].z))
                    goal = PointStamped()
                    lon, lat, alt = act[entity_id][0].toLongLatAlt()
                    goal.point = Point(x=lat, y=lon, z=alt)
                    if entity_id == 'UGV':
                        self.attack_goal(entity_id, goal)
                    else:
                        self.move_entity_to_goal(entity_id, goal)

        for act in self._actions['TAKE_PATH']:
            if len(act) > 0:
                for elm in act:
                    entity_id = elm
                    # entity_id = elm.popitem()[0]  # get key of dictionary
                    path_name = act[entity_id][0]
                    path = OPath()
                    path.name = path_name
                    # Here goal is Point
                    lon, lat, alt = act[entity_id][1].toLongLatAlt()
                    goal_point = Point(x=lat, y=lon, z=alt)
                    self.take_goal_path(entity_id, path, goal_point)
                    # self._actions['TAKE_PATH'].remove(elm)

    def fill_straight(self, action_type, entity_id, parameter):
        # Parameter is a tupple
        if action_type not in LIST_ACTIONS:
            print("Strange action requested:" + action_type)
            return
        todo = {entity_id: parameter}
        self._actions[action_type].append(todo)

    def fill(self, act_name):
        # self._actions = {'MOVE_TO': [{}], 'LOOK_AT': [{}], 'ATTACK': [{}], 'TAKE_PATH': [{}]}
        if act_name == 'action_0':
            goal = Point(x=0.96, y=-0.00045, z=-0.00313)
            ex1 = {'UGV': ('Path1', goal)}
            self._actions['TAKE_PATH'].append(ex1)
            return True
        if act_name == 'action_1':
            goal = Point(x=0.96, y=-0.00045, z=-0.00313)
            ex1 = {'SensorDrone': (goal)}
            self._actions['MOVE_TO'].append(ex1)
            return True
        if act_name == 'action_2':
            goal = Point(x=0.96, y=-0.0, z=1.2)
            ex1 = {'Suicide': (goal)}
            self._actions['MOVE_TO'].append(ex1)
            return True
        if act_name == 'action_3':
            goal = Point(x=0.96, y=-0.00045, z=-0.00313)
            ex1 = {'SensorDrone': (0.96, -0.00045, -0.00313)}
            self._actions['LOOK_AT'].append(ex1)
            return True
        if act_name == 'action_4':
            goal = Point(x=0.96, y=-0.00045, z=-0.00313)
            ex1 = {'Suicide': (goal)}
            self._actions['MOVE_TO'].append(ex1)
            return True
        if act_name == 'action_5':
            goal = Point(x=0.96, y=-0.00045, z=-0.00313)
            ex1 = {'Suicide': (goal)}
            self._actions['MOVE_TO'].append(ex1)
            goal = Point(x=0.96, y=-0.00045, z=-0.00313)
            ex1 = {'SensorDrone': (goal)}
            self._actions['MOVE_TO'].append(ex1)
            goal = Point(x=0.96, y=-0.00045, z=-0.00313)
            ex1 = {'UGV': (goal)}
            self._actions['ATTACK'].append(ex1)
            return True
        if act_name == 'action_6':
            goal = Point(x=0.96, y=-0.00045, z=-0.00313)
            ex1 = {'UGV': (goal)}
            self._actions['MOVE_TO'].append(ex1)
            goal = Point(x=0.96, y=-0.00045, z=-0.00313)
            ex1 = {'UGV': (goal)}
            self._actions['ATTACK'].append(ex1)
            return True
