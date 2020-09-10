from logic_simulator.entity import Entity
from logic_simulator.pos import Pos
import numpy as np
import copy
import logging

class Ugv(Entity):
    MAX_ACC_MPS2 = 2.1
    MAX_DECC_MPS2 = 12.1
    MAX_YAW_RATE_DEG_SEC = 90.0
    MAX_SPEED_MPS = 5.5556 # 20.0 Kmh
    MAX_RANGE_OF_VIEW = 30
    FIELD_OF_VIEW = 0.1745  # radians.   approx. 10 deg
    paths = {}

    def __init__(self, id, pos: Pos):
        super().__init__(id, pos)
        self._current_path = ''
        self._current_path_wp_index = 0
    
    def reset(self):
        super().reset()
        self._current_path = ''
        self._current_path_wp_index = 0


    @property
    def pos(self) -> Pos:
        return self._pos

    @property
    def velocity(self):
        return self._speed * self._velocity_dir

    def look_at(self, pos):
        logging.info('Ugv look_at ({},{},{})'.format(pos.X, pos.Y, pos.Z))
        self._looking_at = copy.copy(pos)

    def go_to(self, path_id, target_wp):
        logging.info('Ugv go_to path_id{} ({},{},{})'.format(path_id, target_wp.X, target_wp.Y, target_wp.Z))
        if self._reached_target(target_wp):
            logging.info('Ugv go_to command satisfied')
            # go_to command satisfied
            self._hover_in_place()
        else:
            # adjust path
            if path_id != self._current_path:
                logging.info('Ugv go_to change path from {} to {}'.format(self._current_path, path_id))
                self._current_path = path_id
                self._current_path_wp_index = 0
            assert path_id in Ugv.paths.keys()
            waypoints = Ugv.paths[path_id]
            num_of_waypoints = len(waypoints)
            assert isinstance(waypoints, list) and num_of_waypoints > 0
            # next waypoint in cuurent path
            target_pos = waypoints[self._current_path_wp_index]
            if not self._target_pos.equals(target_pos):
                # commanded target has changed
                self._change_target(target_pos)
            elif self._reached_target():
                if self._current_path_wp_index == (num_of_waypoints - 1):
                    # last waypoint in path
                    self._hover_in_place()
                else:
                    logging.debug('Ugv go_to waypoint index {} achieved'.format(self._current_path_wp_index))
                    self._current_path_wp_index += 1
                    target_pos = waypoints[self._current_path_wp_index]
                    self._change_target(target_pos)
            else:
                logging.debug('Ugv go_to continue to index {}'.format(self._current_path_wp_index))
                self._continue_to_current_target()
    
    def is_line_of_sight_to(self, pos):
         
        range_to_target =  self.pos.distance_to(pos)

        is_los = range_to_target < Ugv.MAX_RANGE_OF_VIEW

        if is_los:
            range_to_look_at = self.pos.distance_to(self.looking_at)
            direction_to_target = self.pos.direction_vector(pos)
            direction_to_look_at = self.pos.direction_vector(self.looking_at)

            # cos alpha = A dot B / (norm A * norm B)
            cos_angle = np.dot(direction_to_target , direction_to_look_at) / (range_to_target * range_to_look_at)

            # first quater  - cos function decreasing
            is_los  = cos_angle > np.cos(Ugv.FIELD_OF_VIEW)

        return is_los

    def attack(self, pos, enemies_in_danger):
        logging.info('Ugv Attack on {} {} {}'.format(pos.X, pos.Y, pos.Z))
        # TODO logic for uncertainty and CTE
        for e in enemies_in_danger:
            e.health = 0.0
        self.health -= 0.05

    def _reached_target(self, pos = None)->bool:
        p = pos if (not (pos is None)) and isinstance(pos, Pos) else self._target_pos
        return self._pos.distance_to(p) <= self._speed

    def _change_target(self, target_wp: Pos):
        logging.debug("start pos {} velocity {} target_wp {}".format(self.pos, self.velocity, target_wp))
        self._velocity_dir = np.array([target_wp.X - self._pos.X,target_wp.Y - self._pos.Y,target_wp.Z - self._pos.Z])
        self._velocity_dir = self._velocity_dir / np.linalg.norm(self._velocity_dir)
        self._speed = 0.0
        self._target_pos = copy.copy(target_wp)
        logging.debug("end pos {} velocity {} target_wp {}".format(self.pos, self.velocity, target_wp))

    def _hover_in_place(self):
        logging.debug("start pos {} velocity {} _target_pos {}".format(self.pos, self.velocity, self._target_pos))
        self._velocity_dir = np.array([0.0,0.0,0.0],dtype=float)
        self._speed = 0.0
        # self._target_pos = self._pos
        logging.debug("end pos {} velocity {} _target_pos {}".format(self.pos, self.velocity, self._target_pos))

    def _continue_to_current_target(self):
        logging.debug("start pos {} velocity {} _target_pos {}".format(self.pos, self.velocity, self._target_pos))
        self._speed  = max(self._speed + Ugv.MAX_ACC_MPS2, Ugv.MAX_SPEED_MPS)
        self._velocity_dir = np.array([self._target_pos.X - self._pos.X,self._target_pos.Y - self._pos.Y,self._target_pos.Z - self._pos.Z])
        self._velocity_dir = self._velocity_dir / np.linalg.norm(self._velocity_dir)
        velocity = self._speed * self._velocity_dir
        self._pos.add(velocity)
        logging.debug("end pos {} velocity {} _target_pos {}".format(self.pos, self.velocity, self._target_pos))

    def step(self, *args):
        assert len(args) == 2
        assert isinstance(args[0], Pos)
        assert isinstance(args[1], int) 
        target_wp = args[0]
        path_id = args[1]
        self._t += 1.0
        print("Ugv step", self._t, self.pos)
        if not self._target_pos.equals(target_wp):
            self._change_target(target_wp)
        elif self._reached_target():
            self._hover_in_place()
        else:
            self._continue_to_current_target()