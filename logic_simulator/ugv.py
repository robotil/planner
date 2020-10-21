from logic_simulator.entity import Entity
from logic_simulator.pos import Pos
import numpy as np
import copy
import logging


class Ugv(Entity):
    MAX_ACC_MPS2 = 3.5 * Entity.STEP_TIME * (Entity.STEP_TIME ** 2.0)
    MAX_DECC_MPS2 = 4 * Entity.STEP_TIME * (Entity.STEP_TIME ** 2.0)
    MAX_YAW_RATE_DEG_SEC = 90.0
    MAX_SPEED_MPS = 2.78 * Entity.STEP_TIME  # 20.0 Kmh
    MAX_RANGE_OF_VIEW = 10
    FIELD_OF_VIEW = 0.1745  # radians.   approx. 10 deg

    paths = {}

    def __init__(self, id, pos: Pos):
        super().__init__(id, pos)
        self._current_path = ''
        self._current_path_wp_index = 0
        self._fov = Ugv.FIELD_OF_VIEW
        self._max_range_of_view = Ugv.MAX_RANGE_OF_VIEW
        self._final_wp = None

    def reset(self):
        super().reset()
        self._current_path = ''
        self._current_path_wp_index = 0
        self._final_wp = None

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
        logging.debug('Ugv go_to path_id{} ({},{},{})'.format(path_id, target_wp.x, target_wp.y, target_wp.z))
        self._final_wp = target_wp
        if self._reached_target(target_wp):
            logging.debug('Ugv go_to command satisfied')
            # go_to command satisfied
            self._final_wp = None
            self._hover_in_place()
        else:
            # adjust path
            if path_id != self._current_path:
                logging.debug('Ugv go_to change path from {} to {}'.format(self._current_path, path_id))
                self._current_path = path_id
                self._current_path_wp_index = 0
            assert path_id in Ugv.paths.keys()
            waypoints = Ugv.paths[path_id]
            num_of_waypoints = len(waypoints)
            assert isinstance(waypoints, list) and num_of_waypoints > 0
            # next waypoint in current path
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

    def attack(self, pos, enemies_in_danger):
        logging.debug('Ugv Attack on {} {} {}'.format(pos.x, pos.y, pos.z))
        # TODO logic for uncertainty and CTE
        for e in enemies_in_danger:
            e.health *= 0.5
            e.health = 0.0 if e.health <= 0.1 else e.health
            logging.info('enemy {} health {}'.format(e.id, e.health))
        self.health -= 0.05

    def _reached_target(self, pos=None) -> bool:
        p = pos if (not (pos is None)) and isinstance(pos, Pos) else self._target_pos
        return self._pos.distance_to(p) <= Ugv.MAX_SPEED_MPS * 2.0

    def _change_target(self, target_wp: Pos):
        logging.debug("start pos {} velocity {} target_wp {}".format(self.pos, self.velocity, target_wp))
        self._velocity_dir = np.array([target_wp.x - self._pos.x, target_wp.y - self._pos.y, target_wp.z - self._pos.z])
        self._velocity_dir = self._velocity_dir / np.linalg.norm(self._velocity_dir)
        self._speed = 0.0
        self._target_pos = copy.copy(target_wp)
        logging.debug("end pos {} velocity {} target_wp {}".format(self.pos, self.velocity, target_wp))

    def _hover_in_place(self):
        logging.debug("start pos {} velocity {} _target_pos {}".format(self.pos, self.velocity, self._target_pos))
        self._velocity_dir = np.array([0.0, 0.0, 0.0], dtype=float)
        self._speed = 0.0
        # self._target_pos = self._pos
        logging.debug("end pos {} velocity {} _target_pos {}".format(self.pos, self.velocity, self._target_pos))

    def _continue_to_current_target(self):
        logging.debug("start pos {} velocity {} _target_pos {}".format(self.pos, self.velocity, self._target_pos))
        self._speed = max(self._speed + Ugv.MAX_ACC_MPS2, Ugv.MAX_SPEED_MPS)
        self._velocity_dir = np.array(
            [self._target_pos.x - self._pos.x, self._target_pos.y - self._pos.y, self._target_pos.z - self._pos.z])
        self._velocity_dir = self._velocity_dir / np.linalg.norm(self._velocity_dir)
        velocity = self._speed * self._velocity_dir
        self._pos.add(velocity)
        logging.debug("end pos {} velocity {} _target_pos {}".format(self.pos, self.velocity, self._target_pos))

    def update(self):
        if self._final_wp is not None:
            self.go_to(self._current_path, self._final_wp)


    # def step(self, *args):
    #     assert len(args) == 2
    #     assert isinstance(args[0], Pos)
    #     assert isinstance(args[1], int)
    #     target_wp = args[0]
    #     path_id = args[1]
    #     self._t += 1.0
    #     print("Ugv step", self._t, self.pos)
    #     if not self._target_pos.equals(target_wp):
    #         self._change_target(target_wp)
    #     elif self._reached_target():
    #         self._hover_in_place()
    #     else:
    #         self._continue_to_current_target()
