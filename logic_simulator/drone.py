from logic_simulator.entity import Entity
from logic_simulator.pos import Pos
import numpy as np
import copy
import logging


class Drone(Entity):
    MAX_ACC_MPS2 = 2.1 * Entity.STEP_TIME * (Entity.STEP_TIME ** 2.0)
    MAX_DECC_MPS2 = 12.1 * (Entity.STEP_TIME ** 2.0)
    MAX_YAW_RATE_DEG_SEC = 90.0
    MAX_SPEED_MPS = 5.5556 * Entity.STEP_TIME  # 20.0 Kmh
    MAX_RANGE_OF_VIEW = 45  # meters


    def __init__(self, id, pos: Pos):
        super().__init__(id, pos)
        self._max_range_of_view = Drone.MAX_RANGE_OF_VIEW

    def _is_same_args(self, *args):
        assert len(args) == 1
        assert isinstance(args[0], Pos)
        target_wp = args[0]
        return self._target_pos.equals(target_wp)

    def _change_args(self, *args):
        assert len(args) == 1
        assert isinstance(args[0], Pos)
        target_wp = args[0]
        self._change_target(target_wp)

    def go_to(self, target_wp):
        logging.debug('Drone go_to {} {} {}'.format(target_wp.x, target_wp.y, target_wp.z))
        assert target_wp.z < 30
        if not self._target_pos.equals(target_wp):
            # commanded target has changed
            self._change_target(target_wp)
        elif self._reached_target():
            self._hover_in_place()
        else:
            self._continue_to_current_target()

    def look_at(self, pos):
        logging.debug('Drone look_at {} {} {}'.format(pos.x, pos.y, pos.z))
        assert pos.z < 30
        self._looking_at = copy.copy(pos)

    # def step(self, *args):
    #     # verify input
    #     assert len(args) == 1
    #     assert isinstance(args[0], Pos)
    #     target_wp = args[0]
    #     self._t += 1.0
    #     # print("Drone step", self._t, self.pos)
    #
    #     if not self._target_pos.equals(target_wp):
    #         # commanded target has changed
    #         self._change_target(target_wp)
    #     elif self._reached_target():
    #         self._hover_in_place()
    #     else:
    #         self._continue_to_current_target()

    def clone(self):
        d = Drone(self.id, self.pos)
        d._velocity_dir = self._velocity_dir
        d._speed = self._speed
        return d

    def predict(self, t: float) -> (Pos, float):
        '''
        Predict Drone's state in t timesteps from now
        '''
        dist_to_target = self._pos.distance_to(self._target_pos)
        delta_vel = Drone.MAX_SPEED_MPS - self._velocity
        time_to_max_vel = delta_vel / Drone.MAX_ACC_MPS2
        if time_to_max_vel > t:
            # Not enough time to get to max speed.
            # After t drone wil still be accelerating.
            total_dist = (Drone.MAX_ACC_MPS2 * t ** 2.0) / 2.0
            predicted_pos = copy.copy(self.pos)
            predicted_pos.add(total_dist * self._velocity_dir)
            predicted_speed = self._speed + Drone.MAX_ACC_MPS2 * t
        else:
            # Drone will reach max speed before t
            dist_to_max_vel = (Drone.MAX_ACC_MPS2 * time_to_max_vel ** 2.0) / 2.0
            dist_in_max_vel = Drone.MAX_SPEED_MPS * (t - time_to_max_vel)
            total_dist = dist_to_max_vel + dist_in_max_vel
            predicted_pos = copy.copy(self.pos)
            predicted_pos.add(total_dist * self._velocity_dir)
            predicted_speed = Drone.MAX_SPEED_MPS

        if total_dist > dist_to_target:
            # Drone will reach target before t
            predicted_pos = self._target_pos
            predicted_speed = 0.0

        return predicted_pos, predicted_speed

    def _reached_target(self) -> bool:
        return self._pos.distance_to(self._target_pos) <= Drone.MAX_SPEED_MPS * 2.0

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
        self._speed = max(self._speed + Drone.MAX_ACC_MPS2, Drone.MAX_SPEED_MPS)
        self._velocity_dir = np.array(
            [self._target_pos.x - self._pos.x, self._target_pos.y - self._pos.y, self._target_pos.z - self._pos.z])
        self._velocity_dir = self._velocity_dir / np.linalg.norm(self._velocity_dir)
        velocity = self._speed * self._velocity_dir
        self._pos.add(velocity)
        logging.debug("end pos {} velocity {} _target_pos {}".format(self.pos, self.velocity, self._target_pos))
