from logic_simulator.entity import Entity
from logic_simulator.pos import Pos
import numpy as np
import copy
import logging

class Drone(Entity):

    # for L.L.A.
    MAX_ACC_MPS2 = 2.1 / 100000.0
    MAX_DECC_MPS2 = 12.1 / 100000.0
    MAX_YAW_RATE_DEG_SEC = 90.0 / 100000.0
    MAX_SPEED_MPS = 5.5556 / 100000.0  # 20.0 Kmh
    MAX_RANGE_OF_VIEW = 30 / 100000.0
    FIELD_OF_VIEW = 0.1745  # radians.   approx. 10 deg

    # MAX_ACC_MPS2 = 2.1
    # MAX_DECC_MPS2 = 12.1
    # MAX_YAW_RATE_DEG_SEC = 90.0
    # MAX_SPEED_MPS = 5.5556 # 20.0 Kmh
    # MAX_RANGE_OF_VIEW = 50    #meters
    # FIELD_OF_VIEW = 0.349  # radians.   approx. 20 deg

    def __init__(self,id, pos: Pos):
        super().__init__(id,pos)

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
        logging.info('Drone go_to {} {} {}'.format(target_wp.X, target_wp.Y, target_wp.Z))
        if not self._target_pos.equals(target_wp):
            # commanded target has changed
            self._change_target(target_wp)
        elif self._reached_target():
            self._hover_in_place()
        else:
            self._continue_to_current_target()

    def is_line_of_sight_to(self, pos):
         
        range_to_target =  self.pos.distance_to(pos)

        is_los = range_to_target < Drone.MAX_RANGE_OF_VIEW

        if is_los:
            range_to_look_at = self.pos.distance_to(self.looking_at)
            direction_to_target = self.pos.direction_vector(pos)
            direction_to_look_at = self.pos.direction_vector(self.looking_at)

            # cos alpha = A dot B / (norm A * norm B)
            cos_angle = np.dot(direction_to_target , direction_to_look_at) / (range_to_target * range_to_look_at)

            # first quater  - cos function decreasing
            is_los  = abs(cos_angle) > np.cos(Drone.FIELD_OF_VIEW)

        return is_los


    def look_at(self, pos):
        logging.info('Drone look_at {} {} {}'.format(pos.X, pos.Y, pos.Z))
        assert pos.Z <  30
        self._looking_at = copy.copy(pos)
    
    def step(self, *args):
        # verify input
        assert len(args) == 1
        assert isinstance(args[0], Pos) 
        target_wp = args[0]
        self._t += 1.0
        # print("Drone step", self._t, self.pos)
        
        if not self._target_pos.equals(target_wp):
            # commanded target has changed
            self._change_target(target_wp)
        elif self._reached_target():
            self._hover_in_place()
        else:
            self._continue_to_current_target()
                
    
    def clone(self):
        d = Drone(self.id, self.pos)
        d._velocity_dir = self._velocity_dir
        d._speed = self._speed
        return d
    
    def predict(self, t: float)-> (Pos, float):
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
            dist_to_max_vel = (Drone.MAX_ACC_MPS2  * time_to_max_vel ** 2.0) / 2.0
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

    
    def _reached_target(self)->bool:
        return self._pos.distance_to(self._target_pos) <= self._speed

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
        self._speed  = max(self._speed + Drone.MAX_ACC_MPS2, Drone.MAX_SPEED_MPS)
        self._velocity_dir = np.array([self._target_pos.X - self._pos.X,self._target_pos.Y - self._pos.Y,self._target_pos.Z - self._pos.Z])
        self._velocity_dir = self._velocity_dir / np.linalg.norm(self._velocity_dir)
        velocity = self._speed * self._velocity_dir
        self._pos.add(velocity)
        logging.debug("end pos {} velocity {} _target_pos {}".format(self.pos, self.velocity, self._target_pos))
       

    