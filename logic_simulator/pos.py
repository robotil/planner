import numpy as np

class Pos:

    EPSILON_DISTANCE = 0.1

    def __init__(self,x = 0.0, y = 0.0, z = 0.0):
        self._x = x
        self._y = y
        self._z = z
    
    @property
    def X(self):
        return self._x
    
    @property
    def Y(self):
        return self._y

    @property
    def Z(self):
        return self._z

    def equals(self, other)-> bool:
        return self.distance_to(other) <= Pos.EPSILON_DISTANCE

    def add(self, vec: np.array):
        self._x += vec[0]
        self._y += vec[1]
        self._z += vec[2]

    def distance_to(self, other)->float:
        return np.linalg.norm(np.array([self.X, self.Y, self.Z]) - np.array([other.X,other.Y,other.Z]))


    def direction_vector(self, other):
        return np.array([other.X - self.X, other.Y - self.Y, other.Z - self.Z])
    
    def __str__(self):
        return "({X},{Y},{Z})".format(X=self.X,Y=self.Y,Z=self.Z)