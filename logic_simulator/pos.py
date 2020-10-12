import numpy as np
from geodesy import utm


class Pos:
    EPSILON_DISTANCE = 0.1
    old_school = False
    def __init__(self, x=0.0, y=0.0, z=0.0):
        """
            x=lat, y=long, z=alt
        Returns:
            object:Pos
        """
        if self.old_school:
            self._x = float(x)
            self._y = float(y)
            self._z = float(z)
        else:
            my_utm = utm.fromLatLong(x, y, z)
            my_point = my_utm.toPoint()
            self._x = my_point.x
            self._y = my_point.y
            self._z = my_point.z

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def equals(self, other) -> bool:
        return self.distance_to(other) <= Pos.EPSILON_DISTANCE

    def add(self, vec: np.array):
        self._x += vec[0]
        self._y += vec[1]
        self._z += vec[2]

    def distance_to(self, other) -> float:
        return np.linalg.norm(np.array([self.x, self.y, self.z]) - np.array([other.x, other.y, other.z]))

    def direction_vector(self, other):
        direction = np.array([other.x - self.x, other.y - self.y, other.z - self.z])
        return direction / np.linalg.norm(direction)

    def __str__(self):
        return "({X},{Y},{Z})".format(X=self.x, Y=self.y, Z=self.z)
