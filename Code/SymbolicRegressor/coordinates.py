import numpy as np 

class Coordinate:

    def __init__(self, x, y):

        if not isinstance(x, (int, float, np.float64)):
            raise TypeError("x must be an int, float, or np.float64")
        if not isinstance(y, (int, float, np.float64)):
            raise TypeError("y must be an int, float, or np.float64")

        self._x = x 
        self._y = y 
    
    
    # Getter and Setter for x
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    # Getter and Setter for y
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

 

