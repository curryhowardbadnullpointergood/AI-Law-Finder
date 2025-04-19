

class Point:
    def __init__(self, x, y, data=None, id=None):
        self._x = x 
        self._y = y 
        self._data = data 
        self._id = id 


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

    # Getter and Setter for data
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    # Getter and Setter for id
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value 



