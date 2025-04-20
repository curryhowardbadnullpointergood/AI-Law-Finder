from coordinates import Coordinate

class Point:
    def __init__(self, coordinate, data=None, id=None):

        if not isinstance(coordinate, Coordinate):
            raise TypeError("must be an coordinate")

        self._coordinate = coordinate
        self._data = data 
        self._id = id 


    @property
    def coordinates(self):
        return self._coordinate

    @coordinates.setter
    def coordinates(self, value):
        if not isinstance(value, Coordinate):
            raise TypeError("Must be a Coordinates object")
        self._coordinate = value

    # Access x directly via Point
    @property
    def x(self):
        return self._coordinate.x

    @x.setter
    def x(self, value):
        self._coordinate.x = value

    # Access y directly via Point
    @property
    def y(self):
        return self._coordinate.y

    @y.setter
    def y(self, value):
        self._coordinate.y = value
    

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



