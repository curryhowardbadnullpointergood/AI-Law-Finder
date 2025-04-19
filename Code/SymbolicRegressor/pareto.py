import numpy as np 
from points import Point

class paretoList:

    def __init__(self):
        self.points = [] 
    
    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

    # this gets the ids of all the points in the list 
    def get_ids(self):
        return [point.id for point in self]

    # get the values of the points 
    def get_points(self):
        return [[p.x, p.y, p.data] for p in self.points]


    def has_point(self, p):
        if not isinstance(p, Point):
            raise TypeError("Input is not a Point!")

        left = next((i for i, pt in enumerate(self.points) if pt.x >= p.x), len(self.points))

        return any(self[left + i].y == p.y for i in range(len(self) - left) if self[left + i].x == p.x)
    
    def np_array(self):
        return np.array([[p.x, p.y] for p in self])

    def insert_points(self, p):
        if not isinstance(p, Point):
            raise TypeError("Input is not a Point!")
        
         # sorts out the points based on x valye 
        self.points.sort(key=lambda pt: pt.x)

        
        # Find insert position (sorted by x)
        i = 0
        while i < len(self.points) and self.points[i].x < p.x:
            i += 1

        # Check if the new point is dominated by any left-point (domination check based on y value)
        if i > 0 and self.points[i - 1].y >= p.y:
            return False  # The point is dominated, don't add it

        # Remove any points that are dominated by p (if any points on the right of i are dominated by p)
        self.points = [
            pt for j, pt in enumerate(self.points)
            if not (j >= i and pt.y >= p.y and not (pt.x == p.x and pt.y == p.y))
        ]

        # Insert the new point at the correct position (sorted by x)
        self.points.insert(i, p)
        return True




    # this inserts the valid points onto the list 
    def inser_points(self, p):

        if not isinstance(p, Point):
            raise TypeError("Input is not a Point! ")
    
        # sorts out the points based on x valye 
        self.points.sort(key=lambda pt: pt.x)

        # Find the insert position 
        i = 0
        while i < len(self.points) and self.points[i].x < p.x:
            i += 1

        # Removes all the dominated points to the right
        self.points = [
            pt for j, pt in enumerate(self.points)
            if not (j >= i and pt.y >= p.y and not (pt.x == p.x and pt.y == p.y))
        ]

        # Recalculate the insert position 
        i = 0
        while i < len(self.points) and self.points[i].x < p.x:
            i += 1

        
        if i > 0 and self.points[i - 1].y <= p.y:
            return False

        # If there is no dominating point insert it 
        self.points.insert(i, p)
        
        return True
