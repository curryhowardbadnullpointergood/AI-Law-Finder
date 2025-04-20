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


    def point_distance(self, p):

        if not isinstance(p, Point):
            raise TypeError("Input is not a Point!")

        coord = np.array([p.coordinate.x, p.coordinate.y]) 
        dominant = self.dominant_point(p)

        if not len(dom):
            return 0.0

        candidates = [np.maximum(dom[i], dom[i + 1]) for i in range(len(dom) - 1)]
        candidates.append([p.coordinate.x, np.min(dom[:, 1])])
        candidates.append([np.min(dom[:, 0]), p.coordinate.y])

        candidates = np.array(candidates)

        return np.min(np.sqrt(np.sum((candidates - point) ** 2, axis=1)))


    
    def dominant_point(self, p):

        if not isinstance(p, Point):
            raise TypeError("Input is not  a Point!")
        
        
        x = p.coordinate[0]
        low = 0
        high = len(self.points)

        while low < high:
            mid = (low + high) // 2
            if self.points[mid].coordinate[0] < x:
                low = mid + 1
            else:
                high = mid

        low -= 1
        
        return np.array([self[i][0:2] for i in range(low, -1, -1) if self[i][1] < p[1]])




    def insert_points(self, p):
        if not isinstance(p, Point):
            raise TypeError("Input is not a Point!")
        
        idx_left = self.bisect_right(p) - 1
        idx_right = self.bisect_left(p)

        # Remove dominated points to the right
        self.points = [
            pt for i, pt in enumerate(self.points)
            if not (i >= idx_right and pt.y >= p.y and not (pt.x == p.x and pt.y == p.y))
        ]

        # Recompute left index after removal
        idx_left = self.bisect_right(p) - 1

            # Check if point is Pareto-optimal (not dominated)
        if len(self.points) == 0 or idx_left < 0 or self.points[idx_left].y > p.y:
            insert_pos = self.bisect_left(p)
            self.points.insert(insert_pos, p)
            return True

        return False


    


    def left_search(self, p):
        x = p.x
        low, high = 0, len(self.points)

        while low < high:
            mid = (low + high) // 2
            if self.points[mid].x < x:
                low = mid + 1
            else:
                high = mid
        return low

    def right_search(self, p):
        x = p.x
        low, high = 0, len(self.points)

        while low < high:
            mid = (low + high) // 2
            if self.points[mid].x <= x:
                low = mid + 1
            else:
                high = mid
        return low



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
