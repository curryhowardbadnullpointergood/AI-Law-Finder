
import numpy as np 
from points import Point
from coordinates import Coordinate
from sortedcontainers import SortedKeyList

class paretoList(SortedKeyList):

    def __init__(self):
        self.points = [] 
        super().__init__(key=lambda p: p.x)
    
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
    
    
    def get_coord(self):
        return [[p.x, p.y] for p in self.points]
        


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
        dom = self.dominant_point(p)

        if not len(dom):
            return 0.0

        candidates = [np.maximum(dom[i], dom[i + 1]) for i in range(len(dom) - 1)]
        candidates.append([p.x, np.min(dom[:, 1])])
        candidates.append([np.min(dom[:, 0]), p.y])

        candidates = np.array(candidates)

        return np.min(np.sqrt(np.sum((candidates - coord) ** 2, axis=1)))


    
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

        
        is_pareto = False
        
        right = self.bisect_left(p)

        while len(self) > right and self[right].y >= p.y and not (self[right].x == p.x and self[right].y == p.y):
            self.pop(right)
            is_pareto = True

       
        left = self.bisect_right(p) - 1

        if left == -1 or self[left][1] > p[1]:
            is_pareto = True

        
        if len(self) == 0:
            is_pareto = True

        if is_pareto:
            print("working!!!")
            super().add(p)

        return is_pareto
    
    
    
    
    def bisect_left1(self, value):
        _maxes = self.get_coord()

        if not _maxes:
            return 0

        pos = bisect_left(_maxes, value)

        if pos == len(_maxes):
            return self._len

        idx = bisect_left(self._lists[pos], value)
        return self._loc(pos, idx)
    


    def bisect_right1(self, value):
        _maxes = self.get_coord()

        if not _maxes:
            return 0

        pos = bisect_right(_maxes, value)

        if pos == len(_maxes):
            return self._len

        idx = bisect_right(self._lists[pos], value)
        return self._loc(pos, idx)







    def mergePareto(self, p):
        [self.insert_points(item) for item in p]
        return self

    def pointToList(self, p):
        [self.insert_points(a) for a in p]




def add():
    x= 1
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    x= x+ x 
    



def main():

    PL = paretoList()

    # A = np.random.rand(30, 2)
    
    # A = np.array([
    #     [0.33808567, 0.71078249],
    #     [0.53230935, 0.59861609],
    #     [0.14323923, 0.47040819],
    #     [0.81952712, 0.18329565],
    #     [0.32554314, 0.87905565],
    #     [0.837702, 0.03594707],
    #     [0.1555251, 0.61746657],
    #     [0.98466255, 0.24328891],
    #     [0.03506797, 0.72585188],
    #     [0.79812042, 0.12123644],
    #     [0.19997897, 0.83034173],
    #     [0.58609006, 0.27569143],
    #     [0.74290196, 0.85328874],
    #     [0.39965015, 0.2417549],
    #     [0.42588738, 0.57296336],
    #     [0.41389509, 0.99475464],
    #     [0.60322994, 0.64352742],
    #     [0.29886826, 0.05768122],
    #     [0.22382844, 0.81081495],
    #     [0.55465334, 0.22880699],
    #     [0.19405967, 0.65398404],
    #     [0.86381059, 0.23313568],
    #     [0.63044668, 0.01319088],
    #     [0.50990191, 0.27314109],
    #     [0.4934025, 0.79724437],
    #     [0.95615977, 0.29072778],
    #     [0.58282988, 0.13413821],
    #     [0.90627228, 0.24103038],
    #     [0.41002558, 0.32865168],
    #     [0.82831987, 0.88281003]
    # ])
    
    
    # #print(A)
    # print("#############")
    # [PL.insert_points(Point(Coordinate(x=a[0], y=a[1]))) for a in A]

    # paretoA = PL.np_array()
    # print(paretoA)
    
    PL.insert_points(Point(Coordinate(x=1, y=1)))
    PL.insert_points(Point(Coordinate(x=2, y=2)))
    
    print(PL.get_points())
    
    idx = PL.bisect_left(Point(Coordinate(0.1,5)))
    print(idx)
    PL.insert_points(Point(Coordinate(x=0.1, y=5), data=None))
    
    print(PL.get_points())
    






# Entry point of the program
if __name__ == "__main__":
    main()
