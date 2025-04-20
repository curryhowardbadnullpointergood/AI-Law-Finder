from coordinates import Coordinate
from points import Point 
from pareto import paretoList 




####################################################################
def test_pareto_methods():

    p1 = paretoList()
    print("Testing insert_points")
    p1.insert_points(Point(Coordinate(1,1)))
    p1.insert_points(Point(Coordinate(1,2)))
    p1.insert_points(Point(Coordinate(2,3)))

    print("Testing get points")
    # so the has points method works as intended 
    print(p1.get_points())

    print("Testing has point")
    print(p1.has_point(Point(Coordinate(1,1))))
    print(p1.has_point(Point(Coordinate(1,3))))
    
    # so the none works 
    print("Testing get ids")
    print(p1.get_ids())

    #np_array
    print("Testing np array conversion")
    print(p1.np_array())




if __name__ == "__main__":
    test_pareto_methods()
