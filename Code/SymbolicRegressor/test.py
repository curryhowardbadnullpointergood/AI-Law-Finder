from points import Point 
from pareto import paretoList 



####################################################################
def test_pareto_methods():

    p1 = paretoList()

    p1.insert_points(Point(1,1))
    p1.insert_points(Point(1,2))
    p1.insert_points(Point(2,3))

    
    # so the has points method works as intended 
    print(p1.get_points())
    print(p1.has_point(Point(1,1)))
    print(p1.has_point(Point(1,3)))
    
    # so the none works 
    print(p1.get_ids())

    #np_array
    print(p1.np_array())




if __name__ == "__main__":
    test_pareto_methods()
