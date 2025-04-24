from coordinates import Coordinate
from points import Point 
from pareto import paretoList 
import numpy as np 
import polynomialFit as pf 
import sympy as sp 
import numpy as np
from sympy import symbols, simplify, S
from dimensionalAnalysis import apply_dimensional_analysis 


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


####################################################################
def test_polynomial_fit_data_loading():

    print("########################################################")
    print("Testing load data: ")
    print("########################################################")
    data = np.array([[1, 2], [3, 4], [5, 6]])
    coeffs = [1, 2]
    variables = ['x0', 'x1']
    r = pf.load_data(data, variables)
    print(r)

def test_polynomial_fit_generate_expressions():
    print("########################################################")
    print("Testing Polynomial Generating Expressions: ")
    print("########################################################")
    coeffs=[2, 3]
    variables=['x0', 'x1']
    operators=['+', '-']
    max_degree=2
    r = pf.generate_expressions(coeffs, variables, operators, max_degree)
    print(r)

def test_polynomial_filter_expressions():
    print("########################################################")
    print("Testing Polynomial Filtering Expressions: ")
    print("########################################################")
    coeffs=[2, 3]
    variables=['x0', 'x1']
    operators=['+', '-']
    max_degree=2
    r = pf.generate_expressions(coeffs, variables, operators, max_degree)
    re = pf.filter_expressions(r, variables, coeffs, 2)
    print(re)


def test_polynomial_evaluate_expression():
    print("########################################################")
    print("Testing Polynomial Evaluating Expressions: ")
    print("########################################################")
    coeffs=[2, 3]
    variables=['x0', 'x1']
    operators=['+', '-']
    max_degree=2
    r = pf.generate_expressions(coeffs, variables, operators, max_degree)
    re = pf.filter_expressions(r, variables, coeffs, 2)
    X = np.array([
    [1, 2],  
    [2, 0],  
    [3, 1],  
    [0, 4],  
    ])
    y = 2 * X[:, 0]**2 + 3 * X[:, 1] 
    ree = pf.evaluate_expressions(re, variables, X, y )
    print(ree)





def test_polynomial_evaluate_expression():
    print("########################################################")
    print("Testing Polynomial Evaluating Expressions: ")
    print("########################################################")
    coeffs=[2, 3]
    variables=['x0', 'x1']
    operators=['+', '-']
    max_degree=2
    r = pf.generate_expressions(coeffs, variables, operators, max_degree)
    re = pf.filter_expressions(r, variables, coeffs, 2)
    X = np.array([
    [1, 2],  
    [2, 0],  
    [3, 1],  
    [0, 4],  
    ])
    y = 2 * X[:, 0]**2 + 3 * X[:, 1] 
    ree = pf.evaluate_expressions(re, variables, X, y )
    reee = pf.bestFit(ree)
    print(reee)




def test_all_dimensionless():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    var_names = ['x1', 'x2']
    
    units_db = {
        'x1': [0, 0, 0, 0, 0, 0],
        'x2': [0, 0, 0, 0, 0, 0],
        'y':  [0, 0, 0, 0, 0, 0]
    }

    tx, ty, names, solved_expr, exprs = apply_dimensional_analysis(X, y, var_names, units_db)
    
    assert np.allclose(tx, X)
    assert np.allclose(ty, y)
    assert names == var_names
    assert solved_expr == S(1)
    assert all(str(e) == v for e, v in zip(exprs, var_names))

def test_basic_dimensional_reduction():
    # Example: F = m * a (Force = mass * acceleration)
    X = np.array([[2, 10], [3, 20], [4, 40]])  # m, a
    y = np.array([20, 60, 160])  # F
    var_names = ['m', 'a']

    # units: [m, s, kg, T, V, cd]
    units_db = {
        'm': [0, 0, 1, 0, 0, 0],     # kg
        'a': [1, -2, 0, 0, 0, 0],    # m/s^2
        'y': [1, -2, 1, 0, 0, 0]     # Force = m*a => kg*m/s^2
    }

    tx, ty, names, solved_expr, exprs = apply_dimensional_analysis(X, y, var_names, units_db)
    
    m, a = symbols('m a')
    print("Expected:", m * a)
    print("Got:", solved_expr)
    print("Difference:", simplify(solved_expr - m * a))

    #assert simplify(solved_expr - m * a) == 0
    #assert np.allclose(ty, np.ones_like(y))  # because y = m*a
    #assert len(tx[0]) == 0  # no additional dimensionless variables needed

def test_with_additional_pi_term():
    # Fake case: y = x1 * x2, but x1 and x2 have overlapping but non-identical units
    X = np.array([[1, 2], [2, 4], [3, 6]])
    y = np.array([2, 8, 18])
    var_names = ['x1', 'x2']

    units_db = {
        'x1': [1, -1, 0, 0, 0, 0],  # m/s
        'x2': [-1, 1, 0, 0, 0, 0],  # s/m
        'y':  [0, 0, 0, 0, 0, 0]    # dimensionless
    }

    tx, ty, names, solved_expr, exprs = apply_dimensional_analysis(X, y, var_names, units_db)

    assert simplify(solved_expr) == S(1)  # target is already dimless
    assert len(tx[0]) == 1  # one pi term
    assert names[0] == 'pi1'
    assert np.allclose(ty, y)







if __name__ == "__main__":
    #test_pareto_methods()
    print("########################################################")
    print("Testing Polynomial Fit")
    print("########################################################")
    test_polynomial_fit_data_loading()
    test_polynomial_fit_generate_expressions()
    test_polynomial_filter_expressions()
    test_polynomial_evaluate_expression()
    print("########################################################")
    print("Testing Dimensional Analysis: ")
    print("########################################################")
    test_all_dimensionless()
    print("########################################################")
    test_basic_dimensional_reduction()
    print("########################################################")
    test_with_additional_pi_term()
    print("########################################################")



