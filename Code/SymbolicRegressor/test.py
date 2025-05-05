#from coordinates import Coordinate
#from points import Point 
#from pareto import paretoList 
import numpy as np 
import polynomialFit as pf 
import sympy as sp 
import numpy as np
from sympy import symbols, simplify, S
import dimensionalAnalysis as da 
import neuralNetwork as nn 
from pareto import Point 
import pareto as p 
import plots as plots














UNIT_TABLE = {

    'mass':                 [1, 0, 0, 0, 0, 0, 0], 
    'length':               [0, 1, 0, 0, 0, 0, 0],
    'time':                 [0, 0, 1, 0, 0, 0, 0],
    'temperature':          [0, 0, 0, 1, 0, 0, 0],
    'current':              [0, 0, 0, 0, 1, 0, 0],
    'amount':               [0, 0, 0, 0, 0, 1, 0], 
    'luminous_intensity':   [0, 0, 0, 0, 0, 0, 1],


    'area':                   [0, 2, 0, 0, 0, 0, 0],   
    'acceleration':           [0, 1, -2, 0, 0, 0, 0],  
    'velocity':               [0, 1, -1, 0, 0, 0, 0],  
    'force':                  [1, 1, -2, 0, 0, 0, 0],  
    'energy':                 [1, 2, -2, 0, 0, 0, 0],  
    'power':                  [1, 2, -3, 0, 0, 0, 0], 
    'pressure':               [1, -1, -2, 0, 0, 0, 0],
    'charge':                 [0, 0, 1, 0, 1, 0, 0],  
    'voltage':                [1, 2, -3, 0, -1, 0, 0],
    'resistance':             [1, 2, -3, 0, -2, 0, 0],
    'capacitance':            [-1, -2, 4, 0, 2, 0, 0],
    'inductance':             [1, 2, -2, 0, -2, 0, 0], 
    'current':                [0, 0, 0, 0, 1, 0, 0],   
    'potential':              [1, 2, -3, 0, -1, 0, 0], 
    'magnetic_field':         [1, 0, -2, 0, -1, 0, 0], 
    'magnetic_flux':          [1, 2, -2, 0, -1, 0, 0], 
    'electric_field':         [1, 1, -3, 0, -1, 0, 0], 
    'permittivity':           [-1, -3, 4, 0, 2, 0, 0],
    'permeability':           [1, 1, -2, 0, -2, 0, 0],  
    'conductance':            [-1, -2, 3, 0, 2, 0, 0], 
    'density':                [1, -3, 0, 0, 0, 0, 0],   
    'frequency':              [0, 0, -1, 0, 0, 0, 0],   
    'wavenumber':             [0, -1, 0, 0, 0, 0, 0],   
    'momentum':               [1, 1, -1, 0, 0, 0, 0],   
    'angular_momentum':       [1, 2, -1, 0, 0, 0, 0],   
    'torque':                 [1, 2, -2, 0, 0, 0, 0],
    'specific_heat':          [0, 2, -2, -1, 0, 0, 0], 
    'thermal_conductivity':   [1, 1, -3, -1, 0, 0, 0], 
    'boltzmann_constant':     [1, 2, -2, -1, 0, 0, 0], 
    'entropy':                [1, 2, -2, -1, 0, 0, 0],  
    'temperature':            [0, 0, 0, 1, 0, 0, 0],    
    'time':                   [0, 0, 1, 0, 0, 0, 0],    
    'length':                 [0, 1, 0, 0, 0, 0, 0],    
    'mass':                   [1, 0, 0, 0, 0, 0, 0],    
    'volume':                 [0, 3, 0, 0, 0, 0, 0],    
    'surface_charge_density': [0, -2, 1, 0, 1, 0, 0],   
    'volume_charge_density':  [0, -3, 1, 0, 1, 0, 0],   
    'current_density':        [0, -2, 0, 0, 1, 0, 0],   
    'light_intensity':        [0, 0, 0, 0, 0, 0, 1],    
    'amount':                 [0, 0, 0, 0, 0, 1, 0],    
    'luminous_intensity':     [0, 0, 0, 0, 0, 0, 1],    
    'dimensionless':          [0, 0, 0, 0, 0, 0, 0],    
}











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






def test_dimensional_matrix_target():
    print("#############################################")
    print("Testing Dimensional Matrix Target: ")
    print("#############################################")
    independent = ['mass', 'acceleration']
    dependent = 'force'

    M, b = da.get_matrix_target(independent, dependent, UNIT_TABLE)

    print("M =\n", M)
    print("b =", b)



def test_dimensional_solve_Dimension():
    print("#############################################")
    print("Testing Dimensional Solve Dimension: ")
    print("#############################################")
    independent = ['mass', 'acceleration']
    dependent = 'force'

    M, b = da.get_matrix_target(independent, dependent, UNIT_TABLE)

    #print("M =\n", M)
    #print("b =", b)
    p, U = da.solveDimension(M, b)

    print("Particular solution p =", p)
    print("Nullspace basis U =")
    for vec in U:
        print(vec)


def test_dimensional_generate_Dimension():
    print("#############################################")
    print("Testing Dimensional Generate Dimension: ")
    print("#############################################")
    data_x = np.array([[1, 2, 3, 4, 5],    # mass
                       [1, 2, 3, 4, 5]])   # acceleration

# Example dependent variable (force corresponding to each data point)
    data_y = np.array([10, 20, 30, 40, 50])

# p (particular solution exponents from earlier)
    p = np.array([1, 1])

# U (nullspace basis - empty in this case)
    U = []

# Call the function to generate dimensionless data
    data_x_prime, data_y_prime = da.generate_dimensionless_data(data_x, data_y, p, U)

    print("Dimensionless independent variables (data_x_prime):")
    print(data_x_prime)

    print("Dimensionless dependent variable (data_y_prime):")
    print(data_y_prime) 

def test_dimensional_symbolic_transformation():
    print("#############################################")
    print("Testing Dimensional Symbolic Transformation: ")
    print("#############################################")

# Example Input
    independent_vars = ['a', 'm']
    p = [1, 1]
    U = np.array([[1, -1], [-1, 1]])

# Call the function
    symbolic_p, symbolic_U = da.symbolicTransformation(independent_vars, p, U)

# Output the results
    print(f"Symbolic Scaling Part (Π xᵢ^pᵢ): {symbolic_p}")
    print("Symbolic Dimensionless Groups (Π xᵢ^uᵢⱼ):")
    for u in symbolic_U:
        print(u)

    print("#####################################################")
    print("Pendulum")
    independent_vars = ['length', 'acceleration']
# Dependent variable: T
    dependent_var = 'time'
    M, b = da.get_matrix_target(independent_vars, dependent_var, UNIT_TABLE)
    p ,U = da.solveDimension(M,b)
    symbolic_p, symbolic_U = da.symbolicTransformation(independent_vars, p, U)


    print(f"Symbolic Scaling Part (Π xᵢ^pᵢ): {symbolic_p}")
    print("Symbolic Dimensionless Groups (Π xᵢ^uᵢⱼ):")
    for u in symbolic_U:
        print(u)

def test_neural_network():
    print("#############################################")
    print("Testing Neural Network: ")
    print("#############################################")

    units_dict = {
        'distance':     [0, 1, 0, 0, 0, 0, 0],   # L
        'acceleration': [0, 1, -2, 0, 0, 0, 0],  # L / T^2
        'time':         [0, 0, 1, 0, 0, 0, 0],   # T
    }
    
    independent_vars = ['acceleration', 'time']
    dependent_var = 'distance'

    M, b = da.get_matrix_target(independent_vars, dependent_var, units_dict)
    p, U = da.solveDimension(M, b)
    symbolic_p, symbolic_U = da.symbolicTransformation(independent_vars, p, U)

    g_vals = np.array([9.8, 9.8, 9.8, 9.8, 9.8])
    t_vals = np.array([1, 2, 3, 4, 5])
    s_vals = 0.5 * g_vals * t_vals**2

    data_x = np.vstack([g_vals, t_vals])  
    data_y = s_vals

    data_x_prime, data_y_prime = da.generate_dimensionless_data(data_x, data_y, p, U)

    train_loader, val_loader = nn.prepare_data(data_x_prime.T, data_y_prime, batch_size=2)
    model = nn.SymbolicNetwork(n_input=data_x_prime.shape[0], n_output=1)
    nn.train_network(model, train_loader, val_loader, epochs=600, learning_rate=1e-3, device='cpu')

    predictions = nn.predict(model, data_x_prime.T, device='cpu')
    gradients = nn.get_gradient(model, data_x_prime.T, device='cpu')
    
    print("Predictions: ")
    print(predictions)
    print("Gradients: ")
    print(gradients)




def test_pareto_points(): 

    print("########################################################")
    print("Testing Pareto Points: ")
    print("########################################################")
    x = symbols('x')

    current = []

    new_exprs = [sp.sin(x), x, sp.sin(x) + 0.1*x, sp.cos(x)]
    losses =    [0.01,    0.5,  0.009,        0.8]

    updated = p.update_pareto_points(current, new_exprs, losses)

    for c, l, f in updated:
        print(f"Complexity: {c:.2f}, Loss: {l:.4f}, Formula: {f}")

def test_pareto_plot(): 

    print("########################################################")
    print("Testing Plots: ")
    print("########################################################")
    
    pareto_points = [
        (2.0, 0.5, "x"),
        (3.5, 0.3, "x + y"),
        (5.0, 0.15, "sin(x) + y"),
        (7.0, 0.12, "log(x*y)"),
        (9.0, 0.11, "exp(x) * sin(y)")
    ]

# Generate plot
    #plot = plots.plot_pareto_frontier(pareto_points, title="Mock Pareto Frontier")
    #fig = plot.draw()
    #plt.show()

# Display plot (Jupyter/interactive only)
    #print(plot)




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
    # test_all_dimensionless()
    print("########################################################")
    # test_basic_dimensional_reduction()
    print("########################################################")
    # test_with_additional_pi_term()
    print("########################################################")
    test_dimensional_matrix_target()
    test_dimensional_solve_Dimension()
    test_dimensional_generate_Dimension()
    test_dimensional_symbolic_transformation()
    #test_neural_network()
    test_pareto_points()
    test_pareto_plot()














    print("End")
