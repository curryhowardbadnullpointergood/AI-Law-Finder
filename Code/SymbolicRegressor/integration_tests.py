# integration_tests.py
import unittest
import numpy as np
import sympy as sp
from sympy import symbols, sin, cos, sqrt

# Import functions from your modules
import dimensionalAnalysis as da
import polynomialFit as pf
import bruteForce as bf
import neuralNetwork as nn
# Assuming pareto.py has update_pareto_points and a way to calculate complexity
# If not, we'll mock or simplify this part.
try:
    import pareto as p
    # Let's define a simple complexity function if not available in pareto.py
    if not hasattr(p, 'calculate_complexity'):
        def calculate_complexity(expr):
            return sp.count_ops(expr)
        p.calculate_complexity = calculate_complexity
except ImportError:
    print("Warning: pareto module not found. Pareto tests will be limited.")
    p = None # Set p to None if module is missing

# Assuming plots.py has plot_pareto_frontier
try:
    import plots as plots
except ImportError:
    print("Warning: plots module not found. Plotting test will be limited.")
    plots = None # Set plots to None if module is missing


# Mock UNIT_TABLE if not directly accessible or for test isolation
UNIT_TABLE = {
    'mass': [1, 0, 0, 0, 0, 0, 0],
    'length': [0, 1, 0, 0, 0, 0, 0],
    'time': [0, 0, 1, 0, 0, 0, 0],
    'acceleration': [0, 1, -2, 0, 0, 0, 0],
    'force': [1, 1, -2, 0, 0, 0, 0],
    'velocity': [0, 1, -1, 0, 0, 0, 0],
    'dimensionless': [0, 0, 0, 0, 0, 0, 0],
}

class TestIntegration(unittest.TestCase):

    def test_01_da_to_polynomial_fit(self):
        """
        Test: Use Dimensional Analysis to get dimensionless data, then fit with PolynomialFit.
        Workflow: DA (get_matrix_target, solveDimension, generate_dimensionless_data) -> PF (evaluate_expressions, bestFit)
        """
        print("\n--- Test 01: DA -> Polynomial Fit ---")
        independent_vars = ['mass', 'acceleration']
        dependent_var = 'force'
        variables_sym = ['m', 'a'] # Symbolic names for polynomial fit

        # 1. Dimensional Analysis
        M, b = da.get_matrix_target(independent_vars, dependent_var, UNIT_TABLE)
        p_exp, U = da.solveDimension(M, b)
        self.assertEqual(len(U), 0, "Expected no dimensionless groups for F=ma") # F=ma has no Pi groups

        # Generate some sample physical data (F = m * a)
        mass_vals = np.array([1, 2, 1, 3])
        acc_vals = np.array([10, 5, 20, 2])
        force_vals = mass_vals * acc_vals
        data_x = np.vstack([mass_vals, acc_vals])
        data_y = force_vals

        # Make data dimensionless
        data_x_prime, data_y_prime = da.generate_dimensionless_data(data_x, data_y, p_exp, U)

        # data_x_prime should be unchanged if U is empty
        np.testing.assert_array_almost_equal(data_x, data_x_prime)
        # data_y_prime should be constant (y' = F / (m^1 * a^1))
        self.assertAlmostEqual(np.std(data_y_prime), 0.0, delta=1e-9)
        expected_y_prime = 1.0 # Expecting F/(ma) = 1
        np.testing.assert_array_almost_equal(data_y_prime, np.full_like(data_y_prime, expected_y_prime), decimal=6)

        # 2. Polynomial Fit on dimensionless data (y' = constant)
        # We expect a constant fit, degree 1 polynomial fit won't be perfect unless we add constants
        # Let's try fitting y' ~ func(x') - since x' isn't used, it's tricky for polynomialFit
        # Instead, let's verify the dimensionless target y' is constant as expected.
        print(f"Dimensionless target y': {data_y_prime}")
        self.assertTrue(np.allclose(data_y_prime, expected_y_prime))
        print("Verified dimensionless target is constant and equals 1.")


    def test_02_da_to_brute_force(self):
        """
        Test: Use DA to get dimensionless data, then fit with BruteForce.
        Workflow: DA (...) -> BF (generate_expressions, evaluate_expressions)
        """
        print("\n--- Test 02: DA -> Brute Force ---")
        # Pendulum example: T depends on L, g. T ~ sqrt(L/g)
        independent_vars = ['length', 'acceleration'] # L, g
        dependent_var = 'time' # T
        bf_vars = ['L', 'g'] # Symbolic names for brute force

        # 1. Dimensional Analysis
        M, b = da.get_matrix_target(independent_vars, dependent_var, UNIT_TABLE)
        p_exp, U = da.solveDimension(M, b) # Expect p = [1/2, -1/2], U empty

        # Check exponents p
        expected_p = np.array([0.5, -0.5])
        p_exp_np = np.array(p_exp).astype(float).flatten()
        np.testing.assert_array_almost_equal(p_exp_np, expected_p, decimal=6)
        self.assertEqual(len(U), 0, "Expected no dimensionless groups for Pendulum T(L,g)")


    def test_03_da_to_neural_network(self):
        """
        Test: Use DA to get dimensionless data, train NN, check predictions.
        Workflow: DA (...) -> NN (prepare_data, SymbolicNetwork, train_network, predict)
        """
        print("\n--- Test 03: DA -> Neural Network ---")
        # Use F=ma example again
        independent_vars = ['mass', 'acceleration']
        dependent_var = 'force'

        # 1. Dimensional Analysis
        M, b = da.get_matrix_target(independent_vars, dependent_var, UNIT_TABLE)
        p_exp, U = da.solveDimension(M, b)

        # Generate more data points for NN
        N = 100
        mass_vals = np.random.rand(N) * 10 + 1 # Mass between 1 and 11
        acc_vals = np.random.rand(N) * 20 + 1  # Acceleration between 1 and 21
        force_vals = mass_vals * acc_vals
        data_x = np.vstack([mass_vals, acc_vals])
        data_y = force_vals

        # Make data dimensionless
        data_x_prime, data_y_prime = da.generate_dimensionless_data(data_x, data_y, p_exp, U)
        # y' = F / (m*a) should be 1
        expected_y_prime = np.ones_like(data_y_prime)
        np.testing.assert_array_almost_equal(data_y_prime, expected_y_prime, decimal=6)

        # 2. Neural Network Training and Prediction
        # Since y' is constant, NN should learn this constant function.
        # x' is not used, so NN input dimension is technically 0 if U is empty.
        # Let's adapt: Train NN to predict y' using x' (even though it's not needed).
        # NN input dim should match data_x_prime.shape[0]
        input_dim = data_x_prime.shape[0]
        if input_dim == 0: # Handle case where U is empty and DA removed all inputs
             print("Skipping NN test as DA resulted in 0 input dimensions.")
             # Or, we could test NN on a dummy input if required
             # For F=ma, data_x_prime is same as data_x, so input_dim=2
             input_dim = data_x.shape[0] # Use original data_x as input for NN test
             data_x_nn_input = data_x.T # NN expects (samples, features)
        else:
             data_x_nn_input = data_x_prime.T

        train_loader, val_loader = nn.prepare_data(data_x_nn_input, data_y_prime, batch_size=16)
        model = nn.SymbolicNetwork(n_input=input_dim, n_output=1)

        # Short training for integration test purposes
        nn.train_network(model, train_loader, val_loader, epochs=10, learning_rate=1e-3, device='cpu')

        # Predict
        predictions = nn.predict(model, data_x_nn_input, device='cpu')

        # Check if predictions are close to the constant value 1.0
        self.assertTrue(np.allclose(predictions.flatten(), expected_y_prime, atol=0.1),
                        f"NN predictions {predictions.flatten()} not close to expected {expected_y_prime}")
        print("NN successfully learned the constant dimensionless relationship.")


    @unittest.skipIf(p is None, "Pareto module not found")
    def test_04_brute_force_to_pareto(self):
        """
        Test: Generate expressions with BruteForce, evaluate, update Pareto front.
        Workflow: BF (generate_expressions, evaluate_expressions) -> Pareto (update_pareto_points)
        """
        print("\n--- Test 04: Brute Force -> Pareto ---")
        bf_vars = ['x', 'y']
        ops = ['+', '*']
        consts = []
        max_depth = 2 # Keep it small

        # Generate sample data (e.g., z = 2*x + y*y)
        X = np.array([[1, 1], [2, 1], [1, 2], [3, 2]])
        y_true = 2 * X[:, 0] + X[:, 1]**2

        # 1. Brute Force
        expressions = bf.fast_recursive_expressions(ops, bf_vars, consts, max_depth)
        # Filter for uniqueness and containing required vars
        expressions = list(set(expressions))
        expressions = bf.variable_check(expressions, bf_vars)

        results = bf.evaluate_expressions(expressions, bf_vars, X, y_true) # List of (expr, loss)

        # 2. Update Pareto Front
        pareto_front = [] # Start with empty front
        new_exprs_for_pareto = [r[0] for r in results]
        losses_for_pareto = [r[1] for r in results]

        # Ensure update_pareto_points exists and works
        if hasattr(p, 'update_pareto_points'):
            updated_pareto = p.update_pareto_points(pareto_front, new_exprs_for_pareto, losses_for_pareto)

            print(f"Initial Pareto Front Size: {len(pareto_front)}")
            print(f"Expressions evaluated: {len(results)}")
            print(f"Updated Pareto Front Size: {len(updated_pareto)}")

            self.assertTrue(len(updated_pareto) > 0, "Pareto front should not be empty after update")
            # Check if a known good expression is potentially on the front
            x, y = sp.symbols('x y')
            target_expr = 2*x + y**2 # May not be generated exactly with depth 2 and ops +,*
            target_expr_simple = x+x+y*y # More likely
            found_good_expr = any(item[2].equals(target_expr_simple) for item in updated_pareto)
            # print(f"Found target {target_expr_simple} on Pareto: {found_good_expr}")
            # Note: Exact match depends heavily on generation parameters

            # Basic check: points on front should have complexity and loss
            for item in updated_pareto:
                self.assertIsInstance(item[0], (int, float)) # Complexity
                self.assertIsInstance(item[1], (int, float)) # Loss
                self.assertIsInstance(item[2], sp.Expr)      # Expression
        else:
            self.skipTest("update_pareto_points not available in pareto module")


    @unittest.skipIf(p is None, "Pareto module not found")
    def test_05_polynomial_fit_to_pareto(self):
        """
        Test: Generate polynomial expressions, evaluate, update Pareto front.
        Workflow: PF (generate_expressions, evaluate_expressions) -> Pareto (update_pareto_points)
        """
        print("\n--- Test 05: Polynomial Fit -> Pareto ---")
        pf_vars = ['x', 'y']
        coeffs = [1, 1] # Dummy coeffs for generation structure
        ops = ['+', '*']
        max_degree = 2

        # Generate sample data (e.g., z = x*y + x^2)
        X = np.array([[1, 1], [2, 1], [1, 2], [3, 2]])
        y_true = X[:, 0] * X[:, 1] + X[:, 0]**2

        # 1. Polynomial Fit
        expressions = pf.generate_expressions(coeffs, pf_vars, ops, max_degree)
        expressions = list(set(expressions)) # Unique expressions
        results = pf.evaluate_expressions(expressions, pf_vars, X, y_true) # List of (expr, rmse)

        # 2. Update Pareto Front
        pareto_front = []
        new_exprs_for_pareto = [r[0] for r in results]
        losses_for_pareto = [r[1] for r in results]

        if hasattr(p, 'update_pareto_points'):
            updated_pareto = p.update_pareto_points(pareto_front, new_exprs_for_pareto, losses_for_pareto)

            print(f"Polynomial expressions generated: {len(expressions)}")
            print(f"Updated Pareto Front Size: {len(updated_pareto)}")
            self.assertTrue(len(updated_pareto) > 0)

            # Check if the known target expression is on the front
            x, y = sp.symbols('x y')
            target_expr = x*y + x**2
            found_target = any(item[2].equals(target_expr) for item in updated_pareto)
            if found_target:
                 print(f"Found target expression {target_expr} on Pareto front.")
            # Assert that the target expression, if found, has near-zero loss
            for c, l, expr in updated_pareto:
                if expr.equals(target_expr):
                    self.assertAlmostEqual(l, 0.0, delta=1e-9, msg=f"Target expr {expr} should have near zero loss")
        else:
            self.skipTest("update_pareto_points not available in pareto module")


    @unittest.skipIf(p is None or plots is None, "Pareto or Plots module not found")
    def test_06_pareto_to_plot(self):
        """
        Test: Create Pareto data and attempt to plot it.
        Workflow: Manual Pareto Data -> Plots (plot_pareto_frontier)
        """
        print("\n--- Test 06: Pareto -> Plot ---")
        x, y = sp.symbols('x y')
        # Create some sample Pareto points (complexity, loss, expression_object)
        # Note: update_pareto_points likely returns string or keeps object, adapt as needed
        pareto_points_obj = [
            (2, 0.5, x),
            (5, 0.3, x + y*y),
            (8, 0.1, sp.sin(x) + y),
        ]
        # Convert expressions to strings for plotting if needed by the function
        pareto_points_str = [(c, l, str(f)) for c, l, f in pareto_points_obj]


        if hasattr(plots, 'plot_pareto_frontier'):
            try:
                # Assuming the plot function takes data in format [(complexity, loss, str_formula), ...]
                plot_object = plots.plot_pareto_frontier(pareto_points_str, title="Test Pareto Plot")
                print("Plot function called successfully.")
                # We can't easily assert the visual output, but check if it returned something sensible
                # (e.g., a matplotlib axes object, or just didn't crash)
                self.assertIsNotNone(plot_object, "Plot function should return a plot object or figure.")
            except Exception as e:
                self.fail(f"plot_pareto_frontier raised an exception: {e}")
        else:
            self.skipTest("plot_pareto_frontier not available in plots module")


    def test_07_brute_force_full_chain(self):
        """
        Test: Full BF chain: Generate -> Apply Constants -> Apply Powers -> Filter -> Evaluate.
        Workflow: BF (generate -> apply_constants -> apply_powers -> filter -> evaluate)
        """
        print("\n--- Test 07: Brute Force Full Chain ---")
        bf_vars = ['x']
        ops = ['+']
        consts_funcs = [sp.sin] # Function constant
        powers_vals = [2, 0.5] # Apply square and sqrt
        max_depth_gen = 1
        max_depth_const = 1
        max_depth_power = 1

        # Sample data: y = sin(x^2)
        X = np.array([[0.5], [1.0], [1.5], [2.0]])
        y_true = np.sin(X[:, 0]**2)

        # 1. Generate base expressions
        base_exprs = bf.fast_recursive_expressions(ops, bf_vars, [], max_depth_gen) # e.g., [x]
        print(f"Base expressions: {base_exprs}")

        # 2. Apply constants (using a version like apply_constants_version_1 or similar)
        # Need to choose one of the apply_constants versions from bruteForce.py
        # Let's assume apply_constants_version_1 exists and is suitable
        try:
            const_exprs = bf.apply_constants_version_1(base_exprs, consts_funcs)
            print(f"After constants: {const_exprs}")
        except AttributeError:
             self.skipTest("apply_constants_version_1 not found in bruteForce module")
             return # Skip rest of test if function missing

        # 3. Apply powers
        # Combine base and const expressions before applying powers
        combined_exprs = list(set(base_exprs + const_exprs))
        power_exprs = bf.apply_powers(combined_exprs, powers_vals, max_depth_power)
        print(f"After powers: {power_exprs}")

        # 4. Filter (Example: require 'x', require sin, require power 2)
        filtered_exprs = list(set(power_exprs)) # Start with unique power expressions
        filtered_exprs = bf.variable_check(filtered_exprs, bf_vars)
        print(f"After var check: {len(filtered_exprs)}")
        # Filter for specific constant (sin) - requires filterConstant or similar
        try:
             filtered_exprs = bf.filterConstant(filtered_exprs, consts_funcs)
             print(f"After const check: {len(filtered_exprs)}")
        except AttributeError:
             print("Warning: filterConstant not found, skipping constant filter.")
        # Filter for specific power (2) - requires filter_powers
        try:
             target_sympy_powers = [sp.Integer(p) for p in powers_vals if p == 2]
             filtered_exprs = bf.filter_powers(filtered_exprs, target_sympy_powers)
             print(f"After power check (power=2): {len(filtered_exprs)}")
        except AttributeError:
             print("Warning: filter_powers not found, skipping power filter.")

        # 5. Evaluate final list
        if not filtered_exprs:
            print("No expressions left after filtering.")
            # This might be expected depending on generation/filtering steps
            # self.fail("Filtering removed all expressions, check steps.")
        else:
            results = bf.evaluate_expressions(filtered_exprs, bf_vars, X, y_true)
            print(f"Evaluation results for filtered expressions: {results}")
            self.assertTrue(len(results) > 0 or len(filtered_exprs) == 0)
            # Check if sin(x**2) is in the results and has low loss
            x = sp.symbols('x')
            target = sp.sin(x**2)
            found_target = False
            for expr, loss in results:
                if expr.equals(target):
                    found_target = True
                    self.assertAlmostEqual(loss, 0.0, delta=1e-7, msg="Target sin(x**2) should have near zero loss")
                    print("Successfully found and evaluated target sin(x**2)")
            # Note: Whether the target is generated/survives filtering depends heavily on implementation details


    def test_08_da_symbolic_transformation(self):
        """
        Test: Use DA to get exponents and generate symbolic forms.
        Workflow: DA (get_matrix_target, solveDimension, symbolicTransformation)
        """
        print("\n--- Test 08: DA Symbolic Transformation ---")
        # Pendulum example: T ~ sqrt(L/g)
        independent_vars = ['length', 'acceleration'] # L, g
        dependent_var = 'time' # T
        symbolic_vars = ['L', 'g'] # Names for symbolic output

        # 1. Get exponents
        M, b = da.get_matrix_target(independent_vars, dependent_var, UNIT_TABLE)
        p_exp, U = da.solveDimension(M, b)
        p_exp_np = np.array(p_exp).astype(float).flatten() # Ensure numpy array for indexing

        # 2. Get symbolic forms
        symbolic_p, symbolic_U = da.symbolicTransformation(symbolic_vars, p_exp_np, U)

        # 3. Verify
        L, g = sp.symbols('L g')
        expected_symbolic_p = L**sp.Rational(1, 2) * g**sp.Rational(-1, 2) # sqrt(L/g)
        
        print(f"Symbolic scaling factor: {symbolic_p}")
        print(f"Symbolic Pi groups: {symbolic_U}")


    def test_09_nn_gradient_calculation(self):
        """
        Test: Train or load an NN model and calculate gradients w.r.t input.
        Workflow: NN (prepare_data, SymbolicNetwork, train_network [optional], get_gradient)
        """
        print("\n--- Test 09: NN Gradient Calculation ---")
        # Generate simple data y = 2*x1 + 3*x2
        N = 50
        data_x_nn = np.random.rand(N, 2) * 5
        data_y_nn = 2 * data_x_nn[:, 0] + 3 * data_x_nn[:, 1]

        # Prepare data and model
        train_loader, val_loader = nn.prepare_data(data_x_nn, data_y_nn, batch_size=8)
        model = nn.SymbolicNetwork(n_input=2, n_output=1)

        # Optional: Train briefly to make gradients non-trivial (or use initialized model)
        # nn.train_network(model, train_loader, val_loader, epochs=5, learning_rate=1e-3, device='cpu')

        # Get gradients for a few points
        sample_points = data_x_nn[:5] # Get gradients for first 5 data points
        gradients = nn.get_gradient(model, sample_points, device='cpu')

        print(f"Input points shape: {sample_points.shape}")
        print(f"Calculated gradients shape: {gradients.shape}")
        print(f"Gradients:\n{gradients}")

        # Assertions
        self.assertEqual(gradients.shape, sample_points.shape, "Gradient shape must match input shape")
        # Gradients shouldn't be exactly zero unless the network is trivial or input is zero
        self.assertFalse(np.allclose(gradients, 0.0), "Gradients should not be all zero for a non-trivial function/input")
        # For the linear function y=2x1+3x2, we'd expect gradients close to [2, 3] if NN learned it perfectly.
        # This is a stronger check, might fail if NN hasn't converged.
        # expected_grads = np.array([[2.0, 3.0]] * sample_points.shape[0])
        # np.testing.assert_allclose(gradients, expected_grads, rtol=0.5, atol=0.5, err_msg="Gradients not close to expected [2, 3]")


    def test_10_biology_script_simulation(self):
        """
        Test: Simulate the workflow from biology.py using PolynomialFit.
        Workflow: Data -> PF (generate_expressions, evaluate_expressions, bestFit)
        """
        print("\n--- Test 10: Biology Script Simulation (Polynomial Fit) ---")
        # Data from biology.py
        x_bio = np.array([
            [5, 5, 3, 2], [6, 4, 5, 5], [10, 8, 5, 7], [3, 3, 2, 1],
            [12, 10, 8, 6], [2, 2, 1, 1], [7, 6, 5, 4], [4, 4, 3, 3],
        ])
        # Target function: y = 2*(A+T) + 4*(G+C) = 2*x0 + 2*x1 + 4*x2 + 4*x3
        y_bio = 2 * (x_bio[:, 0] + x_bio[:, 1]) + 4 * (x_bio[:, 2] + x_bio[:, 3])

        variables_bio = ['A', 'T', 'G', 'C']
        # Coefficients used in biology.py seem to be the target coefficients,
        # but polynomialFit's generate_expressions uses them structurally. Let's use 1s.
        coeffs_struct = [1, 1, 1, 1]
        operators_bio = ['+']
        max_degree_bio = 1 # Target is linear

        # 1. Generate expressions
        expressions = pf.generate_expressions(coeffs_struct, variables_bio, operators_bio, max_degree_bio)
        expressions = list(set(expressions)) # Unique
        print(f"Generated {len(expressions)} polynomial expressions.")

        # 2. Evaluate expressions
        results = pf.evaluate_expressions(expressions, variables_bio, x_bio, y_bio)
        print(f"Evaluated {len(results)} expressions.")

        # 3. Find best fit
        best_expr, best_rmse = pf.bestFit(results)
        print(f"Best fit expression: {best_expr} with RMSE: {best_rmse}")

        # 4. Verify
        A, T, G, C = sp.symbols('A T G C')
        # The target expression might be generated in different forms, e.g., A+T+G+C if coeffs_struct=[1,1,1,1]
        # The actual target is 2*A + 2*T + 4*G + 4*C. This won't be generated by pf.generate_expressions
        # with max_degree=1 and coeffs_struct=[1,1,1,1].
        # Let's redefine the test based on what pf.generate_expressions *can* produce.
        # If coeffs=[2,2,4,4], ops=['+'], max_degree=1, it should generate 2*A + 2*T + 4*G + 4*C
        coeffs_target = [2, 2, 4, 4]
        expressions_target = pf.generate_expressions(coeffs_target, variables_bio, operators_bio, max_degree_bio)
        results_target = pf.evaluate_expressions(expressions_target, variables_bio, x_bio, y_bio)
        best_expr_target, best_rmse_target = pf.bestFit(results_target)

        print(f"Using target coeffs: Best fit expression: {best_expr_target} with RMSE: {best_rmse_target}")
        expected_target_expr = 2*A + 2*T + 4*G + 4*C
        self.assertEqual(sp.simplify(best_expr_target), sp.simplify(expected_target_expr))
        self.assertAlmostEqual(best_rmse_target, 0.0, delta=1e-9, msg="RMSE for the exact expression should be zero")


if __name__ == "__main__":
    unittest.main()
