import time
import psutil
import os
import numpy as np
import torch # Assuming torch is used by neuralNetwork
import sympy as sp

# Import necessary functions from your modules
# Add error handling in case some modules are not fully implemented yet
try:
    import neuralNetwork as nn
    NN_AVAILABLE = True
except ImportError:
    print("Warning: neuralNetwork module not found or incomplete. Skipping NN tests.")
    NN_AVAILABLE = False

try:
    import bruteForce as bf
    BF_AVAILABLE = True
except ImportError:
    print("Warning: bruteForce module not found or incomplete. Skipping BF tests.")
    BF_AVAILABLE = False

try:
    import dimensionalAnalysis as da
    DA_AVAILABLE = True
    # Define a dummy UNIT_TABLE if needed by DA functions and not globally accessible
    if not hasattr(da, 'UNIT_TABLE'):
         da.UNIT_TABLE = {
            'mass': [1, 0, 0], 'length': [0, 1, 0], 'time': [0, 0, 1],
            'velocity': [0, 1, -1], 'acceleration': [0, 1, -2],
            'force': [1, 1, -2], 'dimensionless': [0, 0, 0]
         }
except ImportError:
    print("Warning: dimensionalAnalysis module not found or incomplete. Skipping DA tests.")
    DA_AVAILABLE = False

# --- Helper Function to Get Memory Usage ---
def get_memory_usage_mb():
    """Returns the resident set size (RSS) memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # Convert bytes to MB

# --- Test Functions ---

def performance_test_nn_training(sample_sizes=[100, 1000, 5000], epochs=5):
    """
    Measures training time and memory usage for the Neural Network
    with varying sample sizes.
    """
    if not NN_AVAILABLE:
        print("Skipping NN Training Performance Test.")
        return

    print("\n--- Performance Test: Neural Network Training ---")
    print(f"Testing with epochs={epochs}")
    results = {}

    for n_samples in sample_sizes:
        print(f"\nTesting with {n_samples} samples...")

        # 1. Generate synthetic data
        n_features = 5
        data_x = np.random.rand(n_samples, n_features).astype(np.float32)
        # Simple linear target for testing
        coeffs = np.random.rand(n_features)
        data_y = (data_x @ coeffs).astype(np.float32)

        # 2. Prepare data loaders and model
        try:
            train_loader, val_loader = nn.prepare_data(data_x, data_y, batch_size=64)
            model = nn.SymbolicNetwork(n_input=n_features, n_output=1)
        except Exception as e:
            print(f"  Error during NN setup for {n_samples} samples: {e}")
            continue

        # 3. Measure Time and Memory
        mem_before = get_memory_usage_mb()
        process = psutil.Process(os.getpid())
        cpu_before = process.cpu_times()
        start_time = time.perf_counter()

        try:
            # Run training (use 'cpu' for consistent testing unless GPU is specifically tested)
            nn.train_network(model, train_loader, val_loader, epochs=epochs, learning_rate=1e-3, device='cpu')
        except Exception as e:
            print(f"  Error during NN training for {n_samples} samples: {e}")
            continue

        end_time = time.perf_counter()
        cpu_after = process.cpu_times()
        mem_after = get_memory_usage_mb()

        duration = end_time - start_time
        cpu_used = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
        mem_used = mem_after # Report total memory after, as peak is harder to track precisely
        mem_increase = mem_after - mem_before

        print(f"  Samples: {n_samples}")
        print(f"  Training Time: {duration:.4f} seconds")
        print(f"  CPU Time Used: {cpu_used:.4f} seconds")
        print(f"  Memory Usage (End): {mem_used:.2f} MB")
        print(f"  Memory Increase: {mem_increase:.2f} MB")
        results[n_samples] = {'time_s': duration, 'cpu_s': cpu_used, 'mem_mb': mem_used}

    # 4. Analyze results (Example interpretation)
    print("\nExample Interpretation:")
    if len(results) > 1:
        sizes = sorted(results.keys())
        time_ratio = results[sizes[-1]]['time_s'] / results[sizes[0]]['time_s']
        size_ratio = sizes[-1] / sizes[0]
        print(f"  Time scaled by ~{time_ratio:.2f}x for a {size_ratio:.2f}x increase in sample size.")
        # Add more sophisticated analysis like fitting a line if needed
    print("-" * 40)
    return results


try:
    import bruteForce as bf
    BF_AVAILABLE = True
    print("Successfully imported bruteForce module.")
except ImportError as e:
    print(f"Error: Failed to import bruteForce module: {e}")
    print("Please ensure bruteForce.py is in the Python path.")
    BF_AVAILABLE = False
    sys.exit(1) # Exit if the core module can't be imported

# --- Helper Function ---
def get_memory_usage_mb():
    """Returns the resident set size (RSS) memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # Convert bytes to MB

# --- Test Function: Expression Generation ---
def performance_test_bf_generation(depths=[2, 3, 4],
                                   variables=['x', 'y'],
                                   operators=['+', '*', '-']):
    """
    Measures time and memory usage for brute force expression generation
    at different maximum depths.
    """
    if not BF_AVAILABLE:
        print("Skipping Brute Force Generation Performance Test (module not available).")
        return

    print("\n--- Performance Test: Brute Force Expression Generation ---")
    print(f"Testing with variables={variables}, operators={operators}")
    results = {}
    constants = [] # Keep constants empty for focus on structural complexity

    for depth in depths:
        print(f"\nTesting with max_depth={depth}...")

        mem_before = get_memory_usage_mb()
        start_time = time.perf_counter()

        try:
            # --- Call the generation function ---
            # NOTE: fast_recursive_expressions uses lru_cache internally.
            # Each call creates a new inner function scope, so cache is fresh per depth.
            generated_expressions = bf.fast_recursive_expressions(
                operators=operators,
                variables=variables,
                constants=constants,
                max_depth=depth
            )
            num_expressions = len(generated_expressions)
            # --- End of measured section ---

        except Exception as e:
            print(f"  Error during BF generation for depth={depth}: {e}")
            # If generation fails (e.g., runs out of memory), record failure
            end_time = time.perf_counter()
            mem_after = get_memory_usage_mb()
            duration = end_time - start_time
            mem_used = mem_after
            mem_increase = mem_after - mem_before
            print(f"  Generation Failed!")
            print(f"  Time Elapsed before failure: {duration:.4f} seconds")
            print(f"  Memory Usage at failure: {mem_used:.2f} MB")
            print(f"  Memory Increase before failure: {mem_increase:.2f} MB")
            results[depth] = {'time_s': duration, 'mem_mb': mem_used, 'num_expr': -1, 'status': 'failed'}
            continue # Skip to next depth

        end_time = time.perf_counter()
        mem_after = get_memory_usage_mb()

        duration = end_time - start_time
        mem_used = mem_after # Total memory after generation
        mem_increase = mem_after - mem_before

        print(f"  Max Depth: {depth}")
        print(f"  Generated Expressions: {num_expressions}")
        print(f"  Generation Time: {duration:.4f} seconds")
        print(f"  Memory Usage (End): {mem_used:.2f} MB")
        print(f"  Memory Increase: {mem_increase:.2f} MB")
        results[depth] = {'time_s': duration, 'mem_mb': mem_used, 'num_expr': num_expressions, 'status': 'success'}

        # Optional: Add a check for excessive time/memory and break early
        if duration > 60: # Example: Stop if a depth takes over a minute
             print(f"  Stopping generation tests early: Depth {depth} took > 60 seconds.")
             break
        if mem_increase > 1024: # Example: Stop if memory increase exceeds 1GB
             print(f"  Stopping generation tests early: Depth {depth} increased memory by > 1024 MB.")
             break


    # --- Analysis ---
    print("\nGeneration Performance Summary:")
    last_num_expr = 0
    last_time = 0
    for depth in sorted(results.keys()):
        res = results[depth]
        if res['status'] == 'success':
            print(f"  Depth {depth}: {res['num_expr']} expressions, {res['time_s']:.4f}s, {res['mem_mb']:.2f}MB total mem")
            if last_num_expr > 0:
                expr_ratio = res['num_expr'] / last_num_expr if last_num_expr else float('inf')
                time_ratio = res['time_s'] / last_time if last_time else float('inf')
                print(f"    -> Expr Ratio (vs D{depth-1}): {expr_ratio:.2f}x, Time Ratio: {time_ratio:.2f}x")
            last_num_expr = res['num_expr']
            last_time = res['time_s']
        else:
            print(f"  Depth {depth}: Failed after {res['time_s']:.4f}s, {res['mem_mb']:.2f}MB total mem")

    print("-" * 50)
    return results


# --- Test Function: Expression Evaluation ---
def performance_test_bf_evaluation(sample_sizes=[100, 1000, 10000],
                                   n_expressions_to_eval=50,
                                   generation_depth=3):
    """
    Measures evaluation time for a fixed number of Brute Force expressions
    with varying sample sizes.
    """
    if not BF_AVAILABLE:
        print("Skipping Brute Force Evaluation Performance Test (module not available).")
        return

    print("\n--- Performance Test: Brute Force Expression Evaluation ---")
    print(f"Evaluating {n_expressions_to_eval} expressions (generated up to depth {generation_depth})")
    results = {}

    # --- 1. Generate a pool of expressions first ---
    print("Generating expression pool...")
    eval_vars = ['x', 'y', 'z']
    eval_ops = ['+', '*', '-']
    try:
        expression_pool = bf.fast_recursive_expressions(
            operators=eval_ops,
            variables=eval_vars,
            constants=[],
            max_depth=generation_depth
        )
        expression_pool = list(set(expression_pool)) # Unique
        # Optional: Filter for expressions containing all variables
        # expression_pool = bf.variable_check(expression_pool, eval_vars)

        if len(expression_pool) < n_expressions_to_eval:
            print(f"  Warning: Generated only {len(expression_pool)} unique expressions.")
            expressions_to_test = expression_pool
            actual_n_expr = len(expression_pool)
            if not expressions_to_test:
                 print("  Error: No expressions generated for evaluation pool. Skipping test.")
                 return
        else:
            expressions_to_test = expression_pool[:n_expressions_to_eval]
            actual_n_expr = n_expressions_to_eval
        print(f"  Using {actual_n_expr} expressions for evaluation test.")

    except Exception as e:
        print(f"  Error generating expression pool: {e}. Skipping evaluation test.")
        return

    # --- 2. Test evaluation with varying sample sizes ---
    for n_samples in sample_sizes:
        print(f"\nTesting evaluation with {n_samples} samples...")

        # Generate synthetic data
        n_features = len(eval_vars)
        try:
            # Use float64 for potentially better numerical stability in sympy/numpy
            X_data = np.random.rand(n_samples, n_features).astype(np.float64) * 10.0 + 0.1 # Avoid zeros
            y_true = np.random.rand(n_samples).astype(np.float64) # Dummy target
        except MemoryError:
            print(f"  MemoryError generating data for {n_samples} samples. Skipping.")
            results[n_samples] = {'time_s': -1, 'status': 'data_oom'}
            continue


        # Measure Time for evaluation
        start_time = time.perf_counter()
        try:
            # --- Call the evaluation function ---
            evaluation_results = bf.evaluate_expressions(
                expressions=expressions_to_test,
                variables=eval_vars,
                X=X_data,
                y_true=y_true
            )
            # --- End of measured section ---
            status = 'success'
        except MemoryError:
             print(f"  MemoryError during evaluation for {n_samples} samples.")
             status = 'eval_oom'
             evaluation_results = [] # No results if OOM
        except Exception as e:
            print(f"  Error during BF evaluation for {n_samples} samples: {e}")
            status = 'eval_error'
            evaluation_results = [] # No results if error

        end_time = time.perf_counter()
        duration = end_time - start_time

        print(f"  Samples: {n_samples}")
        print(f"  Evaluation Time: {duration:.4f} seconds")
        print(f"  Status: {status}")
        if status == 'success':
             print(f"  Evaluated expressions (successfully): {len(evaluation_results)}")
        results[n_samples] = {'time_s': duration, 'status': status}

        # Optional: Stop early if evaluation takes too long
        if duration > 120: # Stop if evaluation takes > 2 minutes
            print(f"  Stopping evaluation tests early: {n_samples} samples took > 120 seconds.")
            break

    # --- Analysis ---
    print("\nEvaluation Performance Summary:")
    last_time = 0
    last_n_samples = 0
    for n_samples in sorted(results.keys()):
         res = results[n_samples]
         print(f"  Samples {n_samples}: Time {res['time_s']:.4f}s, Status: {res['status']}")
         if res['status'] == 'success' and last_time > 0:
             time_ratio = res['time_s'] / last_time if last_time else float('inf')
             size_ratio = n_samples / last_n_samples if last_n_samples else float('inf')
             print(f"    -> Time Ratio (vs N={last_n_samples}): {time_ratio:.2f}x for Size Ratio: {size_ratio:.2f}x")
         if res['status'] == 'success':
             last_time = res['time_s']
             last_n_samples = n_samples
         else: # Reset comparison point if a test failed
             last_time = 0
             last_n_samples = 0


    print("-" * 50)
    return results



def performance_test_da_transformation(sample_sizes=[1000, 10000, 100000]):
    """
    Measures time and memory for Dimensional Analysis data transformation.
    """
    if not DA_AVAILABLE:
        print("Skipping Dimensional Analysis Transformation Performance Test.")
        return

    print("\n--- Performance Test: Dimensional Analysis Transformation ---")
    results = {}

    # Setup for DA (e.g., F = m*a)
    independent_vars = ['mass', 'acceleration']
    dependent_var = 'force'
    try:
        M, b = da.get_matrix_target(independent_vars, dependent_var, da.UNIT_TABLE)
        p_exp, U = da.solveDimension(M, b)
        p_exp_np = np.array(p_exp).astype(float).flatten()
    except Exception as e:
        print(f"  Error during DA setup: {e}")
        return

    for n_samples in sample_sizes:
        print(f"\nTesting with {n_samples} samples...")

        # 1. Generate synthetic physical data
        n_features = len(independent_vars)
        data_x = np.random.rand(n_features, n_samples) + 0.1 # Avoid zeros
        data_y = np.prod(data_x, axis=0) # Dummy target y = x1*x2...

        # 2. Measure Time and Memory
        mem_before = get_memory_usage_mb()
        start_time = time.perf_counter()

        try:
            # Perform the transformation
            _, _ = da.generate_dimensionless_data(data_x, data_y, p_exp_np, U)
        except Exception as e:
            print(f"  Error during DA transformation for {n_samples} samples: {e}")
            continue

        end_time = time.perf_counter()
        mem_after = get_memory_usage_mb()

        duration = end_time - start_time
        mem_used = mem_after
        mem_increase = mem_after - mem_before

        print(f"  Samples: {n_samples}")
        print(f"  Transformation Time: {duration:.4f} seconds")
        print(f"  Memory Usage (End): {mem_used:.2f} MB")
        print(f"  Memory Increase: {mem_increase:.2f} MB")
        results[n_samples] = {'time_s': duration, 'mem_mb': mem_used}

    # 3. Analyze results
    print("\nExample Interpretation:")
    if len(results) > 1:
        sizes = sorted(results.keys())
        time_ratio = results[sizes[-1]]['time_s'] / results[sizes[0]]['time_s']
        mem_ratio = results[sizes[-1]]['mem_mb'] / results[sizes[0]]['mem_mb']
        size_ratio = sizes[-1] / sizes[0]
        print(f"  Transformation time scaled by ~{time_ratio:.2f}x for a {size_ratio:.2f}x increase in sample size.")
        print(f"  Memory usage scaled by ~{mem_ratio:.2f}x for a {size_ratio:.2f}x increase in sample size.")
        # Expect near linear scaling for both time and memory with N
    print("-" * 40)
    return results



try:
    import polynomialFit as pf
    PF_AVAILABLE = True
    print("Successfully imported polynomialFit module.")
except ImportError as e:
    print(f"Error: Failed to import polynomialFit module: {e}")
    print("Please ensure polynomialFit.py is in the Python path.")
    PF_AVAILABLE = False
    sys.exit(1) # Exit if the core module can't be imported

# --- Helper Function ---
def get_memory_usage_mb():
    """Returns the resident set size (RSS) memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # Convert bytes to MB

# --- Test Function: Expression Generation ---
def performance_test_pf_generation(max_degrees=[1, 2, 3],
                                   num_variables=3,
                                   operators=['+', '*']):
    """
    Measures time and memory usage for polynomial expression generation
    with varying maximum degrees.
    """
    if not PF_AVAILABLE:
        print("Skipping Polynomial Fit Generation Performance Test (module not available).")
        return

    print("\n--- Performance Test: Polynomial Fit Expression Generation ---")
    variables = [f'x{i}' for i in range(num_variables)]
    # Use dummy coefficients (e.g., all 1s) for structure
    coeffs = [1] * num_variables
    print(f"Testing with num_variables={num_variables}, operators={operators}")
    results = {}

    for degree in max_degrees:
        print(f"\nTesting with max_degree={degree}...")

        mem_before = get_memory_usage_mb()
        start_time = time.perf_counter()

        try:
            # --- Call the generation function ---
            generated_expressions = pf.generate_expressions(
                coeffs=coeffs,
                variables=variables,
                operators=operators,
                max_degree=degree
            )
            num_expressions = len(generated_expressions)
            # --- End of measured section ---
            status = 'success'

        except MemoryError:
            # Handle potential out-of-memory errors for high degrees/vars
            end_time = time.perf_counter()
            mem_after = get_memory_usage_mb()
            duration = end_time - start_time
            mem_used = mem_after
            mem_increase = mem_after - mem_before
            print(f"  Generation Failed (MemoryError)!")
            print(f"  Time Elapsed before failure: {duration:.4f} seconds")
            print(f"  Memory Usage at failure: {mem_used:.2f} MB")
            results[degree] = {'time_s': duration, 'mem_mb': mem_used, 'num_expr': -1, 'status': 'oom'}
            continue # Skip to next degree

        except Exception as e:
            end_time = time.perf_counter()
            mem_after = get_memory_usage_mb()
            duration = end_time - start_time
            mem_used = mem_after
            mem_increase = mem_after - mem_before
            print(f"  Generation Failed (Error: {e})!")
            print(f"  Time Elapsed before failure: {duration:.4f} seconds")
            print(f"  Memory Usage at failure: {mem_used:.2f} MB")
            results[degree] = {'time_s': duration, 'mem_mb': mem_used, 'num_expr': -1, 'status': 'error'}
            continue # Skip to next degree


        end_time = time.perf_counter()
        mem_after = get_memory_usage_mb()

        duration = end_time - start_time
        mem_used = mem_after # Total memory after generation
        mem_increase = mem_after - mem_before

        print(f"  Max Degree: {degree}")
        print(f"  Generated Expressions: {num_expressions}")
        print(f"  Generation Time: {duration:.4f} seconds")
        print(f"  Memory Usage (End): {mem_used:.2f} MB")
        print(f"  Memory Increase: {mem_increase:.2f} MB")
        results[degree] = {'time_s': duration, 'mem_mb': mem_used, 'num_expr': num_expressions, 'status': 'success'}

        # Optional: Add a check for excessive time/memory and break early
        # Polynomial generation complexity grows very rapidly with degree and num_vars
        if duration > 60: # Stop if a degree takes over a minute
             print(f"  Stopping generation tests early: Degree {degree} took > 60 seconds.")
             break
        if mem_increase > 1024: # Stop if memory increase exceeds 1GB
             print(f"  Stopping generation tests early: Degree {degree} increased memory by > 1024 MB.")
             break

    # --- Analysis ---
    print("\nGeneration Performance Summary:")
    last_num_expr = 0
    last_time = 0
    for degree in sorted(results.keys()):
        res = results[degree]
        if res['status'] == 'success':
            print(f"  Degree {degree}: {res['num_expr']} expressions, {res['time_s']:.4f}s, {res['mem_mb']:.2f}MB total mem")
            if last_num_expr > 0:
                # Avoid division by zero if previous step failed or generated 0 expressions
                expr_ratio = (res['num_expr'] / last_num_expr) if last_num_expr else float('inf')
                time_ratio = (res['time_s'] / last_time) if last_time else float('inf')
                print(f"    -> Expr Ratio (vs D{degree-1}): {expr_ratio:.2f}x, Time Ratio: {time_ratio:.2f}x")
            last_num_expr = res['num_expr']
            last_time = res['time_s']
        else:
            print(f"  Degree {degree}: Failed ({res['status']}) after {res['time_s']:.4f}s, {res['mem_mb']:.2f}MB total mem")
            # Reset comparison point if a test failed
            last_num_expr = 0
            last_time = 0

    print("-" * 50)
    return results


# --- Test Function: Expression Evaluation ---
def performance_test_pf_evaluation(sample_sizes=[100, 1000, 10000],
                                   num_variables=3,
                                   generation_degree=2,
                                   operators=['+', '*']):
    """
    Measures evaluation time for a fixed pool of Polynomial Fit expressions
    with varying sample sizes.
    """
    if not PF_AVAILABLE:
        print("Skipping Polynomial Fit Evaluation Performance Test (module not available).")
        return

    print("\n--- Performance Test: Polynomial Fit Expression Evaluation ---")
    eval_vars = [f'x{i}' for i in range(num_variables)]
    eval_coeffs = [1] * num_variables
    print(f"Generating expression pool (degree={generation_degree}, vars={num_variables})...")
    results = {}

    # --- 1. Generate a fixed pool of expressions first ---
    try:
        expression_pool = pf.generate_expressions(
            coeffs=eval_coeffs,
            variables=eval_vars,
            operators=operators,
            max_degree=generation_degree
        )
        # Make unique, although generate_expressions might already do simplification
        expression_pool = list(set(expression_pool))
        n_expressions_to_eval = len(expression_pool)

        if not expression_pool:
            print("  Error: No expressions generated for evaluation pool. Skipping test.")
            return
        print(f"  Using {n_expressions_to_eval} unique expressions for evaluation test.")

    except Exception as e:
        print(f"  Error generating expression pool: {e}. Skipping evaluation test.")
        return

    # --- 2. Test evaluation with varying sample sizes ---
    for n_samples in sample_sizes:
        print(f"\nTesting evaluation with {n_samples} samples...")

        # Generate synthetic data
        try:
            # Use float64 for potentially better numerical stability
            X_data = np.random.rand(n_samples, num_variables).astype(np.float64) * 5.0 + 0.1 # Avoid zeros/extremes
            # Generate a dummy target based on a simple polynomial
            target_coeffs = np.random.rand(num_variables)
            y_true = X_data @ target_coeffs + np.sum(X_data**2, axis=1) # Example: linear + quadratic term
            y_true = y_true.astype(np.float64)
        except MemoryError:
            print(f"  MemoryError generating data for {n_samples} samples. Skipping.")
            results[n_samples] = {'time_s': -1, 'status': 'data_oom'}
            continue
        except Exception as e:
             print(f"  Error generating data for {n_samples} samples: {e}. Skipping.")
             results[n_samples] = {'time_s': -1, 'status': 'data_error'}
             continue

        # Measure Time for evaluation
        start_time = time.perf_counter()
        try:
            # --- Call the evaluation function ---
            # Note: pf.evaluate_expressions uses lambdify internally
            evaluation_results = pf.evaluate_expressions(
                expressions=expression_pool,
                variables=eval_vars,
                data=X_data, # Pass data directly as expected by the function
                y_true=y_true
            )
            # --- End of measured section ---
            status = 'success'
        except MemoryError:
             print(f"  MemoryError during evaluation for {n_samples} samples.")
             status = 'eval_oom'
             evaluation_results = [] # No results if OOM
        except Exception as e:
            print(f"  Error during PF evaluation for {n_samples} samples: {e}")
            status = 'eval_error'
            evaluation_results = [] # No results if error

        end_time = time.perf_counter()
        duration = end_time - start_time

        print(f"  Samples: {n_samples}")
        print(f"  Evaluation Time: {duration:.4f} seconds")
        print(f"  Status: {status}")
        if status == 'success':
             # Check how many expressions evaluated without error (lambdify can fail)
             print(f"  Evaluated expressions (successfully): {len(evaluation_results)}/{n_expressions_to_eval}")
        results[n_samples] = {'time_s': duration, 'status': status}

        # Optional: Stop early if evaluation takes too long
        if duration > 120: # Stop if evaluation takes > 2 minutes
            print(f"  Stopping evaluation tests early: {n_samples} samples took > 120 seconds.")
            break

    # --- Analysis ---
    print("\nEvaluation Performance Summary:")
    last_time = 0
    last_n_samples = 0
    for n_samples in sorted(results.keys()):
         res = results[n_samples]
         print(f"  Samples {n_samples}: Time {res['time_s']:.4f}s, Status: {res['status']}")
         if res['status'] == 'success' and last_time > 0:
             time_ratio = res['time_s'] / last_time if last_time else float('inf')
             size_ratio = n_samples / last_n_samples if last_n_samples else float('inf')
             print(f"    -> Time Ratio (vs N={last_n_samples}): {time_ratio:.2f}x for Size Ratio: {size_ratio:.2f}x")
         if res['status'] == 'success':
             last_time = res['time_s']
             last_n_samples = n_samples
         else: # Reset comparison point if a test failed
             last_time = 0
             last_n_samples = 0

    print("-" * 50)
    return results 




# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Performance Tests...")

    # Run the tests
    nn_results = performance_test_nn_training(sample_sizes=[100, 500, 1500], epochs=3)
    # bf_results = performance_test_bf_evaluation(sample_sizes=[100, 1000, 5000], n_expressions=15, depth=2)
    da_results = performance_test_da_transformation(sample_sizes=[1000, 10000, 50000])

    if not BF_AVAILABLE:
        print("Cannot run performance tests because bruteForce module failed to import.")
    else:
        print("="*60)
        print(" Starting Brute Force Performance Tests")
        print("="*60)

        # --- Run Generation Test ---
        # Be cautious with depths > 4, they can take a very long time/memory
        generation_results = performance_test_bf_generation(
            depths=[2, 3], # Start with smaller depths
            variables=['x', 'y'],
            operators=['+', '*', '-']
        )

        # --- Run Evaluation Test ---
        evaluation_results = performance_test_bf_evaluation(
            sample_sizes=[100, 1000, 5000], # Test scaling with data size
            n_expressions_to_eval=50,      # Number of formulas to check
            generation_depth=3             # Complexity of formulas to check
        )

        print("\nPerformance Tests Completed.")
        print("="*60)


    if not PF_AVAILABLE:
        print("Cannot run performance tests because polynomialFit module failed to import.")
    else:
        print("="*60)
        print(" Starting Polynomial Fit Performance Tests")
        print("="*60)

        # --- Run Generation Test ---
        # Be cautious with max_degree > 3 or num_variables > 4
        generation_results = performance_test_pf_generation(
            max_degrees=[1, 2], # Keep degrees low initially
            num_variables=3,    # Number of input variables (e.g., x, y, z)
            operators=['+', '*'] # Common operators
        )

        # --- Run Evaluation Test ---
        evaluation_results = performance_test_pf_evaluation(
            sample_sizes=[100, 1000, 5000], # Test scaling with data size
            num_variables=3,               # Must match generation pool if reused
            generation_degree=2,           # Complexity of formulas to evaluate
            operators=['+', '*']
        )

        print("\nPerformance Tests Completed.")
        print("="*60)

    print("\nPerformance Tests Completed.")

