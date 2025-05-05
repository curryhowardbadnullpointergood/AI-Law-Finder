import time
import psutil
import os
import numpy as np
import torch 
import sympy as sp

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
    if not hasattr(da, 'UNIT_TABLE'):
         da.UNIT_TABLE = {
            'mass': [1, 0, 0], 'length': [0, 1, 0], 'time': [0, 0, 1],
            'velocity': [0, 1, -1], 'acceleration': [0, 1, -2],
            'force': [1, 1, -2], 'dimensionless': [0, 0, 0]
         }
except ImportError:
    print("Warning: dimensionalAnalysis module not found or incomplete. Skipping DA tests.")
    DA_AVAILABLE = False

def get_memory_usage_mb():
    """Returns the resident set size (RSS) memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) 


def performance_test_nn_training(sample_sizes=[100, 1000, 5000], epochs=5):
    if not NN_AVAILABLE:
        print("Skipping NN Training Performance Test.")
        return

    print("\n--- Performance Test: Neural Network Training ---")
    print(f"Testing with epochs={epochs}")
    results = {}

    for n_samples in sample_sizes:
        print(f"\nTesting with {n_samples} samples...")

        n_features = 5
        data_x = np.random.rand(n_samples, n_features).astype(np.float32)
        coeffs = np.random.rand(n_features)
        data_y = (data_x @ coeffs).astype(np.float32)

        try:
            train_loader, val_loader = nn.prepare_data(data_x, data_y, batch_size=64)
            model = nn.SymbolicNetwork(n_input=n_features, n_output=1)
        except Exception as e:
            print(f"  Error during NN setup for {n_samples} samples: {e}")
            continue

        mem_before = get_memory_usage_mb()
        process = psutil.Process(os.getpid())
        cpu_before = process.cpu_times()
        start_time = time.perf_counter()

        try:
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

    print("\nExample Interpretation:")
    if len(results) > 1:
        sizes = sorted(results.keys())
        time_ratio = results[sizes[-1]]['time_s'] / results[sizes[0]]['time_s']
        size_ratio = sizes[-1] / sizes[0]
        print(f"  Time scaled by ~{time_ratio:.2f}x for a {size_ratio:.2f}x increase in sample size.")
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
    sys.exit(1) 

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

def performance_test_bf_generation(depths=[2, 3, 4],
                                   variables=['x', 'y'],
                                   operators=['+', '*', '-']):
    if not BF_AVAILABLE:
        print("Skipping Brute Force Generation Performance Test (module not available).")
        return

    print("\n--- Performance Test: Brute Force Expression Generation ---")
    print(f"Testing with variables={variables}, operators={operators}")
    results = {}
    constants = []

    for depth in depths:
        print(f"\nTesting with max_depth={depth}...")

        mem_before = get_memory_usage_mb()
        start_time = time.perf_counter()

        try:
            generated_expressions = bf.fast_recursive_expressions(
                operators=operators,
                variables=variables,
                constants=constants,
                max_depth=depth
            )
            num_expressions = len(generated_expressions)

        except Exception as e:
            print(f"  Error during BF generation for depth={depth}: {e}")
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
            continue 

        end_time = time.perf_counter()
        mem_after = get_memory_usage_mb()

        duration = end_time - start_time
        mem_used = mem_after 
        mem_increase = mem_after - mem_before

        print(f"  Max Depth: {depth}")
        print(f"  Generated Expressions: {num_expressions}")
        print(f"  Generation Time: {duration:.4f} seconds")
        print(f"  Memory Usage (End): {mem_used:.2f} MB")
        print(f"  Memory Increase: {mem_increase:.2f} MB")
        results[depth] = {'time_s': duration, 'mem_mb': mem_used, 'num_expr': num_expressions, 'status': 'success'}

        if duration > 60: 
             print(f"  Stopping generation tests early: Depth {depth} took > 60 seconds.")
             break
        if mem_increase > 1024: 
             print(f"  Stopping generation tests early: Depth {depth} increased memory by > 1024 MB.")
             break


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


def performance_test_bf_evaluation(sample_sizes=[100, 1000, 10000],
                                   n_expressions_to_eval=50,
                                   generation_depth=3):
    if not BF_AVAILABLE:
        print("Skipping Brute Force Evaluation Performance Test (module not available).")
        return

    print("\n--- Performance Test: Brute Force Expression Evaluation ---")
    print(f"Evaluating {n_expressions_to_eval} expressions (generated up to depth {generation_depth})")
    results = {}

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
        expression_pool = list(set(expression_pool))

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

    for n_samples in sample_sizes:
        print(f"\nTesting evaluation with {n_samples} samples...")

        n_features = len(eval_vars)
        try:
            X_data = np.random.rand(n_samples, n_features).astype(np.float64) * 10.0 + 0.1
            y_true = np.random.rand(n_samples).astype(np.float64) 
        except MemoryError:
            print(f"  MemoryError generating data for {n_samples} samples. Skipping.")
            results[n_samples] = {'time_s': -1, 'status': 'data_oom'}
            continue


        start_time = time.perf_counter()
        try:
            evaluation_results = bf.evaluate_expressions(
                expressions=expressions_to_test,
                variables=eval_vars,
                X=X_data,
                y_true=y_true
            )
            status = 'success'
        except MemoryError:
             print(f"  MemoryError during evaluation for {n_samples} samples.")
             status = 'eval_oom'
             evaluation_results = [] 
        except Exception as e:
            print(f"  Error during BF evaluation for {n_samples} samples: {e}")
            status = 'eval_error'
            evaluation_results = [] 

        end_time = time.perf_counter()
        duration = end_time - start_time

        print(f"  Samples: {n_samples}")
        print(f"  Evaluation Time: {duration:.4f} seconds")
        print(f"  Status: {status}")
        if status == 'success':
             print(f"  Evaluated expressions (successfully): {len(evaluation_results)}")
        results[n_samples] = {'time_s': duration, 'status': status}

        if duration > 120:
            print(f"  Stopping evaluation tests early: {n_samples} samples took > 120 seconds.")
            break

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
         else: 
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

        n_features = len(independent_vars)
        data_x = np.random.rand(n_features, n_samples) + 0.1 # Avoid zeros
        data_y = np.prod(data_x, axis=0) # Dummy target y = x1*x2...

        mem_before = get_memory_usage_mb()
        start_time = time.perf_counter()

        try:
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

    print("\nExample Interpretation:")
    if len(results) > 1:
        sizes = sorted(results.keys())
        time_ratio = results[sizes[-1]]['time_s'] / results[sizes[0]]['time_s']
        mem_ratio = results[sizes[-1]]['mem_mb'] / results[sizes[0]]['mem_mb']
        size_ratio = sizes[-1] / sizes[0]
        print(f"  Transformation time scaled by ~{time_ratio:.2f}x for a {size_ratio:.2f}x increase in sample size.")
        print(f"  Memory usage scaled by ~{mem_ratio:.2f}x for a {size_ratio:.2f}x increase in sample size.")
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
    sys.exit(1)

def get_memory_usage_mb():
    """Returns the resident set size (RSS) memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

def performance_test_pf_generation(max_degrees=[1, 2, 3],
                                   num_variables=3,
                                   operators=['+', '*']):
    if not PF_AVAILABLE:
        print("Skipping Polynomial Fit Generation Performance Test (module not available).")
        return

    print("\n--- Performance Test: Polynomial Fit Expression Generation ---")
    variables = [f'x{i}' for i in range(num_variables)]
    coeffs = [1] * num_variables
    print(f"Testing with num_variables={num_variables}, operators={operators}")
    results = {}

    for degree in max_degrees:
        print(f"\nTesting with max_degree={degree}...")

        mem_before = get_memory_usage_mb()
        start_time = time.perf_counter()

        try:
            generated_expressions = pf.generate_expressions(
                coeffs=coeffs,
                variables=variables,
                operators=operators,
                max_degree=degree
            )
            num_expressions = len(generated_expressions)
            status = 'success'

        except MemoryError:
            end_time = time.perf_counter()
            mem_after = get_memory_usage_mb()
            duration = end_time - start_time
            mem_used = mem_after
            mem_increase = mem_after - mem_before
            print(f"  Generation Failed (MemoryError)!")
            print(f"  Time Elapsed before failure: {duration:.4f} seconds")
            print(f"  Memory Usage at failure: {mem_used:.2f} MB")
            results[degree] = {'time_s': duration, 'mem_mb': mem_used, 'num_expr': -1, 'status': 'oom'}
            continue 

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
            continue


        end_time = time.perf_counter()
        mem_after = get_memory_usage_mb()

        duration = end_time - start_time
        mem_used = mem_after
        mem_increase = mem_after - mem_before

        print(f"  Max Degree: {degree}")
        print(f"  Generated Expressions: {num_expressions}")
        print(f"  Generation Time: {duration:.4f} seconds")
        print(f"  Memory Usage (End): {mem_used:.2f} MB")
        print(f"  Memory Increase: {mem_increase:.2f} MB")
        results[degree] = {'time_s': duration, 'mem_mb': mem_used, 'num_expr': num_expressions, 'status': 'success'}

        if duration > 60: 
             print(f"  Stopping generation tests early: Degree {degree} took > 60 seconds.")
             break
        if mem_increase > 1024: 
             print(f"  Stopping generation tests early: Degree {degree} increased memory by > 1024 MB.")
             break

    print("\nGeneration Performance Summary:")
    last_num_expr = 0
    last_time = 0
    for degree in sorted(results.keys()):
        res = results[degree]
        if res['status'] == 'success':
            print(f"  Degree {degree}: {res['num_expr']} expressions, {res['time_s']:.4f}s, {res['mem_mb']:.2f}MB total mem")
            if last_num_expr > 0:
                expr_ratio = (res['num_expr'] / last_num_expr) if last_num_expr else float('inf')
                time_ratio = (res['time_s'] / last_time) if last_time else float('inf')
                print(f"    -> Expr Ratio (vs D{degree-1}): {expr_ratio:.2f}x, Time Ratio: {time_ratio:.2f}x")
            last_num_expr = res['num_expr']
            last_time = res['time_s']
        else:
            print(f"  Degree {degree}: Failed ({res['status']}) after {res['time_s']:.4f}s, {res['mem_mb']:.2f}MB total mem")
            last_num_expr = 0
            last_time = 0

    print("-" * 50)
    return results


def performance_test_pf_evaluation(sample_sizes=[100, 1000, 10000],
                                   num_variables=3,
                                   generation_degree=2,
                                   operators=['+', '*']):
    if not PF_AVAILABLE:
        print("Skipping Polynomial Fit Evaluation Performance Test (module not available).")
        return

    print("\n--- Performance Test: Polynomial Fit Expression Evaluation ---")
    eval_vars = [f'x{i}' for i in range(num_variables)]
    eval_coeffs = [1] * num_variables
    print(f"Generating expression pool (degree={generation_degree}, vars={num_variables})...")
    results = {}

    try:
        expression_pool = pf.generate_expressions(
            coeffs=eval_coeffs,
            variables=eval_vars,
            operators=operators,
            max_degree=generation_degree
        )
        expression_pool = list(set(expression_pool))
        n_expressions_to_eval = len(expression_pool)

        if not expression_pool:
            print("  Error: No expressions generated for evaluation pool. Skipping test.")
            return
        print(f"  Using {n_expressions_to_eval} unique expressions for evaluation test.")

    except Exception as e:
        print(f"  Error generating expression pool: {e}. Skipping evaluation test.")
        return

    for n_samples in sample_sizes:
        print(f"\nTesting evaluation with {n_samples} samples...")

        try:
            X_data = np.random.rand(n_samples, num_variables).astype(np.float64) * 5.0 + 0.1 # Avoid zeros/extremes
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

        start_time = time.perf_counter()
        try:
            evaluation_results = pf.evaluate_expressions(
                expressions=expression_pool,
                variables=eval_vars,
                data=X_data, # Pass data directly as expected by the function
                y_true=y_true
            )
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
             print(f"  Evaluated expressions (successfully): {len(evaluation_results)}/{n_expressions_to_eval}")
        results[n_samples] = {'time_s': duration, 'status': status}

        if duration > 120: 
            print(f"  Stopping evaluation tests early: {n_samples} samples took > 120 seconds.")
            break

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
         else:
             last_time = 0
             last_n_samples = 0

    print("-" * 50)
    return results 




if __name__ == "__main__":
    print("Starting Performance Tests...")

    nn_results = performance_test_nn_training(sample_sizes=[100, 500, 1500], epochs=3)
    da_results = performance_test_da_transformation(sample_sizes=[1000, 10000, 50000])

    if not BF_AVAILABLE:
        print("Cannot run performance tests because bruteForce module failed to import.")
    else:
        print("="*60)
        print(" Starting Brute Force Performance Tests")
        print("="*60)

        generation_results = performance_test_bf_generation(
            depths=[2, 3],
            variables=['x', 'y'],
            operators=['+', '*', '-']
        )

        evaluation_results = performance_test_bf_evaluation(
            sample_sizes=[100, 1000, 5000],
            n_expressions_to_eval=50,     
            generation_depth=3           
        )

        print("\nPerformance Tests Completed.")
        print("="*60)


    if not PF_AVAILABLE:
        print("Cannot run performance tests because polynomialFit module failed to import.")
    else:
        print("="*60)
        print(" Starting Polynomial Fit Performance Tests")
        print("="*60)

        generation_results = performance_test_pf_generation(
            max_degrees=[1, 2],
            num_variables=3,  
            operators=['+', '*'] 
        )

        evaluation_results = performance_test_pf_evaluation(
            sample_sizes=[100, 1000, 5000], 
            num_variables=3,               
            generation_degree=2,          
            operators=['+', '*']
        )

        print("\nPerformance Tests Completed.")
        print("="*60)

    print("\nPerformance Tests Completed.")

