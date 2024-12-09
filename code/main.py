import random
import numpy as np
from deap import creator, base, tools, algorithms
import operator
import sympy

# Define symbols
x, g, L = sympy.symbols('x g L')

# Set up DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", sympy.Expr, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Primitive set
pset = sympy.sets.Set([x, g, L, sympy.sin, sympy.cos, operator.add, operator.sub, operator.mul, operator.truediv])

# Corrected generate_expression function with argument
def generate_expression(pset, max_depth, current_depth=0):
    if current_depth >= max_depth:
        return random.choice([x, g, L, random.uniform(-10, 10)])

    if random.random() < 0.5:
        return random.choice([sympy.sin, sympy.cos])(generate_expression(pset, max_depth, current_depth + 1))
    else:
        return random.choice([operator.add, operator.sub, operator.mul, operator.truediv])(
            generate_expression(pset, max_depth, current_depth + 1),
            generate_expression(pset, max_depth, current_depth + 1)
        )

#Register generate_expression correctly with pset argument
toolbox.register("individual", generate_expression, pset=pset, max_depth=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function
def evaluate(individual, theta, alpha):
    try:
        predicted_alpha = np.array([individual.subs({x: t, g: 9.81, L: 1.0}) for t in theta])
        fitness = -np.nanmean(np.abs(predicted_alpha - alpha))
    except (ZeroDivisionError, OverflowError, TypeError, ValueError) as e:
        print(f"Error evaluating expression: {e}, expression: {individual}")
        fitness = -np.inf
    return fitness,

#Pendulum data generation
def generate_pendulum_data(length, g, dt, num_steps, initial_angle=np.pi / 4, initial_velocity=0.0):
    theta = np.zeros(num_steps)
    omega = np.zeros(num_steps)
    alpha = np.zeros(num_steps)
    theta[0] = initial_angle
    omega[0] = initial_velocity
    for i in range(num_steps - 1):
        alpha[i] = -g / length * np.sin(theta[i])
        omega[i + 1] = omega[i] + alpha[i] * dt
        theta[i + 1] = theta[i] + omega[i + 1] * dt
        theta[i + 1] = (theta[i + 1] + np.pi) % (2 * np.pi) - np.pi
    return np.vstack((theta, omega, alpha))

# Example run
length = 1.0
g = 9.81
dt = 0.05
num_steps = 500

data = generate_pendulum_data(length, g, dt, num_steps)
theta = data[0, :]
alpha = data[2, :]

pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)

best_individual = hof[0]
print(f"Best individual: {best_individual}")
print(f"Best fitness: {best_individual.fitness.values[0]}")
print(f"Equation: alpha = {best_individual}")