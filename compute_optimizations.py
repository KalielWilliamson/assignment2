import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import six
from sklearn.metrics import mean_squared_error, accuracy_score

sys.modules['sklearn.externals.six'] = six
import mlrose


# data class to hold the results of an optimization algorithm
@dataclass
class OptimizationResult:
    best_state: np.ndarray
    best_fitness: float
    fitness_curve: np.ndarray
    wall_time: float
    algorithm: str


@dataclass
class NeuralNetResult:
    algorithm: str
    wall_time: float
    y_train_error: float
    y_test_error: float
    fitness_curve: np.ndarray


# randomized hill climbing
def randomized_hill_climbing(problem, max_attempts, max_iters, restarts):
    # start time
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, max_attempts=max_attempts,
                                                                       max_iters=max_iters, restarts=restarts,
                                                                       curve=True)
    # calculate wall time
    wall_time = time.time() - start_time

    return OptimizationResult(best_state, best_fitness, fitness_curve, wall_time, "Randomized Hill Climbing")


# simulated annealing
def simulated_annealing(problem, max_attempts, max_iters, schedule):
    # start time
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=schedule,
                                                                         max_attempts=max_attempts,
                                                                         max_iters=max_iters, curve=True)
    # calculate wall time
    wall_time = time.time() - start_time

    return OptimizationResult(best_state, best_fitness, fitness_curve, wall_time, "Simulated Annealing")


# genetic algorithm
def genetic_algorithm(problem, max_attempts, max_iters, pop_size, mutation_prob):
    # start time
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=pop_size,
                                                                 mutation_prob=mutation_prob,
                                                                 max_attempts=max_attempts, max_iters=max_iters,
                                                                 curve=True)
    # calculate wall time
    wall_time = time.time() - start_time
    # to dataframe
    return OptimizationResult(best_state, best_fitness, fitness_curve, wall_time, "Genetic Algorithm")


# mimic optimization algorithm
def mimic(problem, max_attempts, max_iters, pop_size, keep_pct):
    # start time
    start_time = time.time()
    best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct,
                                                           max_attempts=max_attempts, max_iters=max_iters,
                                                           curve=True)
    # calculate wall time
    wall_time = time.time() - start_time

    return OptimizationResult(best_state, best_fitness, fitness_curve, wall_time, "MIMIC")


# neural network randomized optimization
def neural_network(x_train, y_train, x_test, y_test,
                   timestamp,
                   algorithm,
                   args) :
    hyperparameters = args.copy()
    schedule = mlrose.GeomDecay()
    # if temperature is specified in args dict, use it
    if 'temperature' in args:
        schedule = mlrose.ExpDecay(init_temp=args['temperature'])
        # remove temperature from args dict
        del args['temperature']

    # get start timestamp
    start = time.time()
    nn_model = mlrose.NeuralNetwork(hidden_nodes=[16, 32, 32, 2],
                                    activation='relu',
                                    is_classifier=True,
                                    curve=True,
                                    schedule=schedule,
                                    bias=True,
                                    **args)

    nn_model = nn_model.fit(x_train, y_train)
    fitness_curve = nn_model.fitness_curve
    wall_time = time.time() - start

    # predict labels for train set and assess accuracy
    y_train_error = accuracy_score(y_train, nn_model.predict(x_train))
    y_test_error = accuracy_score(y_test, nn_model.predict(x_test))

    results = {
        'algorithm': [algorithm],
        'wall_time': [wall_time],
        'y_train_accuracy': [y_train_error],
        'y_test_accuracy': [y_test_error],
        'fitness_curve': [fitness_curve.tolist()]
    }
    results.update(hyperparameters)
    df = pd.DataFrame(results, index=[0])
    df.to_csv(f"results/nn_results/x_nn_results_{timestamp}_{algorithm}.csv", index=False, header=True)
