import random
import sys

import six

sys.modules['sklearn.externals.six'] = six
import mlrose


# traveling salesman problem
def traveling_salesman_problem(n):
    coords = []
    for i in range(n):
        coords.append((random.uniform(0, 1), random.uniform(0, 1)))

    fitness = mlrose.TravellingSales(coords=coords)
    problem = mlrose.TSPOpt(length=n, fitness_fn=fitness, maximize=True)
    return problem


def k_color_problem(n=100, k=5):
    edges = []
    # generate edges for a complete graph with n nodes
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    fitness = mlrose.MaxKColor(edges=edges)
    problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=True, max_val=k)
    return problem


def continuous_peaks_problem(t_pct):
    fitness = mlrose.ContinuousPeaks(t_pct=t_pct)
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    return problem
