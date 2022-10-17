import dataclasses
import glob
import sys
from datetime import datetime

import six
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.modules['sklearn.externals.six'] = six

from compute_optimizations import *
from optimization_problems import *
from visualization import *


# Course: CS 7641 - Machine Learning
# Semester: Spring 2022
# Assignment: #2
# Assignment for the course: Machine Learning
# Author: Kaliel L. Williamson
# Date: 10/12/2022


# four peaks solution
def run_opt_iteration(problem, problem_name):
    # get timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # get random hyperparameters in the range of the problem
    max_attempts = random.randint(1, 100)
    max_iters = random.randint(1, 3000)
    restarts = random.randint(1, 100)
    pop_size = random.randint(1, 100)
    mutation_prob = random.uniform(0, 1)
    temperature = random.uniform(0.002, 1)
    keep_pct = random.uniform(0, 1)

    # randomized hill climbing
    rhc_result = randomized_hill_climbing(problem=problem, max_attempts=max_attempts, max_iters=max_iters,
                                          restarts=restarts)
    rhc_df = pd.DataFrame([dataclasses.asdict(rhc_result)])
    rhc_df['max_attempts'] = max_attempts
    rhc_df['max_iters'] = max_iters
    rhc_df['restarts'] = restarts

    # simulated annealing
    schedule = mlrose.ExpDecay(init_temp=temperature)
    sa_result = simulated_annealing(problem=problem, max_attempts=max_attempts, max_iters=max_iters, schedule=schedule)
    sa_df = pd.DataFrame([dataclasses.asdict(sa_result)])
    sa_df['max_attempts'] = max_attempts
    sa_df['max_iters'] = max_iters
    sa_df['temperature'] = temperature

    # genetic algorithm
    # problem, max_attempts, max_iters, pop_size, mutation_prob
    ga_result = genetic_algorithm(problem=problem, max_attempts=max_attempts, max_iters=max_iters, pop_size=pop_size,
                                  mutation_prob=mutation_prob)
    ga_df = pd.DataFrame([dataclasses.asdict(ga_result)])
    ga_df['max_attempts'] = max_attempts
    ga_df['max_iters'] = max_iters
    ga_df['pop_size'] = pop_size
    ga_df['mutation_prob'] = mutation_prob

    # mimic
    mimic_result = mimic(problem=problem, max_attempts=max_attempts, max_iters=max_iters, pop_size=pop_size,
                         keep_pct=keep_pct)
    mimic_df = pd.DataFrame([dataclasses.asdict(mimic_result)])
    mimic_df['max_attempts'] = max_attempts
    mimic_df['max_iters'] = max_iters
    mimic_df['pop_size'] = pop_size
    mimic_df['keep_pct'] = keep_pct

    df = pd.concat([rhc_df, sa_df, ga_df, mimic_df])

    # add hyperparameters to the dataframe
    df['problem'] = problem_name

    # visualize fitness curves
    visualize_opt_optimization_results(optimization_results=df, timestamp=timestamp)

    # save the results to a csv file
    df.to_csv(f"results/opt_results/{problem_name.replace(' ', '_')}_results_{timestamp}.csv", index=False)


def run_all_opt_problems(iterations=300):
    # include loading bar
    print(f"Running {iterations} iterations on the 3 problems...")
    for _ in tqdm(range(iterations)):
        # # continuous peaks
        # continuous_peaks_problem_instance = continuous_peaks_problem(t_pct=0.1)
        # run_opt_iteration(problem=continuous_peaks_problem_instance, problem_name="Continuous Peaks")

        # # queens
        # queens_problem_instance = queens_problem(n=100)
        # run_opt_iteration(problem=queens_problem_instance, problem_name="Queens")
        #
        # traveling salesman
        traveling_salesman_problem_instance = traveling_salesman_problem(n=100)
        run_opt_iteration(problem=traveling_salesman_problem_instance, problem_name="Traveling Salesman")
    print("Done.")

    df = pd.concat([pd.read_csv(f) for f in glob.glob("results/opt_results/*.csv")])
    visualize_all_opt_results(df)


def run_neural_network_iteration(x_train, y_train, x_test, y_test):
    # get random hyperparameters in the range of the problem
    max_attempts = random.randint(1, 100)
    max_iters = 3000
    pop_size = random.randint(1, 200)
    mutation_prob = random.uniform(0, 1)
    restarts = random.randint(1, 100)
    temperature = random.uniform(0.002, 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # run randomized hill climbing
    args = {
        'max_attempts': max_attempts,
        'max_iters': max_iters,
        'restarts': restarts,
    }
    neural_network(x_train, y_train, x_test, y_test, timestamp=timestamp, algorithm='random_hill_climb', args=args)

    # run simulated annealing
    args = {
        'max_attempts': max_attempts,
        'max_iters': max_iters,
        'temperature': temperature
    }
    neural_network(x_train, y_train, x_test, y_test, timestamp=timestamp, algorithm='simulated_annealing', args=args)

    # run genetic algorithm
    args = {
        'max_attempts': max_attempts,
        'pop_size': pop_size,
        'mutation_prob': mutation_prob
    }
    neural_network(x_train, y_train, x_test, y_test, timestamp=timestamp, algorithm='genetic_alg', args=args)

    # run genetic algorithm
    args = {
        'max_attempts': max_attempts,
        'max_iters': max_iters
    }
    neural_network(x_train, y_train, x_test, y_test, timestamp=timestamp, algorithm='gradient_descent', args=args)


def run_all_nn_iterations(iterations=100):
    print(f"Running {iterations} iterations of neural networks...")
    x, y = fetch_openml(data_id='41138', cache=True, return_X_y=True)  # APSFailure dataset
    y = np.where(y == 'neg', 0, 1)

    x.fillna(x.mean(), inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    for _ in tqdm(range(iterations)):
        run_neural_network_iteration(x_train, y_train, x_test, y_test)
    print("Done.")


# main
if __name__ == '__main__':
    run_all_nn_iterations()
    run_all_opt_problems()
