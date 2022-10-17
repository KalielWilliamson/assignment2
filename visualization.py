import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import six

sys.modules['sklearn.externals.six'] = six


# visualize & save the fitness curve
def visualize_opt_fitness_curve(fitness_curve, optimization_problem, optimization_algorithm):
    plt.plot(fitness_curve)
    plt.title(f"{optimization_problem} - {optimization_algorithm}")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    # remove spaces from optimization problem name
    optimization_problem = optimization_problem.replace(" ", "_")
    # remove spaces from optimization algorithm name
    optimization_algorithm = optimization_algorithm.replace(" ", "_")
    plt.savefig(f"visualizations/{optimization_problem}_{optimization_algorithm}.png")


def visualize_opt_optimization_results(optimization_results: pd.DataFrame, timestamp: str):
    # visualize the fitness curve for each algorithm together
    # for each row in the dataframe
    optimization_problem = optimization_results['problem'].iloc[0]
    for index, row in optimization_results.iterrows():
        # plot the fitness curve
        plt.plot(row['fitness_curve'], label=row['algorithm'])
    plt.title(optimization_problem)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig(f"visualizations/{optimization_problem}_fitness_curve_{timestamp}.png")
    plt.clf()

    # visualize wall time vs best fitness for each algorithm
    # include names of the algorithms
    for index, row in optimization_results.iterrows():
        plt.annotate(row['algorithm'], (row['wall_time'], row['best_fitness']))
    plt.scatter(optimization_results['wall_time'], optimization_results['best_fitness'])
    plt.title(optimization_problem)
    plt.xlabel('Wall Time')
    plt.ylabel('Best Fitness')
    plt.savefig(f"visualizations/{optimization_problem}_wall_time_vs_best_fitness_{timestamp}.png")
    plt.clf()


def visualize_all_opt_results(df: pd.DataFrame):
    problems = df['problem'].unique()
    for problem in problems:
        data = df[df['problem'] == problem]
        sns.violinplot(data=data, x="wall_time", y="best_fitness")
        # set plot title
        plt.savefig(f'visualizations/{problem}_wall_time_vs_best_fit_violin.png')
        plt.clf()

        # plotting correlation heatmap
        sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
        plt.savefig(f'visualizations/{problem}_correlations_heatmap.png')
        plt.clf()


def visualize_all_nn_results(df: pd.DataFrame):
    # get rows where algorithm is random hill climbing or simulated annealing
    df = df[df['algorithm'].isin(['random_hill_climb', 'simulated_annealing'])]
    # create violinplot grouped by algorithm
    sns.violinplot(data=df, x="wall_time", y="test_mse", hue="algorithm")

    sns.violinplot(data=df, x="wall_time", y="test_mse")
    # set plot title
    # plt.savefig(f'visualizations/neural_network_wall_time_vs_test_mse_violin.png')
    plt.show()
    plt.clf()

    # plotting correlation heatmap
    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
    # plt.savefig(f'visualizations/neural_network_correlations_heatmap.png')
    plt.show()
    plt.clf()


def visualize_neural_network_results(df: pd.DataFrame, timestamp: str):
    plt.scatter(df['wall_time'], 1 - df['test_mse'])
    plt.title('Neural Network')
    plt.xlabel('Wall Time')
    plt.ylabel('Inverse Test MSE')
    # plt.savefig(f"visualizations/neural_network_wall_time_vs_inverse_test_mse_{timestamp}.png")
    plt.show()
