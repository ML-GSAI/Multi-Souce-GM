import os
import datetime
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data import ConditionalGaussianMixtureDistribution
from util import seed_torch, calculate_empirical_TV


def generate_shared_means(K, d_dif, d_sim):
    means_dif = (torch.arange(1, K + 1).unsqueeze(1)).repeat(1, d_dif)
    means_sim = torch.zeros((1, d_sim))
    shared_means = torch.cat([means_dif, means_sim.repeat(K, 1)], dim=1)
    # print("True means:\n", shared_means)
    return shared_means


class GaussianMixtureModel(nn.Module):
    def __init__(self, K, d_dif, d_sim, p_y):
        """
        Initializes the Gaussian Mixture Model.

        Parameters:
            K (int): Number of Gaussian components.
            d_dif (int): Dimensionality of source-specific features.
            d_sim (int): Dimensionality of shared features.
            p_y (torch.Tensor): Probability distribution over classes.
        """
        super().__init__()
        self.K = K
        self.d_dif = d_dif
        self.d_sim = d_sim
        self.p_y = p_y
        self.means = nn.Parameter(torch.zeros(K, d_dif + d_sim))  # Initialize means to zero

    def estimate_means(self, x_train, y_train, method="single"):
        """
        Estimates the means of the Gaussian components using training data.

        Parameters:
            x_train (torch.Tensor): Training data of shape (n, d).
            y_train (torch.Tensor): Training labels of shape (n,).
            method (str): "single" for class-specific means, "multi" for shared means.
        """
        with torch.no_grad():
            # Compute the global mean if the method is "multi" and similar-dim > 0
            global_mean = x_train.mean(dim=0) if method == "multi" and self.d_sim > 0 else None
            for k in range(self.K):
                class_mean = x_train[y_train == k].mean(dim=0)
                if method == "multi" and self.d_sim > 0 and global_mean is not None:
                    # Replace the last d_sim dimensions with the global mean
                    class_mean[-self.d_sim:] = global_mean[-self.d_sim:]
                self.means[k] = class_mean

    def forward(self):
        return self.means


# Function to simulate and evaluate the GMM
def gaussian_simulation(seed, K, D, sim, n):
    """
    Runs a single simulation of the GMM experiment.

    Parameters:
        seed (int): Random seed for reproducibility.
        K (int): Number of Gaussian components.
        D (int): Total dimensionality of the data.
        sim (float): Proportion of shared dimensions (0 to 1).
        n (int): Number of training, testing and sampling samples.

    Returns:
        tuple: Empirical TV distances for "single" and "multi" methods.
    """
    seed_torch(seed)

    # Define distribution settings
    p_y = torch.full((K,), 1 / K)
    d_sim = int(sim * D)
    d_dif = D - d_sim

    # Generate training data
    mu = generate_shared_means(K, d_dif, d_sim)
    distribution = ConditionalGaussianMixtureDistribution(mu, p_y)
    x_train, y_train = distribution.generate_samples(n)

    # Initialize models
    model_single = GaussianMixtureModel(K, d_dif, d_sim, p_y)
    model_multi = GaussianMixtureModel(K, d_dif, d_sim, p_y)

    # Estimate means using "single" and "multi" methods
    model_multi.estimate_means(x_train, y_train, method="multi")
    model_single.estimate_means(x_train, y_train, method="single")

    # Generate test data
    x_test, y_test = distribution.generate_samples(500)

    # Compute densities and TV distances
    mean_hat_single = model_single()
    mean_hat_multi = model_multi()
    distribution_hat_single = ConditionalGaussianMixtureDistribution(mean_hat_single, p_y)
    distribution_hat_multi = ConditionalGaussianMixtureDistribution(mean_hat_multi, p_y)
    p_hat_single = distribution_hat_single.calculate_densities(x_test, y_test)
    p_hat_multi = distribution_hat_multi.calculate_densities(x_test, y_test)
    p_true = distribution.calculate_densities(x_test, y_test)
    empirical_TV_single = calculate_empirical_TV(p_hat_single, p_true)
    empirical_TV_multi = calculate_empirical_TV(p_hat_multi, p_true)

    return empirical_TV_single, empirical_TV_multi


# Function to run the experiment for a given parameter
def run_experiment(tag, seeds, Ks, ns, sims, K_default, D_default, sim_default, n_default, results_dir):
    """
    Runs experiments for the specified parameter and records results.

    Parameters:
        tag (str): Parameter to vary ("K", "sim", or "n").
        seeds (list): List of random seeds for reproducibility.
        Ks (list): List of values for the number of components.
        ns (list): List of sample sizes.
        sims (list): List of similarity proportions.
        K_default (int): Default value for K.
        D_default (int): Default dimensionality.
        sim_default (float): Default similarity proportion.
        n_default (int): Default sample size.
        results_dir (str): Directory to save results.
    """
    params = {"K": Ks, "sim": sims, "n": ns}  # Parameter ranges
    results = []  # Store experiment results

    # Default parameters
    K, D, sim, n = K_default, D_default, sim_default, n_default

    # Iterate over seeds and parameter values
    for seed in seeds:
        for value in params[tag]:
            if tag == "K":
                K = value
            elif tag == "sim":
                sim = value
            elif tag == "n":
                n = value

            # Run the simulation
            TV_single, TV_multi = gaussian_simulation(seed, K, D, sim, n)

            # Compute theoretical TV bounds
            TV_bound_single = np.sqrt((K * D) / n)
            TV_bound_multi = np.sqrt(((K - 1) * (D - int(sim * D)) + D) / n)

            # Record results
            results.append({
                "seed": seed,
                "K": K,
                "D": D,
                "sim": sim,
                "n": n,
                "theoretical TV bound of single": TV_bound_single,
                "theoretical TV bound of multi": TV_bound_multi,
                "theoretical advantage (abs)": TV_bound_single - TV_bound_multi,
                "empirical TV of single": TV_single.item(),
                "empirical TV of multi": TV_multi.item(),
                "advantage (abs)": TV_single.item() - TV_multi.item(),
            })

    # Save results as a CSV
    results_df = pd.DataFrame(results).groupby(tag, as_index=False).mean()
    results_dir_tag = os.path.join(results_dir, tag)
    os.makedirs(results_dir_tag, exist_ok=True)
    file_path = os.path.join(results_dir_tag, f"results_{tag}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv")
    results_df.to_csv(file_path, index=False)
    print(results_df)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Gaussian Mixture Model Simulations")
    parser.add_argument("--tag", type=str, choices=["K", "sim", "n"], required=True, help="Experiment tag (K, sim, n)")
    parser.add_argument("--results_dir", type=str, default="./simulations/results/gaussian", help="Directory to save results")
    args = parser.parse_args()

    # Experiment parameters
    seeds = [3, 33, 520, 2025, 25731]
    Ks = [1, 3, 5, 10, 15]
    ns = [100, 300, 500, 1000, 5000]
    sims = [0, 0.3, 0.5, 0.7, 1.0]
    K_default = 5
    sim_default = 0.5
    n_default = 500
    D_default = 10

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    # Run the experiment
    run_experiment(args.tag, seeds, Ks, ns, sims, K_default, D_default, sim_default, n_default, args.results_dir)