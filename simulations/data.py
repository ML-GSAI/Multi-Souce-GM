import torch
import torch.distributions as dist


class ConditionalGaussianMixtureDistribution:
    def __init__(self, means, p_y):
        """
        Initialize the conditional Gaussian Mixture Model.

        Parameters:
            means (torch.Tensor): A (K, d) tensor of means for the K components.
            p_y (torch.Tensor): A (K,) tensor of probabilities for each component.
        """
        self.means = means
        self.p_y = p_y

        # Validate that p_y sums to 1
        if not torch.isclose(p_y.sum(), torch.tensor(1.0)):
            raise ValueError("p_y must sum to 1")

    def generate_samples(self, n):
        """
        Generate samples from the conditional Gaussian mixture.

        Parameters:
            n (int): Number of samples to generate.

        Returns:
            x (torch.Tensor): A (n, d) tensor of generated samples.
            y (torch.Tensor): A (n,) tensor of labels (component indices).
        """
        K, d = self.means.shape

        # Sample labels y based on p_y
        y = torch.multinomial(self.p_y, n, replacement=True)  # Shape: (n,)
        x = torch.zeros((n, d))  # Shape: (n, d)

        # Generate samples for each component
        for k in range(K):
            num_samples = (y == k).sum().item()  # Number of samples for component k
            if num_samples > 0:
                x[y == k] = torch.randn((num_samples, d)) + self.means[k]  # Sample num_samples points from N(mean[k], I)

        return x, y

    def calculate_densities(self, x, y):
        """
        Compute the Gaussian density for the given x and y.

        Parameters:
            x (torch.Tensor): A (n, d) tensor of data points.
            y (torch.Tensor): A (n,) tensor of labels (component indices).

        Returns:
            torch.Tensor: A (n,) tensor of Gaussian densities for each point.
        """
        n, d = x.shape
        selected_means = self.means[y]  # Shape: (n, d)

        # Define a unit covariance matrix
        unit_covariance = torch.eye(d, device=x.device)

        # Compute Gaussian densities
        densities = torch.zeros(n, device=x.device)
        for i in range(n):
            mvn = dist.MultivariateNormal(selected_means[i], covariance_matrix=unit_covariance)
            densities[i] = mvn.log_prob(x[i]).exp()  # Convert log-density to density

        return densities
