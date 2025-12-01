import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# pytest -sv tests/learning/test_deep_learning.py
class TestDeepLearning:

    @staticmethod
    def save_current_plot(filename: str, folder="outputs", close=True) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, folder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        if close:
            plt.close()
        return save_path

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_softmax_numpy
    def test_softmax_numpy(self) -> None:
        np.random.seed(123)
        z = np.random.randint(low=-5, high=15, size=25)
        sigma = np.exp(z) / sum(np.exp(z))
        plt.plot(z, sigma, "ko")
        plt.xlabel("Original number (z)")
        plt.ylabel("Softmax value (σ)")
        plt.title(f"Σσ = {np.sum(sigma)}")
        self.save_current_plot(filename="softmax_dist.png", close=False)
        plt.yscale("log")
        self.save_current_plot(filename="softmax_dist_log_scaled.png")

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_softmax_torch
    def test_softmax_torch(self) -> None:
        np.random.seed(123)
        z = np.random.randint(low=-5, high=15, size=25)
        softmax = nn.Softmax(dim=0)
        """
        forward method
        """
        sigma = softmax(torch.Tensor(z))
        assert np.isclose(torch.sum(sigma).item(), 1)

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_shannon_entropy_numpy
    def test_shannon_entropy_numpy(self) -> None:
        """
        Shannon entropy: A measure of uncertainty within a single probability distribution
        H(p) = -Σp(x)log(p(x))
        """
        P = np.array([0.25, 0.75])
        H = -np.sum(P * np.log(P))
        print(f"\nShannon entropy: {H}")
        """
        Cross entropy: A measure of how well predicted distribution explains true distribution
        H(p,q) = -Σp(x)log(q(x))

        P: true distribution
        Q: predicted distribution
        """
        P = np.array([0, 1])
        Q = np.array([0.25, 0.75])
        H = -np.sum(P * np.log(Q))
        print(f"Cross entropy: {H}")

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_binary_cross_entropy_torch
    def test_binary_cross_entropy_torch(self) -> None:
        P = torch.tensor([0.0, 1.0])
        Q = torch.tensor([0.25, 0.75])
        H = F.binary_cross_entropy(Q, P)
        print(f"\nBinary cross entropy: {H}")
