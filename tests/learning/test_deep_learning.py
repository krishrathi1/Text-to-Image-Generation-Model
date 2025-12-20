import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# pytest -sv tests/learning/test_deep_learning.py
class TestDeepLearning:

    class ModelState:

        _hidden_size = 8
        _learning_rate = 0.01
        _num_epochs = 100

        @property
        def hidden_size(self):
            return self._hidden_size

        @property
        def learning_rate(self):
            return self._learning_rate

        @property
        def num_epochs(self):
            return self._num_epochs

        def set_breadth(self, hidden_size: int):
            assert hidden_size > 0, "_hidden_size must be positive"
            self._hidden_size = hidden_size

        def set_lr(self, learning_rate: float):
            assert learning_rate > 0, "_learning_rate must be positive"
            self._learning_rate = learning_rate

        def set_epochs(self, num_epochs: int):
            assert num_epochs > 0, "_num_epochs must be positive"
            self._num_epochs = num_epochs

    class MLP(nn.Module):

        _net: nn.Sequential

        def __init__(
            self,
            input_size,
            breadth,
            output_size,
            linear=False,
            depth=0,
        ) -> None:
            super().__init__()
            self._net = nn.Sequential(
                nn.Linear(input_size, breadth),
                *(
                    sum(
                        [
                            (
                                [nn.Linear(breadth, breadth)]
                                if linear
                                else [nn.ReLU(), nn.Linear(breadth, breadth)]
                            )
                            for _ in range(depth)
                        ],
                        [],
                    )
                    + ([] if linear else [nn.ReLU()])
                ),
                nn.Linear(breadth, output_size),
                # nn.Sigmoid(),  # nn.BCELoss
            )

        def forward(self, x) -> torch.Tensor:
            return self._net(x)

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

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_iris_classification
    def test_iris_classification(self) -> None:
        iris = load_iris()
        assert isinstance(iris, dict)
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df["species"] = iris.target_names[iris.target]
        sns.pairplot(iris_df, hue="species")
        self.save_current_plot(filename="iris_pairplot.png")
        x = torch.tensor(iris.data).float()
        y = torch.tensor(iris.target)
        partitions = [0.8, 0.1, 0.1]
        assert sum(partitions) == 1
        x_train, x_temp, y_train, y_temp = train_test_split(
            x, y, train_size=partitions[0], stratify=y
        )  # stratify ensures balanced classes in both sets
        x_devset, x_test, y_devset, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=partitions[2] / np.sum(partitions[1:]),
            stratify=y_temp,
        )
        assert isinstance(x_train, torch.Tensor)
        assert isinstance(y_train, torch.Tensor)
        assert isinstance(x_devset, torch.Tensor)
        assert isinstance(y_devset, torch.Tensor)
        assert isinstance(x_test, torch.Tensor)
        assert isinstance(y_test, torch.Tensor)
        train_dataset = TensorDataset(x_train, y_train)
        devset_dataset = TensorDataset(x_devset, y_devset)
        test_dataset = TensorDataset(x_test, y_test)
        batch_size = 4
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        devset_loader = DataLoader(devset_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset, shuffle=False)
        input_size = 4
        output_size = 3
        m_state = self.ModelState()
        m_state.set_breadth(1)  # 1 -> 8
        m_state.set_lr(0.0001)  # 0.0001 -> 0.01
        m_state.set_epochs(10)  # 10 -> 100
        """
        nn.CrossEntropyLoss: works with output_size >= 2 (Multiclass classification)
        """
        criterion = nn.CrossEntropyLoss()
        print()
        while True:
            model = self.MLP(input_size, m_state.hidden_size, output_size, depth=1)
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=m_state.learning_rate)
            print("\nHidden Size:", m_state.hidden_size)
            print("Learning Rate:", m_state.learning_rate)
            print("Number of Epochs:", m_state.num_epochs, "\n")
            losses = torch.zeros(m_state.num_epochs)
            softmax = nn.Softmax(dim=1)
            for epoch_idx in range(m_state.num_epochs):
                for x, y in train_loader:
                    y_hat = model(x)  # forward pass
                    loss = criterion(y_hat, y)  # compute loss
                    losses[epoch_idx] = loss
                    optimizer.zero_grad()
                    loss.backward()  # backprop
                    optimizer.step()
                if (epoch_idx + 1) % (m_state.num_epochs // 10) == 0:
                    print(
                        f"Epoch [{epoch_idx+1}/{m_state.num_epochs}], Loss: {loss.item():.4f}"
                    )
                    total_acc = 100 * torch.mean(
                        (torch.argmax(y_hat, dim=1) == y).float()
                    )
                    print(f"Total Accuracy: {total_acc}\n")
                    assert np.allclose(
                        torch.sum(softmax(y_hat), dim=1).detach(), np.ones(len(y_hat))
                    )
            print()
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for x, y in devset_loader:
                    y_hat = model(x)
                    _, predicted = torch.max(y_hat.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                accuracy = 100 * correct / total
                print(f"Accuracy of the model on the devset data: {accuracy:.2f} %")
            while (
                option := input(
                    "Choose an option (0: Test, 1: Hidden Size, 2: Learning Rate, 3: Number of Epochs): "
                )
            ) not in (
                "0",
                "1",
                "2",
                "3",
            ):
                print("Invalid choice. Please try again.")
            print()
            if int(option):
                set_val = input("Set value : ")
                {
                    "1": lambda: m_state.set_breadth(int(set_val)),
                    "2": lambda: m_state.set_lr(float(set_val)),
                    "3": lambda: m_state.set_epochs(int(set_val)),
                }.get(option, lambda: "Invalid.")()
                continue
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for x, y in test_loader:
                    y_hat = model(x)
                    _, predicted = torch.max(y_hat.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                accuracy = 100 * correct / total
                print(f"Accuracy of the model on the test data: {accuracy:.2f} %")
            break
