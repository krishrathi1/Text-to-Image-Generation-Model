import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
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

        _activation: nn.Module
        _linear = False
        _breadth = 10
        _depth = 0
        _learning_rate = 0.001
        _num_epochs = 100
        _batch_size = 8
        _dropout_rate = 0.1
        _batch_norm = False

        @property
        def activation(self):
            return self._activation

        @property
        def linear(self):
            return self._linear

        @property
        def breadth(self):
            return self._breadth

        @property
        def depth(self):
            return self._depth

        @property
        def learning_rate(self):
            return self._learning_rate

        @property
        def num_epochs(self):
            return self._num_epochs

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def dropout_rate(self):
            return self._dropout_rate

        @property
        def batch_norm(self):
            return self._batch_norm

        def set_linear(self, linear: bool):
            self._linear = linear

        def set_breadth(self, breadth: int):
            assert breadth > 0, "_breadth must be positive"
            self._breadth = breadth

        def set_depth(self, depth: int):
            assert depth >= 0, "_depth must be non-negative"
            self._depth = depth

        def set_lr(self, learning_rate: float):
            assert learning_rate > 0, "_learning_rate must be positive"
            self._learning_rate = learning_rate

        def set_epochs(self, num_epochs: int):
            assert num_epochs > 0, "_num_epochs must be positive"
            self._num_epochs = num_epochs

        def set_batch_size(self, batch_size: int):
            assert batch_size > 0, "_batch_size must be positive"
            self._batch_size = batch_size

        def set_dropout_rate(self, dropout_rate: float):
            assert (dropout_rate >= 0) | (
                dropout_rate <= 1
            ), "_dropout_rate must be between 0 to 1"
            self._dropout_rate = dropout_rate

        def set_batch_norm(self, batch_norm: bool):
            self._batch_norm = batch_norm

        def set_activation(self, activation: nn.Module):
            self._activation = activation

    class MLP(nn.Module):

        _net: nn.Sequential

        def __init__(
            self,
            activation,
            input_size,
            breadth,
            output_size,
            linear=False,
            depth=0,
            dropout_rate=0.0,
            batch_norm=True,
        ) -> None:
            super().__init__()
            self._net = nn.Sequential(
                nn.Linear(input_size, breadth, bias=not batch_norm),
                *(
                    sum(
                        [
                            (
                                [nn.Linear(breadth, breadth)]
                                if linear
                                else [
                                    (
                                        nn.BatchNorm1d(breadth)
                                        if batch_norm
                                        else nn.Identity()
                                    ),
                                    activation,
                                    nn.Dropout(dropout_rate),
                                    nn.Linear(breadth, breadth, bias=not batch_norm),
                                ]
                            )
                            for _ in range(depth)
                        ],
                        [],
                    )
                    + (
                        []
                        if linear
                        else [
                            nn.BatchNorm1d(breadth) if batch_norm else nn.Identity(),
                            activation,
                            nn.Dropout(dropout_rate),
                        ]
                    )
                ),
                nn.Linear(breadth, output_size, bias=True),
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
        sns.violinplot(data=iris_df, inner=None)
        sns.swarmplot(data=iris_df, color="k", size=2)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self.save_current_plot(filename="iris_vln_swarmplot_before.png")
        """
        Z-score normalization: (feature - mean) / std
        """
        cols2zscore = iris_df.keys()
        iris_df[cols2zscore] = iris_df[cols2zscore].apply(stats.zscore)
        sns.violinplot(data=iris_df, inner=None)
        sns.swarmplot(data=iris_df, color="k", size=2)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self.save_current_plot(filename="iris_vln_swarmplot_after.png")
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
        devset_loader = DataLoader(devset_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset, shuffle=False)
        input_size = 4
        output_size = 3
        m_state = self.ModelState()
        m_state.set_activation(nn.ReLU())
        m_state.set_breadth(10)
        m_state.set_batch_size(8)
        """
        nn.CrossEntropyLoss: works with output_size >= 2 (Multiclass classification)
        """
        criterion = nn.CrossEntropyLoss()
        print()
        while True:
            train_loader = DataLoader(
                train_dataset,
                batch_size=m_state.batch_size,
                shuffle=True,
                drop_last=True,
            )
            model = self.MLP(
                m_state.activation,
                input_size,
                m_state.breadth,
                output_size,
                linear=m_state.linear,
                depth=m_state.depth,
                dropout_rate=m_state.dropout_rate,
                batch_norm=m_state.batch_norm,
            )
            model.train()
            """
            Adam combines momentum and RMSprop

            w = w - lr/((s'+eps)^0.5) * v'
            v' = v / (1-b1^t)
            s' = s / (1-b2^t)
            v = (1-b1)*dJ + b1*vt-1 (momentum)
            s = (1-b2)*(dJ^2) + b2*st-1 (RMSprop)

            Adam and RMSprop are robust to small learning rate compared to vanilla SGD
            """
            optimizer = optim.Adam(
                model.parameters(),
                lr=m_state.learning_rate,  # 0.001 recommended
                betas=(0.9, 0.999),  # (b1, b2)
                eps=1e-08,  # prevent s' being zero for numerical stability
            )
            print("\nActivation function:", m_state.activation)
            print("Linearity:", m_state.linear)
            print("Breadth of hidden layers:", m_state.breadth)
            print("Depth of hidden layers:", m_state.depth)
            print("Learning Rate:", m_state.learning_rate)
            print("Number of Epochs:", m_state.num_epochs)
            print("Batch Size:", m_state.batch_size)
            print("Dropout Rate:", m_state.dropout_rate)
            print("Batch Normalization:", m_state.batch_norm, "\n")
            softmax = nn.Softmax(dim=1)
            for epoch_idx in range(m_state.num_epochs):
                for x, y in train_loader:
                    y_hat = model(x)  # forward pass
                    epoch_loss = criterion(y_hat, y)  # compute loss
                    optimizer.zero_grad()
                    epoch_loss.backward()  # backprop
                    optimizer.step()
                    epoch_acc = 100 * torch.mean(
                        (torch.argmax(y_hat, dim=1) == y).float()
                    )
                if (epoch_idx + 1) % (m_state.num_epochs // 10) == 0:
                    print(
                        f"Epoch [{epoch_idx+1}/{m_state.num_epochs}], Epoch Loss: {epoch_loss.item():.4f}, Epoch Accuracy: {epoch_acc}\n"
                    )
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
                (
                    option := input(
                        """Choose an option
    0) Test
    1) Activation Function (0: nn.ReLU(), 1: nn.LeakyReLU(negative_slope=0.01), 2: nn.GELU(approximate="tanh"))
    2) Linearity (0: False, 1: True)
    3) Breadth of hidden layers
    4) Depth of hidden layers
    5) Learning Rate
    6) Number of Epochs
    7) Batch Size
    8) Dropout Rate
    9) Batch Normalization (0: False, 1: True)
    : """
                    )
                )
                not in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
            ):
                print("Invalid choice. Please try again.")
            print()
            if int(option):
                set_val = input("Set value : ")
                {
                    "1": lambda: m_state.set_activation(
                        [
                            nn.ReLU(),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.GELU(approximate="tanh"),
                        ][int(set_val)]
                    ),
                    "2": lambda: m_state.set_linear(bool(int(set_val))),
                    "3": lambda: m_state.set_breadth(int(set_val)),
                    "4": lambda: m_state.set_depth(int(set_val)),
                    "5": lambda: m_state.set_lr(float(set_val)),
                    "6": lambda: m_state.set_epochs(int(set_val)),
                    "7": lambda: m_state.set_batch_size(int(set_val)),
                    "8": lambda: m_state.set_dropout_rate(float(set_val)),
                    "9": lambda: m_state.set_batch_norm(bool(int(set_val))),
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

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_mnist_classification
    def test_mnist_classification(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "assets", "mnist_train_small.csv")
        mnist_df = pd.read_csv(file_path, header=None)
        x = torch.tensor(
            mnist_df.iloc[:, 1:].values.reshape(
                mnist_df.shape[0], mnist_df.shape[1] - 1
            )
        ).float()
        y = torch.tensor(mnist_df.iloc[:, 0].values)
        partitions = [0.8, 0.1, 0.1]
        assert sum(partitions) == 1
        x_train, x_temp, y_train, y_temp = train_test_split(
            x, y, train_size=partitions[0], stratify=y
        )
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
        devset_loader = DataLoader(devset_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset, shuffle=False)
        input_size = mnist_df.shape[1] - 1
        output_size = len(set(mnist_df.iloc[:, 0].values))
        m_state = self.ModelState()
        m_state.set_activation(nn.LeakyReLU(negative_slope=0.01))
        m_state.set_breadth(60)
        m_state.set_batch_size(64)
        criterion = nn.CrossEntropyLoss()
        print()
        while True:
            train_loader = DataLoader(
                train_dataset,
                batch_size=m_state.batch_size,
                shuffle=True,
                drop_last=True,
            )
            model = self.MLP(
                m_state.activation,
                input_size,
                m_state.breadth,
                output_size,
                linear=m_state.linear,
                depth=m_state.depth,
                dropout_rate=m_state.dropout_rate,
                batch_norm=m_state.batch_norm,
            )
            model.train()
            optimizer = optim.Adam(
                model.parameters(),
                lr=m_state.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
            print("\nActivation function:", m_state.activation)
            print("Linearity:", m_state.linear)
            print("Breadth of hidden layers:", m_state.breadth)
            print("Depth of hidden layers:", m_state.depth)
            print("Learning Rate:", m_state.learning_rate)
            print("Number of Epochs:", m_state.num_epochs)
            print("Batch Size:", m_state.batch_size)
            print("Dropout Rate:", m_state.dropout_rate)
            print("Batch Normalization:", m_state.batch_norm, "\n")
            softmax = nn.Softmax(dim=1)
            for epoch_idx in range(m_state.num_epochs):
                for x, y in train_loader:
                    y_hat = model(x)  # forward pass
                    epoch_loss = criterion(y_hat, y)  # compute loss
                    optimizer.zero_grad()
                    epoch_loss.backward()  # backprop
                    optimizer.step()
                    epoch_acc = 100 * torch.mean(
                        (torch.argmax(y_hat, dim=1) == y).float()
                    )
                if (epoch_idx + 1) % (m_state.num_epochs // 10) == 0:
                    print(
                        f"Epoch [{epoch_idx+1}/{m_state.num_epochs}], Epoch Loss: {epoch_loss.item():.4f}, Epoch Accuracy: {epoch_acc}\n"
                    )
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
                (
                    option := input(
                        """Choose an option
    0) Test
    1) Activation Function (0: nn.ReLU(), 1: nn.LeakyReLU(negative_slope=0.01), 2: nn.GELU(approximate="tanh"))
    2) Linearity (0: False, 1: True)
    3) Breadth of hidden layers
    4) Depth of hidden layers
    5) Learning Rate
    6) Number of Epochs
    7) Batch Size
    8) Dropout Rate
    9) Batch Normalization (0: False, 1: True)
    : """
                    )
                )
                not in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
            ):
                print("Invalid choice. Please try again.")
            print()
            if int(option):
                set_val = input("Set value : ")
                {
                    "1": lambda: m_state.set_activation(
                        [
                            nn.ReLU(),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.GELU(approximate="tanh"),
                        ][int(set_val)]
                    ),
                    "2": lambda: m_state.set_linear(bool(int(set_val))),
                    "3": lambda: m_state.set_breadth(int(set_val)),
                    "4": lambda: m_state.set_depth(int(set_val)),
                    "5": lambda: m_state.set_lr(float(set_val)),
                    "6": lambda: m_state.set_epochs(int(set_val)),
                    "7": lambda: m_state.set_batch_size(int(set_val)),
                    "8": lambda: m_state.set_dropout_rate(float(set_val)),
                    "9": lambda: m_state.set_batch_norm(bool(int(set_val))),
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
