import matplotlib.image as mpimg
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
import torchvision
import torchvision.transforms as T

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchsummary import summary
from torchvision.models import vgg19, VGG19_Weights
from typing import cast


# pytest -sv tests/learning/test_deep_learning.py
class TestDeepLearning:

    class ModelState:

        _activation: nn.Module
        _linear = False
        _breadth: int | tuple[int, ...] = 8
        _depth = 0
        _learning_rate = 0.001
        _num_epochs = 100
        _batch_size = 8
        _dropout_rate = 0.1
        _batch_norm = False
        _ksp = (5, 1, 0)  # kernel size, stride, padding

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

        @property
        def ksp(self):
            return self._ksp

        def set_linear(self, linear: bool):
            self._linear = linear

        def set_breadth(self, breadth: int | tuple[int, ...]):
            if isinstance(breadth, int):
                assert breadth > 0, "_breadth must be positive"
                assert (
                    breadth & (breadth - 1)
                ) == 0, "_breadth is recommended to be power of two"
            if isinstance(breadth, tuple):
                assert all(
                    isinstance(elm, int) and (elm > 0) for elm in breadth
                ), "Not all elements in _breadth are positive integers"
                assert all(
                    (elm & (elm - 1)) == 0 for elm in breadth
                ), "All elements in _breadth are recommended to be power of two"
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
            assert (
                batch_size & (batch_size - 1)
            ) == 0, "_batch_size is recommended to be power of two"
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

        def set_ksp(self, ksp: tuple[int, int, int]):
            assert ksp[0] > 0, "kernel size must be positive"
            assert ksp[1] > 0, "stride must be positive"
            assert ksp[2] >= 0, "padding must be non-negative"
            self._ksp = ksp

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

    class CNNV1(nn.Module):

        # class _Debugger(nn.Module):

        #     def forward(self, x):
        #         print(x.shape)
        #         return x

        class _Vectorizer(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.view(-1, int(x.shape.numel() / x.shape[0]))

        _kernel_size: int
        _net: nn.Sequential

        def __init__(
            self,
            activation,
            input_size,
            breadth,
            output_size,
            ksp=(5, 1, 0),
            fc_breadth=64,
            dropout_rate=0.0,
            batch_norm=True,
        ) -> None:
            super().__init__()
            self._kernel_size = input_size[1]
            for _ in range(len(breadth)):
                self._kernel_size = (
                    np.floor((self._kernel_size - ksp[0] + 2 * ksp[2]) / ksp[1]) + 1
                )
                self._kernel_size = int(np.floor(self._kernel_size / 2))
            self._net = nn.Sequential(
                # self._Debugger(),
                nn.Conv2d(
                    input_size[0],
                    breadth[0],
                    kernel_size=ksp[0],
                    stride=ksp[1],
                    padding=ksp[2],
                ),
                # self._Debugger(),
                (nn.BatchNorm2d(breadth[0]) if batch_norm else nn.Identity()),
                activation,
                nn.MaxPool2d(2),
                # self._Debugger(),
                *(
                    sum(
                        [
                            (
                                [
                                    nn.Conv2d(
                                        breadth[idx],
                                        breadth[idx + 1],
                                        kernel_size=ksp[0],
                                        stride=ksp[1],
                                        padding=ksp[2],
                                    ),
                                    # self._Debugger(),
                                    (
                                        nn.BatchNorm2d(breadth[idx + 1])
                                        if batch_norm
                                        else nn.Identity()
                                    ),
                                    activation,
                                    nn.MaxPool2d(2),
                                    # self._Debugger(),
                                ]
                            )
                            for idx in range(len(breadth) - 1)
                        ],
                        [],
                    )
                ),
                self._Vectorizer(),
                nn.Linear(
                    breadth[-1] * int(self._kernel_size**2),
                    fc_breadth,
                    bias=True,
                ),
                (nn.BatchNorm1d(fc_breadth) if batch_norm else nn.Identity()),
                activation,
                nn.Dropout(dropout_rate),
                nn.Linear(fc_breadth, output_size, bias=True),
                # nn.Sigmoid(),  # nn.BCELoss
            )

        def forward(self, x) -> torch.Tensor:
            return self._net(x)

    class CNNDatasetV1(Dataset):

        _inputs: torch.Tensor
        _targets: torch.Tensor
        _transform: T.Compose | None

        def __init__(
            self, inputs: torch.Tensor, targets: torch.Tensor, transform=None
        ) -> None:
            assert inputs.shape[0] == targets.shape[0], "Size mismatch between tensors"
            self._inputs = inputs
            self._targets = targets
            self._transform = transform

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                self._transform(self._inputs[index])
                if self._transform
                else self._inputs[index]
            ), self._targets[index]

        def __len__(self) -> int:
            return self._inputs.shape[0]

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

    @staticmethod
    def extract_features(
        x: torch.Tensor, model_features: nn.Module, layers: list, verbose=False
    ) -> dict[str, torch.Tensor]:
        features = {}
        for name, layer in model_features._modules.items():
            if isinstance(layer, nn.Module):
                x = layer(x)
                if verbose and isinstance(layer, nn.Conv2d):
                    print(f"Name: {name}, Layer: {layer}")
                if name in layers:
                    features[name] = x
        return features

    @staticmethod
    def gram_matrix(mat: torch.Tensor) -> torch.Tensor:
        _, num_featmaps, height, width = mat.shape
        mat = mat.view(num_featmaps, height * width)
        return mat @ mat.t() / (num_featmaps * height * width)

    @staticmethod
    def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(
            torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        )

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

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_iris_ffn_classification
    def test_iris_ffn_classification(self) -> None:
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
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
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
        train_dataset = TensorDataset(x_train.to(device), y_train.to(device))
        devset_dataset = TensorDataset(x_devset.to(device), y_devset.to(device))
        test_dataset = TensorDataset(x_test.to(device), y_test.to(device))
        devset_loader = DataLoader(devset_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset, shuffle=False)
        input_size = 4
        output_size = 3
        m_state = self.ModelState()
        m_state.set_activation(nn.ReLU())
        m_state.set_breadth(8)
        m_state.set_depth(0)
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
            model.to(device)
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

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_mnist_ffn_classification
    def test_mnist_ffn_classification(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "assets", "mnist_train_small.csv")
        mnist_df = pd.read_csv(file_path, header=None)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        x = torch.tensor(
            mnist_df.iloc[:, 1:].values.reshape(
                mnist_df.shape[0], mnist_df.shape[1] - 1
            )
        ).float()
        """
        min-max scaling: (feature - min) / (max - min)
        """
        x = x / torch.max(x)
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
        train_dataset = TensorDataset(x_train.to(device), y_train.to(device))
        devset_dataset = TensorDataset(x_devset.to(device), y_devset.to(device))
        test_dataset = TensorDataset(x_test.to(device), y_test.to(device))
        devset_loader = DataLoader(devset_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset, shuffle=False)
        input_size = mnist_df.shape[1] - 1
        output_size = len(set(mnist_df.iloc[:, 0].values))
        m_state = self.ModelState()
        m_state.set_activation(nn.LeakyReLU(negative_slope=0.01))
        m_state.set_breadth(64)
        m_state.set_depth(3)
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
            model.to(device)
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

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_conv2d_single_in_channel
    def test_conv2d_single_in_channel(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "assets", "cafe_rooted.jpeg")
        pic = mpimg.imread(file_path)
        print(f"\nRaw Image Shape: {pic.shape}")
        pic = np.mean(pic, axis=2)  # 2D image
        pic = pic / np.max(pic)
        print(f"2D Transformed Image Shape: {pic.shape}")
        """
        view(batch_size, in_channels, height, width)
        """
        v_kernel = torch.tensor(
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float
        ).view(1, 1, 3, 3)
        h_kernel = torch.tensor(
            [[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float
        ).view(1, 1, 3, 3)
        pic_tensor = torch.tensor(pic, dtype=torch.float).view(
            1, 1, pic.shape[0], pic.shape[1]
        )
        assert F.conv2d(pic_tensor, v_kernel).shape == torch.Size(
            [1, 1, pic.shape[0] - 2, pic.shape[1] - 2]
        )
        assert F.conv2d(pic_tensor, h_kernel).shape == torch.Size(
            [1, 1, pic.shape[0] - 2, pic.shape[1] - 2]
        )
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].imshow(np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
        axes[0, 0].set_title("Vertical kernel")
        axes[0, 1].imshow(np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
        axes[0, 1].set_title("Horizontal kernel")
        conv_v = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=(1, 1),
            padding=0,
            bias=False,
            dtype=torch.float,
        )
        conv_h = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=(1, 1),
            padding=0,
            bias=False,
            dtype=torch.float,
        )
        with torch.no_grad():
            conv_v.weight.copy_(v_kernel)
            conv_v_out = conv_v(pic_tensor).detach().cpu().numpy().squeeze()
            conv_h.weight.copy_(h_kernel)
            conv_h_out = conv_h(pic_tensor).detach().cpu().numpy().squeeze()
        axes[1, 0].imshow(conv_v_out, cmap="gray", vmin=0, vmax=0.01)
        axes[1, 1].imshow(conv_h_out, cmap="gray", vmin=0, vmax=0.01)
        plt.tight_layout()
        plt.show()
        self.save_current_plot(filename="cafe_rooted_conv2d.png")

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_cifar10_transform
    def test_cifar10_transform(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "assets", "cifar10")
        cdata = torchvision.datasets.CIFAR10(root=file_path, download=True)
        print(f"\n{cdata}")
        print(cdata.data.shape)
        assert cdata.data.shape[0] == len(cdata.targets)
        print(cdata.classes)
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for idx, ax in enumerate(axes.flatten()):
            pic = cdata.data[idx, :, :, :]
            label = cdata.classes[cdata.targets[idx]]
            ax.imshow(pic)
            ax.text(
                16,
                0,
                label,
                ha="center",
                fontweight="bold",
                color="k",
                backgroundcolor="y",
            )
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        self.save_current_plot(filename="cifar10_first_25.png")
        cdata.transform = T.Compose(
            [
                T.Resize(32 * 4),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),  # [0, 255] -> [0.0, 1.0]
            ]
        )
        bird1 = cdata.data[123, :32, :32, :3]
        bird2 = cdata.transform(bird1)
        assert bird2.shape == (1, 32 * 4, 32 * 4)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(bird1)
        axes[1].imshow(bird2.detach().cpu().numpy().squeeze(), cmap="gray")
        plt.tight_layout()
        plt.show()
        self.save_current_plot(filename="cifar10_123_bird_transform.png")
        imgtrans = T.Compose(
            [
                T.ToPILImage(),
                T.RandomVerticalFlip(p=1.0),
                T.ToTensor(),  # [0, 255] -> [0.0, 1.0]
            ]
        )
        cdata_inputs = torch.tensor(cdata.data, dtype=torch.float).transpose(1, 3)
        cdata_inputs = cdata_inputs / torch.max(cdata_inputs)
        print(cdata_inputs.shape)
        test_dataset = self.CNNDatasetV1(
            cdata_inputs,
            torch.tensor(cdata.targets),
            imgtrans,
        )
        batch_size = 8
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        x, y = next(iter(test_loader))
        fig, axes = plt.subplots(2, batch_size, figsize=(16, 4))
        for i in range(batch_size):
            axes[0, i].imshow(
                cdata_inputs[i, 0, :, :].detach().cpu().numpy().squeeze(), cmap="gray"
            )
            axes[1, i].imshow(
                x[i, 0, :, :].detach().cpu().numpy().squeeze(), cmap="gray"
            )
            for row in range(2):
                axes[row, i].set_xticks([])
                axes[row, i].set_yticks([])
        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel("torch dataset")
        plt.tight_layout()
        plt.show()
        self.save_current_plot(filename="cifar10_dataset_transform.png")

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_mnist_cnn_classification
    def test_mnist_cnn_classification(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "assets", "mnist_train_small.csv")
        mnist_df = pd.read_csv(file_path, header=None)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        x = torch.tensor(
            mnist_df.iloc[:, 1:].values.reshape(mnist_df.shape[0], 1, 28, 28)
        ).float()
        """
        min-max scaling: (feature - min) / (max - min)
        """
        x = x / torch.max(x)
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
        train_dataset = TensorDataset(x_train.to(device), y_train.to(device))
        devset_dataset = TensorDataset(x_devset.to(device), y_devset.to(device))
        test_dataset = TensorDataset(x_test.to(device), y_test.to(device))
        devset_loader = DataLoader(devset_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset, shuffle=False)
        input_size = (x.shape[1], x.shape[2])
        output_size = len(set(mnist_df.iloc[:, 0].values))
        m_state = self.ModelState()
        m_state.set_activation(nn.ReLU())
        m_state.set_breadth((8, 16))
        m_state.set_batch_size(128)
        m_state.set_batch_norm(True)
        m_state.set_ksp((5, 1, 1))
        criterion = nn.CrossEntropyLoss()
        print()
        while True:
            train_loader = DataLoader(
                train_dataset,
                batch_size=m_state.batch_size,
                shuffle=True,
                drop_last=True,
            )
            model = self.CNNV1(
                m_state.activation,
                input_size,
                m_state.breadth,
                output_size,
                ksp=m_state.ksp,
                fc_breadth=50,
            )
            model.to(device)
            model.train()
            optimizer = optim.Adam(
                model.parameters(),
                lr=m_state.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
            print("\nActivation function:", m_state.activation)
            print("Breadth of hidden layers:", m_state.breadth)
            print("Learning Rate:", m_state.learning_rate)
            print("Number of Epochs:", m_state.num_epochs)
            print("Batch Size:", m_state.batch_size)
            print("Dropout Rate:", m_state.dropout_rate)
            print("Batch Normalization:", m_state.batch_norm)
            print("Kernel Size, Stride, Padding:", m_state.ksp, "\n")
            print(summary(model, (1, 28, 28)), "\n")
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
    2) Breadth of hidden layers (ex. 10 20: 10 first feature maps, 20 second feature maps)
    3) Learning Rate
    4) Number of Epochs
    5) Batch Size
    6) Dropout Rate
    7) Batch Normalization (0: False, 1: True)
    8) Kernel Size, Stride, Padding (ex. 5 1 0: 5 kernel size, 1 stride, 0 padding)
    : """
                    )
                )
                not in ("0", "1", "2", "3", "4", "5", "6", "7", "8")
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
                    "2": lambda: m_state.set_breadth(tuple(map(int, set_val.split()))),
                    "3": lambda: m_state.set_lr(float(set_val)),
                    "4": lambda: m_state.set_epochs(int(set_val)),
                    "5": lambda: m_state.set_batch_size(int(set_val)),
                    "6": lambda: m_state.set_dropout_rate(float(set_val)),
                    "7": lambda: m_state.set_batch_norm(bool(int(set_val))),
                    "8": lambda: m_state.set_ksp(
                        cast(tuple[int, int, int], tuple(map(int, set_val.split())))
                    ),
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

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_cifar10_cnn_classification
    def test_cifar10_cnn_classification(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "assets", "cifar10")
        cdata = torchvision.datasets.CIFAR10(root=file_path, download=True)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        assert device == torch.device(
            "cuda"
        ), "CUDA device not found. Operation requires a GPU."
        x = torch.tensor(cdata.data, dtype=torch.float).transpose(1, 3)
        """
        min-max scaling: (feature - min) / (max - min)
        """
        x = x / torch.max(x)
        y = torch.tensor(cdata.targets)
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
        train_dataset = TensorDataset(x_train.to(device), y_train.to(device))
        devset_dataset = TensorDataset(x_devset.to(device), y_devset.to(device))
        test_dataset = TensorDataset(x_test.to(device), y_test.to(device))
        devset_loader = DataLoader(devset_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset, shuffle=False)
        input_size = (x.shape[1], x.shape[2])
        output_size = len(set(cdata.targets))
        m_state = self.ModelState()
        m_state.set_activation(nn.ReLU())
        m_state.set_breadth((64, 128, 256))
        m_state.set_batch_size(256)
        m_state.set_batch_norm(True)
        m_state.set_ksp((3, 1, 1))
        criterion = nn.CrossEntropyLoss()
        print()
        while True:
            train_loader = DataLoader(
                train_dataset,
                batch_size=m_state.batch_size,
                shuffle=True,
                drop_last=True,
            )
            model = self.CNNV1(
                m_state.activation,
                input_size,
                m_state.breadth,
                output_size,
                ksp=m_state.ksp,
                fc_breadth=50,
            )
            model.to(device)
            model.train()
            optimizer = optim.Adam(
                model.parameters(),
                lr=m_state.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
            print("\nActivation function:", m_state.activation)
            print("Breadth of hidden layers:", m_state.breadth)
            print("Learning Rate:", m_state.learning_rate)
            print("Number of Epochs:", m_state.num_epochs)
            print("Batch Size:", m_state.batch_size)
            print("Dropout Rate:", m_state.dropout_rate)
            print("Batch Normalization:", m_state.batch_norm)
            print("Kernel Size, Stride, Padding:", m_state.ksp, "\n")
            print(summary(model, x.shape[1:]), "\n")
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
    2) Breadth of hidden layers (ex. 10 20: 10 first feature maps, 20 second feature maps)
    3) Learning Rate
    4) Number of Epochs
    5) Batch Size
    6) Dropout Rate
    7) Batch Normalization (0: False, 1: True)
    8) Kernel Size, Stride, Padding (ex. 5 1 0: 5 kernel size, 1 stride, 0 padding)
    : """
                    )
                )
                not in ("0", "1", "2", "3", "4", "5", "6", "7", "8")
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
                    "2": lambda: m_state.set_breadth(tuple(map(int, set_val.split()))),
                    "3": lambda: m_state.set_lr(float(set_val)),
                    "4": lambda: m_state.set_epochs(int(set_val)),
                    "5": lambda: m_state.set_batch_size(int(set_val)),
                    "6": lambda: m_state.set_dropout_rate(float(set_val)),
                    "7": lambda: m_state.set_batch_norm(bool(int(set_val))),
                    "8": lambda: m_state.set_ksp(
                        cast(tuple[int, int, int], tuple(map(int, set_val.split())))
                    ),
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

    # pytest -sv tests/learning/test_deep_learning.py::TestDeepLearning::test_image_neural_style_transfer_via_vgg19
    def test_image_neural_style_transfer_via_vgg19(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(
            script_dir,
            "assets",
            "cafe_rooted.jpeg",
        )
        content_img = mpimg.imread(file_path)
        file_path = os.path.join(
            script_dir,
            "assets",
            "1000_Three_domes_of_Oia_in_Santorini_Photo_by_Giles_Laurent.jpg",
        )
        style_img = mpimg.imread(file_path)
        """
        target_img = np.random.randint(
            low=0, high=255, size=content_img.shape, dtype=np.uint8
        )
        """
        target_img = np.copy(content_img)
        print("\ncontent image (before):", content_img.shape, content_img.dtype)
        print("style image (before):", style_img.shape, style_img.dtype)
        print("target image (before):", target_img.shape, target_img.dtype)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        assert device == torch.device(
            "cuda"
        ), "CUDA device not found. Operation requires a GPU."
        weights = VGG19_Weights.DEFAULT
        model_features = vgg19(weights=weights).features.to(device).eval()
        """
        weight freezing
        """
        for p in model_features.parameters():
            p.requires_grad = False
        """
        imgtrans = T.Compose([T.ToPILImage(), weights.transforms()])
        imgtrans = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        """
        imgtrans = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(256),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        content_img = imgtrans(content_img)
        assert isinstance(content_img, torch.Tensor)
        content_img = content_img.unsqueeze(0).to(device)
        style_img = imgtrans(style_img)
        assert isinstance(style_img, torch.Tensor)
        style_img = style_img.unsqueeze(0).to(device)
        target_img = imgtrans(target_img)
        assert isinstance(target_img, torch.Tensor)
        target_img = target_img.unsqueeze(0).to(device)
        print("\ncontent image (after):", content_img.shape, content_img.dtype)
        print("style image (after):", style_img.shape, style_img.dtype)
        print("target image (after):", target_img.shape, target_img.dtype)
        print()
        self.extract_features(target_img, model_features, [], verbose=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        pic = content_img.cpu().squeeze().numpy().transpose((1, 2, 0))
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
        axes[0].imshow(pic)
        axes[0].set_title("Content picture")
        pic = style_img.cpu().squeeze().numpy().transpose((1, 2, 0))
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
        axes[1].imshow(pic)
        axes[1].set_title("Style picture")
        pic = target_img.cpu().squeeze().numpy().transpose((1, 2, 0))
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
        axes[2].imshow(pic)
        axes[2].set_title("Target picture")
        plt.tight_layout()
        plt.show()
        self.save_current_plot(filename="style_transfer_before.png")
        print()
        content_feature_dict = self.extract_features(
            content_img, model_features, ["0", "2", "5", "7", "10"]
        )
        content_feature_names = list(content_feature_dict.keys())
        content_feature_maps = list(content_feature_dict.values())
        fig, axes = plt.subplots(2, 5, figsize=(18, 6))
        for i in range(5):
            pic = np.mean(content_feature_maps[i].cpu().squeeze().numpy(), axis=0)
            pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
            axes[0, i].imshow(pic, cmap="gray")
            axes[0, i].set_title("Content layer " + str(content_feature_names[i]))
            pic = self.gram_matrix(content_feature_maps[i]).cpu().numpy()
            pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
            axes[1, i].imshow(pic, cmap="gray", vmax=0.1)
            axes[1, i].set_title("Gram matrix, layer " + str(content_feature_names[i]))
        plt.tight_layout()
        plt.show()
        self.save_current_plot(filename="content_feature_maps.png")
        style_feature_dict = self.extract_features(
            style_img, model_features, ["0", "2", "5", "7", "10"]
        )
        style_feature_names = list(style_feature_dict.keys())
        style_feature_maps = list(style_feature_dict.values())
        fig, axes = plt.subplots(2, 5, figsize=(18, 6))
        for i in range(5):
            pic = np.mean(style_feature_maps[i].cpu().squeeze().numpy(), axis=0)
            pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
            axes[0, i].imshow(pic, cmap="gray")
            axes[0, i].set_title("Style layer " + str(style_feature_names[i]))
            pic = self.gram_matrix(style_feature_maps[i]).cpu().numpy()
            pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
            axes[1, i].imshow(pic, cmap="gray", vmax=0.1)
            axes[1, i].set_title("Gram matrix, layer " + str(style_feature_names[i]))
        plt.tight_layout()
        plt.show()
        self.save_current_plot(filename="style_feature_maps.png")
        content_layers = ["0"]
        content_feature_dict = self.extract_features(
            content_img, model_features, content_layers
        )
        style_layers = ["0", "5", "10", "19"]
        style_feature_dict = self.extract_features(
            style_img, model_features, style_layers
        )
        target_img = target_img.requires_grad_(True)
        optimizer = optim.Adam([target_img], lr=0.01)
        content_weight = 1e4
        style_weight = 1e6
        tv_weight = 1e-5
        num_epochs = 1500
        for epoch_idx in range(num_epochs):
            target_feature_dict = self.extract_features(
                target_img, model_features, list(model_features._modules.keys())
            )
            content_loss = 0
            style_loss = 0
            for layer_name in target_feature_dict.keys():
                if layer_name in content_layers:
                    content_loss += torch.mean(
                        (
                            target_feature_dict[layer_name]
                            - content_feature_dict[layer_name]
                        )
                        ** 2
                    )
                if layer_name in style_layers:
                    style_loss += torch.mean(
                        (
                            self.gram_matrix(target_feature_dict[layer_name])
                            - self.gram_matrix(style_feature_dict[layer_name])
                        )
                        ** 2
                    )
            total_loss = (
                content_weight * content_loss
                + style_weight * style_loss
                + tv_weight * self.total_variation_loss(target_img)
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                target_img.clamp_(0, 1)
            if epoch_idx % 100 == 0:
                print(f"Epoch {epoch_idx} | Total loss: {total_loss.item():.2f}")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        pic = content_img.cpu().squeeze().numpy().transpose((1, 2, 0))
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
        axes[0].imshow(pic)
        axes[0].set_title("Content picture")
        pic = style_img.cpu().squeeze().numpy().transpose((1, 2, 0))
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
        axes[1].imshow(pic)
        axes[1].set_title("Style picture")
        pic = target_img.detach().cpu().squeeze().numpy().transpose((1, 2, 0))
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
        axes[2].imshow(pic)
        axes[2].set_title("Target picture")
        plt.tight_layout()
        plt.show()
        self.save_current_plot(filename="style_transfer_after.png")
