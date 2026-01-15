import matplotlib.pyplot as plt
import os
import torch

from sklearn.datasets import load_sample_image


class Main:

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
    def q_sample(
        x0: torch.Tensor, t: torch.Tensor, betas: torch.Tensor, eps=None, seed=123
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q: true posterior distribution
        x0: clean image
        t: final timestep batches
        betas: small noise schedules
        eps: noise
        """
        torch.manual_seed(seed)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bar_T = alpha_bars[t - 1].view(x0.shape[0], 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x0, dtype=torch.float)
        """
        markov chain: x0 -> x1 -> ... -> xT
        xi = sqrt_alpha_i * xi-1 + sqrt_one_minus_alpha_i * eps_i
        """
        xT = torch.sqrt(alpha_bar_T) * x0 + torch.sqrt(1 - alpha_bar_T) * eps
        return xT, eps

    @staticmethod
    def main():
        img = load_sample_image("china.jpg")
        plt.imshow(img)
        plt.axis("off")
        Main.save_current_plot(filename="china_x0.png")
        B, C, H, W = 1, img.shape[2], img.shape[0], img.shape[1]
        x0 = torch.tensor(img).reshape(B, C, H, W)
        T = 1000
        t = torch.tensor([T]).expand(B, -1)  # .repeat(B)
        betas = torch.linspace(1e-4, 0.02, T)
        xT = Main.q_sample(x0, t, betas)[0]
        xT = xT.detach().cpu()[0, :, :, :]
        """
        min-max scaling
        """
        xT = (xT - torch.min(xT)) / (torch.max(xT) - torch.min(xT))
        plt.imshow(xT.view(H, W, C))
        plt.axis("off")
        Main.save_current_plot(filename="china_xT.png")


if __name__ == "__main__":

    Main.main()
