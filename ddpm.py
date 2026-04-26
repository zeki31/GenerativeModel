import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from loguru import logger
from torch.nn import functional as F
from tqdm import tqdm

from unet import UNet


class Diffuser:
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device="cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all(), f"t should be in [1, {T}]"
        t_idx = t - 1  # t starts from 1, but index starts from 0

        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        eps = torch.randn_like(
            x_0, device=self.device
        )  # gaussian noise whose shape is the same as x_0
        x_t = (
            torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps
        )  # add noise to x_0
        return x_t, eps

    def sample(self, model, x_shape=(20, 1, 28, 28), labels=None):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)  # start from pure noise

        for i in tqdm(range(self.num_timesteps, 0, -1), desc="Sampling"):
            t = torch.tensor(
                [i] * batch_size, device=self.device, dtype=torch.long
            )  # create a tensor of shape (batch_size,) filled with the current timestep
            x = self.denoise(model, x, t, labels)

        images = [self._reverse_to_img(x[i]) for i in range(batch_size)]
        return images

    def denoise(self, model, x, t, labels):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all(), f"t should be in [1, {T}]"

        t_idx = t - 1  # t starts from 1, but index starts from 0
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx - 1]

        N = alpha_bar.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            eps = model(x, t, labels)  # predict noise using the model
        model.train()  # set model back to training mode

        noise = torch.randn_like(
            x, device=self.device
        )  # gaussian noise whose shape is the same as x
        noise[t == 1] = 0  # no noise when t=1

        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(
            alpha
        )  # mean of the posterior
        std = torch.sqrt(
            (1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
        )  # standard deviation of the posterior
        return mu + noise * std

    def _reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def show_images(self, images, rows=2, cols=10):
        fig = plt.figure(figsize=(cols, rows))
        i = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, i + 1)
                plt.imshow(images[i], cmap="gray")
                plt.axis("off")
                i += 1
        plt.show()


if __name__ == "__main__":
    img_size = 28
    batch_size = 128
    num_timesteps = 1000
    ephochs = 10
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    preprocess = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(
        root="./data", transform=preprocess, download=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    diffuser = Diffuser(num_timesteps=num_timesteps, device=device)
    model = UNet(num_labels=10)  # Assuming 10 classes for MNIST
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(ephochs):
        loss_sum = 0
        cnt = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{ephochs}"):
            optimizer.zero_grad()
            x_0s = images.to(device)
            ts = torch.randint(
                1, num_timesteps + 1, (len(x_0s),), device=device
            )  # random timesteps for each image in the batch
            labels = labels.to(device)  # Move labels to the same device as the model

            x_ts, noise = diffuser.add_noise(x_0s, ts)
            noise_pred = model(x_ts, ts, labels)  # predict noise using the model
            loss = F.mse_loss(noise_pred, noise)  # calculate mean squared error loss

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        logger.info(f"Epoch {epoch + 1}/{ephochs}, Loss: {loss_avg:.4f}")

    # Plot training loss
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Sample images from the trained model
    images = diffuser.sample(model)
    diffuser.show_images(images)
