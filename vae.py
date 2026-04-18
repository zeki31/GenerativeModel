import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(
            input_dim, hidden_dim
        )  # Linear has W and b as parameters
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        # Output logvar output (1/2)log(sigma^2) instead of outputting sigma directly to ensure sigma is positive
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)  # Output in [0, 1]
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)  # Sample from standard normal
        z = mu + sigma * eps  # Reparameterization trick
        return z

    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z)

        batch_size = len(x)
        recon_loss = F.mse_loss(x_hat, x, reduction="sum")
        kl_loss = -torch.sum(
            1 + torch.log(sigma**2) - mu**2 - sigma**2
        )  # KL divergence between q(z|x) and p(z)
        return (recon_loss + kl_loss) / batch_size


if __name__ == "__main__":
    # Hyperparameters
    input_dim = 28 * 28  # Flattened MNIST images
    hidden_dim = 200
    latent_dim = 20  # latent vector z
    epochs = 30
    learning_rate = 3e-4
    batch_size = 32

    # Dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                torch.flatten
            ),  # Flatten the 28x28 image into a 784-dimensional vector
        ]
    )
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Model, optimizer
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    # Training loop
    pbar = tqdm(range(epochs), desc="Training VAE")
    for epoch in pbar:
        loss_sum = 0
        cnt = 0
        for x, label in dataloader:
            optimizer.zero_grad()
            loss = model.get_loss(x)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        loss_avg = loss_sum / cnt
        losses.append(loss_avg)

        pbar.set_postfix({"loss_avg": loss_avg})
        pbar.update(1)

    # Plot training loss
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss")
    plt.show()

    # Generate new samples
    with torch.no_grad():
        sample_size = 64
        z = torch.randn(sample_size, latent_dim)  # Sample from standard normal
        x = model.decoder(z)  # Decode to get generated samples
        generated_images = x.view(sample_size, 1, 28, 28)  # Reshape to image format
    # Visualize generated samples
    grid_img = torchvision.utils.make_grid(
        generated_images, nrow=8, padding=2, normalize=True
    )
    plt.imshow(grid_img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    plt.axis("off")
    plt.title("Generated MNIST Samples")
    plt.show()
