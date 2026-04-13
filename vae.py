import os
import time
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from datasets import MulticlassDataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


Loss = NamedTuple("Loss", [("value", torch.Tensor), ("bce", torch.Tensor), ("kl", torch.Tensor)])
Reconstruction = NamedTuple("Reconstruction", [("output", torch.Tensor),
                            ("z", torch.Tensor), ("mu", torch.Tensor), ("log_var", torch.Tensor)])
Snapshot = NamedTuple("Snapshot", [("epoch", int), ("bce", float), ("kl", float)])


def _load_results(path):
    return torch.load(path, weights_only=False)


class VAE(nn.Module):
    def __init__(self, channels, latent_size):
        super(VAE, self).__init__()

        in_channels = channels
        self.latent_size = latent_size

        modules = []
        hidden_dims = [32, 64, 128, 256]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_size)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_size)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_size, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> Reconstruction:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return Reconstruction(self.decode(z), z, mu, log_var)

    def loss_function(self, input: torch.Tensor, recons: Reconstruction, kld_weight: float, epsilon=1e-7) -> Loss:
        bce_loss = torch.flatten(F.binary_cross_entropy(recons.output, input, reduction='none'), 1)
        bce_loss = torch.mean(torch.sum(bce_loss, dim=1), dim=0)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + recons.log_var - recons.mu ** 2 - recons.log_var.exp(), dim=1), dim=0)

        loss = bce_loss + kl_loss
        return Loss(loss, bce_loss.detach(), kl_loss.detach())

    def sample(self, num_samples: int, device: torch.device, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_size)
        z = z.to(device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]


def train_model(dataset: MulticlassDataset, net: VAE, num_epochs=30, batch_size=64, device="cpu", kld_weight=0.00025):
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    snapshots = []
    train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=batch_size, shuffle=False)
    for epoch in range(1, num_epochs + 1):
        print('epoch', epoch)

        # training
        net.train()
        sum_recon = 0
        sum_kl = 0
        total = 0
        start = time.time()
        for x, t in tqdm(train_loader):
            x = x.to(device)
            t = t.to(device)

            optimizer.zero_grad()
            r = net(x)
            loss = net.loss_function(x, r, kld_weight)
            loss.value.backward()
            optimizer.step()

            sum_recon += loss.bce * len(t)
            sum_kl += loss.kl * len(t)
            total += len(t)

        end = time.time()
        elapsed_time = end - start
        throughput = total / elapsed_time
        print('train mean bce={}, kl={}, throughput={} images/sec'.format(
            sum_recon / total, sum_kl / total, throughput))

        # evaluation
        net.eval()
        sum_recon = 0
        sum_kl = 0
        total = 0
        for x, t in val_loader:
            x = x.to(device)
            t = t.to(device)
            r = net(x)
            loss = net.loss_function(x, r, kld_weight)
            sum_recon += loss.bce * len(t)
            sum_kl += loss.kl * len(t)
            total += len(t)

        print('validation  mean bce={}, kl={}'.format(sum_recon / total, sum_kl / total))
        snapshots.append(Snapshot(epoch, sum_recon / total, sum_kl / total))
        scheduler.step()

    return snapshots


def evaluate_model(dataset: MulticlassDataset, net: VAE, device="cpu"):
    num_examples = 4
    col_width = 4
    fig = plt.figure(figsize=(num_examples * col_width, 11))
    gs = plt.GridSpec(3, num_examples, height_ratios=[col_width, 3, col_width])

    num_coded = 2000
    examples, labels = dataset.val[0:num_coded]
    labels = labels.reshape(-1).cpu().detach().numpy()

    net.to(device)
    net.eval()
    x = examples.to(device)
    r = net(x)
    coded = r.z.cpu().detach().numpy()
    for i in range(num_examples):
        ax = fig.add_subplot(gs[0, i])
        plt.imshow(x[i].cpu().detach().numpy().reshape(32, 32), cmap='gray', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, i])
        z = r.z[i].cpu().detach().numpy()
        ax.scatter(coded[:, 0], coded[:, 1], marker='.', c=labels, cmap="jet", sizes=[4]*num_coded)
        ax.plot(z[0], z[1], marker='s', c="magenta", markersize=15)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[2, i])
        plt.imshow(r.output[i].cpu().detach().numpy().reshape(32, 32), cmap='gray', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def sample_model(dataset: MulticlassDataset, net: VAE, device="cpu"):
    net.to(device)
    net.eval()

    num_coded = 2000
    examples, labels = dataset.train[0:num_coded]
    labels = labels.reshape(-1).cpu().detach().numpy()
    x = examples.to(device)
    r = net(x)
    coded = r.z.cpu().detach().numpy()

    rows = 3
    columns = 6
    # do some radial sampling
    r = torch.linspace(.1, 2.5, rows * columns)
    theta = torch.linspace(0, 4 * 3.1415, rows * columns)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.stack([x, y], dim=1)
    z = z.to(device)
    samples = net.decode(z)

    size = 2
    fig = plt.figure(figsize=(columns * size + rows * size, rows * size))
    gs = plt.GridSpec(rows, columns + rows)

    z = z.cpu().detach().numpy()
    ax = fig.add_subplot(gs[:, :rows])
    ax.scatter(coded[:, 0], coded[:, 1], marker='.', c=labels, cmap="jet", sizes=[16]*num_coded)
    ax.plot(z[:, 0], z[:, 1], marker='s', c="magenta", markersize=8)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(rows):
        for j in range(columns):
            ax = fig.add_subplot(gs[i, j + rows])
            plt.imshow(samples[i * columns + j].cpu().detach().numpy().reshape(32, 32),
                       cmap='gray', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def display_image(ax, x, label, vrange=None):
    assert len(x.shape) == 2, str(x.shape)

    if vrange is None:
        vmin = np.min(x)
        vmax = np.max(x)
    else:
        vmin, vmax = vrange

    ax.imshow(x, interpolation='nearest', cmap='gray', extent=[0, x.shape[1], x.shape[0], 0], vmin=vmin, vmax=vmax)
    ax.grid(True)
    ax.set_xticks(np.arange(x.shape[1]))
    ax.xaxis.set_ticklabels([])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(x.shape[0]))
    ax.yaxis.set_ticklabels([])
    ax.set_xlabel(label)


alien = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='float32')


def train_transposed_convolution():
    image = torch.from_numpy(2 * alien - 1).unsqueeze(0).unsqueeze(0)
    conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, stride=2, bias=False, padding_mode="reflect")
    tconv = nn.ConvTranspose2d(2, 1, kernel_size=3, padding=1, stride=2, bias=False)
    module = nn.Sequential(conv, tconv, nn.Tanh())

    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

    for i in range(5000):
        optimizer.zero_grad()
        o = module(image)
        loss = F.mse_loss(o, image)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(i, loss.item())

    c = conv.weight[:, 0].detach().numpy()
    t = tconv.weight[:, 0].detach().numpy()
    z = conv(image).detach().squeeze().numpy()
    result = module(image).detach().squeeze().numpy()
    return {"c": c, "t": t, "z": z, "result": result}


def show_transposed_convolution(data):
    c = data["c"]
    t = data["t"]
    z = data["z"]
    result = data["result"]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(2, 3, width_ratios=[13, 3, 7])

    display_image(fig.add_subplot(gs[:, 0]), alien, "Input")
    display_image(fig.add_subplot(gs[0, 1]), c[0], "C0", (c.min(), c.max()))
    display_image(fig.add_subplot(gs[1, 1]), c[1], "C1", (c.min(), c.max()))
    display_image(fig.add_subplot(gs[0, 2]), z[0], "Z0", (z.min(), z.max()))
    display_image(fig.add_subplot(gs[1, 2]), z[1], "Z1", (z.min(), z.max()))
    fig.tight_layout()

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(2, 3, width_ratios=[7, 3, 13])

    display_image(fig.add_subplot(gs[0, 0]), z[0], "Z0", (z.min(), z.max()))
    display_image(fig.add_subplot(gs[1, 0]), z[1], "Z1", (z.min(), z.max()))
    display_image(fig.add_subplot(gs[0, 1]), t[0], "T0", (t.min(), t.max()))
    display_image(fig.add_subplot(gs[1, 1]), t[1], "T1", (t.min(), t.max()))
    display_image(fig.add_subplot(gs[:, 2]), result, "Output")
    fig.tight_layout()

    plt.show()


def train_mnist():
    dataset = MulticlassDataset.mnist(as_images=32).to_torch()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = VAE(1, 2)
    path = os.path.join(RESULTS_DIR, "vae.results")

    if os.path.exists(path):
        print(path, "already exists, skipping training")
        return

    snapshots = train_model(dataset, net, device=device)
    results = {"snapshots": snapshots, "net": net.state_dict()}
    torch.save(results, path)
    print(f"Saved {path}")


def load_mnist():
    path = os.path.join(RESULTS_DIR, "vae.results")
    return _load_results(path)


def show_mnist(results):
    dataset = MulticlassDataset.mnist(as_images=32).to_torch()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = VAE(1, 2)
    net.load_state_dict(results["net"])
    evaluate_model(dataset, net, device=device)


def show_mnist_samples(results):
    dataset = MulticlassDataset.mnist(as_images=32).to_torch()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = VAE(1, 2)
    net.load_state_dict(results["net"])
    sample_model(dataset, net, device=device)


def latent_slider_data(results, num_steps=60):
    """Precompute latent space traversal data for an interactive slider."""
    dataset = MulticlassDataset.mnist(as_images=32).to_torch()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = VAE(1, 2)
    net.load_state_dict(results["net"])
    net.to(device)
    net.eval()

    num_coded = 2000
    examples, labels = dataset.train[0:num_coded]
    labels = labels.reshape(-1).cpu().detach().numpy()
    x = examples.to(device)
    r = net(x)
    coded = r.z.cpu().detach().numpy()

    # radial spiral through latent space
    r_vals = torch.linspace(0.1, 2.5, num_steps)
    theta = torch.linspace(0, 4 * 3.1415, num_steps)
    zx = r_vals * torch.cos(theta)
    zy = r_vals * torch.sin(theta)
    z = torch.stack([zx, zy], dim=1).to(device)

    with torch.no_grad():
        samples = net.decode(z).cpu().numpy()

    z = z.cpu().numpy()
    return coded, labels, z, samples


def plot_latent_step(fig, coded, labels, z, samples, step):
    """Plot a single frame of the VAE latent space traversal."""
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(coded[:, 0], coded[:, 1], marker='.', c=labels, cmap="jet", sizes=[4]*len(labels))
    ax.plot(z[:step+1, 0], z[:step+1, 1], 'k-', alpha=0.3)
    ax.plot(z[step, 0], z[step, 1], marker='s', c="magenta", markersize=15, zorder=5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(samples[step].reshape(32, 32), cmap='gray', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()


if __name__ == "__main__":
    # Training
    train_transposed_convolution_data = train_transposed_convolution()
    train_mnist()

    # Visualization
    show_transposed_convolution(train_transposed_convolution_data)

    vae_results = load_mnist()
    show_mnist(vae_results)
    show_mnist_samples(vae_results)
