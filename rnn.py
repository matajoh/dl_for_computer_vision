import copy
import os
import pickle
import sys
from typing import List, NamedTuple, Tuple

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MulticlassDataset, SequenceDataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class ElmanRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_classes: int = None, num_layers=1,
                 dropout=0.1, tokens_output=None):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0)

        self.h0 = nn.Parameter(torch.randn((num_layers, 1, hidden_size), dtype=torch.float32))

        if tokens_output is None:
            self.tokens_output = nn.Linear(hidden_size, input_size)
        else:
            self.tokens_output = tokens_output

        self.class_output = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.keep_hiddens = False
        self.hiddens = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(1)
        h = self.h0.repeat(1, batch_size, 1)

        outputs, h = self.rnn(x, h)

        if self.keep_hiddens:
            self.hiddens = outputs.detach().cpu()

        tokens_output = self.tokens_output(self.dropout(outputs))
        class_output = self.class_output(self.dropout(h))[-1]
        return class_output, tokens_output


class BernoulliMixture(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_components: int):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.num_components = num_components
        self.sampling = False
        self.keep_dist = False
        self.dist = None

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2z = nn.Linear(hidden_size, num_components)
        self.h2d = nn.Linear(hidden_size, num_components * output_size)

    def forward(self, input: torch.Tensor):
        h = F.leaky_relu(self.i2h(input))
        z = F.softmax(self.h2z(h), -1)
        d = F.sigmoid(self.h2d(h))
        batch_size = input.size(0)
        length = input.size(1)
        d = d.view(batch_size, length, self.num_components, self.output_size)
        if self.sampling:
            z = z.reshape(-1, self.num_components)
            z = torch.multinomial(z, 1, replacement=True).reshape(-1)
            z = z.reshape(batch_size, length)
            p = torch.zeros(batch_size, length, self.output_size, device=d.device)
            for i in range(batch_size):
                for j in range(length):
                    p[i, j] = d[i, j, z[i, j]]

            return p

        if self.keep_dist:
            self.dist = (z.detach().cpu(), d.detach().cpu())

        z = z.unsqueeze(-1)
        return (z * d).sum(-2)


class PatchDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.h2z = nn.Linear(input_size, 2 * latent_size)
        self.z2d = nn.Linear(latent_size, hidden_size * 4)
        self.d2o = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_size, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, input: torch.Tensor):
        length, batch_size = input.shape[:2]
        input = input.view(-1, self.input_size)
        mu, logvar = torch.chunk(self.h2z(input), 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        result = self.z2d(z)
        result = result.view(-1, self.hidden_size, 2, 2)
        result = self.d2o(result)
        result = result.view(length, batch_size, -1)
        return result


def compute_accuracy(output, expected_labels):
    _, actual_labels = torch.max(output, -1)
    num_correct = (actual_labels == expected_labels).sum().item()
    total = len(actual_labels)
    return num_correct / total


Snapshot = NamedTuple("Snapshot", [("step", int), ("accuracy", float), ("loss", float)])


def _load_results(path):
    _current_module = sys.modules[__name__]

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__":
                obj = getattr(_current_module, name, None)
                if obj is not None:
                    return obj
            return super().find_class(module, name)

    _pickle_module = type(sys)('_pickle_compat')
    _pickle_module.Unpickler = _Unpickler
    _pickle_module.load = pickle.load
    _pickle_module.UnpicklingError = pickle.UnpicklingError
    return torch.load(path, weights_only=False, pickle_module=_pickle_module)


def train_rnn(dataset: SequenceDataset, net: ElmanRNN, criterion, self_criterion,
              batch_size, num_epochs, device="cpu") -> Tuple[dict, List[Snapshot]]:
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    snapshots = []
    best_model = None
    best_loss = float('inf')

    train_loader = DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset.val, batch_size=batch_size, shuffle=False)
    for epoch in range(1, num_epochs + 1):
        net.train()
        sum_accuracy = 0
        sum_loss = 0
        total = 0

        def compute_loss_and_accuracy(x: torch.Tensor, t: torch.Tensor):
            x = x.transpose(0, 1)
            if len(x.shape) == 2:
                x_input = F.one_hot(x, net.input_size).float()
            else:
                x_input = x

            optimizer.zero_grad()
            y, tokens = net(x_input)
            loss = criterion(y, t)
            accuracy = compute_accuracy(y, t)

            y = tokens[:-1].reshape(-1, net.input_size)

            if len(x.shape) == 2:
                t = x[1:].reshape(-1)
            else:
                t = x[1:].reshape(-1, x.size(-1))

            loss += self_criterion(y, t)

            return loss, accuracy

        with tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}') as pbar:
            for x, t in pbar:
                x = x.to(device)
                t = t.to(device)

                loss, accuracy = compute_loss_and_accuracy(x, t)

                loss.backward()
                optimizer.step()

                sum_loss += loss.item() * len(t)
                sum_accuracy += accuracy * len(t)
                total += len(t)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

        print('train mean loss={}, accuracy={}'.format(
            sum_loss / total, sum_accuracy / total))

        net.eval()
        sum_accuracy = 0
        sum_loss = 0
        total = 0
        for x, t in val_loader:
            x = x.to(device)
            t = t.to(device)

            loss, accuracy = compute_loss_and_accuracy(x, t)

            sum_loss += loss.item() * len(t)
            sum_accuracy += accuracy * len(t)
            total += len(t)

        if sum_loss < best_loss:
            best_loss = sum_loss
            net = net.to("cpu")
            best_model = copy.deepcopy(net.state_dict())
            net = net.to(device)

        print('validation  mean loss={}, accuracy={}'.format(sum_loss / total, sum_accuracy / total))
        snapshots.append(Snapshot(epoch, sum_accuracy / total, sum_loss / total))

    return best_model, snapshots


def sample(dataset: SequenceDataset, net: ElmanRNN, num_samples: int, temperature=1):
    net = net.to("cpu")
    net.eval()

    prefix = 1
    samples = [torch.tensor(t) for t in dataset.random_tokens(num_samples, prefix)]
    for _ in range(dataset.max_length - prefix):
        x = torch.stack(samples)
        x = x.transpose(0, 1)
        if dataset.tokens:
            x_input = F.one_hot(x, net.input_size).float()
        else:
            x_input = x

        with torch.no_grad():
            _, tokens = net(x_input)

        next_samples = []
        for seq, o in zip(samples, tokens[-1]):
            t = dataset.to_token(o, temperature=temperature)
            next_samples.append(torch.cat([seq, t]))

        samples = next_samples

    return samples


def assemble(patches: np.ndarray):
    image = np.zeros((28, 28), np.uint8)
    for i, patch in enumerate(patches):
        r = (i // 7) * 4
        c = (i % 7) * 4
        if isinstance(patch, torch.Tensor):
            patch = patch.cpu().numpy()

        image[r:r+4, c:c+4] = patch.reshape(4, 4) * 255

    return image


def evaluate_text_model(dataset: SequenceDataset, net: nn.Module, snapshots: List[Snapshot], device="cpu"):
    label_names = dataset.label_names
    num_examples = 12
    columns = 3

    output = []

    net.to(device)
    net.eval()
    val = DataLoader(dataset.val, batch_size=256, shuffle=False)
    for x, _ in tqdm(val):
        x = x.to(device)
        x = x.transpose(0, 1)
        x_input = F.one_hot(x, net.input_size).float()

        with torch.no_grad():
            y, _ = net(x_input)

        output.append(F.softmax(y, dim=1))

    sequences, labels = dataset.val.tensors
    sequences = sequences.cpu().numpy()
    labels = labels.cpu().numpy()
    output = torch.cat(output).cpu().numpy()

    predicted = np.argmax(output, axis=1)
    correct_index = predicted == labels
    incorrect_index = predicted != labels

    correct = output[correct_index]
    correct_sequences = sequences[correct_index]
    correct_labels = labels[correct_index]
    correct_conf = correct[np.arange(len(correct_labels)), correct_labels]

    incorrect = output[incorrect_index]
    incorrect_sequences = sequences[incorrect_index]
    incorrect_labels = labels[incorrect_index]
    incorrect_conf = incorrect[np.arange(len(incorrect_labels)), incorrect_labels]

    correct_order = np.argsort(correct_conf)[::-1]
    incorrect_order = np.argsort(incorrect_conf)[::-1]

    correct = correct[correct_order]
    correct_sequences = correct_sequences[correct_order]
    correct_labels = correct_labels[correct_order]
    incorrect = incorrect[incorrect_order]
    incorrect_sequences = incorrect_sequences[incorrect_order]
    incorrect_labels = incorrect_labels[incorrect_order]

    examples = np.concatenate([correct_sequences[:3], correct_sequences[-3:],
                               incorrect_sequences[:3], incorrect_sequences[-3:]])
    example_labels = np.concatenate([correct_labels[:3], correct_labels[-3:],
                                     incorrect_labels[:3], incorrect_labels[-3:]])
    output = np.concatenate([correct[:3], correct[-3:], incorrect[:3], incorrect[-3:]])
    top_labels = np.argpartition(output, -2)[:, -2:]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(10, 4))
    gs0 = gridspec.GridSpec(1, 2, wspace=.3)
    gs00 = gridspec.GridSpecFromSubplotSpec(num_examples // columns, columns, gs0[0], hspace=2.5)
    gs0.update(left=0.05, right=0.95, bottom=0.1, top=0.99)

    for i in range(num_examples):
        row = i // columns
        col = i % columns
        ax = fig.add_subplot(gs00[row, col])
        ax.patch.set_alpha(0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        chars = [dataset.tokens[t] for t in examples[i]]
        example = "".join(chars)
        if top_labels[i, 1] == example_labels[i]:
            color = "green"
        else:
            color = "red"
        ax.text(0, 0, example, fontsize=15, color=color, fontweight="bold")
        certainty1, certainty0 = output[i, top_labels[i]]
        text1 = label_names[top_labels[i, 0]]
        text0 = label_names[top_labels[i, 1]]
        ax.set_xlabel("{} ({})\n{} ({})".format(text0[:3], round(float(certainty0), 2),
                                                text1[:3], round(float(certainty1), 2)))
        ax.set_xticks([])
        ax.set_yticks([])

    end_index = 0
    for i, snapshot in enumerate(snapshots):
        if snapshot.loss < snapshots[end_index].loss:
            end_index = i

    print("best snapshot", snapshots[end_index])

    ax1 = fig.add_subplot(gs0[1])
    x = [snapshots[i][0] for i in range(end_index)]
    y1 = [snapshots[i][1] for i in range(end_index)]
    y2 = [snapshots[i][2] for i in range(end_index)]

    acc_line = ax1.plot(x, y1, 'b-')
    ax1.set_xticks([])
    ax1.set_ylim(0, 1.1)
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()

    loss_line = ax2.plot(x, y2, 'g--')
    ax2.tick_params('y', colors='g')

    ax2.legend(acc_line + loss_line, ["Accuracy", "Loss"], loc="center right")
    plt.show()


def evaluate_image_model(dataset: SequenceDataset, net: nn.Module, snapshots: List[Snapshot], device="cpu"):
    label_names = dataset.label_names
    num_examples = 6
    examples, example_labels = dataset.val[0:num_examples]
    columns = num_examples // 2

    net.to(device)
    net.eval()
    x = examples.to(device)
    x = x.transpose(0, 1)
    output = F.softmax(net(x)[0], dim=1).cpu().detach().numpy()
    top_labels = np.argpartition(output, -2)[:, -2:]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(10, 5))
    gs0 = gridspec.GridSpec(1, 2, wspace=.3)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, columns, gs0[0])
    gs0.update(left=0.01, right=0.94, bottom=0.05, top=0.95)

    for i in range(num_examples):
        row = i // columns
        col = i % columns
        ax = fig.add_subplot(gs00[row, col])
        example = assemble(examples[i].cpu().numpy())
        ax.imshow(example, cmap='gray', interpolation='nearest')

        if example_labels[i] != top_labels[i][1]:
            for spine in ax.spines.values():
                spine.set_color('red')
                spine.set_linewidth(2)

        certainty1, certainty0 = output[i, top_labels[i]]
        text1 = label_names[top_labels[i, 0]]
        text0 = label_names[top_labels[i, 1]]
        ax.set_xlabel("{} ({})\n{} ({})".format(text0, round(float(certainty0), 2), text1, round(float(certainty1), 2)))
        ax.set_xticks([])
        ax.set_yticks([])

    end_index = 0
    for i, snapshot in enumerate(snapshots):
        if snapshot.loss < snapshots[end_index].loss:
            end_index = i

    ax1 = fig.add_subplot(gs0[1])
    x = [snapshots[i][0] for i in range(end_index)]
    y1 = [snapshots[i][1] for i in range(end_index)]
    y2 = [snapshots[i][2] for i in range(end_index)]

    acc_line = ax1.plot(x, y1, 'b-')
    ax1.set_xticks([])
    ax1.set_ylim(0, 1.1)
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()

    loss_line = ax2.plot(x, y2, 'g--')
    ax2.tick_params('y', colors='g')

    ax2.legend(acc_line + loss_line, ["Accuracy", "Loss"], loc="center right")
    plt.show()


def vanishing_gradient(word: str):
    length = len(word)
    sequence = torch.tensor([ord(c) - ord('a') for c in word]).unsqueeze(1)
    rnn_model = nn.RNN(26, 128, 1)
    fc = nn.Linear(128, 2)
    h2o = nn.Linear(128, 26)
    x = F.one_hot(sequence, 26).float().requires_grad_(True)

    h0 = torch.zeros(1, 1, 128)
    _, h = rnn_model(x, h0)
    h = h.squeeze(1)
    criterion = nn.CrossEntropyLoss()
    t = torch.tensor(1, dtype=torch.long).unsqueeze(0)
    y = fc(h)
    loss = criterion(y, t)
    loss.backward()

    grad_mag_no_ss = [x_i.sum().abs().item() for x_i in x.grad]

    x.grad = None

    o, h = rnn_model(x, h0)
    o = o.squeeze(1)
    h = h.squeeze(1)
    tokens = h2o(o)
    y = fc(h)
    loss = criterion(y, t)
    y = tokens[:-1].reshape(-1, 26)
    t_ss = sequence[1:].reshape(-1)
    loss += criterion(y, t_ss)
    loss.backward()

    grad_mag_ss = [x_i.sum().abs().item() for x_i in x.grad]

    plt.rc('font', size=15)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(range(length), grad_mag_no_ss, c="black", linestyle='dashed')
    ax1.set_xlim(0, length - 1)
    ax1.set_xlabel("Input position")
    ax1.set_xticks(list(range(length)), list(word))
    ax1.grid(visible=True)
    ax1.set_ylabel("Gradient Magnitude")
    ax1.set_yscale("log")
    ax1.set_title("Without self-supervision")

    ax2.plot(range(length), grad_mag_ss, c="black", linestyle='dashed')
    ax2.set_xlim(0, length - 1)
    ax2.set_xlabel("Input position")
    ax2.set_xticks(list(range(length)), list(word))
    ax2.grid(visible=True)
    ax2.set_ylabel("Gradient Magnitude")
    ax2.set_yscale("log")
    ax2.set_title("With self-supervision")

    fig.tight_layout()
    plt.show()


def train_language_rnn(num_hidden=512, length=6):
    dataset = SequenceDataset.language(length, 990).to_torch()
    rnn_net = ElmanRNN(dataset.input_size, num_hidden, dataset.num_classes)
    path = os.path.join(RESULTS_DIR, f"rnn_language_{length}_{num_hidden}.results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(path):
        print(path, "already exists, skipping training")
        return
    print("training language classifier")
    best_model, snapshots = train_rnn(dataset, rnn_net, nn.CrossEntropyLoss(),
                                      nn.CrossEntropyLoss(),
                                      128, 400, device=device)
    results = {"snapshots": snapshots, "net": best_model,
               "num_hidden": num_hidden, "length": length}
    torch.save(results, path)
    print(f"Saved {path}")


def load_language_rnn(num_hidden=512, length=6):
    path = os.path.join(RESULTS_DIR, f"rnn_language_{length}_{num_hidden}.results")
    return _load_results(path)


def show_language_rnn(results):
    num_hidden = results["num_hidden"]
    length = results["length"]
    dataset = SequenceDataset.language(length, 990).to_torch()
    rnn_net = ElmanRNN(dataset.input_size, num_hidden, dataset.num_classes)
    rnn_net.load_state_dict(results["net"])
    print(results["snapshots"][-1])

    print("sampling")
    samples = sample(dataset, rnn_net, 20)
    for s in samples:
        chars = [dataset.tokens[t] for t in s]
        print("".join(chars))

    evaluate_text_model(dataset, rnn_net, results["snapshots"])


def language_rnn_slider_data(results, num_examples=20):
    """Precompute per-timestep hidden states and outputs for interactive visualization."""
    num_hidden = results["num_hidden"]
    length = results["length"]
    dataset = SequenceDataset.language(length, 990).to_torch()
    net = ElmanRNN(dataset.input_size, num_hidden, dataset.num_classes)
    net.load_state_dict(results["net"])
    net.to("cpu")
    net.eval()

    import random
    random.seed(42)
    indices = random.sample(range(len(dataset.train)), num_examples)

    all_frames = []
    for i in indices:
        word_tensor = dataset.train[i][0]
        word = [dataset.tokens[c.item()] for c in word_tensor]
        label = dataset.train[i][1].item()

        x = word_tensor.unsqueeze(1)
        x_input = F.one_hot(x, net.input_size).float()

        class_outputs = []
        net.keep_hiddens = True
        for j in range(1, length + 1):
            with torch.no_grad():
                output, tokens = net(x_input[:j])
            class_outputs.append(F.softmax(output, -1).cpu().numpy()[0])

        hiddens = net.hiddens.squeeze(1).cpu().numpy()
        token_outputs = F.softmax(tokens, -1).squeeze(1).cpu().numpy()

        for j in range(length):
            all_frames.append({
                "word": word[:j+1],
                "label": label,
                "hidden": hiddens[j],
                "token_output": token_outputs[j],
                "class_output": class_outputs[j],
            })

    return dataset, all_frames


def plot_rnn_frame(fig, dataset, frame):
    """Plot a single frame of the RNN language visualization."""
    word = frame["word"]
    label = frame["label"]
    h = frame["hidden"]
    y = frame["token_output"]
    o = frame["class_output"]

    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.2, hspace=0.5)

    # Top-p token predictions
    top_p = 0.95
    ax = fig.add_subplot(gs[0, 0])
    top_labels = np.argsort(y)[::-1]
    num_predict = 0
    p = 0
    while p < top_p and num_predict < len(top_labels):
        p += y[top_labels[num_predict]]
        num_predict += 1
    top_labels = top_labels[:num_predict]
    xticks = np.arange(num_predict)
    xlabels = [dataset.tokens[t] for t in top_labels]
    ax.bar(xticks, y[top_labels])
    ax.set_xticks(xticks, xlabels)
    ax.set_xlabel("y")
    ax.set_yticks([])

    # Hidden state
    num_hidden = len(h)
    ax = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("viridis")
    colors = cmap(0.5 * (h + 1))
    ax.bar(np.arange(num_hidden), h, width=1, color=colors)
    ax.set_xticks([0, num_hidden], [0, num_hidden - 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_title("h")

    # Class output
    label_names = dataset.label_names
    ax = fig.add_subplot(gs[0, 2])
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(label_names)]
    positions = [1.75 + 0.5*i for i in range(len(label_names))]
    ax.bar(positions, o, width=0.5, color=colors_bar)
    ax.set_xticks(positions, [n[:1] for n in label_names])
    ax.set_yticks([])
    ax.set_title("o")

    # Word display
    ax = fig.add_subplot(gs[1, :2])
    for i, c in enumerate(word):
        ax.text(i, 0, c, fontsize=72, fontfamily="monospace")
    ax.set_xlim(-0.2, 6)
    ax.set_axis_off()

    # Ground truth
    gt = np.zeros(dataset.num_classes)
    gt[label] = 1
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(positions, gt, width=0.5, color=colors_bar)
    ax.set_xticks(positions, [n[:1] for n in label_names])
    ax.set_yticks([])

    fig.tight_layout()


def train_emnist_rnn(num_hidden=128, tokens=False, alpha_num=False):
    dataset = SequenceDataset.emnist(tokens, alpha_num).to_torch()
    self_criterion = nn.CrossEntropyLoss() if tokens else nn.BCELoss()
    tokens_output = None if tokens else BernoulliMixture(num_hidden, num_hidden * 4, 16, 4)
    rnn_net = ElmanRNN(dataset.input_size, num_hidden, dataset.num_classes, tokens_output=tokens_output)
    path = os.path.join(RESULTS_DIR,
                        f"rnn_emnist_{'alnum' if alpha_num else ''}{'token' if tokens else 'patch'}_{num_hidden}.results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(path):
        print(path, "already exists, skipping training")
        return
    print("training emnist classifier")
    best_model, snapshots = train_rnn(dataset, rnn_net, nn.CrossEntropyLoss(), self_criterion,
                                      512, 400, device=device)
    results = {"snapshots": snapshots, "net": best_model,
               "num_hidden": num_hidden, "tokens": tokens, "alpha_num": alpha_num}
    torch.save(results, path)
    print(f"Saved {path}")


def load_emnist_rnn(num_hidden=128, tokens=False, alpha_num=False):
    path = os.path.join(RESULTS_DIR,
                        f"rnn_emnist_{'alnum' if alpha_num else ''}{'token' if tokens else 'patch'}_{num_hidden}.results")
    return _load_results(path)


def show_emnist_rnn(results):
    num_hidden = results["num_hidden"]
    tokens = results["tokens"]
    alpha_num = results["alpha_num"]
    dataset = SequenceDataset.emnist(tokens, alpha_num).to_torch()
    tokens_output = None if tokens else BernoulliMixture(num_hidden, num_hidden * 4, 16, 4)
    rnn_net = ElmanRNN(dataset.input_size, num_hidden, dataset.num_classes, tokens_output=tokens_output)
    rnn_net.load_state_dict(results["net"])
    print(results["snapshots"][-1])
    evaluate_image_model(dataset, rnn_net, results["snapshots"])


def train_emnist_rnn_finetune(num_hidden=128):
    dataset = SequenceDataset.emnist()
    tokens_output = BernoulliMixture(num_hidden, num_hidden * 4, 16, 4)
    self_criterion = nn.BCELoss()
    rnn_net = ElmanRNN(dataset.input_size, num_hidden, dataset.num_classes, tokens_output=tokens_output)
    pretrained = load_emnist_rnn(num_hidden)
    rnn_net.load_state_dict(pretrained["net"])

    dataset = SequenceDataset.emnist(alpha_num=True).to_torch()
    rnn_net.requires_grad_(False)
    rnn_net.class_output = nn.Linear(rnn_net.hidden_size, dataset.num_classes)
    path = os.path.join(RESULTS_DIR, f"rnn_emnist_finetune_{num_hidden}.results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(path):
        print(path, "already exists, skipping training")
        return
    best_model, snapshots = train_rnn(dataset, rnn_net, nn.CrossEntropyLoss(), self_criterion,
                                      512, 50, device=device)
    results = {"snapshots": snapshots, "net": best_model, "num_hidden": num_hidden}
    torch.save(results, path)
    print(f"Saved {path}")


def load_emnist_rnn_finetune(num_hidden=128):
    path = os.path.join(RESULTS_DIR, f"rnn_emnist_finetune_{num_hidden}.results")
    return _load_results(path)


def show_emnist_rnn_finetune(results):
    num_hidden = results["num_hidden"]
    dataset = SequenceDataset.emnist()
    tokens_output = BernoulliMixture(num_hidden, num_hidden * 4, 16, 4)
    rnn_net = ElmanRNN(dataset.input_size, num_hidden, dataset.num_classes, tokens_output=tokens_output)
    pretrained = load_emnist_rnn(num_hidden)
    rnn_net.load_state_dict(pretrained["net"])

    dataset = SequenceDataset.emnist(alpha_num=True).to_torch()
    rnn_net.requires_grad_(False)
    rnn_net.class_output = nn.Linear(rnn_net.hidden_size, dataset.num_classes)
    rnn_net.load_state_dict(results["net"])
    evaluate_image_model(dataset, rnn_net, results["snapshots"])


def image_to_sequence():
    image = MulticlassDataset.emnist().train.values[0][0]
    patches = SequenceDataset.emnist().train.values[0]

    fig = plt.figure(figsize=(9, 4))
    gs = gridspec.GridSpec(8, 18)
    gs.update(left=.01, right=.99, bottom=.01, top=.99)

    ax = fig.add_subplot(gs[:, :8])
    ax.imshow(image.reshape(28, 28), cmap="gray", interpolation="none", extent=[0, 28, 28, 0])
    for i in range(7):
        ax.axhline(i * 4, color="yellow")
        ax.axvline(i * 4, color="yellow")

    ax.set_xlim(0, 28)
    ax.set_ylim(28, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    for i, patch in enumerate(patches):
        row = (i // 10)
        col = 8 + (i % 10)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(patch.reshape(4, 4), cmap="gray", interpolation="none", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def bernoulli_samples():
    patches = SequenceDataset.emnist().train.values
    labels = SequenceDataset.emnist(True).train.values

    index = labels == 1
    patches = patches[index]

    bernoulli = patches.mean(0)

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(2, 6)
    gs.update(left=0.01, right=0.99, bottom=0.01, top=0.99)

    ax = fig.add_subplot(gs[:2, :2])
    ax.imshow(bernoulli.reshape(4, 4), cmap="viridis", interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])

    bernoulli = torch.from_numpy(bernoulli)

    for i in range(8):
        sample_patch = torch.bernoulli(bernoulli).numpy()
        ax = fig.add_subplot(gs[i // 4, i % 4 + 2])
        ax.imshow(sample_patch.reshape(4, 4), cmap="gray", interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


if __name__ == "__main__":
    # Visualization (no training needed)
    vanishing_gradient("backpropagation")
    image_to_sequence()
    bernoulli_samples()

    # Training
    train_language_rnn()
    train_emnist_rnn()

    # Visualization
    lang_results = load_language_rnn()
    show_language_rnn(lang_results)

    emnist_results = load_emnist_rnn()
    show_emnist_rnn(emnist_results)
