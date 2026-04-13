import copy
from datetime import datetime, timedelta
from enum import Enum
import math
import os
import time
from typing import Callable, List, NamedTuple, Tuple, Union

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PositionalDataset, SequenceDataset, TranslationDataset, DATA_DIR

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_1d_frequencies(num_samples: int):
    b_values = np.arange(1, num_samples // 2 + 1).astype(np.float32)
    b_values = torch.from_numpy(b_values).reshape(1, -1)
    a_values = 1 / np.arange(1, num_samples // 2 + 1).astype(np.float32)
    a_values = torch.from_numpy(a_values)
    return a_values, b_values


def generate_2d_frequencies(max_log_scale: float, position_size: int):
    if position_size % 2:
        raise ValueError(f"{position_size} must be a multiple of {2}")

    vals_per_input = position_size // 2
    b_values = torch.zeros(position_size, 2)
    b_values[:vals_per_input, 0] = 2. ** torch.linspace(0, max_log_scale, vals_per_input) - 1
    b_values[vals_per_input:, 1] = 2. ** torch.linspace(0, max_log_scale, vals_per_input) - 1
    b_values = b_values.transpose(0, 1)
    a_values = torch.ones(b_values.size(1))
    return a_values, b_values


class PositionalMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size=256, num_layers=3,
                 position_size=0, max_log_scale=6):
        super(PositionalMLP, self).__init__()
        if position_size:
            if input_size == 1:
                a_values, b_values = generate_1d_frequencies(position_size)
            else:
                a_values, b_values = generate_2d_frequencies(max_log_scale, position_size // 2)

            input_size = 2 * b_values.size(1)
            self.a_values = nn.Parameter(a_values, requires_grad=False)
            self.b_values = nn.Parameter(b_values, requires_grad=False)
        else:
            self.a_values = None
            self.b_values = None

        layers = nn.ModuleList()
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.b_values is not None:
            encoded = (math.pi * x) @ self.b_values
            x = torch.cat([self.a_values * encoded.cos(),
                           self.a_values * encoded.sin()], dim=-1)

        return self.layers(x)


def train_positional_mlp(data: PositionalDataset, net: PositionalMLP, num_steps=10000,
                         device="cpu", lr=5e-4, weight_decay=1e-3, sigmoid=False):
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_x = torch.from_numpy(data.train.values).to(device)
    train_y = torch.from_numpy(data.train.labels).to(device)
    val_x = torch.from_numpy(data.val.values).to(device)
    val_y = torch.from_numpy(data.val.labels).to(device)
    with tqdm(range(num_steps)) as pb:
        for step in pb:
            if step % 100 == 0:
                with torch.no_grad():
                    net.eval()
                    batch_rows = val_x.shape[0] // 4
                    output = []
                    for i in range(4):
                        start = i * batch_rows
                        end = start + batch_rows
                        output.append(net(val_x[start:end]))

                    output = torch.cat(output)
                    if sigmoid:
                        output = torch.sigmoid(output)

                    mse = torch.square(output - val_y).mean().item()
                    psnr_val = -10 * math.log10(mse)
                    print("step", step, "val:", psnr_val)
                    net.train()

            optimizer.zero_grad()

            y = net(train_x)
            if sigmoid:
                y = torch.sigmoid(y)

            loss = 0.5 * torch.square(y - train_y).mean()
            loss.backward()
            optimizer.step()

            pb.set_postfix(loss=loss.item())


def polynomial(position_size: int):
    data = PositionalDataset.polynomial(25, 40)
    net = PositionalMLP(data.input_size, data.output_size, 32, 1,
                        position_size=position_size)

    train_positional_mlp(data, net)

    plt.rc('font', size=14)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)

    x = data.val.values
    y1 = data.val.labels
    y2 = net(torch.from_numpy(x)).detach().numpy()
    mse = np.square(y1 - y2).mean()
    psnr = -10 * math.log10(mse)

    ax.plot(x, y1, label="True")
    ax.plot(x, y2, label=f"Predicted {psnr:.2f}dB")
    ax.legend()
    plt.tight_layout()
    plt.show()


def periodic(num_layers: int, hidden_size: int, position_size: int):
    data = PositionalDataset.periodic(32, 8)
    net = PositionalMLP(data.input_size, data.output_size, hidden_size, num_layers,
                        position_size=position_size)

    train_positional_mlp(data, net)

    plt.rc('font', size=14)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)

    x = data.val.values
    y1 = data.val.labels
    y2 = net(torch.from_numpy(x)).detach().numpy()
    mse = np.square(y1 - y2).mean()
    psnr = -10 * math.log10(mse)

    ax.plot(x, y1, label="True")
    ax.plot(x, y2, label=f"Predicted {psnr:.2f}dB")
    ax.set_yticks([0, 2, 4])
    ax.set_xticks([0, 2])
    ax.legend()
    plt.tight_layout()
    plt.show()


def cat_image(position_size: int):
    data = PositionalDataset.cat()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net_pos = PositionalMLP(data.input_size, data.output_size, position_size=position_size)
    train_positional_mlp(data, net_pos, device=device,
                         num_steps=2000, lr=1e-3, weight_decay=0, sigmoid=True)

    net_nopos = PositionalMLP(data.input_size, data.output_size, position_size=0)
    train_positional_mlp(data, net_nopos, device=device,
                         num_steps=2000, lr=1e-3, weight_decay=0, sigmoid=True)

    uv = torch.from_numpy(data.val.values).to(device)
    pixels = torch.sigmoid(net_nopos(uv)).cpu().detach().numpy()
    psnr_nopos = -10 * math.log10(np.square(data.val.labels - pixels).mean())
    pixels_nopos = (pixels * 255).astype(np.uint8)

    pixels = torch.sigmoid(net_pos(uv)).cpu().detach().numpy()
    psnr_pos = -10 * math.log10(np.square(data.val.labels - pixels).mean())
    pixels_pos = (pixels * 255).astype(np.uint8)

    plt.rc('font', size=14)
    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 3, 1)
    gt = Image.open(os.path.join(DATA_DIR, "cat.jpg"))
    ax.imshow(gt)
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(pixels_nopos)
    ax.text(0, 510, f"{psnr_nopos:.2f}dB", color="yellow", fontsize=24, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(pixels_pos)
    ax.text(0, 510, f"{psnr_pos:.2f}dB", color="yellow", fontsize=24, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.show()


# --- Transformer building blocks ---

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        if hidden_size % num_heads:
            raise ValueError(f"Cannot evenly partition {hidden_size} into {num_heads} heads")

        self.q_fc = nn.Linear(hidden_size, hidden_size)
        self.k_fc = nn.Linear(hidden_size, hidden_size)
        self.v_fc = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.head_size = int(hidden_size // num_heads)
        self.softmax = nn.Softmax(dim=-1)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_size, _ = x.shape
        x = x.reshape(batch_size, sequence_size, self.num_heads, self.head_size)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, sequence_size, _ = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, sequence_size, self.hidden_size)
        return x

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor,
                                     v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.head_size))

        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)

        self.attention = self.softmax(logits)
        return torch.matmul(self.attention, v)

    def forward(self, q_tokens: torch.Tensor, k_tokens: torch.Tensor,
                v_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.split_heads(self.q_fc(q_tokens))
        k = self.split_heads(self.k_fc(k_tokens))
        v = self.split_heads(self.v_fc(v_tokens))

        attention_output = self.scaled_dot_product_attention(q, k, v, mask)

        output = self.output(self.combine_heads(attention_output))
        return output


class BlockType(Enum):
    Encoder = 0
    Decoder = 1
    DecoderOnly = 2


class TransformerBlock(nn.Module):
    def __init__(self, block_type: BlockType, token_size: int, num_heads: int,
                 max_length: int, mlp_ratio: int, dropout: float):
        nn.Module.__init__(self)
        self.block_type = block_type
        self.hidden_size = token_size
        self.num_heads = num_heads

        self.mhsa0 = MultiHeadSelfAttention(token_size, num_heads)
        self.norm0 = nn.LayerNorm(token_size)
        if block_type == BlockType.Decoder:
            self.mhsa1 = MultiHeadSelfAttention(token_size, num_heads)
            self.norm1 = nn.LayerNorm(token_size)
        else:
            self.mhsa1 = None
            self.norm1 = None

        self.mlp = nn.Sequential(
            nn.Linear(token_size, mlp_ratio * token_size),
            nn.GELU(),
            nn.Linear(mlp_ratio * token_size, token_size)
        )
        self.norm2 = nn.LayerNorm(token_size)

        self.dropout = nn.Dropout(dropout)

        if block_type == BlockType.Encoder:
            self.mask = None
        elif block_type in (BlockType.DecoderOnly, BlockType.Decoder):
            mask = torch.triu(torch.ones((1, max_length, max_length)), diagonal=1)
            mask = mask.bool()
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None) -> torch.Tensor:
        if self.block_type == BlockType.Encoder:
            attn0 = self.mhsa0(x, x, x, None)
            self.attention = self.mhsa0.attention
        else:
            length = x.size(1)
            mask = self.mask[:, :length, :length]
            attn0 = self.mhsa0(x, x, x, mask)

        x = self.norm0(x + self.dropout(attn0))

        if self.block_type == BlockType.Decoder:
            self.attn1 = self.mhsa1(x, k, v, None)
            self.attention = self.mhsa1.attention
            x = self.norm1(x + self.dropout(self.attn1))

        out = self.mlp(x)
        x = self.norm2(x + self.dropout(out))
        return x


class SupervisionKind(Enum):
    Supervised = 0,
    Unsupervised = 1,
    SemiSupervised = 2


class VisionTransformer(nn.Module):
    def __init__(self, max_length: int, input_size: int, token_size: int,
                 num_blocks: int, num_heads: int, num_classes: int,
                 position_encoder: Callable[[int], torch.Tensor],
                 mlp_ratio=4, dropout=0.1, emb_dropout=0.1):
        nn.Module.__init__(self)

        self.input_size = input_size
        self.token_size = token_size
        self.max_length = max_length
        self.class_attention = []
        self.store_class_attention = False
        self.supervision = SupervisionKind.Supervised

        self.input_to_token = nn.Linear(input_size, token_size)
        self.dropout = nn.Dropout(emb_dropout)

        pos = [position_encoder(i) for i in range(max_length + 1)]
        self.register_buffer("position_encoding", torch.stack(pos).unsqueeze(0))

        self.blocks = nn.ModuleList([TransformerBlock(BlockType.Encoder, token_size, num_heads,
                                                      max_length, mlp_ratio, dropout)
                                     for _ in range(num_blocks)])

        self.class_token = nn.Parameter(torch.randn(1, 1, token_size))
        self.class_output = nn.Linear(self.token_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.input_to_token(x)

        class_token = self.class_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1)

        position_encoding = self.position_encoding.expand((x.size(0), -1, -1))
        tokens = tokens + position_encoding

        tokens = self.dropout(tokens)

        if self.store_class_attention:
            self.class_attention.clear()

        for block in self.blocks:
            tokens = block(tokens)
            if self.store_class_attention:
                self.class_attention.append(block.attention[:, :, 0, 1:].detach().cpu().numpy())

        class_output = self.class_output(tokens[:, 0])
        return class_output


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, max_length: int, input_size: int, token_size: int,
                 num_blocks: int, num_heads: int,
                 position_encoder: Callable[[int], torch.Tensor],
                 mlp_ratio=4, dropout=0.1, tokens_output=None):
        nn.Module.__init__(self)

        self.input_size = input_size
        self.token_size = token_size
        self.max_length = max_length
        self.supervision = SupervisionKind.Unsupervised

        self.input_to_token = nn.Linear(input_size, token_size)

        pos = [position_encoder(i) for i in range(max_length + 1)]
        self.register_buffer("position_encoding", torch.stack(pos).unsqueeze(0))

        self.start_token = nn.Parameter(torch.randn(1, 1, token_size))

        self.blocks = nn.ModuleList([TransformerBlock(BlockType.DecoderOnly, token_size, num_heads,
                                                      max_length + 1, mlp_ratio, dropout)
                                     for _ in range(num_blocks)])

        if tokens_output is None:
            self.tokens_output = nn.Linear(self.token_size, input_size)
        else:
            self.tokens_output = tokens_output

        self.class_output = None

        self.keep_tokens = False
        self.tokens = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        tokens = self.input_to_token(x)

        start_token = self.start_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([start_token, tokens], dim=1)

        position_encoding = self.position_encoding.expand((x.size(0), -1, -1))
        tokens = tokens + position_encoding[:, :length + 1]

        for block in self.blocks:
            tokens = block(tokens)

        if self.keep_tokens:
            self.tokens = tokens.detach().cpu()

        tokens_output = self.tokens_output(tokens)
        if self.class_output:
            tokens = tokens.mean(-2)
            class_output = self.class_output(tokens)
            if self.supervision == SupervisionKind.SemiSupervised:
                return class_output, tokens_output

            return class_output

        return tokens_output


class Transformer(nn.Module):
    def __init__(self, max_input_length: int, input_size: int, token_size: int,
                 max_output_length: int, output_size: int,
                 num_blocks: int, num_heads: int,
                 position_encoder: Callable[[int], torch.Tensor],
                 mlp_ratio=4, dropout=0.1):
        nn.Module.__init__(self)

        self.input_size = input_size
        self.output_size = output_size
        self.token_size = token_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.attention = []
        self.store_attention = False

        self.input_to_token = nn.Linear(input_size, token_size)
        self.output_to_token = nn.Linear(output_size, token_size)

        pos = [position_encoder(i) for i in range(max_input_length)]
        self.register_buffer("input_position_encoding", torch.stack(pos).unsqueeze(0))

        pos = [position_encoder(i) for i in range(max_output_length + 1)]
        self.register_buffer("output_position_encoding", torch.stack(pos).unsqueeze(0))

        self.encoder_blocks = nn.ModuleList([TransformerBlock(BlockType.Encoder, token_size, num_heads,
                                                              max_input_length, mlp_ratio, dropout)
                                            for _ in range(num_blocks)])
        self.decoder_blocks = nn.ModuleList([TransformerBlock(BlockType.Decoder, token_size, num_heads,
                                                              max_output_length + 1, mlp_ratio, dropout)
                                             for _ in range(num_blocks)])

        self.output = nn.Linear(token_size, output_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        length = y.size(1)
        input_tokens = self.input_to_token(x)
        output_tokens = self.output_to_token(y)

        position_encoding = self.input_position_encoding.expand((x.size(0), -1, -1))
        input_tokens = input_tokens + position_encoding

        position_encoding = self.output_position_encoding.expand((x.size(0), -1, -1))
        output_tokens = output_tokens + position_encoding[:, :length]

        if self.store_attention:
            self.attention.clear()

        for enc, dec in zip(self.encoder_blocks, self.decoder_blocks):
            key_value = enc(input_tokens)
            output_tokens = dec(output_tokens, key_value, key_value)
            if self.store_attention:
                self.attention.append((enc.attention.detach().cpu().numpy(),
                                       dec.attention.detach().cpu().numpy()))

        output_tokens = self.output(output_tokens)

        return output_tokens


# --- Training functions ---

def compute_accuracy(output: torch.Tensor, expected_labels: torch.Tensor) -> float:
    _, actual_labels = torch.max(output, -1)
    num_correct = (actual_labels == expected_labels).sum().item()
    total = len(actual_labels)
    return num_correct / total


Snapshot = NamedTuple("Snapshot", [("step", int), ("accuracy", float), ("loss", float)])


def _load_results(path):
    return torch.load(path, weights_only=False)


def train_transformer(dataset: TranslationDataset, net: Transformer, criterion,
                      batch_size, num_epochs, device="cpu"):
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    snapshots = []
    best_model = None
    best_loss = float("inf")

    def compute_accuracy_loss(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        y_start = torch.full((y.size(0), 1), dataset.start_token, dtype=torch.long, device=device)
        y_end = torch.full((y.size(0), 1), dataset.end_token, dtype=torch.long, device=device)

        y_input = torch.cat([y_start, y], dim=1)
        y_input = F.one_hot(y_input, net.output_size).float()

        output = net(x, y_input)
        t = torch.cat([y, y_end], dim=1)
        loss = criterion(output.view(-1, dataset.output_size), t.view(-1))

        output = output[:, :-1]
        t = t[:, :-1]
        accuracy = compute_accuracy(output[:, :-1], t[:, :-1])
        return loss, accuracy

    train_loader = DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset.val, batch_size=batch_size, shuffle=False)
    for epoch in range(1, num_epochs + 1):
        net.train()
        sum_accuracy = 0
        sum_loss = 0
        total = 0

        with tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}') as pbar:
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                loss, accuracy = compute_accuracy_loss(x, y)
                loss.backward()
                optimizer.step()

                batch_size_actual = x.size(0)
                sum_loss += loss.item() * batch_size_actual
                sum_accuracy += accuracy * batch_size_actual
                total += batch_size_actual

                pbar.set_postfix(**{'loss (batch)': "{:.2f}".format(loss.item())})

        print('train mean loss={}, accuracy={}'.format(
            sum_loss / total,
            sum_accuracy / (total * dataset.output_max_length)))

        net.eval()
        sum_accuracy = 0
        sum_loss = 0
        total = 0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                loss, accuracy = compute_accuracy_loss(x, y)

            batch_size_actual = x.size(0)
            sum_loss += loss.item() * batch_size_actual
            sum_accuracy += accuracy * batch_size_actual
            total += batch_size_actual

        if sum_loss < best_loss:
            best_loss = sum_loss
            net = net.to("cpu")
            best_model = copy.deepcopy(net.state_dict())
            net = net.to(device)

        mean_loss = sum_loss / total
        mean_accuracy = sum_accuracy / (total * dataset.output_max_length)
        print('validation  mean loss={}, accuracy={}'.format(mean_loss, mean_accuracy))
        snapshots.append(Snapshot(epoch, mean_accuracy, mean_loss))

        scheduler.step()

    return best_model, snapshots


def train_partial_transformer(dataset: SequenceDataset, net: Union[VisionTransformer, DecoderOnlyTransformer],
                              criterion, batch_size, num_epochs, device="cpu", token_criterion=None, lr=0.0001):
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    snapshots = []
    best_model = None
    best_loss = float("inf")

    def compute_loss_and_accuracy(x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if x.size(-1) != net.input_size:
            x_input = F.one_hot(x, net.input_size).float()
        else:
            x_input = x

        optimizer.zero_grad()

        if net.supervision == SupervisionKind.Supervised:
            outputs = net(x_input)
            loss = criterion(outputs, t)
            accuracy = compute_accuracy(outputs, t)
        elif net.supervision == SupervisionKind.Unsupervised:
            tokens = net(x_input)
            y = tokens[:, :-1].reshape(-1, net.input_size)
            if len(x.shape) == 2:
                t = x.reshape(-1)
            else:
                t = x.reshape(-1, x.size(-1))
            if token_criterion:
                loss = token_criterion(y, t)
            else:
                loss = criterion(y, t)
            accuracy = None
        elif net.supervision == SupervisionKind.SemiSupervised:
            outputs, tokens = net(x_input)
            y = outputs
            loss = criterion(y, t)
            accuracy = compute_accuracy(y, t)
            y = tokens[:, :-1].reshape(-1, net.input_size)
            if len(x.shape) == 2:
                t = x.reshape(-1)
            else:
                t = x.reshape(-1, x.size(-1))
            if token_criterion:
                loss += token_criterion(y, t)
            else:
                loss += criterion(y, t)
        else:
            raise ValueError(f"Unknown supervision kind {net.supervision}")

        return loss, accuracy

    train_loader = DataLoader(dataset.train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset.val, batch_size=batch_size, shuffle=False)
    for epoch in range(1, num_epochs + 1):
        net.train()
        sum_accuracy = None if net.supervision == SupervisionKind.Unsupervised else 0
        sum_loss = 0
        total = 0

        with tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}') as pbar:
            for x, t in pbar:
                x = x.to(device)
                t = t.to(device)

                loss, accuracy = compute_loss_and_accuracy(x, t)

                loss.backward()
                optimizer.step()

                batch_size_actual = len(t)
                sum_loss += loss.item() * batch_size_actual

                if sum_accuracy is not None:
                    sum_accuracy += accuracy * batch_size_actual

                total += batch_size_actual

                pbar.set_postfix(**{'loss (batch)': "{:.2f}".format(loss.item())})

        print('train mean loss={}, accuracy={}'.format(
            sum_loss / total,
            sum_accuracy / total if sum_accuracy is not None else "N/A"))

        net.eval()
        sum_accuracy = None if net.supervision == SupervisionKind.Unsupervised else 0
        sum_loss = 0
        total = 0
        for x, t in val_loader:
            x = x.to(device)
            t = t.to(device)

            with torch.no_grad():
                loss, accuracy = compute_loss_and_accuracy(x, t)

            batch_size_actual = len(t)
            sum_loss += loss.item() * batch_size_actual

            if sum_accuracy is not None:
                sum_accuracy += accuracy * batch_size_actual

            total += batch_size_actual

        if sum_loss < best_loss:
            best_loss = sum_loss
            net = net.to("cpu")
            best_model = copy.deepcopy(net.state_dict())
            net = net.to(device)

        print('validation  mean loss={}, accuracy={}'.format(
            sum_loss / total,
            sum_accuracy / total if sum_accuracy is not None else "N/A"))
        snapshots.append(Snapshot(epoch, sum_accuracy / total if sum_accuracy is not None else 0, sum_loss / total))

        scheduler.step()

    return best_model, snapshots


# --- Position encodings ---

class SequencePosition:
    def __init__(self, token_size: int, max_length=10000):
        b_values = np.linspace(0, 1, token_size // 2, endpoint=False)
        b_values = 1 / (max_length ** b_values)
        self.b_values = torch.from_numpy(b_values).float()

    def __call__(self, i: int) -> torch.Tensor:
        return torch.cat([torch.sin(i * self.b_values), torch.cos(i * self.b_values)], dim=-1)


class GridPosition:
    def __init__(self, rows: int, cols: int, hidden_size: int):
        self.rows = rows + 1
        self.cols = cols
        max_log_scale = math.log2(rows / (2 * math.pi))
        self.b_values = generate_2d_frequencies(max_log_scale, hidden_size // 2)

    def __call__(self, i: int) -> torch.Tensor:
        r = i // self.cols
        c = i % self.cols
        uv = 2 * torch.tensor([r / self.rows, c / self.cols])
        encoded = (math.pi * uv) @ self.b_values
        pos = torch.cat([encoded.cos(), encoded.sin()], dim=-1)
        return pos


# --- Sampling ---

def sample_text(dataset: SequenceDataset, net: DecoderOnlyTransformer, num_samples: int, top_p: float):
    net = net.to("cpu")
    net.eval()

    samples = torch.zeros(num_samples, dataset.max_length, dtype=torch.long)
    for i in range(0, dataset.max_length):
        x = samples[:, :i]
        x_input = F.one_hot(x, net.input_size).float()

        with torch.no_grad():
            if net.supervision == SupervisionKind.SemiSupervised:
                _, tokens = net(x_input)
            else:
                tokens = net(x_input)

        for s in range(num_samples):
            samples[s, i] = dataset.to_token(tokens[s, i], top_p=top_p)

    return samples


def beam_search(x: torch.Tensor, net: nn.Module, dataset: TranslationDataset, num_beams=5):
    class Beam(NamedTuple("Beam", [("log_prob", float), ("tokens", torch.Tensor)])):
        def __lt__(self, other: "Beam") -> bool:
            return self.log_prob > other.log_prob

        def value(self) -> str:
            return "".join([dataset.output_tokens[t] for t in self.tokens[1:]])

    x_input = x.unsqueeze(0).repeat(num_beams, 1, 1)
    beams = [Beam(0, torch.tensor([dataset.start_token])) for _ in range(num_beams)]
    for i in range(1, dataset.output_max_length + 1):
        y = torch.stack([b.tokens for b in beams], dim=0)
        y_input = F.one_hot(y, net.output_size).float()

        with torch.no_grad():
            tokens = net(x_input, y_input)

        explored = set()
        log_probs = F.log_softmax(tokens[:, i - 1], dim=-1)
        frontier = []
        for b in range(num_beams):
            for k, log_prob in enumerate(log_probs[b]):
                new_log_prob = beams[b].log_prob + log_prob.item()
                new_tokens = torch.cat([beams[b].tokens, torch.tensor([k])])
                new_beam = Beam(new_log_prob, new_tokens)
                if new_beam.value() not in explored:
                    explored.add(new_beam.value())
                    frontier.append(new_beam)

        frontier = sorted(frontier)
        beams = frontier[:num_beams]

    return beams[0].value()


# --- Evaluation ---

def assemble(patches: np.ndarray, size=28):
    color = patches.shape[-1] == 48
    if color:
        image = np.zeros((size, size, 3), np.uint8)
    else:
        image = np.zeros((size, size), np.uint8)

    cols = size // 4
    for i, patch in enumerate(patches):
        r = (i // cols) * 4
        c = (i % cols) * 4
        if isinstance(patch, torch.Tensor):
            patch = patch.cpu().numpy()

        if color:
            patch = patch.reshape(4, 4, 3)
        else:
            patch = patch.reshape(4, 4)

        if patch.dtype == np.float32:
            image[r:r+4, c:c+4] = patch * 255
        else:
            image[r:r+4, c:c+4] = patch

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
        x_input = F.one_hot(x, net.input_size).float()

        with torch.no_grad():
            if net.supervision == SupervisionKind.SemiSupervised:
                y, _ = net(x_input)
            else:
                y = net(x_input)

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
    gs0 = gridspec.GridSpec(1, 2)
    gs00 = gridspec.GridSpecFromSubplotSpec(num_examples // columns, columns, gs0[0], hspace=2.5)
    gs0.update(bottom=0.15, top=0.98)

    for i in range(num_examples):
        row = i // columns
        col = i % columns
        ax = fig.add_subplot(gs00[row, col])
        ax.patch.set_alpha(0)
        for spine in ax.spines.values():
            spine.set_visible(False)

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

    print("Best snapshot", snapshots[end_index])

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

    return ax1, ax2


def evaluate_image_model(dataset: SequenceDataset, net: nn.Module, snapshots: List[Snapshot], device="cpu"):
    label_names = dataset.label_names
    num_examples = 6
    examples, example_labels = dataset.val[0:num_examples]
    columns = num_examples // 2
    size = int(math.sqrt(dataset.max_length)) * 4

    net.to(device)
    net.eval()
    x = examples.to(device)

    if len(x.shape) == 2:
        x_input = F.one_hot(x, net.input_size).float()
    else:
        x_input = x

    if net.supervision == SupervisionKind.SemiSupervised:
        output, _ = net(x_input)
    else:
        output = net(x_input)

    output = F.softmax(output, dim=1).cpu().detach().numpy()
    top_labels = np.argpartition(output, -2)[:, -2:]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(10, 5))
    gs0 = gridspec.GridSpec(1, 2)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, columns, gs0[0])
    gs0.update(left=0.01, right=0.94, bottom=0.05, top=0.95)

    for i in range(num_examples):
        row = i // columns
        col = i % columns
        ax = fig.add_subplot(gs00[row, col])
        example = examples[i]
        if example.dtype == torch.float32:
            example = dataset.unnormalize(examples[i]).cpu().numpy()
            example = assemble(example, size)
        else:
            example = example.cpu().numpy()
            patches = np.stack([dataset.tokens[t] for t in example])
            example = assemble(patches, size)

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

    print("Best snapshot", snapshots[end_index])

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
    return ax1, ax2


# --- Demo functions ---

def train_language_vit(token_size=128, num_blocks=6, num_heads=4):
    dataset = SequenceDataset.language(6, 1000).to_torch()
    net = VisionTransformer(dataset.max_length, dataset.input_size, token_size,
                            num_blocks, num_heads, dataset.num_classes,
                            SequencePosition(token_size))
    path = os.path.join(RESULTS_DIR, "transformer_language_vit.results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(path):
        print(path, "already exists, skipping training")
        return
    print("training language transformer")
    best_model, snapshots = train_partial_transformer(dataset, net, nn.CrossEntropyLoss(), 256, 200, device)
    results = {"snapshots": snapshots, "net": best_model,
               "token_size": token_size, "num_blocks": num_blocks, "num_heads": num_heads}
    torch.save(results, path)
    print(f"Saved {path}")


def load_language_vit():
    path = os.path.join(RESULTS_DIR, "transformer_language_vit.results")
    return _load_results(path)


def show_language_vit(results):
    token_size = results["token_size"]
    num_blocks = results["num_blocks"]
    num_heads = results["num_heads"]
    dataset = SequenceDataset.language(6, 1000).to_torch()
    net = VisionTransformer(dataset.max_length, dataset.input_size, token_size,
                            num_blocks, num_heads, dataset.num_classes,
                            SequencePosition(token_size))
    net.load_state_dict(results["net"])
    evaluate_text_model(dataset, net, results["snapshots"])
    plt.show()


def train_language_gpt(token_size=128, num_blocks=6, num_heads=4):
    dataset = SequenceDataset.language(6, 1000).to_torch()
    net = DecoderOnlyTransformer(dataset.max_length, dataset.input_size, token_size,
                                 num_blocks, num_heads,
                                 SequencePosition(token_size))
    path = os.path.join(RESULTS_DIR, f"transformer_language_{token_size}.results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(path):
        print(path, "already exists, skipping pretraining")
    else:
        print("training language transformer")
        best_model, snapshots = train_partial_transformer(dataset, net, nn.CrossEntropyLoss(), 256, 200, device)
        results = {"snapshots": snapshots, "net": best_model,
                   "token_size": token_size, "num_blocks": num_blocks, "num_heads": num_heads}
        torch.save(results, path)
        print(f"Saved {path}")

    # Fine-tune for classification
    ft_path = os.path.join(RESULTS_DIR, f"transformer_language_{token_size}_finetune.results")
    if os.path.exists(ft_path):
        print(ft_path, "already exists, skipping finetuning")
        return

    pretrained = _load_results(path)
    net.load_state_dict(pretrained["net"])
    net.supervision = SupervisionKind.SemiSupervised
    net.class_output = nn.Linear(net.token_size, dataset.num_classes)
    print("finetuning language transformer")
    best_model, snapshots = train_partial_transformer(dataset, net, nn.CrossEntropyLoss(), 256, 100, device,
                                                      lr=0.00005)
    results = {"snapshots": snapshots, "net": best_model,
               "token_size": token_size, "num_blocks": num_blocks, "num_heads": num_heads}
    torch.save(results, ft_path)
    print(f"Saved {ft_path}")


def load_language_gpt(token_size=128):
    path = os.path.join(RESULTS_DIR, f"transformer_language_{token_size}.results")
    return _load_results(path)


def load_language_gpt_finetune(token_size=128):
    path = os.path.join(RESULTS_DIR, f"transformer_language_{token_size}_finetune.results")
    return _load_results(path)


def show_language_gpt(pretrained_results, finetune_results, top_p=0.95, num_samples=21):
    token_size = pretrained_results["token_size"]
    num_blocks = pretrained_results["num_blocks"]
    num_heads = pretrained_results["num_heads"]
    dataset = SequenceDataset.language(6, 1000).to_torch()
    net = DecoderOnlyTransformer(dataset.max_length, dataset.input_size, token_size,
                                 num_blocks, num_heads,
                                 SequencePosition(token_size))
    net.load_state_dict(pretrained_results["net"])
    samples = sample_text(dataset, net, num_samples, top_p)
    for s in samples:
        chars = [dataset.tokens[t] for t in s]
        print("".join(chars))

    net.supervision = SupervisionKind.SemiSupervised
    net.class_output = nn.Linear(net.token_size, dataset.num_classes)
    net.load_state_dict(finetune_results["net"])
    evaluate_text_model(dataset, net, finetune_results["snapshots"])
    plt.show()


def train_emnist_vit(token_size=128, num_blocks=6, num_heads=4):
    dataset = SequenceDataset.emnist().to_torch()
    net = VisionTransformer(dataset.max_length, dataset.input_size, token_size,
                            num_blocks, num_heads, dataset.num_classes,
                            SequencePosition(token_size))
    path = os.path.join(RESULTS_DIR, "transformer_emnist_vit.results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(path):
        print(path, "already exists, skipping training")
        return
    print("training emnist vit")
    best_model, snapshots = train_partial_transformer(dataset, net, nn.CrossEntropyLoss(), 512, 200, device)
    results = {"snapshots": snapshots, "net": best_model,
               "token_size": token_size, "num_blocks": num_blocks, "num_heads": num_heads}
    torch.save(results, path)
    print(f"Saved {path}")


def load_emnist_vit():
    path = os.path.join(RESULTS_DIR, "transformer_emnist_vit.results")
    return _load_results(path)


def show_emnist_vit(results):
    token_size = results["token_size"]
    num_blocks = results["num_blocks"]
    num_heads = results["num_heads"]
    dataset = SequenceDataset.emnist().to_torch()
    net = VisionTransformer(dataset.max_length, dataset.input_size, token_size,
                            num_blocks, num_heads, dataset.num_classes,
                            SequencePosition(token_size))
    net.load_state_dict(results["net"])
    net.store_class_attention = True
    evaluate_image_model(dataset, net, results["snapshots"])
    plt.show()


def train_emnist_gpt(token_size=128, num_blocks=6, num_heads=4):
    dataset = SequenceDataset.emnist(True).to_torch()
    net = DecoderOnlyTransformer(dataset.max_length, dataset.input_size, token_size,
                                 num_blocks, num_heads,
                                 SequencePosition(token_size))
    path = os.path.join(RESULTS_DIR, f"transformer_emnist_gpt_{token_size}.results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(path):
        print(path, "already exists, skipping pretraining")
    else:
        print("training emnist gpt")
        best_model, snapshots = train_partial_transformer(dataset, net, nn.CrossEntropyLoss(), 512, 200, device)
        results = {"snapshots": snapshots, "net": best_model,
                   "token_size": token_size, "num_blocks": num_blocks, "num_heads": num_heads}
        torch.save(results, path)
        print(f"Saved {path}")

    # Fine-tune
    ft_path = os.path.join(RESULTS_DIR, f"transformer_emnist_{token_size}_finetune.results")
    if os.path.exists(ft_path):
        print(ft_path, "already exists, skipping finetuning")
        return

    pretrained = _load_results(path)
    net.load_state_dict(pretrained["net"])
    net.supervision = SupervisionKind.SemiSupervised
    net.class_output = nn.Linear(net.token_size, dataset.num_classes)
    print("finetuning emnist transformer")
    best_model, snapshots = train_partial_transformer(dataset, net, nn.CrossEntropyLoss(), 512, 100,
                                                      device, lr=0.00005)
    results = {"snapshots": snapshots, "net": best_model,
               "token_size": token_size, "num_blocks": num_blocks, "num_heads": num_heads}
    torch.save(results, ft_path)
    print(f"Saved {ft_path}")


def load_emnist_gpt(token_size=128):
    path = os.path.join(RESULTS_DIR, f"transformer_emnist_gpt_{token_size}.results")
    return _load_results(path)


def load_emnist_gpt_finetune(token_size=128):
    path = os.path.join(RESULTS_DIR, f"transformer_emnist_{token_size}_finetune.results")
    return _load_results(path)


def show_emnist_gpt(pretrained_results, finetune_results):
    token_size = pretrained_results["token_size"]
    num_blocks = pretrained_results["num_blocks"]
    num_heads = pretrained_results["num_heads"]
    dataset = SequenceDataset.emnist(True).to_torch()
    net = DecoderOnlyTransformer(dataset.max_length, dataset.input_size, token_size,
                                 num_blocks, num_heads,
                                 SequencePosition(token_size))
    net.load_state_dict(pretrained_results["net"])
    samples = sample_text(dataset, net, 32, 0.95)
    plt.rc('font', size=15)
    fig = plt.figure(figsize=(8, 4))
    for i, s in enumerate(samples):
        patches = np.stack([dataset.tokens[t] for t in s])
        image = assemble(patches)
        ax = fig.add_subplot(4, 8, i + 1)
        ax.imshow(image, cmap='gray', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.1, hspace=0.1)
    plt.show()

    net.supervision = SupervisionKind.SemiSupervised
    net.class_output = nn.Linear(net.token_size, dataset.num_classes)
    net.load_state_dict(finetune_results["net"])
    evaluate_image_model(dataset, net, finetune_results["snapshots"])
    plt.show()


def train_cifar_vit(token_size=128, num_blocks=6, num_heads=4):
    dataset = SequenceDataset.cifar().normalize().to_torch()
    net = VisionTransformer(dataset.max_length, dataset.input_size, token_size,
                            num_blocks, num_heads, dataset.num_classes,
                            SequencePosition(token_size))
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("trainable parameters:", pytorch_total_params)
    path = os.path.join(RESULTS_DIR, "transformer_cifar_vit.results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(path):
        print(path, "already exists, skipping training")
        return
    print("training cifar vit")
    best_model, snapshots = train_partial_transformer(dataset, net, nn.CrossEntropyLoss(), 512, 200, device)
    results = {"snapshots": snapshots, "net": best_model,
               "token_size": token_size, "num_blocks": num_blocks, "num_heads": num_heads}
    torch.save(results, path)
    print(f"Saved {path}")


def load_cifar_vit():
    path = os.path.join(RESULTS_DIR, "transformer_cifar_vit.results")
    return _load_results(path)


def show_cifar_vit(results):
    token_size = results["token_size"]
    num_blocks = results["num_blocks"]
    num_heads = results["num_heads"]
    dataset = SequenceDataset.cifar().normalize().to_torch()
    net = VisionTransformer(dataset.max_length, dataset.input_size, token_size,
                            num_blocks, num_heads, dataset.num_classes,
                            SequencePosition(token_size))
    net.load_state_dict(results["net"])
    evaluate_image_model(dataset, net, results["snapshots"])
    plt.show()


def train_postcodes(token_size=128, num_blocks=6, num_heads=4, batch_size=220, num_steps=100):
    dataset = TranslationDataset.postcodes().to_torch()
    net = Transformer(dataset.input_max_length, dataset.input_size, token_size,
                      dataset.output_max_length, dataset.output_size,
                      num_blocks, num_heads,
                      SequencePosition(token_size))
    path = os.path.join(RESULTS_DIR, f"transformer_postcodes_{token_size}.results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(path):
        print(path, "already exists, skipping training")
        return
    print("training postcodes transformer")
    best_model, snapshots = train_transformer(dataset, net, nn.CrossEntropyLoss(), batch_size, num_steps, device)
    results = {"snapshots": snapshots, "net": best_model,
               "token_size": token_size, "num_blocks": num_blocks, "num_heads": num_heads}
    torch.save(results, path)
    print(f"Saved {path}")


def load_postcodes(token_size=128):
    path = os.path.join(RESULTS_DIR, f"transformer_postcodes_{token_size}.results")
    return _load_results(path)


def show_postcodes(results):
    token_size = results["token_size"]
    num_blocks = results["num_blocks"]
    num_heads = results["num_heads"]
    dataset = TranslationDataset.postcodes().to_torch()
    net = Transformer(dataset.input_max_length, dataset.input_size, token_size,
                      dataset.output_max_length, dataset.output_size,
                      num_blocks, num_heads,
                      SequencePosition(token_size))
    net.load_state_dict(results["net"])

    import random
    num_examples = 4
    index = random.sample(range(len(dataset.val)), num_examples)
    examples, example_labels = dataset.val[index]

    net.to("cpu")
    net.eval()

    predicted = [beam_search(examples[i], net, dataset) for i in range(num_examples)]

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(4, 2, fig)

    def assemble_postcode(patches_data):
        image = np.zeros((28, 224), np.uint8)
        cols = 224 // 4
        for i, patch in enumerate(patches_data):
            r = (i // cols) * 4
            c = (i % cols) * 4
            if isinstance(patch, torch.Tensor):
                patch = patch.cpu().numpy()
            patch = patch.reshape(4, 4)
            image[r:r+4, c:c+4] = patch * 255
        return image

    for i, example in enumerate(examples):
        ax = fig.add_subplot(gs[i, 0])
        postcode = predicted[i]
        image = assemble_postcode(example)
        ax.imshow(image, cmap='gray', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("".join(postcode).upper())

    snapshots = results["snapshots"]
    end_index = 0
    for i, snapshot in enumerate(snapshots):
        if snapshot.loss < snapshots[end_index].loss:
            end_index = i

    ax1 = fig.add_subplot(gs[:, 1])
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


def vit_attention_slider_data(results, num_examples=20):
    """Precompute ViT attention maps for interactive visualization."""
    token_size = results["token_size"]
    num_blocks = results["num_blocks"]
    num_heads = results["num_heads"]
    dataset = SequenceDataset.emnist().to_torch()
    net = VisionTransformer(dataset.max_length, dataset.input_size, token_size,
                            num_blocks, num_heads, dataset.num_classes,
                            SequencePosition(token_size))
    net.load_state_dict(results["net"])
    net.store_class_attention = True
    net.eval()

    examples, _ = dataset.train[:num_examples]
    output = F.softmax(net(examples), dim=1)
    predicted = output.argmax(1)

    frames = []
    for i in range(num_examples):
        x = examples[i].cpu().numpy()
        t = dataset.label_names[predicted[i].item()]
        for b, attention in enumerate(net.class_attention):
            attn = attention[i].mean(0).reshape(-1, 1)
            frames.append({"block": b, "x": x, "attn": attn, "label": None})
        frames.append({"block": -1, "x": x, "attn": None, "label": t})

    return frames


def plot_vit_attention_frame(fig, frame):
    """Plot a single frame of the ViT attention visualization."""
    x = frame["x"]
    attn = frame["attn"]
    label = frame["label"]
    block = frame["block"]

    ax = fig.add_subplot(1, 2, 1)
    if attn is not None:
        masked = x * attn + 0.5 * (1 - attn)
        ax.imshow(assemble(masked), cmap='viridis', interpolation='nearest')
        ax.set_xlabel(f"block {block}")
    else:
        ax.imshow(assemble(x), cmap='gray', interpolation='nearest')
        ax.set_xlabel("final")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(1, 2, 2)
    ax.set_axis_off()
    if label:
        ax.text(0.5, 0.5, label, fontsize=120, ha="center", va="center")
    fig.tight_layout()


def language_transformer_slider_data(results, num_examples=20):
    """Precompute per-timestep transformer outputs for interactive visualization."""
    token_size = results["token_size"]
    num_blocks = results["num_blocks"]
    num_heads = results["num_heads"]
    dataset = SequenceDataset.language(6, 1000).to_torch()
    net = DecoderOnlyTransformer(dataset.max_length, dataset.input_size, token_size,
                                 num_blocks, num_heads,
                                 SequencePosition(token_size))
    net.supervision = SupervisionKind.SemiSupervised
    net.class_output = nn.Linear(net.token_size, dataset.num_classes)

    net.load_state_dict(results["net"])
    net.to("cpu")
    net.eval()

    import random
    random.seed(42)
    indices = random.sample(range(len(dataset.train)), num_examples)

    frames = []
    for i in indices:
        word_tensor = dataset.train[i][0]
        word = [dataset.tokens[c.item()] for c in word_tensor]
        label = dataset.train[i][1].item()

        x = word_tensor.unsqueeze(0)
        x_input = F.one_hot(x, net.input_size).float()

        class_outputs = []
        net.keep_tokens = True
        for j in range(1, dataset.max_length + 1):
            with torch.no_grad():
                output, tokens_out = net(x_input[:, :j])
            class_outputs.append(F.softmax(output, -1).cpu().numpy()[0])

        tokens = net.tokens.squeeze(0)[1:].cpu().numpy()
        y = F.softmax(tokens_out, -1).squeeze(0)[1:].cpu().numpy()

        for j in range(dataset.max_length):
            frames.append({
                "word": word[:j+1],
                "label": label,
                "token": tokens[j],
                "token_output": y[j],
                "class_output": class_outputs[j],
            })

    return dataset, frames


def plot_language_transformer_frame(fig, dataset, frame):
    """Plot a single frame of the language transformer visualization."""
    word = frame["word"]
    label = frame["label"]
    token = frame["token"]
    y = frame["token_output"]
    o = frame["class_output"]
    token_size = len(token)

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

    # Token embedding
    ax = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("viridis")
    colors = cmap((token + 3) / 6)
    ax.bar(np.arange(token_size), token, width=1, color=colors)
    ax.set_xticks([0, token_size], [0, token_size - 1])
    ax.set_yticks([-3, 0, 3])
    ax.set_ylim(-3, 3)
    ax.set_title("token")

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


def postcodes_slider_data(results, num_examples=8):
    """Precompute postcodes salience data for interactive visualization."""
    token_size = results["token_size"]
    num_blocks = results["num_blocks"]
    num_heads = results["num_heads"]
    dataset = TranslationDataset.postcodes().to_torch()
    net = Transformer(dataset.input_max_length, dataset.input_size, token_size,
                      dataset.output_max_length, dataset.output_size,
                      num_blocks, num_heads,
                      SequencePosition(token_size))
    net.load_state_dict(results["net"])
    net.to("cpu")
    net.eval()

    import random
    random.seed(42)
    indices = random.sample(range(len(dataset.val)), num_examples)
    examples, _ = dataset.val[indices]

    frames = []
    for i in range(num_examples):
        x = examples[i]
        predicted = beam_search(x, net, dataset)

        # Compute salience for each output position
        for pos in range(len(predicted)):
            x_input = x.unsqueeze(0).requires_grad_(True)
            y_prefix = torch.tensor([dataset.output_tokens.index(c) for c in predicted[:pos+1]])
            y_input = F.one_hot(y_prefix, dataset.output_size).float().unsqueeze(0)
            output = net(x_input, y_input)
            target = output[0, pos].max()
            target.backward()
            salience = x_input.grad[0].abs().sum(-1).numpy()
            frames.append({
                "patches": x.numpy(),
                "salience": salience,
                "predicted": "".join(predicted[:pos+1]).upper(),
                "full_prediction": "".join(predicted).upper(),
            })

    return frames


def assemble_postcode(patches_data):
    """Assemble patches into a postcode image."""
    image = np.zeros((28, 224), np.uint8)
    cols = 224 // 4
    for i, patch in enumerate(patches_data):
        if isinstance(patch, torch.Tensor):
            patch = patch.cpu().numpy()
        r = (i // cols) * 4
        c = (i % cols) * 4
        patch = patch.reshape(4, 4)
        image[r:r+4, c:c+4] = (patch * 255).astype(np.uint8)
    return image


def plot_postcodes_frame(fig, frame):
    """Plot a single frame of the postcodes visualization."""
    ax = fig.add_subplot(3, 1, 1)
    ax.imshow(assemble_postcode(frame["patches"]), cmap='gray', interpolation='none')
    ax.set_axis_off()
    ax.set_title(f"Prediction: {frame['full_prediction']}")

    ax = fig.add_subplot(3, 1, 2)
    salience = frame["salience"]
    # Reshape salience to match patch layout
    sal_image = np.zeros((28, 224), np.float32)
    cols = 224 // 4
    for i, s in enumerate(salience):
        r = (i // cols) * 4
        c = (i % cols) * 4
        sal_image[r:r+4, c:c+4] = s
    ax.imshow(sal_image, cmap='viridis', interpolation='none')
    ax.set_axis_off()

    ax = fig.add_subplot(3, 1, 3)
    ax.text(0.5, 0.5, frame["predicted"], fontsize=48, ha="center", va="center", fontfamily="monospace")
    ax.set_axis_off()
    fig.tight_layout()


def temperature_diagram():
    def softmax(x, t):
        exp = np.exp(x / t)
        return exp / np.sum(exp)

    logits = np.random.randn(10)

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(10, 5))

    for i, (tau, pos) in enumerate([(1, 1), (0.75, 2), (0.5, 3), (0.25, 4)]):
        ax = fig.add_subplot(2, 2, pos)
        ax.bar(range(10), softmax(logits, tau))
        ax.set_title(f"$\\tau = {tau}$")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Training
    train_language_vit()
    train_language_gpt()
    train_emnist_vit()
    train_emnist_gpt()
    train_cifar_vit()
    train_postcodes()

    # Visualization
    temperature_diagram()

    lang_vit_results = load_language_vit()
    show_language_vit(lang_vit_results)

    lang_gpt_results = load_language_gpt()
    lang_gpt_ft_results = load_language_gpt_finetune()
    show_language_gpt(lang_gpt_results, lang_gpt_ft_results)

    emnist_vit_results = load_emnist_vit()
    show_emnist_vit(emnist_vit_results)

    emnist_gpt_results = load_emnist_gpt()
    emnist_gpt_ft_results = load_emnist_gpt_finetune()
    show_emnist_gpt(emnist_gpt_results, emnist_gpt_ft_results)

    cifar_results = load_cifar_vit()
    show_cifar_vit(cifar_results)

    postcodes_results = load_postcodes()
    show_postcodes(postcodes_results)
