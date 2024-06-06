import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights


class GradCamModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        self.pretrained = model
        self.layerhook.append(self.pretrained.layer4.register_forward_hook(self.forward_hook()))

        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out


def salience(label: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = read_image("train.jpg")

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.to(device)
    model.eval()
    t = weights.meta["categories"].index(label)
    print(t)

    gcmodel = GradCamModel(model).to(device)

    preprocess = weights.transforms()
    x = preprocess(img).unsqueeze(0).to(device)
    t = torch.tensor([t], dtype=torch.long, device=device)

    out, acts = gcmodel(x)
    loss = F.cross_entropy(out, t)
    loss.backward()

    acts = acts.detach().cpu()
    grads = gcmodel.get_act_grads().detach().cpu()

    pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= -pooled_grads[i]

    heatmap_j = torch.mean(acts, dim=1).squeeze().numpy()
    heatmap_j_max = heatmap_j.max()
    heatmap_j = heatmap_j / heatmap_j_max
    heatmap_j = resize(heatmap_j, (224, 224), preserve_range=True)
    x = x.detach().cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    img = np.clip((x * std + mean)[0].transpose(1, 2, 0), 0, 1)

    plt.figure(figsize=(8, 4))
    ax = plt.subplot(1, 2, 1)
    ax.imshow(img, interpolation='nearest')
    ax.axis('off')
    ax = plt.subplot(1, 2, 2)
    mask = np.where(heatmap_j > 0.25, heatmap_j, 0).reshape(224, 224, 1)
    ax.imshow(img * mask, cmap="jet", interpolation="nearest")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{label.replace(' ', '_')}_salience.pdf")


if __name__ == "__main__":
    salience("steam locomotive")
    salience("stone wall")
