from .utils import SuperTuxDataset, LABEL_NAMES
from .models import load_model

import argparse
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF


def predict(model, inputs, device='cpu'):
    inputs = inputs.to(device)
    logits = model(inputs)
    return F.softmax(logits, -1)


def draw_bar(axis, preds, labels=None):
    y_pos = np.arange(6)
    axis.barh(y_pos, preds, align='center', alpha=0.5)
    axis.set_xticks(np.linspace(0, 1, 10))
    if labels:
        axis.set_yticks(y_pos)
        axis.set_yticklabels(labels)
    else:
        axis.get_yaxis().set_visible(False)
    axis.get_xaxis().set_visible(False)


def main(args):
    model = load_model()
    model.eval()

    dataset = SuperTuxDataset(args.dataset)

    f, axes = plt.subplots(2, args.n)

    idxes = np.random.randint(0, len(dataset), size=args.n)

    for i, idx in enumerate(idxes):
        img, label = dataset[idx]
        preds = predict(model, img[None], device='cpu').detach().cpu().numpy()

        axes[0, i].imshow(TF.to_pil_image(img))
        axes[0, i].axis('off')
        draw_bar(axes[1, i], preds[0], LABEL_NAMES if i == 0 else None)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='val_data')
    parser.add_argument('-n', type=int, default=6)

    args = parser.parse_args()
    main(args)
