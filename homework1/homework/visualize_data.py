import argparse
import matplotlib.pyplot as plt
from .utils import LABEL_NAMES, SuperTuxDataset


def visualize_data(args):
    dataset = SuperTuxDataset(args.dataset)

    f, axes = plt.subplots(args.n, len(LABEL_NAMES))

    counts = [0]*len(LABEL_NAMES)

    for img, label in dataset:
        c = counts[label]
        if c < args.n:
            ax = axes[c][label]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title(LABEL_NAMES[label])
            counts[label] += 1
        if sum(counts) >= args.n * len(LABEL_NAMES):
            break

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('-n', type=int, default=3)
    args = parser.parse_args()

    visualize_data(args)
