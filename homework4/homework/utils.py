import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from . import dense_transforms

DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]


class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor(), min_size=20):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform
        self.min_size = min_size

    def _filter(self, boxes):
        if len(boxes) == 0:
            return boxes
        return boxes[abs(boxes[:, 3] - boxes[:, 1]) * abs(boxes[:, 2] - boxes[:, 0]) >= self.min_size]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import numpy as np
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        nfo = np.load(b + '_boxes.npz')
        data = im, self._filter(nfo['karts']), self._filter(nfo['bombs']), self._filter(nfo['pickup'])
        if self.transform is not None:
            data = self.transform(*data)
        return data


def point_in_box(pred, lbl):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return (x0 <= px) & (px < x1) & (y0 <= py) & (py < y1)


def point_close(pred, lbl, d=5):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return ((x0 + x1 - 1) / 2 - px) ** 2 + ((y0 + y1 - 1) / 2 - py) ** 2 < d ** 2


def box_iou(pred, lbl, t=0.5):
    px, py, pw2, ph2 = pred[:, None, 0], pred[:, None, 1], pred[:, None, 2], pred[:, None, 3]
    px0, px1, py0, py1 = px - pw2, px + pw2, py - ph2, py + ph2
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    iou = (abs(torch.min(px1, x1) - torch.max(px0, x0)) * abs(torch.min(py1, y1) - torch.max(py0, y0))) / \
          (abs(torch.max(px1, x1) - torch.min(px0, x0)) * abs(torch.max(py1, y1) - torch.min(py0, y0)))
    return iou > t


class PR:
    def __init__(self, min_size=20, is_close=point_in_box):
        self.min_size = min_size
        self.total_det = 0
        self.det = []
        self.is_close = is_close

    def add(self, d, lbl):
        lbl = torch.as_tensor(lbl.astype(float), dtype=torch.float32).view(-1, 4)
        d = torch.as_tensor(d, dtype=torch.float32).view(-1, 5)
        all_pair_is_close = self.is_close(d[:, 1:], lbl)

        # Get the box size and filter out small objects
        sz = abs(lbl[:, 2] - lbl[:, 0]) * abs(lbl[:, 3] - lbl[:, 1])

        # If we have detections find all true positives and count of the rest as false positives
        if len(d):
            detection_used = torch.zeros(len(d))
            # For all large objects
            for i in range(len(lbl)):
                if sz[i] >= self.min_size:
                    # Find a true positive
                    s, j = (d[:, 0] - 1e10 * detection_used - 1e10 * ~all_pair_is_close[:, i]).max(dim=0)
                    if not detection_used[j] and all_pair_is_close[j, i]:
                        detection_used[j] = 1
                        self.det.append((float(s), 1))

            # Mark any detection with a close small ground truth as used (no not count false positives)
            detection_used += all_pair_is_close[:, sz < self.min_size].any(dim=1)

            # All other detections are false positives
            for s in d[detection_used == 0, 0]:
                self.det.append((float(s), 0))

        # Total number of detections, used to count false negatives
        self.total_det += int(torch.sum(sz >= self.min_size))

    @property
    def curve(self):
        true_pos, false_pos = 0, 0
        r = []
        for t, m in sorted(self.det, reverse=True):
            if m:
                true_pos += 1
            else:
                false_pos += 1
            prec = true_pos / (true_pos + false_pos)
            recall = true_pos / self.total_det
            r.append((prec, recall))
        return r

    @property
    def average_prec(self, n_samples=11):
        import numpy as np
        pr = np.array(self.curve, np.float32)
        return np.mean([np.max(pr[pr[:, 1] >= t, 0], initial=0) for t in np.linspace(0, 1, n_samples)])


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


def load_detection_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    transform = dense_transforms.Compose(
        [dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor(), dense_transforms.ToHeatmap()])
    dataset = DetectionSuperTuxDataset(dataset_path, transform=transform, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = DetectionSuperTuxDataset('dense_data/train')
    import torchvision.transforms.functional as F
    from pylab import show, subplots
    import matplotlib.patches as patches
    import numpy as np

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[100 + i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='r', lw=2))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='g', lw=2))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='b', lw=2))
        ax.axis('off')
    dataset = DetectionSuperTuxDataset('dense_data/train',
                                       transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),
                                                                           dense_transforms.ToTensor()]))
    fig.tight_layout()
    # fig.savefig('box.png', bbox_inches='tight', pad_inches=0, transparent=True)

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, *dets = dataset[100 + i]
        hm, size = dense_transforms.detections_to_heatmap(dets, im.shape[1:])
        ax.imshow(F.to_pil_image(im), interpolation=None)
        hm = hm.numpy().transpose([1, 2, 0])
        alpha = 0.25 * hm.max(axis=2) + 0.75
        r = 1 - np.maximum(hm[:, :, 1], hm[:, :, 2])
        g = 1 - np.maximum(hm[:, :, 0], hm[:, :, 2])
        b = 1 - np.maximum(hm[:, :, 0], hm[:, :, 1])
        ax.imshow(np.stack((r, g, b, alpha), axis=2), interpolation=None)
        ax.axis('off')
    fig.tight_layout()
    # fig.savefig('heat.png', bbox_inches='tight', pad_inches=0, transparent=True)

    show()
