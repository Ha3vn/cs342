from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):

    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        self.images = []
        img_to_tensor = transforms.ToTensor()
        reader = csv.reader(open(os.path.join(dataset_path, 'labels.csv')))
        for row in reader:
            if row[0] == 'file':
                continue
            self.images.append(row)
        for i in range(len(self.images)):
            self.images[i].append(img_to_tensor(Image.open(os.path.join(dataset_path, self.images[i][0]))))

    def __len__(self):
        """
        Your code here
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.images[idx][3], LABEL_NAMES.index(self.images[idx][1])


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
