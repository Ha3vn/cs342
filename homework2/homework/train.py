from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import load_data
import torch


def train():
    model = CNNClassifier()
    """
    Your code here, modify your HW1 code

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = ClassificationLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95, weight_decay=1e-6)
    epochs = 10

    data = load_data('data/train')

    for epoch in range(epochs):
        model.train()
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optim.step()
            optim.zero_grad()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    args = parser.parse_args()
    train()
