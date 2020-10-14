from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch


def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_func = ClassificationLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6)
    epochs = 5

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

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
