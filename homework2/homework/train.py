import torchvision

from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_func = ClassificationLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6)
    epochs = 5

    data = load_data('data/train')
    global_step = 0

    for epoch in range(epochs):
        # Set the model to training mode.
        model.train()

        for x, y in data:
            x = x.to(device)
            y = y.to(device)

            # if global_step == 0:
            #     train_logger.add_graph(model, x)

            # if global_step % 100 == 0:
            #     image_grid = (torchvision.utils.make_grid(x) * 255).byte()
            #     train_logger.add_image('image', image_grid, global_step=global_step)

            y_pred = model(x)

            # Compute loss and update model weights.
            loss = loss_func(y_pred, y)
            accuracy = (y_pred.argmax(1) == y).float().mean()

            loss.backward()
            optim.step()
            optim.zero_grad()

            # Add loss to TensorBoard.
            # train_logger.add_scalar('accuracy', accuracy, global_step=global_step)
            # train_logger.add_scalar('loss', loss.item(), global_step=global_step)
            global_step += 1

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
