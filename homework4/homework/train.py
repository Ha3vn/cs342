import torch
import numpy as np

from .models import Detector, save_model, ClassificationLoss
from .utils import load_detection_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms as t
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = ClassificationLoss()
    # loss_func = torch.nn.CrossEntropyLoss(torch.tensor([0.01, 0.05, 0.02, 0.46, 0.46]))
    loss_func.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.016, momentum=0.92, weight_decay=1e-4)
    epochs = 30

    train_trans = t.Compose(
        (t.ColorJitter(0.3, 0.3, 0.3, 0.3), t.RandomHorizontalFlip(), t.RandomCrop(96), t.ToTensor()))  # 96
    # val_trans = T.Compose((T.CenterCrop(96), T.ToTensor()))

    data = load_detection_data('dense_data/train', transform=train_trans)
    val = load_detection_data('dense_data/valid')

    for epoch in range(epochs):
        model.train()
        count = 0
        total_loss = 0
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_func(y_pred, y.long())
            total_loss = total_loss + loss.item()
            count += 1
            loss.backward()
            optim.step()
            optim.zero_grad()
        print("Epoch: " + str(epoch) + ", Loss: " + str(total_loss / count))

        model.eval()
        count = 0
        accuracy = 0
        for image, label in val:
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            accuracy = accuracy + (pred.argmax(1) == label).float().mean().item()
            count += 1
        print("Epoch: " + str(epoch) + ", Accuracy: " + str(accuracy / count))
        if accuracy / count > 0.87:
            print("break -> done")
            break

    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
