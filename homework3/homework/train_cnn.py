from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torchvision.transforms as T
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = ClassificationLoss()
    # loss_func = torch.nn.CrossEntropyLoss(torch.tensor([0.05, 0.2, 0.05, 0.35, 0.35]))
    loss_func.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.92, weight_decay=1e-4)
    epochs = 50

    train_trans = T.Compose((T.ToPILImage(), T.ColorJitter(0.8, 0.3), T.RandomHorizontalFlip(), T.RandomCrop(32), T.ToTensor())) # 96
    val_trans = T.Compose((T.ToPILImage(), T.CenterCrop(size=32), T.ToTensor()))

    data = load_data('data/train', transform=train_trans)
    val = load_data('data/valid', transform=val_trans)

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
        print("Epoch: " + str(epoch) + ", Loss: " + str(total_loss/count))

        model.eval()
        count = 0
        accuracy = 0
        for image, label in val:
          image = image.to(device)
          label = label.to(device)
          pred = model(image)
          accuracy = accuracy + (pred.argmax(1) == label).float().mean().item()
          count += 1
        print("Epoch: " + str(epoch) + ", Accuracy: " + str(accuracy/count))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
