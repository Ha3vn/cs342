import torch
import numpy as np

from .models import FCN, save_model, ClassificationLoss
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms as T
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = ClassificationLoss()
    # loss_func = torch.nn.CrossEntropyLoss(torch.tensor([0.01, 0.05, 0.02, 0.46, 0.46]))
    loss_func.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.016, momentum=0.92, weight_decay=1e-4)
    epochs = 30

    train_trans = T.Compose((T.ColorJitter(0.3, 0.3, 0.3, 0.3), T.RandomHorizontalFlip(), T.RandomCrop(96), T.ToTensor())) # 96
    # val_trans = T.Compose((T.CenterCrop(96), T.ToTensor()))

    data = load_dense_data('dense_data/train', transform=train_trans)
    val = load_dense_data('dense_data/valid')

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
        if accuracy/count > 0.87:
          print("break -> done")
          break

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                       convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                            label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                            convert('RGB')), global_step, dataformats='HWC')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
