import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data, DetectionSuperTuxDataset, PR, point_close
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """
    epochs = 20
    lr = 0.02
    global_step = 0
    weight = 100

    weights = torch.FloatTensor([0.5, 0.75, 0,25]).to(device)
    loss = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    d_loss = torch.nn.MSELoss(reduction='none').to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7)

    train_data = load_detection_data('dense_data/train', num_workers=4)
    valid_data = load_detection_data('dense_data/valid', num_workers=4)

    print("Training...")
    for epoch in range(epochs):
        print(epoch, "--------------------------------")
        model.train()
        for image, heatmap, box in train_data:
            image = image.to(device)
            heatmap = heatmap.to(device)

            pred_heatmap = model(image)
            mask = heatmap.sum(1)
            mask[mask > 1] = 1
            mask = mask.round()

            l1 = loss(pred_heatmap, heatmap).mean()
            l2 = (mask[:, None] * d_loss(pred_heatmap, heatmap)).mean()
            l = l1 + weight * l2

            # print(image, "***************************")
            # log(train_logger, image, heatmap, pred_heatmap, global_step)
            # train_logger.add_scalar('loss', l.item(), global_step=global_step)
            global_step += 1

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            pr_box = [PR() for _ in range(3)]
            pr_dist = [PR(is_close=point_close) for _ in range(3)]
            # pr_iou = [PR(is_close=box_iou) for _ in range(3)]
            for img, *gts in DetectionSuperTuxDataset('dense_data/valid', min_size=0):
                detections = model.detect(img.to(device))
                for i, gt in enumerate(gts):
                    pr_box[i].add(detections[i], gt)
                    pr_dist[i].add(detections[i], gt)
            print(pr_box[0].average_prec, pr_box[1].average_prec, pr_box[2].average_prec)
            print(pr_dist[0].average_prec, pr_dist[1].average_prec, pr_dist[2].average_prec)
            # valid_logger.add_scalar('pr_box_0', pr_box[0].average_prec, global_step=global_step)
            # valid_logger.add_scalar('pr_box_1', pr_box[1].average_prec, global_step=global_step)
            # valid_logger.add_scalar('pr_box_2', pr_box[2].average_prec, global_step=global_step)
            # valid_logger.add_scalar('pr_dist_0', pr_dist[0].average_prec, global_step=global_step)
            # valid_logger.add_scalar('pr_dist_1', pr_dist[1].average_prec, global_step=global_step)
            # valid_logger.add_scalar('pr_dist_2', pr_dist[2].average_prec, global_step=global_step)

            if pr_box[0].average_prec > 0.83:
                break
        scheduler.step(l.item())
    print("Done")
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
