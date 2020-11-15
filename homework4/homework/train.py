import torch

from .models import Detector, save_model
from .utils import load_detection_data


def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector().to(device)

    """
    Your code here, modify your HW3 code
    """
    loss = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-6)

    train_data = load_detection_data('dense_data/train', num_workers=4)

    for epoch in range(50):
        model.train()
        for image, heatmap, delta in train_data:
            image = image.to(device)
            heatmap = heatmap.to(device)

            pred_heatmap = model(image)

            l = loss(pred_heatmap, heatmap).mean()
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
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
