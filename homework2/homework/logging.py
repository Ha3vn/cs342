from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        accuracies = list()
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            train_logger.add_scalar('loss', dummy_train_loss, global_step)
            accuracies.append(dummy_train_accuracy.mean().item())
            global_step += 1
        accuracy = torch.FloatTensor(accuracies).mean().item()
        train_logger.add_scalar('accuracy', accuracy, global_step)
        torch.manual_seed(epoch)
        accuracies = list()
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            accuracies.append(dummy_validation_accuracy.mean().item())
        accuracy = torch.FloatTensor(accuracies).mean().item()
        valid_logger.add_scalar('accuracy', accuracy, global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
