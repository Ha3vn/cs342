from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data


def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    raise NotImplementedError('train')

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
