import torch
import torch.nn.functional as f


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        return f.cross_entropy(input, target)


class CNNBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, should_stride=False):
        super().__init__()

        if should_stride:
            stride = 2
        else:
            stride = 1

        self.conv1 = torch.nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(c_out)
        self.conv2 = torch.nn.Conv2d(c_out, c_out, 3, stride=stride, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(c_out)
        self.relu = torch.nn.ReLU()
        self.use_residual = c_in == c_out

    def forward(self, x):
        x_next = self.relu(self.bn1(self.conv1(x)))   # (128, 3, 32, 32)
        x_next = self.conv2(x_next)
        x_next = self.bn2(x_next)
        x_next = self.relu(x_next)
        return x_next


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """
        super().__init__()

        input_channels = 3
        num_classes = 10
        n_layers = 3
        width = 64

        c_in = width
        c_out = width

        layers = list()
        layers.append(torch.nn.Conv2d(input_channels, c_out, 3, padding=1))

        for i in range(n_layers):
            layers.append(CNNBlock(c_in, c_out, should_stride=i % 2 == 0))
            c_in = c_out

        self.feature_extractor = torch.nn.Sequential(*layers)
        self.linear = torch.nn.Linear(c_in, num_classes)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x[:, 0] = (x[:, 0] - 0.5) / 0.5
        x[:, 1] = (x[:, 1] - 0.5) / 0.5
        x[:, 2] = (x[:, 2] - 0.5) / 0.5

        x = self.feature_extractor(x)
        x = x.mean((2, 3))

        return self.linear(x)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
