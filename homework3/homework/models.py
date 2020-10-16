import torch
import torch.nn.functional as F


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
        return F.cross_entropy(input, target)


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
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        input_channels = 3
        num_classes = 10
        n_layers = 6
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
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        x[:, 0] = (x[:, 0] - 0.5) / 0.5
        x[:, 1] = (x[:, 1] - 0.5) / 0.5
        x[:, 2] = (x[:, 2] - 0.5) / 0.5

        x = self.feature_extractor(x)
        x = x.mean((2, 3))

        return self.linear(x)


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
