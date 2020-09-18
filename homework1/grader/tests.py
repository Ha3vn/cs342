"""
EDIT THIS FILE AT YOUR OWN RISK!
It will not ship with your code, editing it will only change the test cases locally, and might make you fail our
remote tests.
"""
import torch
import torch.nn.functional as F

from .grader import Grader, Case, MultiCase

TRAIN_PATH = "data/train"
VALID_PATH = "data/valid"


class DatasetGrader(Grader):
    """SuperTuxDataset"""

    def __init__(self, *a, **ka):
        super().__init__(*a, **ka)
        self.train_data = self.module.utils.SuperTuxDataset(TRAIN_PATH)
        self.valid_data = self.module.utils.SuperTuxDataset(VALID_PATH)

    @Case(score=3)
    def test_size(self):
        """SuperTuxDataset.__len__"""
        assert len(self.train_data) == 21000, 'len(train_data) %d != 21000' % len(self.train_data)
        assert len(self.valid_data) == 9000, 'len(valid_data) %d != 9000' % len(self.valid_data)

    @Case(score=3)
    def test_getitem(self):
        """SuperTuxDataset.__getitem__"""
        for i in range(0, 1000, 100):
            image, label = self.train_data[i]
            assert image.shape == (3, 64, 64), "image.shape==(3,64,64)"
            assert image.dtype == torch.float32, "image.dtype==torch.float32"
            assert isinstance(label, int), "label is int"

    @Case(score=6)
    def test_img_stat(self):
        """image statistics"""
        import numpy as np

        # Test image mean/std
        target_mean, target_std = [0.3521554, 0.30068502, 0.28527516], [0.18182722, 0.18656468, 0.15938024]

        means = [i.mean((1, 2)).numpy() for i, l in self.valid_data]
        mean = np.mean(means, axis=0)
        std = np.std(means, axis=0)

        assert np.allclose(mean, target_mean, rtol=1e-2), "mean %s ~= %s" % (str(mean), str(target_mean))
        assert np.allclose(std, target_std, rtol=1e-2), "std  %s ~= %s" % (str(std), str(target_std))

    @Case(score=3)
    def test_lbl_stat(self):
        """label statistics"""
        import numpy as np

        count = np.bincount([l for i, l in self.valid_data], minlength=6)
        assert np.all(count == 1500), 'label count %s == 1500' % str(count)


class LinearClassifierGrader(Grader):
    """LinearModel"""

    @staticmethod
    def is_linear(cls):
        import numpy as np

        torch.manual_seed(0)
        a = torch.rand(1, 3, 64, 64)
        b = torch.rand(1, 3, 64, 64)
        t = torch.rand(100, 1, 1, 1)
        x = t * a + (1 - t) * b

        v_a = cls(a)
        v_b = cls(b)
        v_x = cls(x)

        return np.allclose((t[:, :, 0, 0] * v_a + (1 - t[:, :, 0, 0]) * v_b).detach().numpy(), v_x.detach().numpy(),
                           atol=1e-2)

    @Case(score=10)
    def test_linearity(self):
        """Linearity"""
        cls = self.module.LinearClassifier()
        assert LinearClassifierGrader.is_linear(cls), "Model is not linear"

    @Case(score=5)
    def test_shape(self):
        """Shape"""

        cls = self.module.LinearClassifier()

        torch.manual_seed(0)
        a = torch.rand(100, 3, 64, 64)
        v_a = cls(a)
        assert v_a.shape == (100, 6), "shape = %s expected (100,6)" % str(v_a.shape)


class LossGrader(Grader):
    """Loss"""

    @MultiCase(score=10, i=range(5), d=range(3, 5))
    def test_forward(self, i, d):
        """ClassificationLoss.forward"""
        import numpy as np
        torch.manual_seed(i)
        label = torch.randint(d, (1,), dtype=torch.int64)
        x = torch.rand(1, d)

        loss = self.module.ClassificationLoss()(x, label)
        true_loss = F.cross_entropy(x, label)

        assert np.isclose(loss.numpy(), true_loss.numpy(), rtol=1e-2), "%s != %s" % (str(loss), str(true_loss))


def accuracy(outputs, labels):
    return (outputs.argmax(1).type_as(labels) == labels).float()


def load_data(dataset, num_workers=0, batch_size=128):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)


class TrainedLinearClassifierGrader(Grader):
    """TrainedLinearModel"""

    @staticmethod
    def accuracy(module, model):
        cls = module.load_model(model)
        cls.eval()
        if model == 'linear':
            assert LinearClassifierGrader.is_linear(cls), "Model is not linear"

        accs = []
        for img, label in load_data(module.utils.SuperTuxDataset(VALID_PATH)):
            accs.extend(accuracy(cls(img), label).numpy())

        return sum(accs) / len(accs)

    @Case(score=30)
    def test_accuracy(self):
        """Accuracy"""
        acc = TrainedLinearClassifierGrader.accuracy(self.module, 'linear')
        return max(min(acc, 0.7) - 0.45, 0) / (0.7 - 0.45), 'accuracy = %0.3f' % acc


class TrainedMLPClassifierGrader(Grader):
    """TrainedMLPModel"""

    @Case(score=30)
    def test_accuracy(self):
        """Accuracy"""
        acc = TrainedLinearClassifierGrader.accuracy(self.module, 'mlp')
        return max(min(acc, 0.8) - 0.5, 0) / (0.8 - 0.5), 'accuracy = %0.3f' % acc
