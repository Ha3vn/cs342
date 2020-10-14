"""
EDIT THIS FILE AT YOUR OWN RISK!
It will not ship with your code, editing it will only change the test cases locally, and might make you fail our
remote tests.
"""
import torch
from .grader import Grader, Case, MultiCase

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_data(dataset, num_workers=0, batch_size=128):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)


class TunedCNNClassifierGrader(Grader):
    """Tuned CNN model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls = self.module.load_model('cnn')
        cls.eval()
        cls = cls.to(device)

        confusion = self.module.ConfusionMatrix(6)
        with torch.no_grad():
            for img, label in load_data(self.module.utils.SuperTuxDataset('data/valid')):
                confusion.add(cls(img.to(device)).argmax(1).cpu(), label)

        self.accuracy = confusion.global_accuracy

    @Case(score=40)
    def test_accuracy(self, min_val=0.86, max_val=0.90):
        """Accuracy"""
        v = self.accuracy
        return max(min(v, max_val) - min_val, 0) / (max_val - min_val), 'accuracy = %0.3f' % v

    @Case(score=10, extra_credit=True)
    def test_accuracy_extra(self, min_val=0.90, max_val=0.94):
        """Accuracy (extra credit)"""
        v = self.accuracy
        return max(min(v, max_val) - min_val, 0) / (max_val - min_val), 'accuracy = %0.3f' % v


class FCNGrader(Grader):
    """FCN Grader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.module.FCN()
        self.model.eval()

    @MultiCase(score=20, shape=[(2**i, 2**i) for i in range(10)] + [(2**(5-i), 2**i) for i in range(5)])
    def test_shape(self, shape):
        """Shape"""
        v = torch.zeros(1, 3, *shape)
        o = self.model(v)
        assert o.shape[2:] == v.shape[2:] and o.size(1) == 5 and o.size(0) == 1,\
            'Output shape (1, 5, %d, %d) expected, got (%d, %d, %d, %d)' % (v.size(2), v.size(3), o.size(0), o.size(1),
                                                                            o.size(2), o.size(3))


class TrainedFCNGrader(Grader):
    """Trained FCN Grader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls = self.module.load_model('fcn')
        cls.eval()
        cls = cls.to(device)

        self.c = self.module.ConfusionMatrix()
        for img, label in load_data(self.module.utils.DenseSuperTuxDataset('dense_data/valid')):
            self.c.add(cls(img.to(device)).argmax(1), label.to(device))

    @Case(score=20)
    def test_global_accuracy(self, min_val=0.70, max_val=0.85):
        """Global accuracy"""
        v = self.c.global_accuracy
        return max(min(v, max_val) - min_val, 0) / (max_val - min_val), '%0.3f' % v

    @Case(score=20)
    def test_iou(self, min_val=0.30, max_val=0.55):
        """Intersection over Union"""
        v = self.c.iou
        return max(min(v, max_val) - min_val, 0) / (max_val - min_val), '%0.3f' % v
