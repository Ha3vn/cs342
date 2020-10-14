"""
EDIT THIS FILE AT YOUR OWN RISK!
It will not ship with your code, editing it will only change the test cases locally, and might make you fail our
remote tests.
"""
import torch
import torch.utils.tensorboard as tb

from .grader import Grader, Case

VALID_PATH = "data/valid"


class CNNClassifierGrader(Grader):
    """CNN model"""

    @staticmethod
    def has_conv(model):
        trace = torch.jit.trace(model, torch.randn((1, 3, 64, 64)))
        if torch.__version__ < '1.5.0':
            graph = trace.graph
        else:
            graph = trace.inlined_graph
        for g in graph.nodes():
            if g.kind() == 'aten::_convolution':
                return True
        return False

    @Case(score=10)
    def test_cnn(self, min_val=0.75, max_val=0.85):
        """is a cnn"""
        assert self.has_conv(self.module.CNNClassifier()), "You model should use convolutions"


class DummyFileWriter(tb.FileWriter):
    def __init__(self):
        self.events = []
        self.log_dir = None

    def add_event(self, e, step=None, walltime=None):
        self.events.append((e, step, walltime))


class DummySummaryWriter(tb.SummaryWriter):
    def __init__(self):
        self.log_dir = None
        self.file_writer = self.all_writers = None
        self._get_file_writer()

    def _get_file_writer(self):
        if self.file_writer is None:
            self.file_writer = DummyFileWriter()
            self.all_writers = {None: self.file_writer}
        return self.file_writer


class LogGrader(Grader):
    """Log correctness"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_logger = DummySummaryWriter()
        valid_logger = DummySummaryWriter()
        self.module.test_logging(train_logger, valid_logger)
        self.train_events = train_logger.file_writer.events
        self.valid_events = valid_logger.file_writer.events

    @staticmethod
    def get_val(events, tag):
        values = {}
        for e, s, _ in events:
            if e.HasField('summary'):
                for v in e.summary.value:
                    if v.tag == tag:
                        values[s] = v.simple_value
        return values

    @Case(score=10)
    def test_train_loss(self, min_val=0, max_val=4):
        """Log training loss"""
        loss = self.get_val(self.train_events, 'loss')
        for step in range(200):
            expect = 0.9 ** (step / 20.)
            assert step in loss, 'no loss found for [epoch=%d, iteration=%d]' % (step // 20, step % 20)
            got = loss[step]
            assert abs(got - expect) < 1e-2, \
                'loss [epoch=%d, iteration=%d] expected %f got %f' % (step // 20, step % 20, expect, got)

    @Case(score=10)
    def test_train_acc(self, min_val=0, max_val=2):
        """Log training accuracies"""
        acc = self.get_val(self.train_events, 'accuracy')
        for epoch in range(10):
            torch.manual_seed(epoch)
            expect = epoch / 10. + torch.mean(torch.cat([torch.randn(10) for i in range(20)]))
            assert 20 * epoch + 19 in acc or 20 * epoch + 20 in acc, 'No accuracy logging found for epoch %d' % epoch
            got = acc[20 * epoch + 19] if 20 * epoch + 19 in acc else acc[20 * epoch + 20]
            assert abs(got - expect) < 1e-2, 'accuracy [epoch=%d] expected %f got %f' % (epoch, expect, got)

    @Case(score=10)
    def test_valid_acc(self, min_val=0, max_val=2):
        """Log valid accuracies"""
        acc = self.get_val(self.valid_events, 'accuracy')
        for epoch in range(10):
            torch.manual_seed(epoch)
            expect = epoch / 10. + torch.mean(torch.cat([torch.randn(10) for i in range(10)]))
            assert 20 * epoch + 19 in acc or 20 * epoch + 20 in acc, 'No accuracy logging found for epoch %d' % epoch
            got = acc[20 * epoch + 19] if 20 * epoch + 19 in acc else acc[20 * epoch + 20]
            assert abs(got - expect) < 1e-2, 'accuracy [epoch=%d] expected %f got %f' % (epoch, expect, got)


def accuracy(outputs, labels):
    return (outputs.argmax(1).type_as(labels) == labels).float()


def load_data(dataset, num_workers=0, batch_size=128):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)


class TrainedCNNClassifierGrader(Grader):
    """Trained CNN model"""

    @staticmethod
    def accuracy(module):
        cls = module.load_model()
        cls.eval()

        accs = []
        for img, label in load_data(module.utils.SuperTuxDataset(VALID_PATH)):
            accs.extend(accuracy(cls(img), label).numpy())

        return sum(accs) / len(accs)

    @Case(score=60)
    def test_accuracy(self, min_val=0.75, max_val=0.85):
        """Accuracy"""
        acc = self.accuracy(self.module)
        return max(min(acc, max_val) - min_val, 0) / (max_val - min_val), 'accuracy = %0.3f' % acc
