from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter


##################
# reservoir ABCs #
##################

class rsvrBase(ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rsvr_size = args.reservoir_size
        self.rsvr = {}
        self.n = 0


    @abstractmethod
    def update(self, **args):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def sample(self, num):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def write(self, writer:SummaryWriter, step):
        writer.add_text(
            'train/reservoir_summary',
            str(self), step
        )
