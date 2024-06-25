from tensorboardX import SummaryWriter
from abc import ABC, abstractmethod
from models.prs_utils.components.component import ComponentE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Encoder(ComponentE, ABC):
    def __init__(self, args):
        super(Encoder, self).__init__(args)

    @abstractmethod
    def forward(self, images):
        pass


class DummyEncoder(Encoder):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = None

    def forward(self, x):
        return x


