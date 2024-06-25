import torch
import torch.nn as nn
from models.prs_utils.components import E
from models.prs_utils.reservoir import reservoir


class Model(nn.Module):
    def __init__(self, args, device) -> None:
        super().__init__()
        self.args = args
        self.device = device
        self.encoder = E[args.encoder](args)
        self.encoder = self.encoder.to(self.device)
        self.rsvr = reservoir[args.reservoir_name](args, device)
        self.rsvr_name = args.reservoir_name
    
    def clip_grad(self, opt):
        if self.args.clip_grad_type == 'value':
            for group in opt.param_groups:
                nn.utils.clip_grad_value_(group['params'], self.args.clip_grad)
        elif self.args.clip_grad_type == 'norm':
            for group in self.optimizer.param_groups:
                nn.utils.clip_grad_norm_(group['params'], self.args.clip_grad, norm_type=2)
        else:
            raise ValueError('Invalid clip_grad type: {}'
                             .format(self.args.clip_grad_type))

    def forward(self, x):
        return self.encoder(x)
    









