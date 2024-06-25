import timm
from utils.args import *
from models.utils.continual_model import ContinualModel
import torch
from utils import none_or_float
from datasets import get_dataset
from utils.conf import get_device
from models.prs_utils.model import Model

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    # add_aux_dataset_args(parser)

    parser.add_argument('--encoder', type=str, default='resnet_encoder', help='Encoder architecture')
    parser.add_argument('--model_name', type=str, default='mlab_reservoir', help='PRS model name')
    parser.add_argument('--pretrained', type=int, default=1, choices=[0, 1], help='Use pretrained encoder')
    parser.add_argument('--fine_tune', type=int, default=1, choices=[0, 1], help='Finetune encoder')
    parser.add_argument('--clip_grad_type', type=str, default='value', choices=['value', 'norm'], help='Clip grad type')

    # reservoir
    parser.add_argument('--reservoir_name', type=str, default='prs_mlab', help='Reservoir name')
    parser.add_argument('--reservoir_size', type=int, default=2000, help='Reservoir size')
    parser.add_argument('--reallocate_num', type=int, default=500, help='When to break and reallocate buffer')
    parser.add_argument('--q_poa', type=float, default=-0.03, help='Power of allocation 0 ~ 1')
    parser.add_argument('--batch_sampler', type=str, default='random', help='uniform | weighted | random')
    parser.add_argument('--replay_multiple', type=int, default=1, help='Replayed batch size relative to input size')
    parser.add_argument('--crs_remove', type=str, default='largest_delta')
    parser.add_argument('--crs_method', type=str, default='weighted')

    return parser


class PRS(ContinualModel):
    NAME = 'prs'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual', 'multi-label']

    def __init__(self, backbone, loss, args, transform):
        args.clip_grad = 5 if args.clip_grad is None else args.clip_grad
        self.dataset = get_dataset(args)
        self.num_classes = self.dataset.N_CLASSES
        self.device = get_device()
        backbone = Model(args, self.device)
        super().__init__(backbone, loss, args, transform)
        self.current_task = 0

    def begin_task(self, dataset):
        self.offset_1, self.offset_2 = self._compute_offsets(self.current_task)

    def end_task(self, dataset):
        if self.args.save_checkpoints:
            self.save_checkpoints()
        self.current_task += 1

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        self.opt.zero_grad()

        if len(self.net.rsvr) > 0:
            k = int(min(len(self.net.rsvr), inputs.size(0) * self.args.replay_multiple))
            replay_dict = self.net.rsvr.sample(online_stream=labels, num=k)
            if self.net.rsvr_name == 'random' or self.net.rsvr_name == 'prs_mlab':
                merged_imgs = torch.cat([inputs, replay_dict['imgs']], dim=0)
                merged_cats = torch.cat([labels, replay_dict['cats']], dim=0)
            elif 'prs' in self.rsvr_name:
                if self.config['batch_sampler'] == 'random':
                    merged_imgs = torch.cat([inputs, replay_dict['imgs']], dim=0)
                    merged_cats = torch.cat([labels, replay_dict['cats']], dim=0)
                else:
                    to_merge_imgs = [inputs]
                    to_merge_imgs.extend(replay_dict['imgs'])
                    to_merge_cats = [labels]
                    to_merge_cats.extend(replay_dict['cats'])
                    merged_imgs = torch.cat(to_merge_imgs, dim=0)
                    merged_cats = torch.cat(to_merge_cats, dim=0)
        else:
            merged_imgs, merged_cats = inputs, labels

        outputs = self.net(merged_imgs)
        targets = merged_cats
        loss = self.args.loss_w * self.loss(outputs, targets)
        loss.backward()
        self.net.clip_grad(self.opt)
        self.opt.step()

        self.net.rsvr.update(imgs=inputs, cats=labels)

        return loss.item()
    
    def forward(self, x):
        outs = self.net(x)
        offset_1, offset_2 = self._compute_offsets(self.current_task-1)
        return outs[:, :offset_2]