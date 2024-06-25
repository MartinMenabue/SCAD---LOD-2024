import torch.nn as nn
import torch.distributed as dist
import torch
import numpy as np
from PIL import Image
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple

from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import NO_LABEL_PLACEHOLDER
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.methods.base_method import BaseMethod
import timm
from my_utils.buffer import MyBuffer
import torchvision
import wandb
import torch.nn.functional as F


class Model(BaseMethod):
    """
    A finetuning (Experience Replay) baseline.
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, config)

        self.net = timm.create_model(self.config['network'], pretrained=True, num_classes=self.num_classes)
        self.current_task = 0
        self.buffer_size = self.config['buffer_size']
        self.device = self.config['device']
        self.buffer = MyBuffer(self.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.seen_y_so_far = torch.zeros(self.num_classes).bool().to(self.device)
        self.transform = self.get_transforms(self.config['dataset'])
        # setup losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def get_transforms(self, dataset_name):
        if "cifar100" in dataset_name:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(224),
                torchvision.transforms.RandomCrop(224, padding=28),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        elif "imagenet" in dataset_name:
            normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ])
        return transform
    
    def get_optimizer(self):
        if self.config['optimizer'] == 'sgd':
            opt = torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.config['weight_decay'])
        else:
            raise ValueError('unsupported optimizer: {}'.format(self.config['optimizer']))
        return opt

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        pass

    def _prepare_model_for_new_task(self, **kwargs) -> None:
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
            prepare_model_for_task function)
        """
        self.opt = self.get_optimizer()

        pass

    def observe(self, x: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
                train: bool = True, epoch=0, not_aug_inputs=None) -> Tuple[torch.Tensor, float]:
        """
        The method used for training and validation, returns a tensor of model predictions and the loss
        This function needs to be defined in the inheriting method class

        Args:
            x (torch.Tensor): The batch of images
            y (torch.Tensor): A 2-d batch indicator tensor of shape (number of samples x number of classes)
            in_buffer (Optional[torch.Tensor]): A 1-d boolean tensor which indicates which sample is from the buffer.
            train (bool): Whether this is training or validation/test

        Returns:
            Tuple[torch.Tensor, float]:
            predictions (torch.Tensor) : a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
            loss (float): the value of the loss
        """

        num_seen_classes = len(self.seen_classes)
        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)
        target = y
        assert target.shape[1] == offset_2
        output, _ = self.forward_net(x)
        output_mask = self.seen_y_so_far.unsqueeze(0).expand_as(output).detach().clone()
        filtered_output = output[:, :offset_2]
        idx = target.sum(0).nonzero().squeeze(1)
        filtered_output = filtered_output[:, idx]
        filtered_target = target[:, idx]
        loss = self.bce(filtered_output / self.temperature, filtered_target)
        self.seen_y_so_far[:offset_2] |= y.any(dim=0).data
        
        loss_der = torch.tensor(0.0).to(self.device)
        loss_re = torch.tensor(0.0).to(self.device)
        if not self.buffer.is_empty():
            buf_inputs_1, _, buf_logits_1, buf_logits_mask_1, _ = self.buffer.get_data(
                self.config['batch_size'], transform=self.transform) 
            buf_outputs_1 = self.net(buf_inputs_1)
            # ignore unseen classes in targets
            buf_logits_1[~buf_logits_mask_1] = 0.0
            buf_outputs_1[~buf_logits_mask_1] = 0.0
            loss_der = F.mse_loss(buf_outputs_1, buf_logits_1)
            loss += self.config['der_alpha'] * loss_der

            buf_inputs_2, buf_labels_2, _, _, _ = self.buffer.get_data(
                self.config['batch_size'], transform=self.transform)
            
            loss_re = self.bce(self.net(buf_inputs_2)[:, :offset_2], buf_labels_2[:, :offset_2].float())
            loss += self.config['der_beta'] * loss_re

        # TODO weigh the buffer loss by the self.memory_strength before getting the loss mean (use in_buffer)
        if train:
            self.opt.zero_grad()
            loss.backward()
            if self.config['clip_grad'] is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config['clip_grad'])
            self.opt.step()

            if output_mask.sum() > 0:
                to_save_labels = torch.cat((target, torch.zeros(target.shape[0], self.num_classes - target.shape[1]).to(self.device)), dim=1)
                self.buffer.add_data(examples=not_aug_inputs, labels=to_save_labels, logits=output.data, logits_mask=output_mask.data,
                                     task_labels=torch.ones(y.shape[0])*self.current_task)
            
            if self.config['wandb_log']:
                wandb.log({"loss": loss.item(), "loss_der": loss_der.item(), "loss_re": loss_re.item()})

        predictions = output[:, :offset_2] > 0.0
        return predictions, loss.item()

    def forward_net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x), None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The method used during inference, returns a tensor of model predictions

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            torch.Tensor: a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        """
        num_seen_classes = len(self.seen_classes)
        output, _ = self.forward_net(x)
        output = output[:, :num_seen_classes]
        predictions = output > 0.0
        return predictions

    def _consolidate_epoch_knowledge(self, **kwargs) -> None:
        """
        A method specific function that takes place after training on each epoch (runs from the
        consolidate_epoch_knowledge function)
        """
        pass

    def consolidate_task_knowledge(self, **kwargs) -> None:
        """Takes place after training on each task"""

        if self.current_task == 1:
            print()

        with torch.no_grad():
            # Update future past logits
            buf_idx, buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_logits_mask = self.buffer.get_data(self.buffer.buffer_size,
                                   transform=self.transform, return_index=True)
            
            #cur_task_labels = torch.ones(buf_labels.shape[0])*self.current_task
            cur_logits_mask = output_mask = self.seen_y_so_far.unsqueeze(0).expand_as(buf_logits_mask).detach().clone()
            buf_outputs = []
            while len(buf_inputs):
                buf_outputs.append(self.net(buf_inputs[:self.config['batch_size']]))
                buf_inputs = buf_inputs[self.config['batch_size']:]
            buf_outputs = torch.cat(buf_outputs)

            chosen = torch.logical_xor(buf_logits_mask, cur_logits_mask)

            if chosen.any() and self.current_task == 1:
                #TODO: gestisci update logits
                to_transplant = self.update_logits(buf_logits, buf_outputs, buf_labels, chosen, self.current_task, self.num_tasks - self.current_task)
                self.buffer.logits[buf_idx[chosen], :] = to_transplant.to(self.buffer.device)
                self.buffer.task_labels[buf_idx[chosen]] = self.task

        self.current_task += 1

        pass

    def update_logits(self, old, new, gt, chosen, task_start, n_tasks=1):


        #transplant = new[:, task_start * self.cpt:(task_start + n_tasks) * self.cpt]

        #gt_values = old[torch.arange(len(gt)), gt]
        max_values = []
        gt_max_values = []
        for i, el in enumerate(gt):
            gt_max_values.append(old[i, el.bool()].max())
            max_values.append(new[i, el.bool()].max())
        gt_max_values = torch.stack(gt_max_values)
        max_values = torch.stack(max_values)
        #gt_max_values = old[gt].max(1).values
        #max_values = new[chosen].max(1).values
        #max_values = transplant.max(1).values
        coeff = self.config['xder_gamma'] * gt_max_values / max_values
        for i, el in enumerate(coeff):
            

        coeff = coeff.unsqueeze(1).repeat(1, sum([self.n_cla_per_tsk[x] for x in range(self.current_task, n_tasks)]))
        mask = (max_values > gt_values).unsqueeze(1).repeat(1, self.cpt * n_tasks)
        transplant[mask] *= coeff[mask]
        old[:, task_start * self.cpt:(task_start + n_tasks) * self.cpt] = transplant

        return old


class Buffer(BufferBase):
    def __init__(self,
                 config: Dict,
                 buffer_dir: Optional[str] = None,
                 map_size: int = 1e9,
                 essential_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 augmentation_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        super(Buffer, self).__init__(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    def _reduce_exemplar_set(self, **kwargs) -> None:
        """remove extra exemplars from the buffer"""
        for label in self.seen_classes:
            if len(self.mem_class_x[label]) > self.n_mems_per_cla:
                n = len(self.mem_class_x[label]) - self.n_mems_per_cla
                self.remove_samples(label, n)

    def _construct_exemplar_set(self, task_data: Dataset, dist_args: Optional[Dict] = None, **kwargs) -> None:
        """
        update the buffer with the new task exemplars, chosen randomly for each class.

        Args:
            new_task_data (Dataset): The new task data
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
        """
        distributed = dist_args is not None
        if distributed:
            rank = dist_args['rank']
        else:
            rank = 0
        new_class_labels = task_data.cur_task

        for class_label in new_class_labels:
            num_images_to_add = min(self.n_mems_per_cla, self.max_mems_pool_size)
            class_images_indices = task_data.get_image_indices_by_cla(class_label, num_images_to_add)
            if distributed:
                device = torch.device(f"cuda:{dist_args['gpu']}")
                class_images_indices_to_broadcast = torch.from_numpy(class_images_indices).to(device)
                torch.distributed.broadcast(class_images_indices_to_broadcast, 0)
                class_images_indices = class_images_indices_to_broadcast.cpu().numpy()

            for image_index in class_images_indices:
                image, label1, label2 = task_data.get_item(image_index)
                if label2 != NO_LABEL_PLACEHOLDER:
                    warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
                self.add_sample(class_label, image, (label1, label2), rank=rank)
