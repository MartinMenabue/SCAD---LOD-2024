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
import wandb
from iirc.datasets_loader import get_lifelong_datasets
from experiments.main import get_transforms
from lifelong_methods.utils import transform_labels_names_to_vector
from my_utils.status import progress_bar


class Model(BaseMethod):
    """
    A finetuning (Experience Replay) baseline.
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, config)

        self.net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=self.num_classes)
        self.current_task = 0
        self.device = self.config['device']
        self.n_tasks = len(self.n_cla_per_tsk)
        self.old_data = []
        self.essential_transforms_fn, self.augmentation_transforms_fn = get_transforms(self.config['dataset'])

        # setup losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
    
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
        
        predictions = torch.zeros_like(y).bool()
        return predictions, 0.0
    
    def forward_net(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def train_joint(self):
        lifelong_datasets, tasks, class_names_to_idx = \
        get_lifelong_datasets(self.config['dataset'], dataset_root=self.config['dataset_path'],
                              tasks_configuration_id=self.config["tasks_configuration_id"],
                              essential_transforms_fn=self.essential_transforms_fn,
                              augmentation_transforms_fn=self.augmentation_transforms_fn, cache_images=False,
                              joint=True)
        loader = torch.utils.data.DataLoader(lifelong_datasets['train'], batch_size=self.config['batch_size'], shuffle=True)

        self.seen_classes = [y for x in tasks for y in x]
        self.cur_task_id = 21
        self.current_task = 21
        
        self.opt = self.get_optimizer()

        for e in range(self.config['epochs_per_task']):
            for i, minibatch in enumerate(loader):
                if self.config['debug_mode'] and i > 10:
                    break
                labels_names = list(zip(minibatch[1], minibatch[2]))
                labels = transform_labels_names_to_vector(
                    labels_names, 115, class_names_to_idx
                )
                images = minibatch[0]
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.opt.zero_grad()
                outputs = self.net(images)
                loss = self.bce(outputs, labels)
                if self.config['wandb_log']:
                    wandb.log({'loss': loss.item()})
                loss.backward()
                if self.config['clip_grad'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config['clip_grad'])
                self.opt.step()
                progress_bar(i, len(loader), e, 'J', loss.item())
        if self.config['save_checkpoints']:
            self.save_checkpoints()


    def consolidate_task_knowledge(self, **kwargs) -> None:
        """Takes place after training on each task"""
        self.current_task += 1
        if self.current_task != self.n_tasks:
                return
        
        lifelong_datasets, tasks, class_names_to_idx = \
        get_lifelong_datasets(self.config['dataset'], dataset_root=self.config['dataset_path'],
                              tasks_configuration_id=self.config["tasks_configuration_id"],
                              essential_transforms_fn=self.essential_transforms_fn,
                              augmentation_transforms_fn=self.augmentation_transforms_fn, cache_images=False,
                              joint=True)
        loader = torch.utils.data.DataLoader(lifelong_datasets['train'], batch_size=self.config['batch_size'], shuffle=True)
        
        self.opt = self.get_optimizer()

        for e in range(self.config['epochs_per_task']):
            for i, minibatch in enumerate(loader):
                labels_names = list(zip(minibatch[1], minibatch[2]))
                labels = transform_labels_names_to_vector(
                    labels_names, self.num_classes, class_names_to_idx
                )
                images = minibatch[0]
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.opt.zero_grad()
                outputs = self.net(images)
                loss = self.bce(outputs, labels)
                if self.config['wandb_log']:
                    wandb.log({'loss': loss.item()})
                loss.backward()
                if self.config['clip_grad'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config['clip_grad'])
                self.opt.step()
                progress_bar(i, len(loader), e, 'J', loss.item())
        pass


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
