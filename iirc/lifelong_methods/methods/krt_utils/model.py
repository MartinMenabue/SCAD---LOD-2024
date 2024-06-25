import torch
from lifelong_methods.methods.krt_utils.tresnet import TResnetM
import os
from pathlib import Path
import urllib.request
import torch.nn as nn
from lifelong_methods.methods.krt_utils.loss_functions.losses import AsymmetricLoss
from lifelong_methods.methods.krt_utils.loss_functions.distillation import pod, embeddings_similarity
from torch.optim import lr_scheduler
from copy import deepcopy
import math
from lifelong_methods.methods.krt_utils.sample_proto import icarl_sample_protos, random_sample_protos
import sys
import time
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import default_collate, DataLoader
import importlib
import inspect
from iirc.datasets_loader import get_lifelong_datasets
from my_utils.inspect_utils import get_transforms


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=tqdm(total=100)

        downloaded = block_num * block_size
        perc = int(downloaded * 100 / total_size)
        self.pbar.update(perc - self.pbar.n)
        if downloaded >= total_size:
            self.pbar.close()

class Model(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        essential_transforms_fn, augmentation_transforms_fn = get_transforms(self.args.dataset)
        self._lifelong_datasets, self._tasks, self._class_names_to_idx = \
            get_lifelong_datasets(self.args.dataset, dataset_root=self.args.dataset_path,
                                    tasks_configuration_id=self.args.tasks_configuration_id,
                                    essential_transforms_fn=essential_transforms_fn,
                                    augmentation_transforms_fn=augmentation_transforms_fn, cache_images=False,
                                    joint=self.args.joint)
        cpt = [len(self._tasks[i]) for i in range(len(self._tasks))]
        self.num_classes = cpt[0]
        model_params = {'args': args, 'num_classes': self.num_classes}
        model = TResnetM(model_params)
        ckpt_path = Path.home().joinpath('.cache', 'iirc', 'tresnet_m_224_21k.pth')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path.parent, exist_ok=True)
            print('Downloading TResNet weights...', file=sys.stderr)
            urllib.request.urlretrieve('https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ASL/MS_COCO_TRresNet_M_224_81.8.pth', ckpt_path, MyProgressBar())
            print('Done.', file=sys.stderr)
        state = torch.load(ckpt_path, map_location='cpu')
        if '21k' in str(ckpt_path):
                #state = {(k if 'body.' not in k else k[5:]): v for k, v in state['state_dict'].items()}
                state = {(k if 'body.' not in k else k[5:]): v for k, v in state['model'].items()}
                filtered_dict = {k: v for k, v in state.items() if
                                (k in model.state_dict() and 'head.fc' not in k)}
        else:
            state = {(k if 'body.' not in k else k[5:]): v for k, v in state['model'].items()}
            filtered_dict = {k: v for k, v in state.items() if
                            (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
        model = model.to(self.device)
        self.model = model
        self.old_model = None
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # normalize, # no need, toTensor does normalization
        ])
        self.image_id_set = set()
        self.old_dataset = []
        self.kd_loss = self.args.kd_loss
        if self.kd_loss == 'pod_spatial':
            self.lambda_c = self.args.lambda_c
            self.lambda_f = self.args.lambda_f
            self.lambda_f_TDL = self.args.lambda_f_TDL if 'lambda_f_TDL' in self.args else 0
        
    
    def parameters(self):
        return self.model.parameters()
    
    def to(self, device):
        self.model = self.model.to(device)
    
    def add_classes(self, increment_classes):
        """
        Expanding the Classifier
        """
        in_dimension = self.model.head.fc.in_features
        old_classes = self.model.head.fc.out_features

        # Expand the full-connected layer for learned classes
        new_fc = nn.Linear(in_dimension, old_classes + increment_classes)
        new_fc.weight.data[:old_classes] = self.model.head.fc.weight.data
        new_fc.bias.data[:old_classes] = self.model.head.fc.bias.data
        new_fc.to(self.device)

        self.model.head.fc = new_fc

    def add_weight_decay(self, weight_decay=1e-4, skip_list=()):
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
    
    def get_train_vars(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        parameters = self.add_weight_decay(self.args.weight_decay)
        self.cls_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(train_loader),
                                                 epochs=self.args.epochs_per_task,
                                                 pct_start=0.2)
        return self.optimizer, self.scheduler, self.cls_criterion
    
    def compute_loss(self, output, target, old_output=None):
        logits = output['logits'].float()
        cls_loss = self.cls_criterion(logits, target)
        kd_loss = 0.
        pod_spatial = torch.zeros(1)
        pod_flat = torch.zeros(1)

        if self.kd_loss == 'pod_spatial' and self.old_model:
            lambda_c = self.lambda_c * math.sqrt(self.num_classes / self.task_size)
            lambda_f = self.lambda_f * math.sqrt(self.num_classes / self.task_size)

            # Only use the last layer output for distillation loss
            old_features = old_output['attentions']
            new_features = output['attentions']
            pod_spatial = pod(old_features, new_features, 'spatial')

            # Only use flat loss
            pod_flat = embeddings_similarity(old_output['embeddings'], output['embeddings'])

            pod_flat = lambda_f * pod_flat
            pod_spatial = lambda_c * pod_spatial
            kd_loss = pod_flat + pod_spatial
        
        loss = cls_loss + kd_loss
        return loss, cls_loss, pod_spatial, pod_flat
    
    def before_task(self, lifelong_datasets, task_id, low_range, high_range):
        for lifelong_dataset in self._lifelong_datasets.values():
                lifelong_dataset.choose_task(task_id)
        train_dataset_without_old = self._lifelong_datasets['train']
        
        if self.args.replay and self.old_dataset:
            train_dataset_with_old = [train_dataset_without_old]
            train_dataset_with_old.extend(self.old_dataset)
            train_dataset_with_old = torch.utils.data.ConcatDataset(train_dataset_with_old)
            lifelong_datasets['train'] = train_dataset_with_old

        self.num_classes = high_range
        self.task_size = high_range - low_range
        if low_range != 0:
            self.add_classes(self.task_size)

        if self.old_dataset:
            for subset in self.old_dataset:
                dataset = subset.dataset
                old_low_range = dataset.included_cats[0]
                new_retrieve_classes = range(old_low_range, self.num_classes)
                dataset.included_cats = new_retrieve_classes
        
    
    def after_task(self, low_range, high_range, train_loader=None):
        train_dataset = train_loader.dataset
        if (isinstance(train_dataset, torch.utils.data.ConcatDataset)):
            train_dataset = train_dataset.datasets[0]
        train_transforms = train_dataset.transform

        if self.kd_loss:
            model_params = {'args': self.args, 'num_classes': high_range}
            self.old_model = TResnetM(model_params)
            self.old_model = self.old_model.to(self.device)
            self.old_model.load_state_dict(self.model.state_dict())
            self.old_model.eval()

        if self.args.replay:
            status = self.model.training
            self.model.eval()
            train_dataset.transform = self.val_transforms  # sample protos without data augmentation

            # Random or Herding
            if self.args.sample_method == 'random':

                # Random sample protos from train dataset
                sample_ds = random_sample_protos(train_dataset, low_range, high_range, self.args.num_protos)
            elif self.args.sample_method == 'herding':

                # Sample protos from train dataset by Herding
                loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, collate_fn=train_loader.collate_fn)
                sample_ds = icarl_sample_protos(self.model, low_range, high_range,
                                                train_dataset, loader, self.args.num_protos, self.device, self.image_id_set)
            else:
                raise ValueError(f"sample_method {self.sample_method} not supported !!!")
            sample_ds.dataset.included_cats = range(low_range, high_range)
            # Concatenate protos to old dataset
            self.old_dataset.append(sample_ds)

            num_all_protos = 0
            for dataset in self.old_dataset:
                num_all_protos += len(dataset)

            train_dataset.transform = train_transforms  # resume the train transform for training
            self.model.train(status)
    
    def get_old_output(self, x):
        old_output = None
        if self.kd_loss and self.old_model is not None:
            with torch.no_grad():
                old_output = self.old_model(x)
        return old_output
        
    
    def __call__(self, x):
        return self.model(x)

    