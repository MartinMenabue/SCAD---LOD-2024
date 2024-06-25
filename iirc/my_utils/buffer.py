

import torch
import numpy as np
from typing import Tuple
from torch.functional import Tensor
from torchvision import transforms
from torch import nn

class bn_track_stats:
    def __init__(self, module: nn.Module, condition=True):
        self.module = module
        self.enable = condition

    def __enter__(self):
        if self.module is not None and not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = False

    def __exit__(self ,type, value, traceback):
        if self.module is not None and not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = True


def reservoir(num_seen_examples: int, buffer_size: int, **kwargs) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :param labels: the set of buffer labels
    :param proposed_class: the class of the current example
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

def balancoir(num_seen_examples: int, buffer_size: int, labels: np.array, proposed_class: int, unique_map: np.array) -> int:
    """
    balancoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :param labels: the set of buffer labels
    :param proposed_class: the class of the current example
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples
    
    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size or len(unique_map) <= proposed_class or unique_map[proposed_class] < np.median(unique_map[unique_map > 0]):
        target_class = np.argmax(unique_map)
        e = rand % unique_map.max()
        idx = np.arange(buffer_size)[labels.cpu() == target_class][rand % unique_map.max()]
        return idx
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class MyBuffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir', 'balancoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.mode = mode
        self.sampling_policy = {'reservoir': reservoir, 'ring': reservoir, 'balancoir': balancoir}[mode]
        # self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'logits_mask', 'attn_probes']
        self.attention_maps = [None] * buffer_size

        self.balanced_class_perm  = None
        self.unique_map = np.empty((0,), dtype=np.int32)

    def update_unique_map(self, label_in, label_out=None):
        while len(self.unique_map) <= label_in:
            self.unique_map = np.concatenate((self.unique_map, np.zeros((len(self.unique_map) * 2 + 1), dtype=np.int32)), axis=0)
        self.unique_map[label_in] += 1
        if label_out is not None:
            self.unique_map[label_out] -= 1

    def class_stratified_add_data(self, dataset, cpt, model=None, desired_attrs=None):
        if not hasattr(self, 'task'):
            self.task = 0
        # Reduce Memory Buffer
        if self.task:
            examples_per_class = self.buffer_size // ((self.task + 1) * cpt)
            assert set(desired_attrs) == {x for x in self.attributes if hasattr(self, x)}
            ret_tuples = self.get_all_data()
            self.empty()
            for tl in ret_tuples[1].unique():
                idx = tl == ret_tuples[1]
                ret_tuple = [a[idx] for a in ret_tuples]
                first = min(ret_tuple[0].shape[0], examples_per_class)
                self.add_data(**{a: ret_tuple[i][:first] for i, a in enumerate(
                    [x for x in self.attributes if x in desired_attrs])})
        
        # Add new task data
        examples_last_task = self.buffer_size - self.num_seen_examples
        examples_per_class = examples_last_task // cpt
        ce = torch.tensor([examples_per_class] * cpt).int()
        ce[torch.randperm(cpt)[:examples_last_task - (examples_per_class * cpt)]] += 1

        with torch.no_grad():
            with bn_track_stats(model, False):
                for data in dataset.train_loader:
                    inputs, labels, not_aug_inputs = data
                    inputs = inputs.to(self.device)
                    not_aug_inputs = not_aug_inputs.to(self.device)
                    if all(ce == 0):
                        break

                    flags = torch.zeros(len(inputs)).bool()
                    for j in range(len(flags)):
                        if ce[labels[j] % cpt] > 0:
                            flags[j] = True
                            ce[labels[j] % cpt] -= 1

                    add_dict = {
                        'examples': not_aug_inputs[flags]
                    }
                    if hasattr(self, 'labels') or desired_attrs is not None and 'labels' in desired_attrs:
                        add_dict['labels'] = labels[flags]
                    if hasattr(self, 'logits') or desired_attrs is not None and 'logits' in desired_attrs:
                        outputs = model(inputs)
                        add_dict['logits'] = outputs.data[flags]
                    if hasattr(self, 'task_labels') or desired_attrs is not None and 'task_labels' in desired_attrs:
                        add_dict['task_labels'] = (torch.ones(len(not_aug_inputs)) *
                                                    (self.task))[flags]
                    self.add_data(**add_dict)
        self.task += 1

    def generate_class_perm(self):
        self.balanced_class_perm = (self.labels.unique()[torch.randperm(len(self.labels.unique()))]).cpu()
        self.balanced_class_index = 0

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))

        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


    def init_tensors(self, **kwargs) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            if attr_str in kwargs:     
                attr = kwargs[attr_str]
                if attr is not None and not hasattr(self, attr_str):
                    typ = torch.int64 if attr_str.endswith('els') else (torch.bool if attr_str.endswith('mask') else torch.float32)
                    setattr(self, attr_str, torch.zeros((self.buffer_size,
                            *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, compress=True, **kwargs) -> None:
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        assert "examples" in kwargs, "examples not provided!"
        examples = kwargs["examples"]
        if self.mode == 'balancoir':
            assert "labels" in kwargs, "labels required in balancoir!"
            labels = kwargs["labels"]

        if not hasattr(self, 'examples'):
            self.init_tensors(**kwargs)

        rix = []
        for i in range(examples.shape[0]):
            index = self.sampling_policy(self.num_seen_examples, self.buffer_size, unique_map=self.unique_map,
                        labels=self.labels if hasattr(self, 'labels') else None, proposed_class=labels[i] if self.mode == 'balancoir' else None)
            
            if index >= 0:
                if self.examples.device != self.device:
                    self.examples.to(self.device)
                if self.mode == 'balancoir':
                    self.update_unique_map(labels[i], self.labels[index] if index < self.num_seen_examples else None)
                self.examples[index] = examples[i].to(self.device)
                for arg in kwargs:
                    if arg == "attention_maps":
                        if compress:
                            self.attention_maps[index] = [at[i].byte() for at in kwargs[arg]]
                        else:
                            self.attention_maps[index] = [at[i] for at in kwargs[arg]]

                    elif arg in self.attributes:
                        getattr(self, arg)[index] = kwargs[arg][i].to(self.device)
                    else:
                        raise ValueError(f'Attribute {arg} not in {self.attributes}')
            self.num_seen_examples += 1

            rix.append(index)
        return torch.tensor(rix).to(self.device)

    def get_data(self, size: int, transform: transforms=None, return_index=False, to_device=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        L = min(self.num_seen_examples, self.examples.shape[0])

        if size > L:
            size = L

        target_device = self.device if to_device is None else to_device

        choice = torch.from_numpy(np.random.choice(L,
                                    size=size, replace=False)).long()
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee)
                            for ee in self.examples[choice].cpu()]).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice].to(target_device),)

        if not return_index:
          return ret_tuple
        else:
          return (choice.to(target_device), ) + ret_tuple
    
    def get_balanced_data(self, size: int, transform: transforms=None, n_classes=-1, to_device=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        tot_classes, class_counts = torch.unique(self.labels[:self.num_seen_examples], return_counts=True)
        if n_classes == -1:
            n_classes = len(tot_classes)

        
        finished = False
        selected = tot_classes
        while not finished:
            n_classes = min(n_classes, len(selected))
            size_per_class = torch.full([n_classes], size // n_classes)
            size_per_class[:size % n_classes] += 1
            selected = tot_classes[class_counts >= size_per_class[0]]
            if n_classes <= len(selected):
                finished = True
            if len(selected) == 0:
                print('WARNING: no class has enough examples')
                return self.get_data(0, transform)

        selected = selected[torch.randperm(len(selected))[:n_classes]]

        choice = []
        for i, id_class in enumerate(selected):
            choice += np.random.choice(torch.where(self.labels[:self.num_seen_examples] == id_class)[0].cpu(),
                                       size=size_per_class[i].item(),
                                       replace=False).tolist()
        choice = torch.tensor(choice)
        return self.get_data_by_index(choice, transform, to_device)
        
        

    def get_data_by_index(self, indexes: Tensor, transform: transforms=None, to_device=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        target_device = self.device if to_device is None else to_device
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[indexes]]).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(target_device)
                ret_tuple += (attr[indexes],)
        return ret_tuple


    def get_data_balanced(self, n_classes: int, n_instances: int, transform: transforms=None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param n_classes: the number of classes to sample
        :param n_instances: the number of instances to be sampled per class
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        classes_to_sample = torch.tensor([])
        choice = torch.tensor([]).long()

        while len(classes_to_sample) < n_classes:
            if self.balanced_class_perm is None or \
               self.balanced_class_index >= len(self.balanced_class_perm) or \
               len(self.balanced_class_perm.unique()) != len(self.labels.unique()):
                self.generate_class_perm()
            
            classes_to_sample = torch.cat([
                classes_to_sample,
                self.balanced_class_perm[self.balanced_class_index:self.balanced_class_index+n_classes]
                ])
            self.balanced_class_index += n_classes

        for a_class in classes_to_sample:
            candidates = np.arange(len(self.labels))[self.labels.cpu() == a_class]
            candidates = candidates[candidates < self.num_seen_examples]
            choice = torch.cat([
                choice, 
                torch.tensor(
                    np.random.choice(candidates,
                    size=n_instances,
                    replace=len(candidates) < n_instances
                    )
                )
            ])
        
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
          return ret_tuple
        else:
          return (choice.to(self.device), ) + ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self
