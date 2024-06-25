import os
import sys
sys.path.append(os.getcwd())
import torch
import timm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import pickle
import types
from torchvision.utils import make_grid
import re
from torchvision.datasets import CIFAR100
from argparse import Namespace
from torch_cka import CKA
import shlex
from my_utils.hooks_handlers import HooksHandlerViT
from lifelong_methods.methods.scad_utils.vit_afd import TokenizedDistillation
from iirc.datasets_loader import get_lifelong_datasets
from lifelong_methods.methods.scad_utils.vision_transformer import vit_base_patch16_224_twf
from lifelong_methods.methods.scad_utils.clip_embeddings import ClipEmbeddings

def get_transforms(dataset_name):
    essential_transforms_fn = None
    augmentation_transforms_fn = None
    if "cifar100" in dataset_name:
        # essential_transforms_fn = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        # ])
        # augmentation_transforms_fn = torchvision.transforms.Compose([
        #     torchvision.transforms.RandomCrop(32, padding=4),
        #     torchvision.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        # ])
        essential_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomCrop(224, padding=28),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    elif "imagenet" in dataset_name:
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        essential_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    return essential_transforms_fn, augmentation_transforms_fn

def load_checkpoint_args(checkpoint_args_path, job_id, return_args_file=False):
    #args_file = os.popen(f'find {shlex.quote(checkpoint_args_path)} -name "*args*{shlex.quote(job_id)}*"').read()
    #args_file = args_file.strip()
    args_file = [x for x in os.listdir(checkpoint_args_path) if job_id in x and 'args' in x and x.endswith('pkl')][0]
    args_file = os.path.join(checkpoint_args_path, args_file)


    with open(args_file, 'rb') as f:
        args = pickle.load(f)
    
    if return_args_file:
        return args, args_file
    else:
        return args


def calc_cka(m1,m2, dataset, layers1, layers2, device, token_index='all'):
    cka=CKA(m1, m2, model1_name="M1", model2_name="M2",model1_layers=layers1,model2_layers=layers2, device=device)
    cka.compare(dataset) 
    return cka.export()['CKA']

cka_dataset = None

@torch.no_grad()
def compute_cka_vits(config, device, model1, model2, num_samples=1000, flush_dataset=False, token_index='all'):
    global cka_dataset

    normalize_t = seq_dataset.get_normalization_transform()
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_t
    ])

    if flush_dataset or cka_dataset is None:
        if 'cifar100' in config['dataset']:
            dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=t)
        else:
            print(f'Error! dataset not supported')
            return
        cka_dataset = dataset
    dataset = cka_dataset
    
    dataset = torch.utils.data.Subset(dataset, np.random.choice(np.arange(len(dataset)), num_samples, replace=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    layers1 = [f'blocks.{i}' for i in range(12)] if token_index != 'cls' else [f'blocks.{i}' for i in range(1, 12)]
    layers2 = [f'blocks.{i}' for i in range(12)] if token_index != 'cls' else [f'blocks.{i}' for i in range(1, 12)]
    cka = calc_cka(model1, model2, dataloader, layers1, layers2, device, token_index)
    return cka

@torch.no_grad()
def compute_cka_general(config, device, model1, model2, num_samples=1000, flush_dataset=False, dataloader=None):
    if dataloader is None:
        global cka_dataset

        t, _ = get_transforms(config['dataset'])
        
        if flush_dataset or cka_dataset is None:
            if 'cifar100' in config['dataset']:
                dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=t)
            else:
                print(f'Error! dataset not supported')
                return
            cka_dataset = dataset
        dataset = cka_dataset
        
        dataset = torch.utils.data.Subset(dataset, np.random.choice(np.arange(len(dataset)), num_samples, replace=False))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)


    network = config['network']
    #import pdb; pdb.set_trace()

    if network.startswith('resnet'):
        layers1 = ['conv1'] + [f'layer{i}' for i in range(1, 5)]
        layers2 = layers1
    elif network.startswith('vit'):
        layers1 = [f'blocks.{i}' for i in range(12)]
        layers2 = layers1
    elif network.startswith('swin'):
        layers1 = [k for k, v in model1.named_modules() if re.search(r'^layers.\d+.blocks.\d+$', k)]
        layers2 = layers1

    cka = calc_cka(model1, model2, dataloader, layers1, layers2, device)
    return cka

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreterr

@torch.no_grad()
def mini_eval(model, device, dataset):
    tg = model.training
    model.eval()
    correct = 0
    total = 0
    if is_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    for data in tqdm(dataset):
        data, target = data[0], data[1]
        data, target = data.to(device), target.to(device)
        output = model(data)[:, :10]
        correct += (torch.argmax(output, dim=1) == target).sum().item()
        total += len(data)
    model.train(tg)
    return correct / total

def load_exp(job_id, task):
    exp_args = load_checkpoint_args(f'/home/aba/martin_cineca/checkpoints', job_id)
    #exp_args = load_checkpoint_args(f'/home/aba/martin_cineca/checkpoints/{method}_{network}_args_{job_id}.txt')
    args = Namespace(**exp_args)
    continual_dataset = get_dataset(args)
    method = args.model
    network = args.network
    num_classes = continual_dataset.N_CLASSES_PER_TASK * continual_dataset.N_TASKS
    model = timm.create_model(args.network, num_classes=num_classes)
    print(model.load_state_dict(torch.load(f'/home/aba/martin_cineca/checkpoints/{method}_{network}_{task}_{job_id}.pt')))
    model.eval()
    return model, args

clip_embeddings = None

def build_twf_vit(device, config):
    essential_transforms_fn, augmentation_transforms_fn = get_transforms(config['dataset'])
    lifelong_datasets, tasks, class_names_to_idx = \
        get_lifelong_datasets(config['dataset'], dataset_root=config['dataset_path'],
                            tasks_configuration_id=config["tasks_configuration_id"],
                            essential_transforms_fn=essential_transforms_fn,
                            augmentation_transforms_fn=augmentation_transforms_fn, cache_images=False,
                            joint=config["joint"])
    n_cla_per_tsk = [len(task) for task in tasks]
    num_classes = int(sum(n_cla_per_tsk))
    num_tasks = len(n_cla_per_tsk)

    net = timm.create_model('vit_base_patch16_224_twf', pretrained=True, num_classes=num_classes).to(device)
    prenet = deepcopy(net).eval()
    #if args.distillation_layers == 'attention_masks':
    HooksHandlerViT(net)
    HooksHandlerViT(prenet)

    if config['method'] == 'twf_vit_clip':
        global clip_embeddings
        if clip_embeddings is None:
            clip_embeddings = ClipEmbeddings(config, device, class_names_to_idx, net.embed_dim).to(device)

    with torch.no_grad():
        x = torch.randn((1, 3, 224, 224)).to(device)
        res = net(x, returnt='full')
        feats_t = res[config['distillation_layers']]

    #self.dist_indices = list(range(len(feats_t)))[::4]
    dist_indices = [int(x) for x in config['adapter_layers'].split(',')]
    feats_t = [feats_t[i] for i in dist_indices]
    
    for (i, x) in zip(dist_indices, feats_t):
        adapt_shape = x.shape[1:]
        if config['method'] == 'twf_vit_clip':
            adapt_shape = (adapt_shape[0]+1, adapt_shape[1]) # we take into account the clip embedding
    
        setattr(net, f"adapter_{i+1}", TokenizedDistillation(
                    adapt_shape,
                    num_tasks,
                    n_cla_per_tsk,
                    adatype=config['adapter_type'],
                    teacher_forcing_or=False,
                    lambda_forcing_loss=config['lambda_fp_replay'],
                    lambda_diverse_loss=config['lambda_diverse_loss'],
                    use_prompt=config['use_prompt'],
                    use_conditioning=config['use_conditioning'],
                ).to(device))
    return net, prenet

def get_twf_vit_outputs(net, prenet, x, y, config, task_labels):
    attention_maps = []
    logits_maps = []

    with torch.no_grad():
        res_s = net(x, returnt='full')
        feats_s = res_s[config['distillation_layers']]
        res_t = prenet(x, returnt='full')
        feats_t = res_t[config['distillation_layers']]
        

    dist_indices = [int(x) for x in config['adapter_layers'].split(',')]
    if config['adapter_type'] == 'twf_original':
        # we must exclude the class token
        partial_feats_s = [feats_s[i][:, 1:, :] for i in dist_indices]
        partial_feats_t = [feats_t[i][:, 1:, :] for i in dist_indices]
    else:
        partial_feats_s = [feats_s[i] for i in dist_indices]
        partial_feats_t = [feats_t[i] for i in dist_indices]
    
    if config['method'] == 'twf_vit_clip':
        class_idxs = y.nonzero(as_tuple=True)[1]
        with torch.no_grad():
            global clip_embeddings
            embs = clip_embeddings(class_idxs)
        partial_feats_s = [torch.cat([p, embs.unsqueeze(1)], dim=1) for p in partial_feats_s]
        partial_feats_t = [torch.cat([p, embs.unsqueeze(1)], dim=1) for p in partial_feats_t]

    for i, (idx, net_feat, pret_feat) in enumerate(zip(dist_indices, partial_feats_s, partial_feats_t)):
        adapter = getattr(
                net, f"adapter_{idx+1}")

        output_rho, logits = adapter.attn_fn(pret_feat, y, task_labels)
        attention_maps.append(output_rho)
        logits_maps.append(logits)

    return res_s, res_t, attention_maps, logits_maps

def get_attention_map_grid_twf_vit(config, attention_maps, distillation_layers, layer, batch_idx, num_imgs=20):
    if distillation_layers == 'attention_masks':
        # Abbiamo matrici 197x197 e dobbiamo prendere le righe
        return make_grid(attention_maps[layer][batch_idx][:num_imgs, 1:].view(-1, 14, 14).unsqueeze(1), padding=5, pad_value=1.0).cpu().detach().numpy().transpose(1, 2, 0)
    else:
        # Abbiamo matrici 197x768 e dobbiamo prendere le colonne
        if 'attention_probe' in config['adapter_type']:
            return make_grid(attention_maps[layer][batch_idx].squeeze(0)[1:].view(14, 14).unsqueeze(1), padding=5, pad_value=1.0).cpu().detach().numpy().transpose(1, 2, 0)
        if attention_maps[0].shape[1] == 197:
            return make_grid(attention_maps[layer][batch_idx][1:].permute(1, 0).view(-1, 14, 14).unsqueeze(1)[:num_imgs], padding=5, pad_value=1.0).cpu().detach().numpy().transpose(1, 2, 0)
        else:
            return make_grid(attention_maps[layer][batch_idx].permute(1, 0).view(-1, 14, 14).unsqueeze(1)[:num_imgs], padding=5, pad_value=1.0).cpu().detach().numpy().transpose(1, 2, 0)

def get_attention_masks_grid_twf_vit(attention_masks, layer, batch_idx, head, num_imgs=20):
    '''Get a grid of attention masks (obtained with Q @ K) for a given layer, batch index and head'''
    # Q @ K sono matrici 197x197. Bisogna prendere le righe
    x = attention_masks[layer][batch_idx][head][:num_imgs, 1:].view(-1, 14, 14).unsqueeze(1)
    x = (x - x.min()) / (x.max() - x.min())
    img = make_grid(x, padding=5, pad_value=1.0).permute(1,2,0).cpu().detach().numpy()
    #img = img * 255
    return img

def get_outputs_grid_twf_vit(outputs, layer, batch_idx, num_imgs=20):
    '''Get a grid of outputs (obtained from the blocks of ViT) for a given layer and batch index'''
    x = outputs[layer][batch_idx][1:, :num_imgs].view(14, 14, -1).permute(2, 0, 1).unsqueeze(1)
    #x = (x - x.min()) / (x.max() - x.min())
    #x = x * 255
    img =  make_grid(x, padding=5, pad_value=1.0).permute(1,2,0).cpu().detach().numpy()
    #import pdb; pdb.set_trace()
    return img

def get_attention_map_grid_twf_resnet(attention_maps, layer, batch_idx):
    '''Get a grid of attention maps (obtained from the adapters of twf) for a given layer and batch index'''
    return make_grid(attention_maps[layer][batch_idx].cpu().detach().unsqueeze(1)).permute(1,2,0).numpy()