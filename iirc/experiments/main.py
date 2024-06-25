import argparse
import importlib
import logging
import os
import sys
import random
from math import ceil
import numpy as np
import torch
import torchvision
#from ml_logger import logbook as ml_logbook
import time
import torch.multiprocessing as mp
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from iirc.datasets_loader import get_lifelong_datasets
from iirc.utils.utils import print_msg
from iirc.definitions import CIL_SETUP, IIRC_SETUP
import lifelong_methods.utils
import lifelong_methods
import utils
from prepare_config import prepare_config
from train import task_train, tasks_eval
from my_utils.status import ProgressBar
import wandb
from my_utils import none_or_float
import uuid
import socket
import datetime
from pathlib import Path
from wandb_offline_sync import agent


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


def print_task_info(lifelong_dataset):
    class_names_samples = {class_: 0 for class_ in lifelong_dataset.cur_task}
    for idx in range(len(lifelong_dataset)):
        labels = lifelong_dataset.get_labels(idx)
        for label in labels:
            if label in class_names_samples.keys():
                class_names_samples[label] += 1
    print_msg(f"Task {lifelong_dataset.cur_task_id} number of samples: {len(lifelong_dataset)}")
    for class_name, num_samples in class_names_samples.items():
        print_msg(f"{class_name} is present in {num_samples} samples")


def main_worker(gpu, config: dict, dist_args: dict = None):
    distributed = dist_args is not None
    if distributed:
        dist_args["gpu"] = gpu
        device = torch.device(f"cuda:{gpu}")
        dist_args["rank"] = dist_args["node_rank"] * dist_args["ngpus_per_node"] + gpu
        rank = dist_args["rank"]
        print_msg(f"Using GPU {gpu} with rank {rank}")
        dist.init_process_group(backend="nccl", init_method=dist_args["dist_url"],
                                world_size=dist_args["world_size"], rank=dist_args["rank"])
    elif gpu is not None:
        device = torch.device(f"cuda:{gpu}")
        rank = 0
        print_msg(f"Using GPU: {gpu}")
    else:
        device = config["device"]
        rank = 0
        print_msg(f"using {config['device']}\n")

    checkpoint = None
    non_loadable_attributes = ["logging_path", "dataset_path", "batch_size"]
    temp = {key: val for key, val in config.items() if key in non_loadable_attributes}
    checkpoint_path = os.path.join(config['logging_path'], 'latest_model')
    json_logs_file_name = 'jsonlogs.jsonl'
    if os.path.isfile(checkpoint_path):
        logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                            filemode='a+',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
        if distributed:
            print_msg(f"\n\nLoading checkpoint {checkpoint_path} on gpu {dist_args['rank']}")
            checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{dist_args['gpu']}")
        else:
            print_msg(f"\n\nLoading checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
        config = checkpoint['config']
        for key in non_loadable_attributes:
            config[key] = temp[key]
        if distributed:
            print_msg(f"Loaded the checkpoint successfully on gpu {dist_args['rank']}")
        else:
            print_msg(f"Loaded the checkpoint successfully")

        if rank == 0:
            print_msg(f"Resuming from task {config['cur_task_id']} epoch {config['task_epoch']}")
            # Remove logs related to traing after the checkpoint was saved
            utils.remove_extra_logs(config['cur_task_id'], config['task_epoch'],
                                    os.path.join(config['logging_path'], json_logs_file_name))
            if distributed:
                dist.barrier()
        else:
            dist.barrier()
    else:
        if rank == 0:
            os.makedirs(config['logging_path'], exist_ok=True)
            if os.path.isfile(os.path.join(config['logging_path'], json_logs_file_name)):
                os.remove(os.path.join(config['logging_path'], json_logs_file_name))
            logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                                filemode='w',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)
            if distributed:
                dist.barrier()
        else:
            dist.barrier()
            logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)

    # torch.random.manual_seed(config['seed'])
    # np.random.seed(config['seed'])
    # random.seed(config["seed"])

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if config["wandb_log"]:
        wandb_config = dict(project=config["wandb_project"], config=config, allow_val_change=True,
                            name=config["id"], id=config["id"])
        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)
        print(f'Wandb run id: {wandb.run.id}', file=sys.stderr)
        print(f'Wandb entity: {config["wandb_entity"]}', file=sys.stderr)
        print(f'Wandb project: {config["wandb_project"]}', file=sys.stderr)
        if 'WANDB_MODE' in os.environ and os.environ['WANDB_MODE'] == 'offline':
            agent.init(frequency=300)

    # logbook_config = ml_logbook.make_config(
    #     logger_dir=config['logging_path'],
    #     filename=json_logs_file_name,
    #     create_multiple_log_files=False,
    #     #wandb_config=wandb_config,
    #     wandb_config=None,
    # )
    # logbook = ml_logbook.LogBook(config=logbook_config)
    logbook = None
    if checkpoint is None and (not distributed or dist_args['rank'] == 0):
        config_to_write = {str(key): str(value) for key, value in config.items()}
        # logbook.write_config(config_to_write)

    essential_transforms_fn, augmentation_transforms_fn = get_transforms(config['dataset'])
    lifelong_datasets, tasks, class_names_to_idx = \
        get_lifelong_datasets(config['dataset'], dataset_root=config['dataset_path'],
                              tasks_configuration_id=config["tasks_configuration_id"],
                              essential_transforms_fn=essential_transforms_fn,
                              augmentation_transforms_fn=augmentation_transforms_fn, cache_images=False,
                              joint=config["joint"])

    if config["complete_info"]:
        for lifelong_dataset in lifelong_datasets.values():
            lifelong_dataset.enable_complete_information_mode()

    if checkpoint is None:
        n_cla_per_tsk = [len(task) for task in tasks]
        metadata = {}
        metadata['n_tasks'] = len(tasks)
        metadata["total_num_classes"] = len(class_names_to_idx)
        metadata["tasks"] = tasks
        metadata["class_names_to_idx"] = class_names_to_idx
        metadata["n_cla_per_tsk"] = n_cla_per_tsk
        if rank == 0:
            metadata_to_write = {str(key): str(value) for key, value in metadata.items()}
            # logbook.write_metadata(metadata_to_write)
    else:
        metadata = checkpoint['metadata']

    # Assert that methods files lie in the folder "methods"
    method = importlib.import_module('lifelong_methods.methods.' + config["method"])
    model = method.Model(metadata["n_cla_per_tsk"], metadata["class_names_to_idx"], config)

    buffer_dir = None
    map_size = None
    if "imagenet" in config["dataset"]:
        if config['n_memories_per_class'] > 0:
            n_classes = sum(metadata["n_cla_per_tsk"])
            buffer_dir = config['logging_path']
            map_size = int(config['n_memories_per_class'] * n_classes * 1.4e6)
        elif config['total_n_memories'] > 0:
            buffer_dir = config['logging_path']
            map_size = int(config['total_n_memories'] * 1.4e6)
    buffer = method.Buffer(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model.to(device)
        model.net = torch.nn.parallel.DistributedDataParallel(model.net, device_ids=[gpu])
    else:
        model.to(config["device"])

    # If loading a checkpoint, load the corresponding state_dicts
    if checkpoint is not None:
        lifelong_methods.utils.load_model(checkpoint, model, buffer, lifelong_datasets)
        print_msg(f"Loaded the state dicts successfully")
        starting_task = config["cur_task_id"]
    else:
        starting_task = 0
    
    progress_bar = ProgressBar(verbose=True)
    multi_metrics = ['jaccard_sim', 'modified_jaccard', 'strict_acc', 'recall']
    all_metrics = {
        **{k: [] for k in multi_metrics},
        **{f'average_{k}': [] for k in multi_metrics},
    }
    n_tasks = config['stop_after'] or len(tasks)
    for cur_task_id in range(starting_task, n_tasks):
        if config['method'] == 'joint':
            cur_task_id = len(tasks) - 1
            for lifelong_dataset in lifelong_datasets.values():
                lifelong_dataset.choose_task(cur_task_id)
            model.net.train()
            model.train_joint()
        else:
            if checkpoint is not None and cur_task_id == starting_task and config["task_epoch"] > 0:
                new_task_starting = False
            else:
                new_task_starting = True

            if config["incremental_joint"]:
                for lifelong_dataset in lifelong_datasets.values():
                    lifelong_dataset.load_tasks_up_to(cur_task_id)
            else:
                for lifelong_dataset in lifelong_datasets.values():
                    lifelong_dataset.choose_task(cur_task_id)

            if rank == 0:
                print_task_info(lifelong_datasets["train"])

            #model.net.eval()
            model.net.train()
            if new_task_starting:
                model.prepare_model_for_new_task(lifelong_datasets=lifelong_datasets, task_data=lifelong_datasets["train"], dist_args=dist_args, buffer=buffer,
                                                num_workers=config["num_workers"])

            start_time = time.time()
            task_train(model, buffer, lifelong_datasets, config, metadata, logbook=logbook, dist_args=dist_args, progress_bar=progress_bar)
            end_time = time.time()
            print_msg(f"Time taken on device {rank} for training on task {cur_task_id}: "
                    f"{round((end_time - start_time) / 60, 2)} mins")

            model.consolidate_task_knowledge(
                buffer=buffer, device=device, batch_size=config["batch_size"], task_data=lifelong_datasets["train"],
            )

        metrics_dict = tasks_eval(
            model, lifelong_datasets["test"], cur_task_id, config, metadata, logbook=logbook, dataset_type="test",
            dist_args=dist_args
        )
        metrics_gathered = {k: [] for k in multi_metrics}
        for i in range(cur_task_id + 1):
            for m in multi_metrics:
                metrics_gathered[m].append(metrics_dict[f'task_{i}_test_{m}'])
        
        for k, v in metrics_gathered.items():
            all_metrics[k].append(v)

        for k, v in metrics_dict.items():
            if 'average' in k:
                all_metrics[k.replace('_test', '')].append(v)
        
        metrics_dict['current_task'] = cur_task_id
        if config['wandb_log']:
            wandb.log(metrics_dict)

        config["cur_task_id"] = cur_task_id + 1
        config["task_epoch"] = 0

        if config['method'] == 'joint':
            break
    
    return all_metrics


if __name__ == '__main__':
    method_names = [x.stem for x in Path("lifelong_methods/methods").glob("*.py")]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="iirc_cifar100",
                        choices=["incremental_cifar100", "iirc_cifar100", "incremental_imagenet_full",
                                 "incremental_imagenet_lite", "iirc_imagenet_full", "iirc_imagenet_lite"])
    parser.add_argument('--epochs_per_task', type=int, default=70,
                        help="The number of epochs per task. This number is multiplied by 2 for the first task.")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--dataset_path', type=str, default="./data")
    parser.add_argument('--logging_path_root', type=str, default="results",
                        help="The directory where the logs and results will be saved")
    parser.add_argument('--ckpt_path', type=str, default="checkpoints")
    parser.add_argument('--wandb_log', type=int, choices=[0, 1], default=0)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--n_layers', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=6,
                        help="Number of workers used to fetch the data for the dataloader")
    parser.add_argument('--group', type=str, default="main",
                        help="The parent folder of the experiment results, so as to group related experiments easily")
    # Parameters for creating the tasks
    parser.add_argument('--tasks_configuration_id', type=int, default=0, choices=range(0, 10),
                        help="The task configuration id. Ignore for joint training")
    # The training method
    parser.add_argument('--method', type=str, default="finetune",
                        choices=method_names)  # , "icarl", "siamese"])
    parser.add_argument('--complete_info', action='store_true',
                        help='use the complete information during training (a multi-label setting)')
    parser.add_argument('--incremental_joint', action='store_true',
                        help="keep all data from previous tasks, while updating their labels as per the observed "
                             "classes (use only with complete_info and without buffer)")
    parser.add_argument('--joint', action='store_true',
                        help="load all classes during the first task. This option ignores the tasks_configuration_id "
                             "(use only with complete_info and without buffer)")
    # The optimizer parameters
    parser.add_argument('--optimizer', type=str, default="momentum", choices=["adam", "momentum", "sgd"])
    parser.add_argument('--lr', type=float, default=0.1, help="The initial learning rate for each task")
    parser.add_argument('--lr_gamma', type=float, default=.2,
                        help="The multiplicative factor for learning rate decay at the epochs specified")
    parser.add_argument('--lr_schedule', nargs='+', type=int, default=None,
                        help="the epochs per task at which to multiply the current learning rate by lr_gamma "
                             "(resets after each task). This setting is ignored if reduce_lr_on_plateau is specified")
    parser.add_argument('--reduce_lr_on_plateau', action='store_true',
                        help='reduce the lr on plateau based on the validation performance metric')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # Model selection and validation set
    parser.add_argument('--checkpoint_interval', type=int, default=3,
                        help="The number of epochs within each task after which the checkpoint is updated. When a task "
                             "is finished, the checkpoint is updated anyways, so set to 0 for checkpointing only after "
                             "each task")
    parser.add_argument('--use_best_model', action='store_true',
                        help='use the best model after training each task based on the best task validation accuracy')
    parser.add_argument('--save_each_task_model', action='store_true',
                        help='save the model after each task')
    # The buffer parameters
    parser.add_argument('--total_n_memories', type=int, default=0,
                        help="The total replay buffer size, which is divided by the observed number of classes to get "
                             "the number of memories kept per task, note that the number of memories per task here is "
                             "not fixed but rather decreases as the number of tasks observed increases (with a minimum "
                             "of 1). If n_memories_per_class is set to a value greater than -1, the "
                             "n_memories_per_class is used instead.")
    parser.add_argument('--n_memories_per_class', type=int, default=-1,
                        help="The number of samples to keep from each class, if set to -1, the total_n_memories "
                             "argument is used instead")
    parser.add_argument('--buffer_sampling_multiplier', type=float, default=1.0,
                        help="A multiplier for sampling from the buffer more/less times than the size of the buffer "
                             "(for example a multiplier of 2 samples from the buffer (with replacement) twice its size "
                             "per epoch)")
    parser.add_argument('--memory_strength', type=float, default=1.0,
                        help="a weight to be multiplied by the loss from the buffer")
    parser.add_argument('--max_mems_pool_per_class', type=int, default=1e5,
                        help="Maximum size of the samples pool per class from which the buffer chooses the exemplars, "
                             "use -1 for choosing from the whole class samples.")
    parser.add_argument('--buffer_size', type=int, default=0)

    # LUCIR Hyperparameters
    parser.add_argument('--lucir_lambda', type=float, default=5.0,
                        help="a weight to be multiplied by the distillation loss (only for the LUCIR method)")
    parser.add_argument('--lucir_margin_1', type=float, default=0.5,
                        help="The 1st margin used with the margin ranking loss for in the LUCIR method")
    parser.add_argument('--lucir_margin_2', type=float, default=0.5,
                        help="The 2nd margin used with the margin ranking loss for in the LUCIR method")

    # Distributed arguments
    parser.add_argument('--num_nodes', type=int, default=1,
                        help="num of nodes to use")
    parser.add_argument('--node_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url', default="env://", type=str,
                        help='node rank for distributed training')
    parser.add_argument('--debug_mode', type=int, choices=[0, 1], default=0)
    parser.add_argument('--clip_grad', type=none_or_float, default=None)
    parser.add_argument('--pool_size_coda', type=int, default=None)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--der_alpha', type=float, default=None)
    parser.add_argument('--der_beta', type=float, default=None)
    parser.add_argument('--pool_size_l2p', type=int, default=None)
    parser.add_argument('--l2p_pull_constraint_coeff', type=int, default=None)
    parser.add_argument('--save_checkpoints', type=int, choices=[0, 1], default=0)
    parser.add_argument('--distillation_layers', type=str, choices=['MHSA_outputs', 'block_outputs', 'attention_masks'], default=None)
    parser.add_argument('--adapter_layers', type=str, default=None, help='Indices of layers to add adapters. Example: 0,4,8,12')
    parser.add_argument('--adapter_type', default=None, choices=['mixer', 'channel', 'double_mixer', 'double_convit', 'mock', 'mimicking', 'twf_original', 'transformer',
                                                                 'clip_cross_attention', 'clip_cross_attention_v2', 'clip_cross_attention_v3',
                                                                 'transformer_pretrained', 'clip_cross_attention_v4', 'clip_cross_attention_v5', 'clip_cross_attention_v6',
                                                                 'clip_cross_attention_v7', 'transformer_pretrained_clip', 'transformer_pretrained_proj',
                                                                 'tat', 'tat_v2', 'tat_norm', 'transformer_pretrained_layer_norm', 'attention_probe_cls',
                                                                 'attention_probe_cls_norm', 'attention_probe_cls_no_gumbel', 'attention_probe_cls_norm_no_gumbel',
                                                                 'attention_probe_new', 'attention_probe_filter', 'attention_probe_softmax',
                                                                 'attention_probe_v2', 'attention_probe_v2_softmax'], type=str, help='Type of adapter')
    parser.add_argument('--lambda_fp', type=float, default=None,
                        help='weight of feature propagation loss replay') 
    parser.add_argument('--lambda_diverse_loss', type=float, default=None,
                        help='Diverse loss hyperparameter.')
    parser.add_argument('--lambda_fp_replay', type=float, default=None,
                        help='weight of feature propagation loss replay')
    parser.add_argument('--lambda_ignition_loss', type=float, default=None, help='Ignition loss hyperparameter.')
    parser.add_argument('--ignition_loss_temp', type=float, default=None, help='Temperature for ignition loss.')
    parser.add_argument('--use_conditioning', type=int, choices=[0, 1], default=None)
    parser.add_argument('--use_prompt', type=int, choices=[0, 1], default=None)
    parser.add_argument('--exclude_class_token', type=int, choices=[0, 1], default=None, help='Exclude class token from distillation')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--adapter_optimizer', type=str, default=None, choices=['sgd', 'adam'], help='Optimizer for adapters')
    parser.add_argument('--adapter_lr', type=float, default=None, help='Learning rate of adapters')
    parser.add_argument('--l2p_head_type', default=None, choices=['token', 'gap', 'prompt', 'token+prompt'], type=str, help='input type of classification head')
    parser.add_argument('--xder_gamma', type=float, default=None, help='gamma for xder')

    parser.add_argument('--rescale_logits_update', type=int, default=None, choices=[0,1], help='Rescale logits when updating?')
    parser.add_argument('--use_gs_scheduler', type=int, default=None, choices=[0,1], help='Use guumbel softmax tau cosine annealing?')
    parser.add_argument('--diverse_loss_mode', default=None, choices=['original', 'fix_normalized', 'fix_diagonal', 'fix_normalized_diagonal'], type=str, help='Type of diverse loss')

    parser.add_argument('--stop_after', type=int, default=None, help='Task limit')
    parser.add_argument('--lambda_dist', type=float, default=1.0, help='Distillation loss coefficient')

    args = parser.parse_args()

    # job number 
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    config = prepare_config(args)
    print(config, file=sys.stderr)
    if "iirc" in config["dataset"]:
        config["setup"] = IIRC_SETUP
    else:
        config["setup"] = CIL_SETUP

    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config["ngpus_per_node"] = torch.cuda.device_count()
    print_msg(f"number of gpus per node: {config['ngpus_per_node']}")
    all_metrics = main_worker(None, config, None)
    
    if config['wandb_log']:
        for k, v in all_metrics.items():
            wandb.run.summary[k] = v
        wandb.finish()
