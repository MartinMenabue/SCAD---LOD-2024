# An Attention-based Representation Distillation Baseline for Multi-Label Continual Learning - IIRC CIFAR-100

The code in this repository is based on the IIRC repository available [here](https://github.com/chandar-lab/IIRC).

If you are using Anaconda to manage the python environments, to run the experiments in this repository you can create a new environment with the provided iirc.yml file:
```
conda env create -f iirc.yml
```

The main script to run the experiments is [experiments/main.py](experiments/main.py).

Firstly, you need to download the CIFAR100 dataset:
```
python3
>>> from torchvision.datasets import CIFAR100
>>> CIFAR100('./data', download=True)
```

now you can run the experiments. As an example, to run our method (SCAD) you can execute:
```bash
python3 experiments/main.py --method=scad --dataset=iirc_cifar100 --dataset_path=./data/ --batch_size=16 --logging_path_root=./results --weight_decay=0 --wandb_log=0 --distillation_layers=block_outputs --optimizer=sgd --lr=0.03 --clip_grad=1.0  --network=vit_base_patch_16_224 --adapter_type=attention_probe_cls_norm --adapter_optimizer=sgd --adapter_lr=0.03 --adapter_layers=1,4,7,10 --der_alpha=0.3 --der_beta=0.8 --lambda_fp=1.0 --lambda_fp_replay=0.1 --use_conditioning=0 --epochs_per_task=5 --lambda_diverse_loss=0.1 --buffer_size=500
```
To get a list of all the arguments, run:
```bash
python3 experiments/main.py --help
```
Please note that certain arguments are exclusive to specific methods.

The code of the methods is located inside the lifelong_methods/methods folder.
