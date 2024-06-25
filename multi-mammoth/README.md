# An Attention-based Representation Distillation Baseline for Multi-Label Continual Learning - WebVision

The code in this repository is based on the mammoth repository available [here](https://github.com/aimagelab/mammoth).

If you are using Anaconda to manage the python environments, to run the experiments in this repository you can create a new environment with the provided multi_mammoth.yml file:
```
conda env create -f multi_mammoth.yml
```

Firstly, you need the WebVision dataset.
* Create the folder ./data/WebVision
* Download the images [here](https://data.vision.ee.ethz.ch/cvl/webvision/flickr_resized_256.tar).
* Create the folder ./data/WebVision/images and copy all the WebVision images there.
* Extract the zip archive 'webvision_files' in ./data/WebVision

The main script to run the experiments is [utils/main.py](utils/main.py).

As an example, to run our method (SCAD) you can execute:
```bash
python3 utils/main.py --model=scad --dataset=seq-webvision --distillation_layers=block_outputs --batch_size=16 --ignore_other_metrics=1 --optimizer=sgd --lr=0.03 --clip_grad=1.0 --n_epochs=1 --buffer_size=2000 --der_alpha=0.3 --der_beta=0.9 --lambda_fp=1.0 --adapter_type=attention_probe_cls_norm --adapter_optimizer=sgd --adapter_lr=0.03 --adapter_layers=1,4,7,10 --use_conditioning=0 --lambda_fp_replay=0.1 --lambda_diverse_loss=0.1 --nowand=1
```
To get a list of all the arguments, run:
```bash
python3 utils/main.py --help
```

The code of the methods is located inside the 'models' folder.
