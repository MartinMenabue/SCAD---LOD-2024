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
import torchvision
from my_utils.buffer import MyBuffer
from my_utils.hooks_handlers import HooksHandlerViT
from lifelong_methods.methods.scad_utils.vit_afd import TokenizedDistillation
import torch.nn.functional as F
from lifelong_methods.methods.scad_utils.vision_transformer import vit_base_patch16_224_twf

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

    for i, (idx, net_feat, pret_feat) in enumerate(zip(dist_indices, partial_feats_s, partial_feats_t)):
        adapter = getattr(
                net, f"adapter_{idx+1}")

        output_rho, logits = adapter.attn_fn(pret_feat, y, task_labels)
        attention_maps.append(output_rho)
        logits_maps.append(logits)

    return res_s, res_t, attention_maps, logits_maps

def batch_iterate(size: int, batch_size: int):
    n_chunks = size // batch_size
    for i in range(n_chunks):
        yield torch.LongTensor(list(range(i * batch_size, (i + 1) * batch_size)))

class Model(BaseMethod):
    """
    A finetuning (Experience Replay) baseline.
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, config)

        self.current_task = 0
        self.device = self.config['device']
        self.net = timm.create_model(f'vit_base_patch16_224_twf', pretrained=True, num_classes=self.num_classes).to(self.device)
        self.buffer = MyBuffer(self.config['buffer_size'], self.device)
        self.transform = self.get_transforms(self.config['dataset'])
        self.not_aug_transform = self.get_test_transforms(self.config['dataset'])
        self.seen_y_so_far = torch.zeros(self.num_classes).bool().to(self.device)

        # setup losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.compress = False if self.config['adapter_type'] == 'attention_probe_filter' else True
    
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
        if self.current_task == 0:
            self.prenet = timm.create_model(f'vit_base_patch16_224_twf', pretrained=True, num_classes=self.num_classes)
            self.prenet = self.prenet.to(self.device)
            self.prenet.eval()

            self.hh_s = HooksHandlerViT(self.net)
            self.hh_t = HooksHandlerViT(self.prenet)

            # Retrieve features
            # Ci serve sapere la shape delle features per costruire gli adapter
            with torch.no_grad():
                x = torch.randn((1, 3, 224, 224)).to(self.device)
                res = self.net(x, returnt='full')
                feats_t = res[self.config['distillation_layers']]
                prenet_input = x
                res = self.prenet(prenet_input, returnt='full')
                pret_feats_t = res[self.config['distillation_layers']]

            self.dist_indices = [int(x) for x in self.config['adapter_layers'].split(',')]
            feats_t = [feats_t[i] for i in self.dist_indices]
            pret_feats_t = [pret_feats_t[i] for i in self.dist_indices]
            
            for (i, x, pret_x) in zip(self.dist_indices, feats_t, pret_feats_t):
                # clear_grad=self.args.detach_skip_grad == 1
                adapt_shape = x.shape[1:]
                pret_shape = pret_x.shape[1:]
                if len(adapt_shape) == 1:
                    adapt_shape = (adapt_shape[0], 1, 1)  # linear is a cx1x1
                    pret_shape = (pret_shape[0], 1, 1)

                setattr(self.net, f"adapter_{i+1}", TokenizedDistillation(
                    adapt_shape, self.num_tasks, self.n_cla_per_tsk, adatype=self.config['adapter_type'],
                    teacher_forcing_or=False,
                    lambda_forcing_loss=self.config['lambda_fp_replay'],
                    lambda_diverse_loss=self.config['lambda_diverse_loss'],
                    use_prompt=self.config['use_prompt'] == 1,
                    use_conditioning=self.config['use_conditioning'] == 1,
                    lambda_ignition_loss=self.config['lambda_ignition_loss'],
                    ignition_loss_temp=self.config['ignition_loss_temp'],
                    exclude_class_token=self.config['exclude_class_token'] == 1,
                    diverse_loss_mode=self.config['diverse_loss_mode'],
                ).to(self.device))

                if self.config['adapter_type'] in ['transformer_pretrained', 'transformer_pretrained_proj', 'transformer_pretrained_layer_norm']:
                    adapter = getattr(self.net, f"adapter_{i+1}")
                    adapter.attn_fn.self_attn.load_state_dict(self.prenet.blocks[i+1].state_dict())
            
            # if self.args.load_checkpoint is not None:
            #     self.net.load_state_dict(torch.load(self.args.load_checkpoint))

            myparams = dict(self.net.named_parameters())
            net_params = [myparams[s] for s in myparams.keys() if 'adapter' not in s]
            adapter_params = [myparams[s] for s in myparams.keys() if 'adapter' in s]

            # if self.args.optimizer == 'sgd':
            #     self.opt = SGD(net_params, lr=self.args.lr,
            #                 weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
                
            #     try:
            #         self.opt_adapters = SGD(adapter_params, lr=self.args.adapter_lr,
            #                     weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
            #     except:
            #         self.opt_adapters = None
            # elif self.args.optimizer == 'adam':
            #     self.opt = Adam(net_params, lr=self.args.lr, weight_decay=self.args.optim_wd)
                
            #     self.opt_adapters = Adam(adapter_params, lr=self.args.adapter_lr, weight_decay=self.args.optim_wd)

            if self.config['optimizer'] == 'sgd':
                self.opt = torch.optim.SGD(net_params, lr=self.config['lr'],
                            weight_decay=self.config['weight_decay'], momentum=self.config['optim_mom'])
            elif self.config['optimizer'] == 'adam':
                self.opt = torch.optim.Adam(net_params, lr=self.config['lr'], weight_decay=self.config['weight_decay'])
            
            try:
                if self.config['adapter_optimizer'] == 'sgd':
                    self.opt_adapters = torch.optim.SGD(adapter_params, lr=self.config['adapter_lr'],
                                    weight_decay=self.config['weight_decay'], momentum=self.config['optim_mom'])
                elif self.config['adapter_optimizer'] == 'adam':
                    self.opt_adapters = torch.optim.Adam(adapter_params, lr=self.config['adapter_lr'], weight_decay=self.config['weight_decay'])
            except:
                self.opt_adapters = None

        #self.opt = self.get_optimizer()
        self.net.train()
        for p in self.prenet.parameters():
            p.requires_grad = False
        pass

    def partial_distill_loss(self, net_partial_features: list, pret_partial_features: list,
                             targets, teacher_forcing: list = None, extern_attention_maps: list = None, task_labels=None):

        assert len(net_partial_features) == len(
            pret_partial_features), f"{len(net_partial_features)} - {len(pret_partial_features)}"

        if teacher_forcing is None or extern_attention_maps is None:
            assert teacher_forcing is None
            assert extern_attention_maps is None

        loss = 0
        losses = {}
        attention_maps = []

        for i, (idx, net_feat, pret_feat) in enumerate(zip(self.dist_indices, net_partial_features, pret_partial_features)):
            assert net_feat.shape == pret_feat.shape, f"{net_feat.shape} - {pret_feat.shape}"

            adapter = getattr(
                self.net, f"adapter_{idx+1}")

            pret_feat = pret_feat.detach()

            if teacher_forcing is None:
                curr_teacher_forcing = torch.zeros(
                    len(net_feat,)).bool().to(self.device)
                curr_ext_attention_map = torch.ones(
                    (len(net_feat), adapter.embed_dim)).to(self.device)
            else:
                curr_teacher_forcing = teacher_forcing
                curr_ext_attention_map = torch.stack(
                    [b[i] for b in extern_attention_maps], dim=0).float()

            adapt_loss, adapt_attention, inlosses = adapter(net_feat, pret_feat, targets,
                                                  teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map,
                                                  task_labels=task_labels)
            losses = {**losses, **{f'adapter_{idx+1}_{k}': v for k, v in inlosses.items()}}

            loss += adapt_loss
            attention_maps.append(adapt_attention.detach().cpu().clone().data)

        # TODO: Vedere se questo i deve essere idx
        return loss / (i + 1), attention_maps, losses

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

        # if self.args.use_patch_level_aug:
        #     inputs = patch_level_aug(inputs)

        labels = y.long()
        inputs = x
        B = x.shape[0]

        stream_task_labels = torch.ones(B)*self.current_task
        stream_task_labels = stream_task_labels.long().to(self.device)
        with torch.no_grad():
            not_aug_inputs_tmp = torch.stack([self.not_aug_transform(i) for i in not_aug_inputs]).to(self.device)
            _, _, stream_not_aug_attention_maps, _ = get_twf_vit_outputs(self.net, self.prenet, not_aug_inputs_tmp,
                                                                      labels, self.config, stream_task_labels)
            stream_not_aug_attention_maps = [i.detach().cpu().clone().data for i in stream_not_aug_attention_maps]

        loss_attention_maps_replay = torch.tensor(0.0).to(self.device)
        if len(self.buffer) > 0:
            # sample from buffer
            buf_choices, buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_logits_mask = self.buffer.get_data(
                self.config['batch_size'], transform=None, return_index=True)
            buf_attention_maps = [self.buffer.attention_maps[c]
                                  for c in buf_choices]

            buf_attention_maps = [[i.to(self.device) for i in v] for v in buf_attention_maps]
            
            buf_not_aug_inputs_tmp = torch.stack([self.not_aug_transform(i) for i in buf_inputs]).to(self.device)
            _, _, buf_not_aug_attn_maps, buf_not_aug_adapter_logits = get_twf_vit_outputs(self.net, self.prenet, buf_not_aug_inputs_tmp, buf_labels, 
                                                                   self.config, buf_task_labels)
            losses_attn_maps = []
            buf_attention_maps_t = []
            for i in range(len(buf_attention_maps[0])):
                buf_attention_maps_t.append([b[i] for b in buf_attention_maps])
            if self.config['adapter_type'] in ['attention_probe_filter', 'attention_probe_softmax', 'attention_probe_new', 'attention_probe_v2']:
                for gt, pred in zip(buf_attention_maps_t, buf_not_aug_attn_maps):
                    gt = torch.stack(gt, dim=0)
                    losses_attn_maps.append(torch.nn.functional.mse_loss(pred, gt))
                loss_attention_maps_replay = torch.mean(torch.stack(losses_attn_maps))
            else:
                for gt, pred in zip(buf_attention_maps_t, buf_not_aug_adapter_logits):
                    for g, p in zip(gt, pred):
                        losses_attn_maps.append(F.binary_cross_entropy_with_logits(p[:, 0, :], g.float()))
                loss_attention_maps_replay = torch.mean(torch.stack(losses_attn_maps))
            
            aug_inputs = torch.stack([self.transform(buf_input) for buf_input in buf_inputs]).to(self.device)

            # if self.args.use_patch_level_aug:
            #     aug_inputs = patch_level_aug(transforms.Resize(224)(buf_inputs))

            inputs = torch.cat([inputs, aug_inputs])
            all_labels = torch.cat([labels, buf_labels[:, :offset_2]])
            all_task_labels = torch.cat([stream_task_labels, buf_task_labels])

        prenet_input =  inputs
        with torch.no_grad():
            res_t = self.prenet(prenet_input, returnt='full')
            # res_t['attention_masks'] = self.hh_t.attentions
            # self.hh_t.reset()
            all_pret_logits, all_pret_partial_features = res_t['output'], res_t[self.config['distillation_layers']]

        res_s = self.net(inputs, returnt='full')
        # res_s['attention_masks'] = self.hh_s.attentions
        # self.hh_s.reset()
        all_logits, all_partial_features = res_s['output'], res_s[self.config['distillation_layers']]

        all_partial_features = [all_partial_features[i] for i in self.dist_indices]
        all_pret_partial_features = [all_pret_partial_features[i] for i in self.dist_indices]

        stream_logits, buf_outputs = all_logits[:B], all_logits[B:]
        stream_partial_features = [p[:B] for p in all_partial_features]
        stream_pret_partial_features = [p[:B]
                                        for p in all_pret_partial_features]


        output_mask = self.seen_y_so_far.unsqueeze(0).expand_as(stream_logits).detach().clone()
        output = stream_logits[:, :offset_2]
        idx = target.sum(0).nonzero().squeeze(1)
        filtered_output = output[:, idx]
        filtered_target = target[:, idx]
        loss = self.bce(filtered_output / self.temperature, filtered_target)
        #loss = self.bce(output / self.temperature, target)

        loss_clf = loss.detach().clone()

        self.seen_y_so_far[:offset_2] |= y.any(dim=0).data

        loss_er = torch.tensor(0.).to(self.device)
        loss_der = torch.tensor(0.).to(self.device)
        loss_afd = torch.tensor(0.).to(self.device)

        if len(self.buffer) == 0:
            loss_afd, stream_attention_maps, losses = self.partial_distill_loss(
                stream_partial_features[-len(stream_pret_partial_features):], stream_pret_partial_features, labels, task_labels=stream_task_labels)
        else:
            buffer_teacher_forcing = buf_task_labels != self.current_task
            teacher_forcing = torch.cat(
                (torch.zeros((B)).bool().to(self.device), buffer_teacher_forcing))
            attention_maps = [
                [torch.ones_like(map) for map in buf_attention_maps[0]]]*B + buf_attention_maps

            loss_afd, all_attention_maps, losses = self.partial_distill_loss(all_partial_features[-len(
                all_pret_partial_features):], all_pret_partial_features, all_labels,
                teacher_forcing, attention_maps, task_labels=all_task_labels)

            stream_attention_maps = [ap[:B] for ap in all_attention_maps]

            # if self.args.use_erace_optim:
            #     loss_er = self.loss(
            #         buf_outputs[:, :(self.task+1)*self.cpt], buf_labels)
            #     loss_der = F.mse_loss(
            #         buf_outputs, buf_logits[:, :self.num_classes])
            # else:
            loss_er = self.bce(buf_outputs[:, :offset_2], buf_labels[:, :offset_2].float())
            der_buf_outputs = buf_outputs.clone()
            der_buf_outputs[~buf_logits_mask] = 0.0
            der_buf_logits = buf_logits.clone()
            der_buf_logits[~buf_logits_mask] = 0.0
            loss_der = F.mse_loss(der_buf_outputs, der_buf_logits)
        
        loss += self.config['der_beta'] * loss_er
        loss += self.config['der_alpha'] * loss_der
        loss += self.config['lambda_fp'] * loss_afd
        loss += self.config['lambda_fp_replay'] * loss_attention_maps_replay

        # TODO weigh the buffer loss by the self.memory_strength before getting the loss mean (use in_buffer)
        if train:
            self.opt.zero_grad()
            if self.opt_adapters is not None:
                self.opt_adapters.zero_grad()
            
            loss.backward()
            if self.config['clip_grad'] is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config['clip_grad'])
            self.opt.step()
            if self.opt_adapters:
                self.opt_adapters.step()

            if self.config['wandb_log']:
                log_dict = {
                    "loss": loss.item(),
                    "loss_clf": loss_clf.item(),
                    "loss_er": loss_er.item(),
                    "loss_der": loss_der.item(),
                    "loss_afd": loss_afd.item(),
                    "loss_attention_maps_replay": loss_attention_maps_replay.item(),
                    }
                log_dict.update(losses)
                wandb.log(log_dict)
            
            to_save_labels = torch.cat((target, torch.zeros(target.shape[0], self.num_classes - target.shape[1]).to(self.device)), dim=1)
            
            if output_mask.sum() > 0:
                self.buffer.add_data(examples=not_aug_inputs,
                    labels=to_save_labels,
                    logits=stream_logits.data,
                    attention_maps=stream_not_aug_attention_maps,
                    task_labels=torch.ones(B)*self.current_task,
                    logits_mask=output_mask.data,
                    compress=self.compress)

        predictions = output > 0.0
        return predictions, loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The method used during inference, returns a tensor of model predictions

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            torch.Tensor: a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        """
        num_seen_classes = len(self.seen_classes)
        output = self.net(x)
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
        if self.config['save_checkpoints'] and self.current_task in [0, 1, 2, 10, self.num_tasks - 1]:
            self.save_checkpoints()

        self.eval()
        with torch.no_grad():
            # loop over buffer
            for buf_idxs in batch_iterate(len(self.buffer), self.config['batch_size']):

                buf_idxs = buf_idxs.to(self.device)
                buf_labels = self.buffer.labels[buf_idxs].to(self.device)
                buf_task_labels = self.buffer.task_labels[buf_idxs].to(self.device)

                buf_mask = buf_task_labels == self.current_task

                if not buf_mask.any():
                    continue

                buf_inputs = self.buffer.examples[buf_idxs][buf_mask]
                buf_labels = buf_labels[buf_mask]
                buf_task_labels = buf_task_labels[buf_mask]
                buf_inputs = torch.stack([self.not_aug_transform(
                    ee.cpu()) for ee in buf_inputs]).to(self.device)

                res_s = self.net(
                    buf_inputs, returnt='full')
                buf_partial_features = res_s[self.config['distillation_layers']]
                prenet_input = buf_inputs
                res_t = self.prenet(prenet_input, returnt='full')
                pret_buf_partial_features = res_t[self.config['distillation_layers']]


                # buf_partial_features = buf_partial_features[:-1]
                # pret_buf_partial_features = pret_buf_partial_features[:-1]

                buf_partial_features = [buf_partial_features[i] for i in self.dist_indices]
                pret_buf_partial_features = [pret_buf_partial_features[i] for i in self.dist_indices]

                _, attention_masks, _ = self.partial_distill_loss(buf_partial_features[-len(
                    pret_buf_partial_features):], pret_buf_partial_features, buf_labels, task_labels=buf_task_labels)

                for i_of_idx, idx in enumerate(buf_idxs[buf_mask]):
                    self.buffer.attention_maps[idx] = [
                        at[i_of_idx] for at in attention_masks]

        self.net.train()

        self.current_task += 1
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
