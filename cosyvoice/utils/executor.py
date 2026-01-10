# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext
import os

import torch
import torch.distributed as dist

from cosyvoice.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join


class Executor:

    def __init__(self, gan: bool = False, ref_model: torch.nn.Module = None, dpo_loss: torch.nn.Module = None):
        self.gan = gan
        self.ref_model = ref_model
        self.dpo_loss = dpo_loss
        self.step = 0
        self.epoch = 0
        self.epoch_start_step = 0
        self.steps_in_epoch = 0
        self.steps_per_epoch_actual = 0
        self.stop_training = False
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.rank))

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=None):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(self.epoch, lr))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        self.epoch_start_step = self.step
        self.steps_in_epoch = 0
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                info_dict["epoch_start_step"] = self.epoch_start_step
                if self.steps_per_epoch_actual > 0:
                    info_dict["steps_per_epoch_actual"] = self.steps_per_epoch_actual
                    info_dict["total_steps_actual"] = self.steps_per_epoch_actual * info_dict.get("max_epoch", 0)
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict, ref_model=self.ref_model, dpo_loss=self.dpo_loss)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                log_per_step(writer, info_dict)
                eval_per_step = info_dict.get('eval_per_step', 0)
                save_per_step = info_dict.get('save_per_step', 0)
                eval_due = eval_per_step > 0 and (self.step + 1) % eval_per_step == 0 and \
                    (batch_idx + 1) % info_dict["accum_grad"] == 0
                save_due = save_per_step > 0 and (self.step + 1) % save_per_step == 0 and \
                    (batch_idx + 1) % info_dict["accum_grad"] == 0
                if eval_due:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                elif save_due:
                    dist.barrier()
                    self.save_step_checkpoint(model, info_dict)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
                    self.steps_in_epoch += 1
                    # Hook for subclass-specific per-step actions (e.g., TTS eval)
                    self.on_step_end(model, info_dict)
                    max_steps = info_dict.get('max_steps', 0)
                    if max_steps and self.step >= max_steps:
                        logging.info('Reached max_steps %s at step %s; stopping early.',
                                     max_steps, self.step)
                        self.stop_training = True
                        break
        dist.barrier()
        if self.stop_training:
            return
        if self.steps_in_epoch > 0:
            self.steps_per_epoch_actual = self.steps_in_epoch
            info_dict["steps_per_epoch_actual"] = self.steps_per_epoch_actual
            info_dict["total_steps_actual"] = self.steps_per_epoch_actual * info_dict.get("max_epoch", 0)
            info_dict["epoch_start_step"] = self.epoch_start_step
        eval_per_epoch = info_dict.get('eval_per_epoch', 1)
        eval_due_epoch = eval_per_epoch > 0 and (self.epoch + 1) % eval_per_epoch == 0
        if eval_due_epoch:
            self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, group_join):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(self.epoch, lr))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        self.epoch_start_step = self.step
        self.steps_in_epoch = 0
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                info_dict["epoch_start_step"] = self.epoch_start_step
                if self.steps_per_epoch_actual > 0:
                    info_dict["steps_per_epoch_actual"] = self.steps_per_epoch_actual
                    info_dict["total_steps_actual"] = self.steps_per_epoch_actual * info_dict.get("max_epoch", 0)
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    batch_dict['turn'] = 'discriminator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
                optimizer.zero_grad()
                log_per_step(writer, info_dict)
                with context():
                    batch_dict['turn'] = 'generator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                optimizer_d.zero_grad()
                log_per_step(writer, info_dict)
                eval_per_step = info_dict.get('eval_per_step', 0)
                save_per_step = info_dict.get('save_per_step', 0)
                eval_due = eval_per_step > 0 and (self.step + 1) % eval_per_step == 0 and \
                    (batch_idx + 1) % info_dict["accum_grad"] == 0
                save_due = save_per_step > 0 and (self.step + 1) % save_per_step == 0 and \
                    (batch_idx + 1) % info_dict["accum_grad"] == 0
                if eval_due:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                elif save_due:
                    dist.barrier()
                    self.save_step_checkpoint(model, info_dict)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
                    self.steps_in_epoch += 1
                    # Hook for subclass-specific per-step actions (e.g., TTS eval)
                    self.on_step_end(model, info_dict)
        dist.barrier()
        if self.steps_in_epoch > 0:
            self.steps_per_epoch_actual = self.steps_in_epoch
            info_dict["steps_per_epoch_actual"] = self.steps_per_epoch_actual
            info_dict["total_steps_actual"] = self.steps_per_epoch_actual * info_dict.get("max_epoch", 0)
            info_dict["epoch_start_step"] = self.epoch_start_step
        eval_per_epoch = info_dict.get('eval_per_epoch', 1)
        eval_due_epoch = eval_per_epoch > 0 and (self.epoch + 1) % eval_per_epoch == 0
        if eval_due_epoch:
            self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def on_step_end(self, model, info_dict):
        """Hook called after each training step. Override in subclass for custom behavior."""
        pass

    @torch.inference_mode()
    def save_step_checkpoint(self, model, info_dict):
        model_name = 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_model(model, model_name, info_dict)

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV'.format(self.epoch, self.step + 1, on_batch_end))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict['turn'] = 'generator'
            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.mean().item() * num_utts)
            log_per_step(None, info_dict)
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_model(model, model_name, info_dict)
