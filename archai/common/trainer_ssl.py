# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil
from typing import Callable, Tuple, Optional

import os
import wandb
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from archai.common.metrics_ssl import MetricsSimClr
from archai.common.tester_ssl import TesterSimClr
from archai.common.config import Config
from archai.common import utils, ml_utils
from archai.common.common import logger
from archai.datasets import data
from archai.common.checkpoint import CheckPoint
from archai.common.dist_utils import ApexUtils
from archai.common.multi_optim import MultiOptim, OptimSched


class TrainerSimClr(EnforceOverrides):
    def __init__(self, conf_train:Config, model:nn.Module,
                 checkpoint:Optional[CheckPoint]=None)->None:
        # region config vars
        self._conf_train = conf_train
        conf_wandb = conf_train['wandb']
        conf_lossfn = conf_train['lossfn']
        self.is_wandb_enabled = conf_wandb['enabled']
        self._grad_clip = conf_train['grad_clip']
        self._drop_path_prob = conf_train['drop_path_prob']
        self._logger_freq = conf_train['logger_freq']
        self._title = conf_train['title']
        self._epochs = conf_train['epochs']
        self.conf_optim = conf_train['optimizer']
        self.conf_sched = conf_train['lr_schedule']
        self.batch_chunks = conf_train['batch_chunks']
        conf_validation = conf_train['validation']
        conf_apex = conf_train['apex']
        self._validation_freq = 0 if conf_validation is None else conf_validation['freq']

        # endregion
        logger.pushd(self._title + '__init__')

        self._apex = ApexUtils(conf_apex, logger)

        if self.is_wandb_enabled and self._apex.is_master():
            wandb.define_metric("global_step")
            wandb.define_metric("epoch")
            wandb.define_metric("loss_train", step_metric="global_step")
            wandb.define_metric("lr", step_metric="global_step")
            wandb.define_metric("epoch_loss_train", step_metric="epoch")
            wandb.define_metric("epoch_timings_train", step_metric="epoch")
            wandb.define_metric("avg_step_timings_train", step_metric="epoch")

        self._checkpoint = checkpoint
        self.model = model

        self._lossfn = ml_utils.get_lossfn(conf_lossfn)
        # using separate apex for Tester is not possible because we must use
        # same distributed model as Trainer and hence they must share apex
        self._tester = TesterSimClr(conf_train, model, self._apex) \
                        if conf_validation else None
        self._metrics:Optional[MetricsSimClr] = None

        self._droppath_module = self._get_droppath_module()
        if self._droppath_module is None and self._drop_path_prob > 0.0:
            logger.warn({'droppath_module': None})

        self._start_epoch = -1 # nothing is started yet

        logger.popd()

    def fit(self, data_loaders:data.DataLoaders)->MetricsSimClr:
        logger.pushd(self._title)

        assert data_loaders.train_dl is not None

        self._metrics = MetricsSimClr(self._title, self._apex, logger_freq=self._logger_freq)

        # create optimizers and schedulers
        self._multi_optim = self.create_multi_optim(len(data_loaders.train_dl))
        # before checkpoint restore, convert to amp
        self.model = self._apex.to_amp_dist(self.model, self._multi_optim,
                                       batch_size=data_loaders.train_dl.batch_size)

        self._lossfn = self._lossfn.to(self.get_device())

        self.pre_fit(data_loaders)

        # we need to restore checkpoint after all objects are created because
        # restoring checkpoint requires load_state_dict calls on these objects
        self._start_epoch = 0
        # do we have a checkpoint
        checkpoint_avail = self._checkpoint is not None
        checkpoint_val = checkpoint_avail and 'trainer' in self._checkpoint
        resumed = False
        if checkpoint_val:
            # restore checkpoint
            resumed = True
            self.restore_checkpoint()
        elif checkpoint_avail: # TODO: bad checkpoint?
            self._checkpoint.clear()
        logger.warn({'resumed': resumed, 'checkpoint_avail': checkpoint_avail,
                     'checkpoint_val': checkpoint_val,
                     'start_epoch': self._start_epoch,
                     'total_epochs': self._epochs})
        logger.info({'grad_clip': self._grad_clip,
                     'drop_path_prob': self._drop_path_prob,
                     'validation_freq': self._validation_freq,
                     'batch_chunks': self.batch_chunks})

        if self._start_epoch >= self._epochs:
            logger.warn(f'fit done because start_epoch {self._start_epoch}>={self._epochs}')
            self.post_fit(data_loaders)
            return self.get_metrics() # we already finished the run, we might be checkpointed

        logger.pushd('epochs')
        for epoch in range(self._start_epoch, self._epochs):
            logger.pushd(epoch)
            self._set_epoch(epoch, data_loaders)
            self.pre_epoch(data_loaders)
            self._train_epoch(data_loaders.train_dl)
            self.post_epoch(data_loaders,epoch)
            logger.popd()
        logger.popd()
        self.post_fit(data_loaders)

        # make sure we don't keep references to the graph
        del self._multi_optim

        logger.popd()
        return self.get_metrics()

    def create_multi_optim(self, train_len:int)->MultiOptim:
        logger.info({'steps_per_epoch': train_len,
                     'conf_sched': self.conf_sched.to_dict()})
        logger.info({'conf_optim': self.conf_optim.to_dict()})

        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state specific to each run
        optim = self.create_optimizer(self.conf_optim, self.model.parameters())
        # create scheduler for optim before applying amp
        sched, sched_on_epoch = self.create_scheduler(self.conf_sched, optim, train_len)

        multi_optim = MultiOptim()
        multi_optim.append(OptimSched(optim, sched, sched_on_epoch))

        logger.info({'multi_optim_len': len(multi_optim)})

        return multi_optim

    def create_optimizer(self, conf_optim:Config, params)->Optimizer:
        optim = ml_utils.create_optimizer(conf_optim, params)
        return optim

    def create_scheduler(self, conf_sched:Config, optim:Optimizer, steps_per_epoch:int) \
            ->Tuple[Optional[_LRScheduler],bool]:
        return ml_utils.create_lr_scheduler(conf_sched, self._epochs,
            optim, steps_per_epoch)

    def get_optimizer(self, index=0)->Optimizer:
        return self._multi_optim[index].optim
    def get_scheduler(self, index=0)->Optional[_LRScheduler]:
        return self._multi_optim[index].sched

    def get_metrics(self)->MetricsSimClr:
        return self._metrics

    def _set_epoch(self, epoch:int, data_loaders:data.DataLoaders)->None:
        # optimizers such as bi-level may use val set for its own use
        # which causes reshuffling due to automatic epoch counting
        # here we make sure that val_dl has same epoch as train_dl
        if hasattr(data_loaders.train_dl.sampler, 'set_epoch'):
            data_loaders.train_dl.sampler.set_epoch(epoch)
        if data_loaders.val_dl is not None and hasattr(data_loaders.val_dl.sampler, 'set_epoch'):
            data_loaders.val_dl.sampler.set_epoch(epoch)

        # apply droppath
        self._set_drop_path(epoch, self._epochs)

        assert self._metrics.epochs() == epoch

    #########################  hooks #########################
    def reduce_sum(self, val):
        return val
        if self._metrics.is_dist():
            return self._metrics.reduce_sum(val)
        else:
            return val
    def reduce_mean(self, val):
        return val
        if self._metrics.is_dist():
            return self._metrics.reduce_mean(val)
        else:
            return val

    def pre_fit(self, data_loaders:data.DataLoaders)->None:
        self._metrics.pre_run()

    def post_fit(self, data_loaders:data.DataLoaders)->None:
        test_metrics = None
        # first run test before checkpointing, otherwise we won't have val metrics
        if data_loaders.test_dl and self._tester:
            test_metrics = self._tester.test(data_loaders.train_dl, phase='test')
            run_metrics = test_metrics.run_metrics
            if self.is_wandb_enabled and self._apex.is_master():
                wandb.run.summary[f'duration_test'] = self.reduce_sum(run_metrics.duration())

        self._metrics.post_run(test_metrics=test_metrics)
        run_metrics = self._metrics.run_metrics
        best_train, best_val, best_test = run_metrics.best_epoch()
        if self.is_wandb_enabled and self._apex.is_master():
            wandb.run.summary[f'duration_train'] = self.reduce_sum(run_metrics.duration())
            if best_train:
                wandb.run.summary['best_train_epoch'] = self.reduce_mean(best_train.index)
                wandb.run.summary['best_train_loss'] = self.reduce_mean(best_train.loss.avg)
            if best_val:
                wandb.run.summary['best_val_epoch'] = self.reduce_mean(best_val.index)
                wandb.run.summary['best_val_loss'] = self.reduce_mean(best_val.loss.avg)
            if best_test:
                wandb.run.summary['best_test_epoch'] = self.reduce_mean(best_test.index)
                wandb.run.summary['best_test_loss'] = self.reduce_mean(best_test.loss.avg)

            save_intermediate = self._conf_train['save_intermediate']
            intermediatedir = self._conf_train['intermediatedir']
            if save_intermediate:
                logdir = utils.full_path(os.environ['logdir'])
                for folder in os.listdir(logdir):
                    srcdir = os.path.join(logdir,folder)
                    destdir = os.path.join(intermediatedir,folder)
                    if os.path.exists(destdir):
                        if os.path.isdir(destdir):
                            shutil.rmtree(destdir)
                        else:
                            os.remove(destdir)
                    if os.path.isdir(srcdir):
                        shutil.copytree(srcdir,destdir)
                    else:
                        shutil.copy(srcdir,destdir)
                print(f'Copied files from logdir {logdir} to intermediate dir {intermediatedir}')

    def pre_epoch(self, data_loaders:data.DataLoaders)->None:
        self._metrics.pre_epoch(lr=self._multi_optim.get_lr(0, 0))

    def post_epoch(self, data_loaders:data.DataLoaders, epoch:int)->None:
        val_metrics = None
        # first run test before checkpointing, otherwise we won't have val metrics
        if data_loaders.val_dl and self._tester and self._validation_freq > 0:
            if self._metrics.epochs() % self._validation_freq == 0 or \
                    self._metrics.epochs() >= self._epochs: # last epoch

                # these asserts makes sure train and val are not ovrlapiing
                # assert train_dl.sampler.epoch == val_dl.sampler.epoch
                # tidx = list(train_dl.sampler)
                # vidx = list(val_dl.sampler)
                # assert all(ti not in vidx for ti in tidx)

                val_metrics = self._tester.test(data_loaders.val_dl, epochs=self._metrics.epochs(), phase='val')

        # update val metrics
        self._metrics.post_epoch(lr=self._multi_optim.get_lr(0, 0), val_metrics=val_metrics)
        if self.is_wandb_enabled and self._apex.is_master():
            epoch_metric = self._metrics.run_metrics.cur_epoch()
            wandb.log({'epoch_loss_train':self.reduce_mean(epoch_metric.loss.avg),
                       'epoch_timings_train':self.reduce_mean(epoch_metric.duration()),
                       'avg_step_timings_train':self.reduce_mean(epoch_metric.step_time.avg),
                       'epoch':epoch})

        # checkpoint if enabled with given freq or if this is the last epoch
        if self._checkpoint is not None and self._apex.is_master() and \
            self._checkpoint.freq > 0 and (self._metrics.epochs() % self._checkpoint.freq == 0 or \
                    self._metrics.epochs() >= self._epochs):
            print('Running save checkpoint in master process')
            best_train = self._checkpoint["trainer_best_train"] \
                        if "trainer_best_train" in self._checkpoint.data.keys() else None
            best_val = self._checkpoint["trainer_best_val"] \
                        if "trainer_best_val" in self._checkpoint.data.keys() else None
            self._checkpoint.new()
            if best_train:
                self._checkpoint["trainer_best_train"] = best_train
            if best_val:
                self._checkpoint["trainer_best_val"] = best_val
            self._checkpoint["epoch"] = epoch
            print('Updating best checkpoint values')
            self.update_checkpoint(self._checkpoint)
            print('Saving best checkpoint')
            self._checkpoint.commit()
            
            save_intermediate = self._conf_train['save_intermediate']
            intermediatedir = utils.full_path(self._conf_train['intermediatedir']) if self._conf_train['intermediatedir'] \
                                else self._conf_train['intermediatedir']
            if save_intermediate:
                logdir = utils.full_path(os.environ['logdir'])
                for folder in os.listdir(logdir):
                    srcdir = os.path.join(logdir,folder)
                    destdir = os.path.join(intermediatedir,folder)
                    if not os.path.exists(destdir):
                        os.makedirs(destdir,exist_ok=True)
                    if os.path.exists(destdir):
                        if os.path.isdir(destdir):
                            shutil.rmtree(destdir)
                        else:
                            os.remove(destdir)
                    if os.path.isdir(srcdir):
                        shutil.copytree(srcdir,destdir)
                    else:
                        shutil.copy(srcdir,destdir)
                print(f'Copied files from logdir {logdir} to intermediate dir {intermediatedir}')

    def pre_step(self, xi:Tensor, xj:Tensor)->None:
        self._metrics.pre_step(xi,xj)

    def post_step(self, loss:Tensor, batch_size:int)->None:
        self._metrics.post_step(loss, batch_size)
        if self.is_wandb_enabled and self._apex.is_master() \
           and self._logger_freq > 0 and \
                ((self._metrics.global_step+1) % self._logger_freq == 0):
            wandb.log({"lr":self._multi_optim.get_lr(0, 0),
                       "loss_train":self.reduce_mean(loss),
                       "global_step":self._metrics.global_step})
    #########################  hooks #########################

    def get_device(self):
        return self._apex.device

    def restore_checkpoint(self)->None:
        state = self._checkpoint['trainer']
        last_epoch = state['last_epoch']
        assert last_epoch >= 0 and last_epoch < self._epochs

        self._metrics.load_state_dict(state['metrics'])
        assert self._metrics.epochs() == last_epoch+1
        self._apex.load_state_dict(state['amp'])
        self.model.load_state_dict(state['model'])
        self._multi_optim.load_state_dict(state['multi_optim'])

        self._start_epoch = last_epoch + 1

    def epoch(self)->int:
        return self._metrics.epochs()

    def update_checkpoint(self, checkpoint:CheckPoint)->None:
        # TODO: Don't need to pass checkpoint
        # save all necessory state
        state = {
            'last_epoch': self._metrics.epochs()-1,
            'metrics': self._metrics.state_dict(),
            'model': self.model.state_dict(),
            'multi_optim': self._multi_optim.state_dict(),
            'amp': self._apex.state_dict()
        }
        cur_train_loss = self._metrics.run_metrics.cur_epoch().loss.avg
        cur_val_loss = self._metrics.run_metrics.cur_epoch().val_metrics.loss.avg \
                    if self._metrics.run_metrics.cur_epoch().val_metrics is not None else -1
        best_train_loss = self._metrics.run_metrics.best_train.loss.avg if self._metrics.run_metrics.best_train is not None else -1
        best_val_loss = self._metrics.run_metrics.best_val.val_metrics.loss.avg if self._metrics.run_metrics.best_val and self._metrics.run_metrics.best_val.val_metrics else -1
        if cur_train_loss == best_train_loss:
            self._checkpoint['trainer_best_train'] = state
        if cur_val_loss == best_val_loss:
            self._checkpoint['trainer_best_val'] = state

        self._checkpoint['trainer'] = state

    def _train_epoch(self, train_dl: DataLoader)->None:
        steps = len(train_dl)
        self.model.train()

        logger.pushd('steps')
        for step, ((xi,xj,_), _) in enumerate(train_dl):
            logger.pushd(step)
            assert self.model.training # derived class might alter the mode

            # TODO: please check that no algorithm is invalidated by swapping prestep with zero grad
            self._multi_optim.zero_grad()

            self.pre_step(xi,xj)

            # divide batch in to chunks if needed so it fits in GPU RAM
            if self.batch_chunks > 1:
                xi_chunks, xj_chunks = torch.chunk(xi, self.batch_chunks), torch.chunk(xj, self.batch_chunks)
            else:
                xi_chunks, xj_chunks = (xi,), (xj,)

            logits_chunks = []
            loss_sum, loss_count = 0.0, 0
            for xic, xjc in zip(xi_chunks, xj_chunks):
                xic, xjc = xic.to(self.get_device(), non_blocking=True), xjc.to(self.get_device(), non_blocking=True)

                logits_ic, logits_jc = self.model(xic), self.model(xjc)
                loss_c = self.compute_loss(self._lossfn, logits_ic, logits_jc)

                self._apex.backward(loss_c, self._multi_optim)

                loss_sum += loss_c.item() * len(logits_ic)
                loss_count += len(logits_ic)
                logits_chunks.append(logits_ic.detach().cpu())

            # TODO: original darts clips alphas as well but pt.darts doesn't
            self._apex.clip_grad(self._grad_clip, self.model, self._multi_optim)

            self._multi_optim.step()

            # TODO: we possibly need to sync so all replicas are upto date
            self._apex.sync_devices()

            self.post_step(loss_sum/loss_count,
                           ml_utils.join_chunks(logits_chunks).size(0)
                           )
            logger.popd()

            # end of step

        self._multi_optim.epoch()
        logger.popd()

    def compute_loss(self, lossfn:Callable, z1:Tensor, z2:Tensor)->Tensor:
        loss = lossfn(z1, z2)
        return loss

    def _get_droppath_module(self)->Optional[nn.Module]:
        m = self.model
        if hasattr(self.model, 'module'): # for data parallel model
            m = self.model.module
        if hasattr(m, 'drop_path_prob'):
            return m
        return None

    def _set_drop_path(self, epoch:int, epochs:int)->None:
        if self._drop_path_prob and self._droppath_module is not None:
            drop_prob = self._drop_path_prob * epoch / epochs
            # set value as property in model (it will be used by forward())
            # this is necessory when using DataParallel(model)
            # https://github.com/pytorch/pytorch/issues/16885
            m = self.model
            if hasattr(self.model, 'module'): # for data parallel model
                m = self.model.module
            if hasattr(m, 'drop_path_prob'):
                m.drop_path_prob(drop_prob)
            else:
                raise RuntimeError('Drop path value {} was specified but model'
                                   ' does not have drop_path_prob() method'\
                                       .format(self._drop_path_prob))
