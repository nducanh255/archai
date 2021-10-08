# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Optional

import wandb
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics_ssl import MetricsSimClr
from .config import Config
from . import utils, ml_utils
from .common import logger
from archai.common.dist_utils import ApexUtils

class TesterSimClr(EnforceOverrides):
    def __init__(self, conf_train:Config, model:nn.Module, apex:ApexUtils)->None:
        conf_val = conf_train['validation']
        conf_wandb = conf_train['wandb']
        self.is_wandb_enabled = conf_wandb['enabled']
        self._title = conf_val['title']
        self._logger_freq = conf_val['logger_freq']
        conf_lossfn = conf_val['lossfn']
        self.batch_chunks = conf_val['batch_chunks']

        self._apex = apex
        self.model = model
        self._lossfn = ml_utils.get_lossfn(conf_lossfn).to(apex.device)
        self._metrics = None
        if self.is_wandb_enabled and self._apex.is_master():
            wandb.define_metric("epoch_loss_val", step_metric="epoch")
            wandb.define_metric("epoch_timings_val", step_metric="epoch")
            wandb.define_metric("avg_step_timings_val", step_metric="epoch")

    def test(self, test_dl: DataLoader, epochs:int = 0, phase:str = 'val')->MetricsSimClr:
        logger.pushd(self._title)

        self._metrics = self._create_metrics()
        self.epoch = epochs-1
        self.phase = phase

        # recreate metrics for this run
        self._pre_test()
        self._test_epoch(test_dl)
        self._post_test()

        logger.popd()
        return self.get_metrics() # type: ignore

    def _test_epoch(self, test_dl: DataLoader)->None:
        self._metrics.pre_epoch()
        self.model.eval()
        steps = len(test_dl)

        with torch.no_grad(), logger.pushd('steps'):
            for step, ((xi,xj,_), _) in enumerate(test_dl):
                # derived class might alter the mode through pre/post hooks
                assert not self.model.training
                logger.pushd(step)

                self._pre_step(self._metrics, xi, xj)

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
                    loss_c = self._lossfn(logits_ic, logits_jc)

                    loss_sum += loss_c.item() * len(logits_ic)
                    loss_count += len(logits_ic)
                    logits_chunks.append(logits_ic.detach().cpu())

                self._post_step(loss_sum/loss_count,
                            ml_utils.join_chunks(logits_chunks).size(0),
                            self._metrics
                            )

                # TODO: we possibly need to sync so all replicas are upto date
                self._apex.sync_devices()

                logger.popd()
        self._metrics.post_epoch() # no "val" dataset for the test phase

    def get_metrics(self)->Optional[MetricsSimClr]:
        return self._metrics

    def state_dict(self)->dict:
        return {
            'metrics': self._metrics.state_dict()
        }

    def get_device(self):
        return self._apex.device

    def load_state_dict(self, state_dict:dict)->None:
        self._metrics.load_state_dict(state_dict['metrics'])

    def _pre_test(self)->None:
        self._metrics.pre_run()

    def _post_test(self)->None:
        self._metrics.post_run()
        if self.is_wandb_enabled and self._apex.is_master() and self.phase == 'val':
            wandb.log({'epoch_loss_val':self.reduce_mean(self._metrics.run_metrics.cur_epoch().loss.avg),
                    'epoch_timings_val':self.reduce_sum(self._metrics.run_metrics.epoch_time_avg()),
                    'avg_step_timings_val': self.reduce_mean(self._metrics.run_metrics.step_time_avg()),
                    'epoch':self.epoch})

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

    def _pre_step(self, metrics:MetricsSimClr, xi:Tensor, xj: Tensor)->None:
        metrics.pre_step(xi, xj)

    def _post_step(self, loss:float, batch_size:int, metrics:MetricsSimClr)->None:
        metrics.post_step(loss, batch_size)

    def _create_metrics(self)->MetricsSimClr:
        return MetricsSimClr(self._title, self._apex, logger_freq=self._logger_freq)

