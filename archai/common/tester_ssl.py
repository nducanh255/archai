# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics_ssl import MetricsSimClr
from .config import Config
from . import utils, ml_utils
from .common import logger
from archai.common.apex_utils import ApexUtils

class TesterSimClr(EnforceOverrides):
    def __init__(self, conf_val:Config, model:nn.Module, apex:ApexUtils)->None:
        self._title = conf_val['title']
        self._logger_freq = conf_val['logger_freq']
        conf_lossfn = conf_val['lossfn']
        self.batch_chunks = conf_val['batch_chunks']

        self._apex = apex
        self.model = model
        self._lossfn = ml_utils.get_lossfn(conf_lossfn).to(apex.device)
        self._metrics = None

    def test(self, test_dl: DataLoader)->MetricsSimClr:
        logger.pushd(self._title)

        self._metrics = self._create_metrics()

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

                self._pre_step(self._metrics)

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

    def _pre_step(self, metrics:MetricsSimClr)->None:
        metrics.pre_step()

    def _post_step(self, loss:float, batch_size:int, metrics:MetricsSimClr)->None:
        metrics.post_step(loss, batch_size)

    def _create_metrics(self)->MetricsSimClr:
        return MetricsSimClr(self._title, self._apex, logger_freq=self._logger_freq)

