# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Mapping, Optional, Union
import copy
import os

import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn, autograd
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides

from archai.common.config import Config
from archai.nas.arch_trainer import ArchTrainer
from archai.common import utils, ml_utils
from archai.nas.model import Model
from archai.common.checkpoint import CheckPoint
from archai.datasets import data
from archai.common.common import logger
from archai.algos.darts.bilevel_optimizer import BilevelOptimizer

class BilevelArchTrainer(ArchTrainer):
    def __init__(self, conf_train: Config, model: Model,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)

        self._model = model
        self.epoch_model_desc = conf_train['epoch_model_desc']
        self._conf_w_optim = conf_train['optimizer']
        self._conf_w_lossfn = conf_train['lossfn']
        self._conf_alpha_optim = conf_train['alpha_optimizer']

    @overrides
    def pre_fit(self, data_loaders:data.DataLoaders)->None:
        super().pre_fit(data_loaders)

        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state
        assert data_loaders.val_dl is not None
        w_momentum = self._conf_w_optim['momentum']
        w_decay = self._conf_w_optim['decay']
        lossfn = ml_utils.get_lossfn(self._conf_w_lossfn).to(self.get_device())

        self._bilevel_optim = BilevelOptimizer(self._conf_alpha_optim, w_momentum,
                                                w_decay, self.model, lossfn,
                                                self.get_device(), self.batch_chunks)

    @overrides
    def post_fit(self, data_loaders:data.DataLoaders)->None:
        # delete state we created in pre_fit
        del self._bilevel_optim
        return super().post_fit(data_loaders)

    @overrides
    def pre_epoch(self, data_loaders:data.DataLoaders)->None:
        super().pre_epoch(data_loaders)

        # prep val set to train alphas
        assert data_loaders.val_dl is not None
        self._val_dl = data_loaders.val_dl
        self._valid_iter = iter(data_loaders.val_dl)  # type: ignore

    @overrides
    def post_epoch(self, data_loaders:data.DataLoaders)->None:
        del self._val_dl
        del self._valid_iter # clean up
        super().post_epoch(data_loaders)
        if self.epoch_model_desc['freq']>0 and ((self._metrics.epochs()+1)%self.epoch_model_desc['freq'])==0 and self._apex.is_master():
            print('Saving model desc in master process..')
            if self._apex.is_dist():
                model_copy = copy.deepcopy(self.model.module)
            else:
                model_copy = copy.deepcopy(self.model)
            model_desc = model_copy.finalizers.finalize_model(model_copy)
            filename, savedir = self.epoch_model_desc['filename'], utils.full_path(self.epoch_model_desc['savedir'])
            desc_filename = os.path.join(savedir,f'{filename}_{self._metrics.epochs()+1}.yaml')
            model_desc.save(desc_filename)
            print('Saved model desc in master process')

    @overrides
    def pre_step(self, x: Tensor, y: Tensor) -> None:
        super().pre_step(x, y)

        # reset val loader if we exausted it
        try:
            x_val, y_val = next(self._valid_iter)
        except StopIteration:
            # reinit iterator
            self._valid_iter = iter(self._val_dl)
            x_val, y_val = next(self._valid_iter)

        # update alphas
        self._bilevel_optim.step(x, y, x_val, y_val, super().get_optimizer())

    @overrides
    def update_checkpoint(self, check_point:CheckPoint)->None:
        super().update_checkpoint(check_point)
        check_point['bilevel_optim'] = self._bilevel_optim.state_dict()

    @overrides
    def restore_checkpoint(self)->None:
        super().restore_checkpoint()
        self._bilevel_optim.load_state_dict(self.check_point['bilevel_optim'])

