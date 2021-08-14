# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple

from overrides import overrides

import os
import yaml
from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import TArchTrainer
from archai.algos.darts.darts_model_desc_builder import DartsModelDescBuilder
from archai.algos.darts.bilevel_arch_trainer_sup import BilevelArchTrainer
from archai.nas.searcher_ssl import SearcherSimClr, SearchResult
from archai.nas.evaluater_ssl import EvaluaterSimClr, EvalResult
from archai.common.config import Config
from archai.common import utils, common


class DartsExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->DartsModelDescBuilder:
        return DartsModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return BilevelArchTrainer

    @overrides
    def run_eval(self, conf:Config)->EvalResult:
        evaler = self.evaluater()
        return evaler.evaluate(conf,
                               model_desc_builder=self.model_desc_builder())

    def update_eval_desc_path(self, conf_eval:Config):
        root_dirname = conf_eval['search_desc_dir']
        train_config_path = os.path.join(root_dirname,'config_used.yaml')
        with open(train_config_path) as f:
            conf_train = yaml.load(f, Loader=yaml.Loader)
        epoch_save_dirname = conf_train['nas']['search']['epoch_model_desc']['savedir'].split('/')[-1]
        filename = conf_train['nas']['search']['epoch_model_desc']['filename']
        epoch_no = conf_eval['load_epoch_no']

        model_desc_filepath = os.path.join(root_dirname,epoch_save_dirname,f'{filename}_{epoch_no}.yaml')
        if not os.path.exists(root_dirname):
            raise Exception(f'Root results directory from search {root_dirname} not found')
        elif not os.path.exists(os.path.join(root_dirname,epoch_save_dirname)):
            raise Exception(f'Epoch model description save directory from search {os.path.join(root_dirname,epoch_save_dirname)} not found')
        else:
            print(f'Loading model description from epoch {epoch_no}')
            conf_eval['final_desc_filename'] = model_desc_filepath

    @overrides
    def _init_conf(self, is_search_or_eval:bool, clean_expdir:bool)->Config:
        config_filename = self.config_filename

        if utils.is_main_process():
            conf = common.common_init(config_filepath=config_filename,
                param_args=['--common.experiment_name', self.get_expname(is_search_or_eval),
                            ], clean_expdir=clean_expdir)
            print('Running main process')
        else:
            conf = common.create_conf(config_filepath=config_filename,
                param_args=['--common.experiment_name', self.get_expname(is_search_or_eval),
                            ])
            Config.set_inst(conf)
            common.update_envvars(conf)
            commonstate = common.get_state()
            common.init_from(commonstate,recreate_logger=False)
            print('Running child process')

        return conf

    @overrides
    def run(self, search=True, eval=True) \
            ->Tuple[Optional[SearchResult], Optional[EvalResult]]:

        search_result, eval_result = None, None

        if search: # run search
            conf = self._init_conf(True, clean_expdir=self.clean_expdir)
            if utils.is_main_process():
                common.create_epoch_desc_dir(conf)
            search_result = self.run_search(conf['nas']['search'])

        if eval:
            conf = self.get_conf(False)
            if utils.is_main_process():
                common.clean_ensure_expdir(conf, clean_dir=self.clean_expdir, ensure_dir=True)
                if search:
                    # first copy search result to eval, otherwise we expect eval config to point to results
                    self.copy_search_to_eval()

            conf = self._init_conf(False, clean_expdir=False)
            self.update_eval_desc_path(conf['nas']['eval'])
            eval_result = self.run_eval(conf)

        return search_result, eval_result

