# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple

from overrides import overrides

from archai.common import utils, common
from archai.common.config import Config
from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import TArchTrainer
from archai.algos.darts.darts_model_desc_builder_ssl import DartsModelDescBuilderSimClr
from archai.algos.darts.bilevel_arch_trainer_ssl import BilevelArchTrainerSimClr
from archai.nas.searcher_ssl import SearcherSimClr, SearchResult
from archai.nas.evaluater_ssl import EvaluaterSimClr, EvalResult


class DartsExperimentRunnerSimClr(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->DartsModelDescBuilderSimClr:
        return DartsModelDescBuilderSimClr()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return BilevelArchTrainerSimClr

    @overrides
    def searcher(self)->SearcherSimClr:
        return SearcherSimClr()

    @overrides
    def evaluater(self)->EvaluaterSimClr:
        return EvaluaterSimClr()

    @overrides
    def run(self, search=True, eval=True) \
            ->Tuple[Optional[SearchResult], Optional[EvalResult]]:

    # if utils.is_main_process():
    #     conf = common_init(config_filepath='confs/algos/simclr.yaml')
    #     print('Running main process')
    # else:
    #     conf = create_conf(config_filepath='confs/algos/simclr.yaml')
    #     Config.set_inst(conf)
    #     update_envvars(conf)
    #     commonstate = get_state()
    #     init_from(commonstate)
    #     print('Running child process')

        search_result, eval_result = None, None

        if search: # run search
            conf = self._init_conf(True, clean_expdir=self.clean_expdir)
            search_result = self.run_search(conf['nas']['search'])

        if eval:
            conf = self.get_conf(False)
            if utils.is_main_process():
                common.clean_ensure_expdir(conf, clean_dir=self.clean_expdir, ensure_dir=True)
                if search or True:
                    # first copy search result to eval, otherwise we expect eval config to point to results
                    self.copy_search_to_eval()

            conf = self._init_conf(False, clean_expdir=False)
            eval_result = self.run_eval(conf['nas']['eval'])

        return search_result, eval_result

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
            common.init_from(commonstate)
            print('Running child process')
        return conf

