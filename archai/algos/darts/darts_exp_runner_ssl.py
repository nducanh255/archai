# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from overrides import overrides

from archai.nas.exp_runner import ExperimentRunner
from archai.nas.arch_trainer import TArchTrainer
from archai.algos.darts.darts_model_desc_builder_ssl import DartsModelDescBuilderSimClr
from archai.algos.darts.bilevel_arch_trainer_ssl import BilevelArchTrainerSimClr
from archai.nas.searcher_ssl import SearcherSimClr


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

