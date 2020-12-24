# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from typing import Dict, List, Type, Iterator, Tuple
import glob
import os
import pathlib
from collections import OrderedDict
import yaml
from inspect import getsourcefile
import re
from tqdm import tqdm

from scipy.stats import kendalltau, spearmanr

from runstats import Statistics

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from archai.common import utils
from archai.common.ordereddict_logger import OrderedDictLogger
from analysis_utils import epoch_nodes, fix_yaml, remove_seed_part, group_multi_runs, collect_epoch_nodes, EpochStats, FoldStats, stat2str, get_epoch_stats, get_summary_text, get_details_text, plot_epochs, write_report

import re

def main():
    parser = argparse.ArgumentParser(description='Report creator')
    parser.add_argument('--results-dir', '-d', type=str,
                        default=r'~/logdir/proxynas_test_0001',
                        help='folder with experiment results from pt')
    parser.add_argument('--out-dir', '-o', type=str, default=r'~/logdir/reports',
                        help='folder to output reports')
    args, extra_args = parser.parse_known_args()

    # root dir where all results are stored
    results_dir = pathlib.Path(utils.full_path(args.results_dir))
    print(f'results_dir: {results_dir}')

    # extract experiment name which is top level directory
    exp_name = results_dir.parts[-1]

    # create results dir for experiment
    out_dir = utils.full_path(os.path.join(args.out_dir, exp_name))
    print(f'out_dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    # get list of all structured logs for each job
    logs = {}
    job_count = 0
    for job_dir in tqdm(results_dir.iterdir()):
        if job_dir.is_dir():
            job_count += 1
            for subdir in job_dir.iterdir():
                if not subdir.is_dir():
                    continue
                # currently we expect that each job was ExperimentRunner job which should have
                # _search or _eval folders
                if subdir.stem.endswith('_search'):
                    sub_job = 'search'
                elif subdir.stem.endswith('_eval'):
                    sub_job = 'eval'
                else:
                    raise RuntimeError(f'Sub directory "{subdir}" in job "{job_dir}" must '
                                    'end with either _search or _eval which '
                                    'should be the case if ExperimentRunner was used.')

                logs_filepath = os.path.join(str(subdir), 'log.yaml')
                if os.path.isfile(logs_filepath):
                    fix_yaml(logs_filepath)
                    with open(logs_filepath, 'r') as f:
                        key = job_dir.name + ':' + sub_job
                        logs[key] = yaml.load(f, Loader=yaml.Loader)
                        all_good = True
                        if 'eval_arch' not in logs[key].keys():
                            print(f'eval_arch not in {key}')
                            all_good = False
                        if 'freeze_evaluate' not in logs[key].keys():
                            print(f'freeze_evaluate not in {key}')
                            all_good = False
                        if all_good:
                            print(f'{key} is all good')



    # logs['proxynas_blahblah:eval']['naswotrain_evaluate']['eval_arch']['eval_train']['naswithouttraining']
    # logs['proxynas_blahblah:eval']['regular_evaluate']['regtrainingtop1']
    # logs['proxynas_blahblah:eval']['freeze_evaluate']['eval_arch']['eval_train']['epochs']['9']['val']['top1']
    # last_epoch_key = list(logs['proxynas_blahblah:eval']['freeze_evaluate']['eval_arch']['eval_train']['epochs'].keys())[-1]
    # last_val_top1 = logs['proxynas_blahblah:eval']['freeze_evaluate']['eval_arch']['eval_train']['epochs'][last_epoch_key]['val']['top1']

    all_reg_evals = []
    all_naswotrain_evals = []
    all_freeze_evals = []
    for key in logs.keys():
        if 'eval' in key:
            try:
                # naswotrain
                naswotrain_top1 = logs[key]['naswotrain_evaluate']['eval_arch']['eval_train']['naswithouttraining']

                # regular evaluation
                reg_eval_top1 = logs[key]['regular_evaluate']['regtrainingtop1']

                # freeze evaluationj
                last_epoch_key = list(logs[key]['freeze_evaluate']['eval_arch']['eval_train']['epochs'].keys())[-1]
                freeze_eval_top1 = logs[key]['freeze_evaluate']['eval_arch']['eval_train']['epochs'][last_epoch_key]['val']['top1']
                
                all_naswotrain_evals.append(naswotrain_top1)
                all_reg_evals.append(reg_eval_top1)
                all_freeze_evals.append(freeze_eval_top1)
            except KeyError as err:
                print(f'KeyError {err} in {key}')
        
    tau, p_value = kendalltau(all_reg_evals, all_freeze_evals)
    spe, sp_value = spearmanr(all_reg_evals, all_freeze_evals)
    print(f'Kendall Tau score: {tau}, p_value {p_value}')
    print(f'Spearman corr: {spe}, p_value {sp_value}')
    results_savename = os.path.join(results_dir, 'results.txt')
    with open(results_savename, 'w') as f:
        f.write(f'Kendall Tau score: {tau}, p_value {p_value}')
        f.write(f'Spearman corr: {spe}, p_value {sp_value}')

    
    plt.scatter(all_reg_evals, all_freeze_evals)
    plt.xlabel('Val top1 at 600 epochs')
    plt.ylabel('Freeze training')
    plt.title('Freeze training at 0.75 val top1 followed by 200 epochs')
    savename = os.path.join(results_dir, 'proxynas_0.75_freeze_training_200_epochs.png')
    plt.savefig(savename, dpi=plt.gcf().dpi, bbox_inches='tight')



if __name__ == '__main__':
    main()