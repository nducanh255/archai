# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
import os
import shutil
from distutils.dir_util import copy_tree
from typing import List, Iterable, Union, Optional, Tuple
import atexit
import subprocess
import datetime
import yaml
import sys

import torch
import torch.distributed
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import psutil

from .config import Config
from . import utils
from .ordereddict_logger import OrderedDictLogger
from .dist_utils import ApexUtils
from send2trash import send2trash

class SummaryWriterDummy:
    def __init__(self, log_dir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass
    def flush(self):
        pass

SummaryWriterAny = Union[SummaryWriterDummy, SummaryWriter]

logger = OrderedDictLogger(None, None, yaml_log=False)
_tb_writer: SummaryWriterAny = None
_atexit_reg = False # is hook for atexit registered?


def get_conf(conf:Optional[Config]=None)->Config:
    if conf is not None:
        return conf
    return Config.get_inst()

def get_conf_common(conf:Optional[Config]=None)->Config:
    return get_conf(conf)['common']

def get_conf_dataset(conf:Optional[Config]=None)->Config:
    return get_conf(conf)['dataset']

def get_conf_search(conf:Optional[Config]=None)->Config:
    return get_conf(conf)['nas']['search']

def get_conf_eval(conf:Optional[Config]=None)->Config:
    return get_conf(conf)['nas']['eval']

def get_experiment_name(conf:Optional[Config]=None)->str:
    return get_conf_common(conf)['experiment_name']

def get_expdir(conf:Optional[Config]=None)->Optional[str]:
    return get_conf_common(conf)['expdir']

def get_datadir(conf:Optional[Config]=None)->Optional[str]:
    return get_conf(conf)['dataset']['dataroot']

def get_logger() -> OrderedDictLogger:
    global logger
    if logger is None:
        raise RuntimeError('get_logger call made before logger was setup!')
    return logger

def get_tb_writer() -> SummaryWriterAny:
    global _tb_writer
    return _tb_writer

class CommonState:
    def __init__(self) -> None:
        global logger, _tb_writer
        self.logger = logger
        self.tb_writer = _tb_writer
        self.conf = get_conf()

def on_app_exit():
    print('Process exit:', os.getpid(), flush=True)
    writer = get_tb_writer()
    writer.flush()
    if isinstance(logger, OrderedDictLogger):
        logger.close()

def _pt_dirs()->Tuple[str, str]:
    # dirs for pt infrastructure are supplied in env vars

    pt_data_dir = os.environ.get('PT_DATA_DIR', '')
    # currently yaml should be copying dataset folder to local dir
    # so below is not needed. The hope is that less reads from cloud
    # storage will reduce overall latency.

    # if pt_data_dir:
    #     param_args = ['--nas.eval.loader.dataset.dataroot', pt_data_dir,
    #                   '--nas.search.loader.dataset.dataroot', pt_data_dir,
    #                   '--nas.search.seed_train.loader.dataset.dataroot', pt_data_dir,
    #                   '--nas.search.post_train.loader.dataset.dataroot', pt_data_dir,
    #                   '--autoaug.loader.dataset.dataroot', pt_data_dir] + param_args

    pt_output_dir = os.environ.get('PT_OUTPUT_DIR', '')

    return pt_data_dir, pt_output_dir

def _pt_params(param_args: list)->list:
    pt_data_dir, pt_output_dir = _pt_dirs()

    if pt_output_dir:
        # prepend so if supplied from outside it takes back seat
        param_args = ['--common.logdir', pt_output_dir] + param_args

    return param_args

def get_state()->CommonState:
    return CommonState()

def init_from(state:CommonState, recreate_logger=True)->None:
    global logger, _tb_writer

    Config.set_inst(state.conf)

    if recreate_logger:
        create_logger(state.conf)
    else:
        logger = state.logger

    logger.info({'common_init_from_state': True})

    _tb_writer = state.tb_writer


def create_conf(config_filepath: Optional[str]=None,
                param_args: list = [], use_args=True)->Config:

    # modify passed args for pt infrastructure
    # if pt infrastructure doesn't exit then param_overrides == param_args
    param_overrides = _pt_params(param_args)

    # create env vars that might be used in paths in config
    if 'default_dataroot' not in os.environ:
        os.environ['default_dataroot'] = default_dataroot()

    conf = Config(config_filepath=config_filepath,
                  param_args=param_overrides,
                  use_args=use_args)
    _update_conf(conf)

    return conf


def common_init_dist(config_filepath: Optional[str]=None,
                param_args: list = [], use_args=True,
                clean_expdir=False)->Config:

        # is_clean = None
        # conf = create_conf(config_filepath='confs/algos/simclr.yaml')
        # Config.set_inst(conf)
        # update_envvars(conf)
        # commonstate = get_state()
        # init_from(commonstate,recreate_logger=False)
        # conf['common']['is_clean'] = True
        # print('Running child process')
    conf = create_conf(config_filepath, param_args, use_args)

    # setup global instance
    Config.set_inst(conf)

    # setup env vars which might be used in paths
    update_envvars(conf)

    # create apex to know distributed processing paramters
    conf_apex = get_conf_common(conf)['apex']
    apex = ApexUtils(conf_apex, logger=logger)

    if apex.is_master():
        # create experiment dir
        create_dirs(conf, clean_expdir)
        # copy from resume dir if exists
        is_clean = copy_resume_dirs(conf)
    else:
        is_clean = False

    if apex.is_dist():
        is_clean = torch.Tensor([is_clean]).to(torch.device(f'cuda:{apex.global_rank}'))
        torch.distributed.broadcast(is_clean, src=0)
        is_clean = is_clean.item()
    # if resume set to False, regenerate conf
    if not is_clean:
        print('Resume directory not clean, disabling resume')
        update_resume_args(param_args)
        conf = create_conf(config_filepath, param_args, use_args)
        Config.set_inst(conf)
        update_envvars(conf)

    if apex.is_master():
        # create intermediate exp dir
        create_intermediate_dirs(conf)

    # create global logger
    create_logger(conf)

    _create_sysinfo(conf)


    # setup tensorboard
    global _tb_writer
    _tb_writer = create_tb_writer(conf, apex.is_master())

    # create hooks to execute code when script exits
    global _atexit_reg
    if not _atexit_reg:
        atexit.register(on_app_exit)
        _atexit_reg = True

    return conf

# TODO: rename this simply as init
# initializes random number gen, debugging etc
def common_init(config_filepath: Optional[str]=None,
                param_args: list = [], use_args=True,
                clean_expdir=False)->Config:

    # TODO: multiple child processes will create issues with shared state so we need to
    # detect multiple child processes but allow if there is only one child process.
    # if not utils.is_main_process():
    #     raise RuntimeError('common_init should not be called from child process. Please use Common.init_from()')

    conf = create_conf(config_filepath, param_args, use_args)

    # setup global instance
    Config.set_inst(conf)

    # setup env vars which might be used in paths
    update_envvars(conf)

    # create experiment dir
    create_dirs(conf, clean_expdir)

    # copy from resume dir if exists
    is_clean = copy_resume_dirs(conf)

    # if resume set to False, regenerate conf
    if not is_clean:
        update_resume_args(param_args)
        conf = create_conf(config_filepath, param_args, use_args)
        Config.set_inst(conf)
        update_envvars(conf)

    # create intermediate exp dir
    create_intermediate_dirs(conf)

    # create global logger
    create_logger(conf)

    _create_sysinfo(conf)

    # create apex to know distributed processing paramters
    conf_apex = get_conf_common(conf)['apex']
    apex = ApexUtils(conf_apex, logger=logger)

    # setup tensorboard
    global _tb_writer
    _tb_writer = create_tb_writer(conf, apex.is_master())

    # create hooks to execute code when script exits
    global _atexit_reg
    if not _atexit_reg:
        atexit.register(on_app_exit)
        _atexit_reg = True

    return conf

def update_resume_args(param_args:list)->None:
    if '--common.resume' in param_args:
        param_args[param_args.index('--common.resume')+1] = 'false'
    else:
        param_args+=['--common.resume','false']
    if '--common.resumedir' in param_args:
        param_args[param_args.index('--common.resumedir')+1] = ''
    else:
        param_args+=['--common.resumedir','']
                            
def _create_sysinfo(conf:Config)->None:
    expdir = get_expdir(conf)

    if expdir and not utils.is_debugging():
        # copy net config to experiment folder for reference
        with open(expdir_abspath('config_used.yaml'), 'w') as f:
            yaml.dump(conf.to_dict(), f)
        if not utils.is_debugging():
            sysinfo_filepath = expdir_abspath('sysinfo.txt')
            subprocess.Popen([f'./scripts/sysinfo.sh "{expdir}" > "{sysinfo_filepath}"'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True)

def expdir_abspath(path:str, create=False)->str:
    """Returns full path for given relative path within experiment directory."""
    return utils.full_path(os.path.join('$expdir',path), create=create)

def create_tb_writer(conf:Config, is_master=True)-> SummaryWriterAny:
    conf_common = get_conf_common(conf)

    tb_dir, conf_enable_tb = utils.full_path(conf_common['tb_dir']), conf_common['tb_enable']
    tb_enable = conf_enable_tb and is_master and tb_dir is not None and len(tb_dir) > 0

    logger.info({'conf_enable_tb': conf_enable_tb,
                 'tb_enable': tb_enable,
                 'tb_dir': tb_dir})

    if tb_enable and conf_common['resume']:
        raise NotImplementedError('Not implemented resuming tensorboard summary writer!') #TODO
    WriterClass = SummaryWriter if tb_enable else SummaryWriterDummy

    return WriterClass(log_dir=tb_dir)

def is_pt()->bool:
    """Is this code running in pt infrastrucuture"""
    return os.environ.get('PT_OUTPUT_DIR', '') != ''

def default_dataroot()->str:
    # the home folder on ITP VMs is super slow so use local temp directory instead
    return '/var/tmp/dataroot' if is_pt() else '~/dataroot'

def _update_conf(conf:Config)->None:
    """Updates conf with full paths resolving enviromental vars"""

    conf_common = get_conf_common(conf)
    conf_dataset = get_conf_dataset(conf)

    experiment_name = conf_common['experiment_name']

    # make sure dataroot exists
    dataroot = conf_dataset['dataroot']
    dataroot = utils.full_path(dataroot)

    # make sure logdir and expdir exists
    logdir = conf_common['logdir']
    if logdir:
        logdir = utils.full_path(logdir)
        expdir = os.path.join(logdir, experiment_name)

        # directory for non-master replica logs
        distdir = os.path.join(expdir, 'dist')
    else:
        expdir = distdir = logdir

    # update conf so everyone gets expanded full paths from here on
    # set environment variable so it can be referenced in paths used in config
    conf_common['logdir'] = logdir
    conf_dataset['dataroot'] = dataroot
    conf_common['expdir'] = expdir
    conf_common['distdir'] = distdir

def update_envvars(conf)->None:
    """Get values from config and put it into env vars"""
    conf_common = get_conf_common(conf)
    logdir = conf_common['logdir']
    expdir = conf_common['expdir']
    distdir = conf_common['distdir']

    conf_dataset = get_conf_dataset(conf)
    dataroot = conf_dataset['dataroot']

    # update conf so everyone gets expanded full paths from here on
    # set environment variable so it can be referenced in paths used in config
    os.environ['logdir'] = logdir
    os.environ['dataroot'] = dataroot
    os.environ['expdir'] = expdir
    os.environ['distdir'] = distdir

def clean_ensure_expdir(conf:Optional[Config], clean_dir:bool, ensure_dir:bool)->None:
    expdir = get_expdir(conf)
    if clean_dir and os.path.exists(expdir):
        send2trash(expdir)
    if ensure_dir:
        os.makedirs(expdir, exist_ok=True)

def create_dirs(conf:Config, clean_expdir:bool)->Optional[str]:
    conf_common = get_conf_common(conf)
    logdir = conf_common['logdir']
    expdir = conf_common['expdir']
    distdir = conf_common['distdir']

    conf_dataset = get_conf_dataset(conf)
    dataroot = utils.full_path(conf_dataset['dataroot'])

    # make sure dataroot exists
    os.makedirs(dataroot, exist_ok=True)

    # make sure logdir and expdir exists
    if logdir:
        clean_ensure_expdir(conf, clean_dir=clean_expdir, ensure_dir=True)
        os.makedirs(distdir, exist_ok=True)
    else:
        raise RuntimeError('The logdir setting must be specified for the output directory in yaml')

    # get cloud dirs if any
    pt_data_dir, pt_output_dir = _pt_dirs()

    # validate dirs
    assert not pt_output_dir or not expdir.startswith(utils.full_path('~/logdir'))

    logger.info({'expdir': expdir,
                 # create info file for current system
                 'PT_DATA_DIR': pt_data_dir, 'PT_OUTPUT_DIR': pt_output_dir})

def create_epoch_desc_dir(conf:Config)->None:
    conf_common = get_conf_common(conf)
    conf_search = get_conf_search(conf)
    epoch_desc_dir = utils.full_path(conf_search['epoch_model_desc']['savedir'])
    os.makedirs(epoch_desc_dir, exist_ok=True)
    
def create_intermediate_dirs(conf:Config)->None:
    conf_common = get_conf_common(conf)
    if conf_common['save_intermediate']:
        os.makedirs(conf_common['intermediatedir'], exist_ok=True)

        # conf_checkpoint = conf_common['checkpoint']
        # intermediatedir = os.path.join(conf_common['intermediatedir'],get_experiment_name(conf))
        # logdir = os.path.join(conf_common['logdir'],get_experiment_name(conf))
        # print(f'Copying logs and ckpts from working directory {logdir} to intermediate directory {intermediatedir}')
        # os.makedirs(intermediatedir, exist_ok=True)
        # log_suffix = ''
        # log_prefix = conf_common['log_prefix']
        # intermediate_ckpt_path = os.path.join(intermediatedir,os.path.basename(utils.full_path(conf_checkpoint['filename'])))
        # intermediate_sys_log_filepath = utils.full_path(os.path.join(intermediatedir, f'{log_prefix}{log_suffix}.log'))
        # intermediate_logs_yaml_filepath = utils.full_path(os.path.join(intermediatedir, f'{log_prefix}{log_suffix}.yaml'))
        # sys_log_filepath = utils.full_path(os.path.join(logdir, f'{log_prefix}{log_suffix}.log'))
        # logs_yaml_filepath = utils.full_path(os.path.join(logdir, f'{log_prefix}{log_suffix}.yaml'))
        # ckpt_path = os.path.join(logdir,os.path.basename(utils.full_path(conf_checkpoint['filename'])))
        # if os.path.exists(ckpt_path):
        #     shutil.copy(ckpt_path, intermediate_ckpt_path)
        # if os.path.exists(logs_yaml_filepath):
        #     shutil.copy(logs_yaml_filepath, intermediate_logs_yaml_filepath)
        # if os.path.exists(sys_log_filepath):
        #     shutil.copy(sys_log_filepath, intermediate_sys_log_filepath)

def copy_resume_dirs(conf:Config)->bool:
    conf_common = get_conf_common(conf)
    if conf_common['resume']:
        conf_checkpoint = conf_common['checkpoint']
        logdir = utils.full_path(os.environ['logdir'])
        resumedir = os.path.join(conf_common['resumedir'],get_experiment_name(conf))
        resume_ckpt_path = os.path.join(resumedir,os.path.basename(utils.full_path(conf_checkpoint['filename'])))
        log_suffix = ''
        log_prefix = conf_common['log_prefix']
        resume_sys_log_filepath = utils.full_path(os.path.join(resumedir, f'{log_prefix}{log_suffix}.log'))
        resume_logs_yaml_filepath = utils.full_path(os.path.join(resumedir, f'{log_prefix}{log_suffix}.yaml'))
        found, check_message = check_resume_files(resume_ckpt_path, resume_sys_log_filepath, resume_logs_yaml_filepath)
        if found:
            for folder in os.listdir(conf_common['resumedir']):
                srcdir = os.path.join(conf_common['resumedir'],folder)
                destdir = os.path.join(logdir,folder)
                if os.path.exists(destdir):
                    if os.path.isdir(destdir):
                        shutil.rmtree(destdir)
                    else:
                        os.remove(destdir)
                if os.path.isdir(srcdir):
                    shutil.copytree(srcdir,destdir)
                else:
                    shutil.copy(srcdir,os.path.join(logdir,folder))
            return True
        else:
            print(check_message)
            return False

    return True

    # conf_checkpoint = conf_common['checkpoint']
    # resumedir = os.path.join(conf_common['resumedir'],get_experiment_name(conf))
    # logdir = os.path.join(conf_common['logdir'],get_experiment_name(conf))
    # if conf_common['resume']:
    #     resume_ckpt_path = os.path.join(resumedir,os.path.basename(utils.full_path(conf_checkpoint['filename'])))
    #     log_suffix = ''
    #     log_prefix = conf_common['log_prefix']
    #     resume_sys_log_filepath = utils.full_path(os.path.join(resumedir, f'{log_prefix}{log_suffix}.log'))
    #     resume_logs_yaml_filepath = utils.full_path(os.path.join(resumedir, f'{log_prefix}{log_suffix}.yaml'))
    #     found, check_message = check_resume_files(resume_ckpt_path, resume_sys_log_filepath, resume_logs_yaml_filepath)
    #     if found:
    #         print(f'Copying logs and ckpts from resume directory {resumedir} to working directory {logdir}')
    #         sys_log_filepath = utils.full_path(os.path.join(logdir, f'{log_prefix}{log_suffix}.log'))
    #         logs_yaml_filepath = utils.full_path(os.path.join(logdir, f'{log_prefix}{log_suffix}.yaml'))
    #         ckpt_path = os.path.join(logdir,os.path.basename(utils.full_path(conf_checkpoint['filename'])))
    #         shutil.copy(resume_sys_log_filepath, sys_log_filepath)
    #         shutil.copy(resume_logs_yaml_filepath, logs_yaml_filepath)
    #         shutil.copy(resume_ckpt_path, ckpt_path)
    #         # if os.path.exists(os.path.join(logdir,'dist')):
    #         #     os.makedirs(os.path.join(logdir,'dist'),exist_ok=True)
    #         #     copy_tree(os.path.join(resumedir,'dist'),os.path.join(logdir,'dist'))
    #         return True
    #     else:
    #         print(check_message)
    #         # if os.path.exists(conf_common['resumedir']):
    #         #     shutil.rmtree(conf_common['resumedir'])
    #         # conf_common['resume'] = conf_checkpoint['resume'] = conf_common['apex']['resume'] = \
    #         # conf_common['apex']['resume'] = conf_common['apex']['resume'] = False
    #         # conf_common['resumedir'] = conf_checkpoint['resumedir'] = ''
    #         return False
    # return True

def check_resume_files(ckpt_path:str, log_path:str, yaml_path:str)->Tuple[bool,str]:
    if os.path.exists(ckpt_path) and os.path.exists(log_path) and os.path.exists(yaml_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            with open(yaml_path, 'r') as f:
                log_yaml = yaml.load(f, Loader=yaml.Loader)
        except Exception as e:
            return False, e
        return True, 'Resume file paths found!'
    else:
        if not os.path.exists(ckpt_path):
            return False, f'{ckpt_path} does not exist'
        if not os.path.exists(log_path):
            return False, f'{log_path} does not exist'
        if not os.path.exists(yaml_path):
            return False, f'{yaml_path} does not exist'

def create_logger(conf:Config):
    conf_common = get_conf_common(conf)

    global logger
    logger.close()  # close any previous instances

    expdir = conf_common['expdir']
    distdir = conf_common['distdir']
    log_prefix = conf_common['log_prefix']
    yaml_log = conf_common['yaml_log']
    log_level = conf_common['log_level']

    if utils.is_main_process():
        logdir, log_suffix = expdir, ''
    else:
        logdir, log_suffix = distdir, '_' + str(os.environ["RANK"])

    # ensure folders
    os.makedirs(logdir, exist_ok=True)

    # file where logger would log messages
    sys_log_filepath = utils.full_path(os.path.join(logdir, f'{log_prefix}{log_suffix}.log'))
    logs_yaml_filepath = utils.full_path(os.path.join(logdir, f'{log_prefix}{log_suffix}.yaml'))
    experiment_name = get_experiment_name(conf) + log_suffix
    #print(f'experiment_name={experiment_name}, log_stdout={sys_log_filepath}, log_file={sys_log_filepath}')

    sys_logger = utils.create_logger(filepath=sys_log_filepath,
                                     name=experiment_name, level=log_level,
                                     enable_stdout=True)
    if not sys_log_filepath:
        sys_logger.warn(
            'log_prefix not specified, logs will be stdout only')

    # reset to new file path
    logger.reset(logs_yaml_filepath, sys_logger, yaml_log=yaml_log, save_delay=conf_common['save_delay'],
                 load_existing_file=conf_common['resume'], backup_existing_file=False)
    logger.info({'command_line': ' '.join(sys.argv) if utils.is_main_process() else f'Child process: {utils.process_name()}-{os.getpid()}'})
    logger.info({'process_name': utils.process_name(), 'is_main_process': utils.is_main_process(),
                 'main_process_pid':utils.main_process_pid(), 'pid':os.getpid(), 'ppid':os.getppid(), 'is_debugging': utils.is_debugging()})
    logger.info({'experiment_name': experiment_name, 'datetime:': datetime.datetime.now()})
    logger.info({'logs_yaml_filepath': logs_yaml_filepath, 'sys_log_filepath': sys_log_filepath})
