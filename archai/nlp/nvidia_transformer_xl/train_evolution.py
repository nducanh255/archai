import argparse
import functools
import itertools
import logging
import logging.config
import math
import os
import shutil
import sys
import time
import warnings

import dllogger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
try:
    from apex import amp
except ModuleNotFoundError:
    warnings.warn('APEX AMP is unavailable')

from torch.nn.parallel import DistributedDataParallel

from archai.nlp.nvidia_transformer_xl import lamb
from archai.nlp.nvidia_transformer_xl.data_utils import get_lm_corpus
from archai.nlp.nvidia_transformer_xl.nvidia_utils import distributed as nv_distributed
from archai.nlp.nvidia_transformer_xl.nvidia_utils.data_parallel import BalancedDataParallel
from archai.nlp.nvidia_transformer_xl.nvidia_utils import exp_utils
from archai.nlp.nvidia_transformer_xl.nvidia_utils.exp_utils import AverageMeter
from archai.nlp.nvidia_transformer_xl.nvidia_utils.exp_utils import l2_promote
from archai.common import utils, common

def get_args(config, config_file, max_step, experiment_name, scheduler):
    config_file_path = utils.full_path(os.path.join('.', 'archai', 'nlp', 'nvidia_transformer_xl', config_file))
    with open(config_file_path) as f:
        config_from_yaml = yaml.load(f, Loader=yaml.FullLoader)[config]['train']

    parent_parser = argparse.ArgumentParser(
        description='PyTorch Transformer-XL Language Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        )
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
    
    general = parser.add_argument_group('general setup')
    general.add_argument('--work_dir', default='~/logdir', type=str,
                         help='Directory for the results')
    general.add_argument('--experiment_name', default='nv_xformer_xl', type=str,
                         help='Directory for the results')
    general.add_argument('--append_dataset', action='store_true',
                         help='Automatically append dataset name to work_dir')
    general.add_argument('--append_time', action='store_true',
                         help='Automatically append current time to work_dir')
    general.add_argument('--cuda', action='store_true',
                         help='Run training on a GPU using CUDA')
    general.add_argument('--fp16', action='store_true',
                         help='Run training in fp16/mixed precision')
    general.add_argument('--restart', type=str, default='',
                         help='Restart training from the saved checkpoint')
    general.add_argument('--debug', action='store_true', default=None,
                         help='Run in debug mode (do not create exp dir)')
    general.add_argument('--log_all_ranks', action='store_true',
                         help='Enable logging from all distributed ranks')
    general.add_argument('--dllog_file', type=str, default='train_log.json',
                         help='Name of the DLLogger output file')
    general.add_argument('--txtlog_file', type=str, default='train_log.log',
                         help='Name of the txt log file')
    general.add_argument('--save_all', action='store_true',
                         help='Save all checkpoints')
    general.add_argument('--no_env', action='store_false',
                         help='Do not print info on execution env')
    general.add_argument('--no_eval', action='store_true',
                         help='Disable model evaluation')
    general.add_argument('--log_interval', type=int, default=10,
                         help='Report interval')
    general.add_argument('--target_throughput', type=float, default=None,
                         help='Target training throughput (for benchmarking)')
    general.add_argument('--target_perplexity', type=float, default=None,
                         help='Target validation perplexity (for benchmarking)')
    general.add_argument('--apex_amp_opt_level', type=str, default='O2',
                         choices=['O0', 'O1', 'O2', 'O3'],
                         help='Optimization level for apex amp')
    general.add_argument('--amp', choices=['apex', 'pytorch'], default='apex',
                         help='Implementation of automatic mixed precision')
    general.add_argument('--affinity', type=str,
                         default='socket_unique_interleaved',
                         choices=['socket', 'single', 'single_unique',
                                  'socket_unique_interleaved',
                                  'socket_unique_continuous',
                                  'disabled'],
                         help='type of CPU affinity')
    general.add_argument('--get_params', action='store_true',
                         help='Get parameter breakdown for different layers')

    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--data', type=str, default=None,
                         help='Location of the data corpus')
    dataset.add_argument('--dataset', type=str, default='wt103',
                         choices=['wt103', 'wt2', 'lm1b', 'enwik8', 'text8'],
                         help='Dataset name')
    dataset.add_argument('--vocab', type=str, default='word', choices=['word', 'bpe'],
                         help='Type of vocabulary')
    dataset.add_argument('--vocab_size', type=int, default=None,
                         help='Size of vocabulary')

    model = parser.add_argument_group('model setup - defaults are for base model')
    model.add_argument('--init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--emb_init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--init_range', type=float, default=0.1,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--emb_init_range', type=float, default=0.01,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--init_std', type=float, default=0.02,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--proj_init_std', type=float, default=0.01,
                       help='Parameters initialized by N(0, init_std)')

    opt = parser.add_argument_group('optimizer setup')
    opt.add_argument('--optim', default='jitlamb', type=str,
                     choices=['adam', 'sgd', 'adagrad', 'lamb', 'jitlamb'],
                     help='Optimizer to use')
    opt.add_argument('--lr', type=float, default=0.01,
                     help='Initial learning rate')
    opt.add_argument('--mom', type=float, default=0.0,
                     help='Momentum for sgd')
    opt.add_argument('--scheduler', default='cosine', type=str,
                     choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                     help='LR scheduler to use')
    opt.add_argument('--max_step_scheduler', type=int, default=None,
                     help='Max number of training steps for LR scheduler')
    opt.add_argument('--warmup_step', type=int, default=1000,
                     help='Number of iterations for LR warmup')
    opt.add_argument('--decay_rate', type=float, default=0.5,
                     help='Decay factor when ReduceLROnPlateau is used')
    opt.add_argument('--lr_min', type=float, default=0.0,
                     help='Minimum learning rate during annealing')
    opt.add_argument('--clip', type=float, default=0.25,
                     help='Gradient clipping')
    opt.add_argument('--weight_decay', type=float, default=0.0,
                     help='Weight decay for adam|lamb')
    opt.add_argument('--clip_nonemb', action='store_true',
                     help='Only clip the gradient of non-embedding params')
    opt.add_argument('--patience', type=int, default=0,
                     help='Patience')
    opt.add_argument('--eta_min', type=float, default=0.001,
                     help='Min learning rate for cosine scheduler')

    training = parser.add_argument_group('training setup')
    training.add_argument('--max_step', type=int, default=40000,
                          help='Max number of training steps')
    training.add_argument('--batch_size', type=int, default=256,
                          help='Global batch size')
    training.add_argument('--local_batch_size', type=int, default=None,
                          help='Local (per-device) batch size, this setting \
                          overrides global --batch_size and sets batch_size \
                          to local_batch_size * world_size')
    training.add_argument('--batch_chunk', type=int, default=1,
                          help='Split batch into chunks and train with '
                          'gradient accumulation. 16GB V100 FP16 requires 1 chunk, FP32 requires 2 chunks')
    training.add_argument('--roll', action='store_true',
                          help='Enable random shifts within each data stream')
    training.add_argument('--tgt_len', type=int, default=192,
                          help='Number of tokens to predict')
    training.add_argument('--ext_len', type=int, default=0,
                          help='Length of the extended context')
    training.add_argument('--mem_len', type=int, default=192,
                          help='Length of the retained previous heads, number of tokens cached from previous iterations during training')
    training.add_argument('--seed', type=int, default=1111,
                          help='Random seed')
    training.add_argument('--multi_gpu', default=None, type=str,
                          choices=['ddp', 'dp'],
                          help='Use multiple GPU')
    training.add_argument('--gpu0_bsz', type=int, default=-1,
                          help='Batch size on gpu 0 (for "dp" backend)')
    training.add_argument('--same_length', action='store_true',
                          help='Use the same attn length for all tokens')
    training.add_argument('--varlen', action='store_true',
                          help='Use variable length')
    training.add_argument('--swap_mem', action='store_true',
                          help='Swap memory tensors to cpu')

    val = parser.add_argument_group('validation setup')
    val.add_argument('--eval_tgt_len', type=int, default=192,
                     help='Number of tokens to predict for evaluation')
    val.add_argument('--eval_batch_size', type=int, default=16,
                     help='Eval batch size')
    val.add_argument('--eval_max_steps', type=int, default=-1,
                     help='Max eval steps')
    val.add_argument('--eval_interval', type=int, default=5000,
                     help='Evaluation interval')

    fear = parser.add_argument_group('FEAR setup')
    fear.add_argument('--ppl_threshold', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='Delimited list of perplexity thresholds for extracting the FEAR checkpoint')
    fear.add_argument('--use_train', action='store_true',
                          help='Use training ppl for extracting FEAR checkpoint, if not selected, uses validation ppl')
    fear.add_argument('--fear_terminate', action='store_true',
                          help='Terminate training after extracting the FEAR checkpoint')

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank',  type=int,
                      default=os.getenv('LOCAL_RANK', 0),
                      help='Used for multi-process training.')

    parser.set_defaults(**config_from_yaml)
    args, _ = parser.parse_known_args()
    if args.ppl_threshold:
        args.ppl_threshold = np.sort(args.ppl_threshold)[::-1].tolist()

    args.max_step = max_step
    args.experiment_name = experiment_name
    args.scheduler = scheduler

    if args.ext_len < 0:
        raise RuntimeError('Extended context length must be non-negative')

    # default mem_len==192, eval_tgt_len==192, tgt_len==192
    if args.mem_len == 0:
        if args.eval_tgt_len > args.ext_len + args.tgt_len:
            raise RuntimeError('eval_tgt_len should be <= tgt_len + ext_len; '
                               f'eval_tgt_len: {args.eval_tgt_len}, '
                               f'tgt_len: {args.tgt_len}, '
                               f'ext_len: {args.ext_len}')
    else:
        if args.eval_tgt_len > args.mem_len + args.tgt_len:
            raise RuntimeError('eval_tgt_len should be <= tgt_len + mem_len; '
                               f'eval_tgt_len: {args.eval_tgt_len}, '
                               f'tgt_len: {args.tgt_len}, '
                               f'mem_len: {args.mem_len}')

    if args.batch_size % args.batch_chunk != 0:
        raise RuntimeError('Batch size needs to be divisible by batch chunk')

    if args.fp16 and args.amp == 'apex' and 'apex' not in sys.modules:
        raise RuntimeError(
            'APEX AMP unavailable, install APEX or switch to pytorch AMP'
        )

    if args.debug is None:
        args.debug = utils.is_debugging()

    return args


def save_checkpoint(args, model, model_config, optimizer, scheduler, scaler,
                    vocab, epoch, batch, last_iter, train_step, best_val_loss,
                    is_best, work_dir, is_fear=False, ppl_threshold=None):
    if args.fp16:
        if args.amp == 'pytorch':
            amp_state = scaler.state_dict()
        elif args.amp == 'apex':
            amp_state = amp.state_dict()
        else:
            raise RuntimeError(f'args.amp should be pytorch or apex but was "{args.amp}"')
    else:
        amp_state = None

    state = {
        'args': args,
        'model_config': model_config,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        # 'scheduler_state': scheduler.state_dict(),
        'vocab': vocab,
        'amp_state': amp_state,
        'epoch': epoch,
        'batch': batch,
        'last_iter': last_iter,
        'train_step': train_step,
        'best_val_loss': best_val_loss,
        }
    if scheduler:
        state['scheduler_state'] = scheduler.state_dict()

    # Saving intermediate checkpoint for FEAR step 1
    if is_fear:
        with nv_distributed.sync_workers() as rank:
            fear_chkpt_fname = 'checkpoint_fear_threshold_'+str(ppl_threshold)+'.pt'
            fear_chkpt_path = os.path.join(work_dir, fear_chkpt_fname)
            if rank == 0:
                # always save last checkpoint
                logging.info(f'Saving checkpoint to {fear_chkpt_path}')
                torch.save(state, fear_chkpt_path)
    
        return

    last_chkpt_fname = 'checkpoint_last.pt'
    with nv_distributed.sync_workers() as rank:
        last_chkpt_path = os.path.join(work_dir, last_chkpt_fname)
        if rank == 0:
            # always save last checkpoint
            logging.info(f'Saving checkpoint to {last_chkpt_path}')
            torch.save(state, last_chkpt_path)

            # save best checkpoint if better than previous best
            if is_best:
                best_chkpt_fname = 'checkpoint_best.pt'
                best_chkpt_path = os.path.join(work_dir, best_chkpt_fname)
                logging.info(f'Saving checkpoint to {best_chkpt_path}')
                shutil.copy(last_chkpt_path, best_chkpt_path)

            # save every checkpoint if save_all is true
            if args.save_all:
                step_chkpt_fname = f'checkpoint_{train_step}.pt'
                step_chkpt_path = os.path.join(work_dir, step_chkpt_fname)
                logging.info(f'Saving checkpoint to {step_chkpt_path}')
                shutil.copy(last_chkpt_path, step_chkpt_path)


def load_checkpoint(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint_last.pt')

    dst = f'cuda:{torch.cuda.current_device()}'
    logging.info(f'Loading checkpoint from {path}')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint


def init_weight(weight, args):
    """Intialize given parameters using specified strategy"""
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal': # default
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m, args):
    """Initialize weights of module using specified strategy"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight, args)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight, args)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight, args)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
        if hasattr(m, 'out_layers_weights'):
            for i in range(len(m.out_layers_weights)):
                if m.out_layers_weights[i] is not None:
                    init_weight(m.out_layers_weights[i], args)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            if isinstance(m.r_emb, list):
                for r_emb in m.r_emb:
                    init_weight(r_emb, args)
            else:
                init_weight(m.r_emb, args)
        if hasattr(m, 'r_w_bias'):
            if isinstance(m.r_w_bias, list):
                for r_w_bias in m.r_w_bias:
                    init_weight(r_w_bias, args)
            else:
                init_weight(m.r_w_bias, args)
        if hasattr(m, 'r_r_bias'):
            if isinstance(m.r_r_bias, list):
                for r_r_bias in m.r_r_bias:
                    init_weight(r_r_bias, args)
            else:
                init_weight(m.r_r_bias, args)
        if hasattr(m, 'r_bias'):
            if isinstance(m.r_bias, list):
                for r_bias in m.r_bias:
                    init_weight(r_bias, args)
            else:
                init_weight(m.r_bias, args)


def update_dropout(m, args):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout


def update_dropatt(m, args):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt


def evaluate(eval_iter, model, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    # default mem_len==192, eval_tgt_len==192, tgt_len==192
    if args.mem_len == 0:
        model.reset_length(tgt_len=args.eval_tgt_len,
                           ext_len=args.ext_len + args.tgt_len - args.eval_tgt_len,
                           mem_len=args.mem_len
                           )
    else:
        model.reset_length(tgt_len=args.eval_tgt_len,
                           ext_len=args.ext_len,
                           mem_len=args.mem_len + args.tgt_len - args.eval_tgt_len,
                           )

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = None
        for i, (data, target, seq_len, warm) in enumerate(eval_iter):
            if args.eval_max_steps > 0 and i >= args.eval_max_steps:
                break
            loss, mems = model(data, target, mems)
            loss = loss.float().mean()
            if warm:
                # assert (mems is None) or mems.size(1) == model.mem_len
                total_loss += seq_len * loss.item()
                total_len += seq_len

    # Switch back to the training mode
    model.reset_length(tgt_len=args.tgt_len,
                       ext_len=args.ext_len,
                       mem_len=args.mem_len
                       )
    model.train()

    return total_loss / total_len


def train_iteration(model, i, mems, data_chunks, target_chunks, scaler,
                    optimizer, device, delay_unscale, args):
    # trains a given chunk
    cpu = torch.device('cpu')
    data_i = data_chunks[i].contiguous()
    target_i = target_chunks[i].contiguous()

    if args.swap_mem and mems[i] is not None:
        mems[i] = mems[i].to(device, non_blocking=True)

    enable_autocast = args.fp16 and args.amp == 'pytorch'
    with torch.cuda.amp.autocast(enable_autocast):
        loss, mems[i] = model(data_i, target_i, mems[i])
        loss = loss.float().mean().type_as(loss) / args.batch_chunk

    if args.swap_mem and mems[i] is not None:
        mems[i] = mems[i].to(cpu, non_blocking=True)

    if args.fp16:
        if args.amp == 'pytorch':
            scaler.scale(loss).backward()
        elif args.amp == 'apex':
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward()
        else:
            raise RuntimeError(f'args.amp should be pytorch or apex but was "{args.amp}"')
    else:
        loss.backward()

    train_loss = loss.float().item()
    return train_loss


def train(tr_iter, va_iter, model, para_model, optimizer,
          optimizer_sparse, scheduler, scheduler_sparse, scaler, vocab, epoch,
          last_batch, last_iter, train_step, best_val_loss, meters,
          device, args, fear_activated=0):
    # Turn on training mode which enables dropout.
    model.train()

    train_loss = 0
    target_tokens = 0
    log_step = 0
    log_start_time = time.time()

    mems = [None for _ in range(args.batch_chunk)]
    if args.varlen:
        train_iter = tr_iter.get_varlen_iter(start=last_iter)
    else:
        train_iter = tr_iter.get_fixlen_iter(start=last_iter)

    for batch, (data, target, seq_len, _) in enumerate(train_iter, start=last_batch+1):
        log_step += 1
        target_tokens += target.numel()

        for param in model.parameters():
            param.grad = None

        # Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.
        data_chunks = torch.chunk(data, args.batch_chunk, 1)
        target_chunks = torch.chunk(target, args.batch_chunk, 1)

        for i in range(args.batch_chunk):
            # if this is last chunk and distribued mode then use delay_unscale=True for amp
            if i < args.batch_chunk - 1 and isinstance(para_model, DistributedDataParallel):
                with para_model.no_sync():
                    train_loss_chunk = train_iteration(
                        para_model, i, mems, data_chunks, target_chunks, scaler,
                        optimizer, device, True, args
                    )
            else:
                train_loss_chunk = train_iteration(
                    para_model, i, mems, data_chunks, target_chunks, scaler,
                    optimizer, device, False, args
                )

            train_loss += train_loss_chunk

        if args.fp16:
            if args.amp == 'pytorch':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            elif args.amp == 'apex':
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if args.fp16 and args.amp == 'pytorch':
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            if optimizer_sparse:
                optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step <= args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if optimizer_sparse:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step - args.warmup_step)
                    if scheduler_sparse:
                        scheduler_sparse.step(train_step - args.warmup_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)
            if scheduler_sparse:
                scheduler_sparse.step(train_step)
            
        if train_step % args.log_interval == 0:
            cur_loss = train_loss / log_step
            cur_loss = nv_distributed.all_reduce_item(cur_loss, op='mean')
            train_loss = 0

            elapsed = time.time() - log_start_time
            avg_elapsed = elapsed / log_step
            avg_elapsed = nv_distributed.all_reduce_item(avg_elapsed, op='max')
            log_start_time = time.time()
            log_step = 0

            lr = optimizer.param_groups[0]['lr']
            throughput = target_tokens / elapsed
            throughput = nv_distributed.all_reduce_item(throughput, op='sum')
            meters['train_throughput'].update(throughput)
            target_tokens = 0

            print('| epoch {:3d} step {:>8d} | batches {:>6d} / {:d} | lr {:.3e} ' \
                '| ms/batch {:5.1f} | tok/s {:7.0f} | loss {:5.2f}'.format(
                    epoch,
                    train_step,
                    batch,
                    tr_iter.n_batch,
                    lr,
                    avg_elapsed * 1000,
                    throughput,
                    cur_loss,
                    ))

            if args.ppl_threshold is not None and len(args.ppl_threshold): # check to see if fear is enabled
                if args.use_train:
                    curr_ppl = math.exp(cur_loss)
                
                else: # use validation perplexity
                    val_loss = evaluate(va_iter, model, args)
                    val_loss = nv_distributed.all_reduce_item(val_loss, op='mean')
                    curr_ppl = math.exp(val_loss)

        do_periodic_eval = train_step % args.eval_interval == 0
        is_final_step = train_step == args.max_step
        interrupted = False #timeout_handler.interrupted

        if (do_periodic_eval or is_final_step or interrupted) and not args.no_eval:
            eval_start_time = time.time()
            val_loss = evaluate(va_iter, model, args)
            val_loss = nv_distributed.all_reduce_item(val_loss, op='mean')

            print('| Eval {:3d} at step {:>8d} | time: {:5.2f}s | valid loss {:5.2f}'.format(
                          train_step // args.eval_interval,
                          train_step, (time.time() - eval_start_time), val_loss,))

            last_iter = tr_iter.last_iter

            # Check if the validation loss is the best we've seen so far.
            is_best = False
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)
                if scheduler_sparse:
                    scheduler_sparse.step(val_loss)

            # subtract eval time from timers for training
            log_start_time += time.time() - eval_start_time
        
        if interrupted:
            print(f'Received SIGTERM, exiting')
            sys.exit(0)

        if is_final_step:
            break
    
    return train_step, best_val_loss, fear_activated


def train_during_evolution(model, config, config_file, max_step, experiment_name, scheduler):
    # Disable profiling executor
    try:
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    except AttributeError:
        pass

    # Before we do anything with models, we want to ensure that we get fp16
    # execution of torch.einsum in APEX AMP.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations.
    # Note that running `--apex_amp_opt_level O2` will remove the need for this
    # code, but it is still valid.
    if 'apex' in sys.modules:
        amp.register_half_function(torch, 'einsum')
        
    args = get_args(config, config_file, max_step, experiment_name, scheduler)

    # Initialize device and distributed backend
    torch.cuda.set_device(args.local_rank)
    l2_promote()
    device = torch.device('cuda' if args.cuda else 'cpu')
    nv_distributed.init_distributed(args.cuda)

    pt_data_dir, pt_output_dir = common.pt_dirs()
    args.data = args.data or pt_data_dir or common.default_dataroot()
    args.data = utils.full_path(os.path.join(args.data,'textpred', exp_utils.dataset_dir_name(args.dataset)))
    
    if args.local_batch_size is not None: # default is None
        world_size = nv_distributed.get_world_size()
        args.batch_size = world_size * args.local_batch_size
        
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###########################################################################
    # Load data
    ###########################################################################
    corpus = get_lm_corpus(args.data, args.dataset, args.vocab, max_size=args.vocab_size)
    ntokens = len(corpus.vocab)
    vocab = corpus.vocab
    args.n_token = ntokens

    if args.mem_len == 0: # default is 192
        eval_mem_len = 0
    else:
        eval_mem_len = args.mem_len + args.tgt_len - args.eval_tgt_len

    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
                                  device=device, ext_len=args.ext_len)
    va_iter = corpus.get_iterator('valid', args.eval_batch_size,
                                  args.eval_tgt_len, device=device,
                                  mem_len=eval_mem_len, ext_len=args.ext_len)
    te_iter = corpus.get_iterator('test', args.eval_batch_size,
                                  args.eval_tgt_len, device=device,
                                  mem_len=eval_mem_len, ext_len=args.ext_len)

    model.apply(functools.partial(weights_init, args=args))
    # ensure embedding init is not overridden by out_layer in case of weight sharing
    model.word_emb.apply(functools.partial(weights_init, args=args))

    # optimizer
    if args.optim.lower() == 'sgd':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
            optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.mom)
            optimizer_sparse = None    
    elif args.optim.lower() == 'adam':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
            optimizer = optim.Adam(dense_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
            optimizer_sparse = None   
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
        optimizer_sparse = None    
    elif args.optim.lower() == 'lamb':
        optimizer = lamb.Lamb(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
        optimizer_sparse = None    
    elif args.optim.lower() == 'jitlamb':
        optimizer = lamb.JITLamb(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
        optimizer_sparse = None

    model = model.to(device)
    if hasattr(model, 'r_emb'):
        if isinstance(model.r_emb, list):
            for idx, _ in enumerate(model.r_emb):
                model.r_emb[idx] = model.r_emb[idx].to(device)
    if hasattr(model, 'r_w_bias'):
        if isinstance(model.r_w_bias, list):
            for idx, _ in enumerate(model.r_w_bias):
                model.r_w_bias[idx] = model.r_w_bias[idx].to(device)
    if hasattr(model, 'r_r_bias'):
        if isinstance(model.r_r_bias, list):
            for idx, _ in enumerate(model.r_r_bias):
                model.r_r_bias[idx] = model.r_r_bias[idx].to(device)
    if hasattr(model, 'r_bias'):
        if isinstance(model.r_bias, list):
            for idx, _ in enumerate(model.r_bias):
                model.r_bias[idx] = model.r_bias[idx].to(device)

    scaler = None
    if args.fp16:
        if args.amp == 'pytorch':
            scaler = torch.cuda.amp.GradScaler()
        elif args.amp == 'apex':
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_amp_opt_level,)

    # by default this argument is not used, instead we spawn multiple instances
    # using command line:
    # python -m torch.distributed.launch --nproc_per_node="$2" train.py \
    #         --config_file wt103_base.yaml \
    #         "${@:3}"
    if args.multi_gpu == 'ddp' and torch.distributed.is_initialized():
        para_model = DistributedDataParallel(model,
                                             device_ids=[args.local_rank],
                                             output_device=args.local_rank,
                                             broadcast_buffers=False,
                                             find_unused_parameters=True,)
    elif args.multi_gpu == 'dp':
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                              model, dim=1).to(device)
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model

    # scheduler
    if args.scheduler == 'cosine':
        if args.max_step_scheduler:
            max_step = args.max_step_scheduler
        else:
            max_step = args.max_step

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step - args.warmup_step, eta_min=args.eta_min)
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_sparse, max_step - args.warmup_step,
                eta_min=args.eta_min)
        else:
            scheduler_sparse = None
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step \
                    else step / (args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.LambdaLR(
                optimizer_sparse,
                lr_lambda=lr_lambda
                )
        else:
            scheduler_sparse = None
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.decay_rate, patience=args.patience,
            min_lr=args.lr_min,
            )
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_sparse, factor=args.decay_rate, patience=args.patience,
                min_lr=args.lr_min,
                )
        else:
            scheduler_sparse = None
    elif args.scheduler == 'constant':
        scheduler = None
        scheduler_sparse = None
        pass
    
    train_step = 0
    start_epoch = 1
    last_batch = 0
    last_iter = 0
    best_val_loss = None

    meters = {}
    warmup = args.mem_len // args.tgt_len + 2
    meters['train_throughput'] = AverageMeter(warmup=warmup)
    ###########################################################################
    # Train
    ###########################################################################
    # Loop over epochs.
    # At any point you can hit Ctrl + C to break out of training early.
    start_time = time.time()
    fear_activated = 0
    print('here', device)
    try:
        for epoch in itertools.count(start=start_epoch):
            if args.roll: # enable random shifts in datasets
                tr_iter.roll(seed=args.seed + epoch)
            train_step, best_val_loss, fear_activated = train(
                                            tr_iter, va_iter, model, para_model,
                                            optimizer, optimizer_sparse, scheduler,
                                            scheduler_sparse, scaler, vocab, epoch, last_batch,
                                            last_iter, train_step, best_val_loss, meters,
                                            device, args, fear_activated
                                            )

            last_batch = 0
            last_iter = 0

            if train_step == args.max_step:
                logging.info('-' * 100)
                logging.info('End of training')
                break

            if fear_activated and args.fear_terminate:
                logging.info('-' * 100)
                logging.info('End of training')
                break

    except KeyboardInterrupt:
        print('Exiting from training early')
    elapsed = time.time() - start_time

    ###########################################################################
    # Test
    ###########################################################################
    summary = {}
    # Run on test data.
    test_start_time = time.time()
    test_loss = evaluate(te_iter, model, args)
    test_loss = nv_distributed.all_reduce_item(test_loss, 'mean')
    test_elapsed = time.time() - test_start_time

    if args.dataset in ['enwik8', 'text8']:
        print('| End of training | test time: {:5.2f}s | test loss {:5.2f} | test bpc {:9.5f}'.format(
            test_elapsed, test_loss, test_loss / math.log(2)))
    else:
        print('| End of training | test time: {:5.2f}s | test loss {:5.2f} | test ppl {:9.3f}'.format(
            test_elapsed, test_loss, math.exp(test_loss)))

        if args.dataset in ['enwik8', 'text8']:
            summary['test_bits_per_character'] = test_loss / math.log(2)
        else:
            summary['test_perplexity'] = math.exp(test_loss)

    print(f'Training time: {(elapsed / 60):.2f} minutes')
    print(f'Training throughput: {meters["train_throughput"].avg:.2f} tok/s')
    
    if best_val_loss:
        val_perplexity = math.exp(best_val_loss)
    else:
        val_perplexity = None

    summary.update({
        'train_throughput': meters['train_throughput'].avg,
        'train_elapsed': elapsed / 60,
        'valid_loss': best_val_loss,
        'valid_perplexity': val_perplexity,
        })

    return summary
