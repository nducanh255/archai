export NCCL_P2P_LEVEL=NVL

python -m torch.distributed.launch \
--nproc_per_node=4 \
archai/nlp/nvidia_transformer_xl/train.py \
--cuda \
--dataset lm1b \
--adaptive \
--div_val 4 \
--n_layer 18 \
--d_model 1280 \
--n_head 8 \
--d_head 128 \
--d_inner 4096 \
--dropout 0.00 \
--dropatt 0.00 \
--optim adam \
--warmup_step 20000 \
--max_step 500000 \
--lr 0.00025 \
--eta_min 0.0 \
--tgt_len 32 \
--mem_len 32 \
--eval_tgt_len 32 \
--batch_size 224 \
--multi_gpu ddp \
--gpu0_bsz 32