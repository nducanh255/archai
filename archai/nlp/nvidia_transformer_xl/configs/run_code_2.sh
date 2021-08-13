export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --master_port=1234 --nproc_per_node="4" archai/nlp/nvidia_transformer_xl/train.py --config dgx1_4gpu_fp32 --config_file wt103_base_FEAR.yaml --n_layer 4 --n_head 2,4,2,2 --d_model 128 --d_head 64,32,64,64 --d_inner 1229,1618,1901,952 --d_embed 128 --div_val 4 --max_step 100 --experiment_name job_2 &
wait
