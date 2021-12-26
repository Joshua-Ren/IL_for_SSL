export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python -m torch.distributed.launch --nproc_per_node=4 --master_port 10086 INK1_interact_MAE.py (--run_name testMulti --enable_amp)