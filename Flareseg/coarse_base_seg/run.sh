# !/bin/bash -e
nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 29504 run.py --local_rank=0 &