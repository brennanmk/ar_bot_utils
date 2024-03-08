# #!/bin/sh
python3 train.py --workers 4 --gpu 1 --timesteps 400 --lr 1e-3 --stop_return -3.0 --obstacles 0 --num_expirements 3 --expirement_name "basic-4"
python3 train.py --workers 6 --gpu 1 --timesteps 400 --lr 1e-3 --stop_return -3.0 --obstacles 0 --num_expirements 3 --expirement_name "basic-6"
python3 train.py --workers 8 --gpu 1 --timesteps 400 --lr 1e-3 --stop_return -3.0 --obstacles 0 --num_expirements 3 --expirement_name "basic-8"
python3 train.py --workers 14 --gpu 1 --timesteps 400 --lr 1e-3 --stop_return -3.0 --obstacles 0 --num_expirements 3 --expirement_name "basic-large-batch" --batch_size=16000
