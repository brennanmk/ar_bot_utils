# #!/bin/sh
python3 train.py --workers 14 --gpu 1 --timesteps 400 --lr 1e-3 --stop_return -3.0 --obstacles 0 --num_expirements 3 --expirement_name "basic"
python3 train.py --workers 14 --gpu 1 --timesteps 400 --lr 1e-3 --stop_return -3.0 --obstacles 5 --cirriculum True --num_expirements 3 --save_location "cirr"
python3 train.py --workers 14 --gpu 1 --timesteps 400 --lr 1e-3 --stop_return -3.0 --obstacles 5 --num_expirements 3 --expirement_name "full"
