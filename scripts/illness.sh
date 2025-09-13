model_name=ParallelTime
dataset_name=illness

uv run run_hydra.py --config-name $dataset_name -m pred_len=24,36,48,60
