model_name=ParallelTime
dataset_name=weather

uv run run_hydra.py --config-name $dataset_name -m pred_len=96,192,336,720
