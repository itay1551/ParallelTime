model_name=ParallelTime
dataset_name=weather

python run_hydra.py --config-name $dataset_name -m pred_len=96,192,336,720
