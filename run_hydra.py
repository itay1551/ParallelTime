import random
import numpy as np
import torch
import hydra
from experiments.exp_long_term_forecasting_parallel_time import Exp_Long_Term_Forecast
import logging

logger = logging.getLogger(__name__)

def set_seed(fix_seed: int):
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: dict):
    logger.info(cfg)
    set_seed(cfg.fix_seed)

    cfg.use_gpu = True if torch.cuda.is_available() and cfg.use_gpu else False
    cfg.comment = '_' + cfg.comment if cfg.comment != '' else ''
    
    if cfg.use_gpu and cfg.get("use_multi_gpu", False):
        # Process GPU device ids if needed
        cfg.device_ids = [int(x) for x in cfg.devices.replace(' ', '').split(',')]
        cfg.gpu = cfg.device_ids[0]

    # For each experiment iteration (you can loop over cfg.itr if needed)
    for ii in range(cfg.itr if "itr" in cfg else 1):
        setting = (
            f"{cfg.model}_"
            f"{cfg.model_id.split('_')[0]}_"
            f"sl{cfg.seq_len}_"
            f"pl{cfg.pred_len}_"
            f"dm{cfg.dim}_"
            f"nblayers{cfg.n_block_layers}_"
            f"psize{cfg.patch_len}_"
            f"pwin{cfg.patches_window_len}_"
            f"numepo{cfg.train_epochs}_"
            f"seed{cfg.patch_len}_"
            f"nre{cfg.num_register_tokens}_"
            f"expr{cfg.expend_ratio_scaler}_"
            f'batch_s{cfg.batch_size}_'
            f'prj_ex{cfg.proj_expend_ratio}_'
            f'prj_sq{cfg.proj_squeeze_ratio}_'
            f'drop{cfg.dropout}_'
            f"comm{cfg.comment}"
        )
        exp = Exp_Long_Term_Forecast(cfg)
        logger.info(f'>>>>>>> start training: {setting} <<<<<<<<<<<<<<<<<<<<<<')
        exp.train(setting)

        logger.info(f'>>>>>>> testing: {setting} <<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)

        if cfg.get("do_predict", False):
            logger.info(f'>>>>>>> predicting: {setting} <<<<<<<<<<<<<<<<<<<<<<')
            exp.predict(setting, True)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
