import logging
import random
import uuid

import hydra
import numpy as np
import torch

from experiments.exp_long_term_forecasting_parallel_time import Exp_Long_Term_Forecast

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
    cfg.comment = "_" + cfg.comment if cfg.comment != "" else ""

    if cfg.use_gpu and cfg.get("use_multi_gpu", False):
        # Process GPU device ids if needed
        cfg.device_ids = [int(x) for x in cfg.devices.replace(" ", "").split(",")]
        cfg.gpu = cfg.device_ids[0]

    cfg.model_number = str(uuid.uuid4())

    # For each experiment iteration (you can loop over cfg.itr if needed)
    for ii in range(cfg.itr if "itr" in cfg else 1):
        exp = Exp_Long_Term_Forecast(cfg)
        logger.info(f">>>>>>> start training: {cfg.model_id}, {cfg.pred_len} <<<<<<<<<<<<<<<<<<<<<<")
        exp.train()

        logger.info(f">>>>>>> testing: {cfg.model_id}, {cfg.pred_len} <<<<<<<<<<<<<<<<<<<<<<")
        exp.test()

        if cfg.get("do_predict", False):
            logger.info(f">>>>>>> predicting: {cfg.model_id}, {cfg.pred_len} <<<<<<<<<<<<<<<<<<<<<<")
            exp.predict(True)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
