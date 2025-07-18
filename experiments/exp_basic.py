import logging
import os

import torch

from model import ParallelTime

logger = logging.getLogger(__name__)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "ParallelTime": ParallelTime,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"{total_params:,}")
        logger.info(self.model)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            logger.info("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            logger.info("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
