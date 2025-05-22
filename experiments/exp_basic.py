import os
import torch
from model import ParallelTime
import sys
import logging

logger = logging.getLogger(__name__)

# Define a custom exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("Unhandled exception occurred", exc_info=(exc_type, exc_value, exc_traceback))
    # Ensure logs are written immediately by flushing handlers
    for handler in logger.handlers:
        handler.flush()

# Set the custom handler as the global exception hook
sys.excepthook = handle_exception

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'ParallelTime': ParallelTime,
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
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            logger.info('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
