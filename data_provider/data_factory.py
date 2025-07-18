import logging
import sys

from torch.utils.data import DataLoader

from data_provider.data_loader import (Dataset_Custom, Dataset_ETT_hour,
                                       Dataset_ETT_minute, Dataset_PEMS,
                                       Dataset_Pred, Dataset_Solar)

logger = logging.getLogger(__name__)

# Define a custom exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("Unhandled exception occurred", exc_info=(exc_type, exc_value, exc_traceback))
    # Ensure logs are written immediately by flushing handlers
    for handler in logger.handlers:
        handler.flush()

# Set the custom handler as the global exception hook
sys.excepthook = handle_exception

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.num_total_nvars > args.nvars_training:
            batch_size = int(args.batch_size * (args.nvars_training / args.num_total_nvars))
        else: 
            batch_size = args.batch_size  # bsz=1 for evaluation
        if 'traffic' in args.data_path or 'electricity' in args.data_path:
            batch_size = 10
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag =='val':
        shuffle_flag = True
        drop_last = True
        if args.num_total_nvars > args.nvars_val:
            batch_size = int(args.batch_size * (args.nvars_training / args.nvars_val))
        else: 
            batch_size = args.batch_size  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    batch_size = 1 if batch_size < 1 else batch_size
    
    logger.info(f'{flag}, {len(data_set)}, batch_s={batch_size}')
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
