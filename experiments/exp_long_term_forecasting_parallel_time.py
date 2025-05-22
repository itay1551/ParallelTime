from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.optim import lr_scheduler
import logging
import random
import logging
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

FAST_TRANING_LOW_METRICS = True

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        nvars = args.num_total_nvars
        nvars_val = args.nvars_val
        self.val_partial_idx = None
        if nvars > nvars_val:
            self.val_partial_idx = np.stack(random.sample(range(nvars), nvars_val))

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss):
        loss = loss.lower()
        if loss == 'mse':
            criterion = nn.MSELoss()
        elif loss == 'huber':
            criterion = nn.HuberLoss(reduction='mean', delta=self.args.huber_delta)
        else:
            raise ValueError('wrong loss type')
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.val_partial_idx is not None:
                    batch_x = batch_x[:, :, self.val_partial_idx]
                    batch_y = batch_y[:, :, self.val_partial_idx]
                
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true).item()

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        is_partial_training = self.args.num_total_nvars > self.args.nvars_training

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        criterion = self._select_criterion(self.args.loss)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                if is_partial_training:
                    _, _, N = batch_x.shape
                    index = np.stack(random.sample(range(N), self.args.nvars_training))
                    batch_x = batch_x[:, :, index]
                    batch_y = batch_y[:, :, index]
                
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.info(f'\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                
                if self.args.lradj == 'OneCycleLR':
                    adjust_learning_rate(model_optim, scheduler=scheduler, epoch=epoch + 1, args=self.args)
                    scheduler.step()

            if epoch == self.args.train_epochs - 1:
                continue

            if self.args.lradj != 'OneCycleLR':
                adjust_learning_rate(model_optim, epoch=epoch + 1, args=self.args)
            logger.info(f'Epoch: {epoch + 1} cost time: {time.time() - epoch_time}')
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info('Early stopping')
                break
            logger.info(f'Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}')

        return self.model


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            logger.info('Loading model...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))) 

        folder_path = f'./test_results/{setting}/'
        weights_folder = f'./test_results/{setting}/weights/'  # Folder for weights
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(weights_folder, exist_ok=True)  # Create weights subfolder

        self.model.eval()
        total_mae, total_mse, total_rmse, total_mape, total_mspe = 0, 0, 0, 0, 0
        count = 0  # Number of processed batches

        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                del _
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs, weights_by_layer_list = self.model(batch_x, True)

                # Save weights_by_layer_list for this batch
                weights_file = os.path.join(weights_folder, f'weights_batch_{i}.pt')
                torch.save([w.detach().cpu() for w in weights_by_layer_list], weights_file)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Convert to numpy
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # Inverse transform if necessary
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                # Calculate metrics for current batch
                mae, mse, rmse, mape, mspe = metric(outputs, batch_y)
                total_mae += mae
                total_mse += mse
                total_rmse += rmse
                total_mape += mape
                total_mspe += mspe
                count += 1

                # Periodic visualization
                if i % 20 == 0 and FAST_TRANING_LOW_METRICS:
                    input_data = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_data.shape
                        input_data = test_data.inverse_transform(input_data.squeeze(0)).reshape(shape)
                    for j in range(7):
                        gt = np.concatenate((input_data[0, :, j], batch_y[0, :, j]), axis=0)
                        pd = np.concatenate((input_data[0, :, j], outputs[0, :, j]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, f'{i}_{j}.pdf'), batch_x.shape[-2])
                        weights_file = os.path.join(folder_path, 'weights', f'weights_batch_{i}_{j}.pt')
                        torch.save([w.detach().cpu()[0 + j] for w in weights_by_layer_list], weights_file)
                        del gt, pd

        # Compute final averaged metrics
        avg_mae = total_mae / count
        avg_mse = total_mse / count
        avg_rmse = total_rmse / count
        avg_mape = total_mape / count
        avg_mspe = total_mspe / count

        logger.info(f'mse:{avg_mse}, mae:{avg_mae}')

        # Save final results
        # with open(f'result_long_term_forecast_{self.args.comment}.txt', 'a') as f:
        with open(f'result_long_term_forecast_{self.args.model_id}.txt', 'a') as f:
            f.write(f'{setting}\n')
            f.write(f'mse: {avg_mse}, mae: {avg_mae}\n\n')

        results_folder = f'./results/{setting}/'
        os.makedirs(results_folder, exist_ok=True)
        np.save(os.path.join(results_folder, 'final_metrics.npy'), np.array([avg_mae, avg_mse, avg_rmse, avg_mape, avg_mspe]))

        return
    
    
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)

        return