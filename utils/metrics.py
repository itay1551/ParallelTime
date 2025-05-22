import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


# import torch

# def RSE(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#     num = torch.sqrt(torch.sum((true - pred) ** 2))
#     den = torch.sqrt(torch.sum((true - true.mean()) ** 2))
#     return num / den

# def CORR(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#     true_mean = true.mean(dim=0, keepdim=True)
#     pred_mean = pred.mean(dim=0, keepdim=True)

#     num = ((true - true_mean) * (pred - pred_mean)).sum(dim=0)
#     den = torch.sqrt(((true - true_mean) ** 2 * (pred - pred_mean) ** 2).sum(dim=0))

#     return (num / (den + 1e-8)).mean(-1)  # Avoid division by zero

# def MAE(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#     return torch.mean(torch.abs(pred - true))

# def MSE(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#     return torch.mean((pred - true) ** 2)

# def RMSE(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#     return torch.sqrt(MSE(pred, true))  # Avoid redundant computation

# def MAPE(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#     return torch.mean(torch.abs((pred - true) / torch.clamp(true, min=1e-8)))  # Avoid division by zero

# def MSPE(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#     return torch.mean(torch.square((pred - true) / torch.clamp(true, min=1e-8)))  # Avoid division by zero

# def metric(pred: torch.Tensor, true: torch.Tensor):
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)
#     rmse = RMSE(pred, true)
#     mape = MAPE(pred, true)
#     mspe = MSPE(pred, true)

#     return mae, mse, rmse, mape, mspe
