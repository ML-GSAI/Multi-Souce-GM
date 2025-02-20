import os
import random
import numpy as np
import torch


def seed_torch(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def calculate_empirical_TV(p_hat, p_true):
    if isinstance(p_hat, torch.Tensor) and isinstance(p_true, torch.Tensor):
        empirical_TV = 0.5 * torch.abs(p_hat / p_true - 1).mean()
    elif isinstance(p_hat, np.ndarray) and isinstance(p_true, np.ndarray):
        empirical_TV = 0.5 * np.mean(np.abs(p_hat / p_true - 1))
    else:
        raise TypeError("Both p_hat and p_true must be either torch.Tensor or numpy.ndarray.")
    
    return empirical_TV

