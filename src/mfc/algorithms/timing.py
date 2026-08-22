import time

import torch


def synchronize_device(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def synchronized_time(device):
    synchronize_device(device)
    return time.perf_counter()
