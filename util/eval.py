import torch

def compute_delta_accuracy(pred, target, threshold=1.5):
    ratio = torch.max(pred / target, target / pred)
    return (ratio < threshold).float().mean().item() * 100