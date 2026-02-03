import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def dice(pred, gt):
    return (2*(pred*gt).sum()) / ((pred+gt).sum() + 1e-6)

def iou(pred, gt):
    inter = (pred*gt).sum()
    union = pred.sum() + gt.sum() - inter
    return inter / (union + 1e-6)

def precision(pred, gt):
    return (pred*gt).sum() / (pred.sum() + 1e-6)

def recall(pred, gt):
    return (pred*gt).sum() / (gt.sum() + 1e-6)

def hausdorff(pred, gt):
    p = np.argwhere(pred)
    g = np.argwhere(gt)
    if len(p)==0 or len(g)==0:
        return 0
    return max(
        directed_hausdorff(p, g)[0],
        directed_hausdorff(g, p)[0]
    )
