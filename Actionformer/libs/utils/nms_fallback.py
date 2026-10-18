"""Portable CPU NMS when the optional compiled extension is unavailable.

Matches the extension's call contract; enables Linux CPU and Windows packaging.
"""
import torch


def _iou(segment, others):
    intersection = (torch.minimum(segment[1], others[:, 1]) -
                    torch.maximum(segment[0], others[:, 0])).clamp(min=0)
    union = (segment[1]-segment[0]) + (others[:, 1]-others[:, 0]) - intersection
    return intersection / union.clamp(min=1e-8)


def nms(segs, scores, iou_threshold):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]; keep.append(int(i))
        order = order[1:]
        if order.numel():
            order = order[_iou(segs[i], segs[order]) <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long)


def softnms(segs, scores, dets, iou_threshold, sigma, min_score, method):
    scores = scores.clone()
    active = torch.arange(len(scores))
    keep = []
    while active.numel():
        best = scores[active].argmax(); idx = active[best]
        if scores[idx] < min_score:
            break
        dets[len(keep), :2] = segs[idx]; dets[len(keep), 2] = scores[idx]
        keep.append(int(idx))
        active = active[active != idx]
        if not active.numel():
            break
        iou = _iou(segs[idx], segs[active])
        if method == 2:
            weight = torch.exp(-iou*iou/sigma)
        elif method == 1:
            weight = torch.where(iou > iou_threshold, 1-iou, torch.ones_like(iou))
        else:
            weight = (iou <= iou_threshold).to(scores.dtype)
        scores[active] *= weight
    return torch.tensor(keep, dtype=torch.long)
