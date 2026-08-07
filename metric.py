import torch
from torch import nn


DEFAULT_SRANK_TAU = 0.01
_EXCLUDED_NORM_LAYERS = (nn.LayerNorm, nn.modules.batchnorm._BatchNorm)


@torch.no_grad()
def weight_norm(model):
    """
    layernorm을 제외한 weight의 l2 norm
    """
    squared_sum = None

    for module in model.modules():
        if isinstance(module, _EXCLUDED_NORM_LAYERS):
            continue

        weight = getattr(module, "weight", None)  # bias는 제외
        if not isinstance(weight, torch.Tensor):
            continue

        value = weight.detach().float().square().sum()
        squared_sum = value if squared_sum is None else squared_sum + value

    if squared_sum is None:
        return 0.0
    return squared_sum.sqrt().item()


@torch.no_grad()
def srank(feature, tau=DEFAULT_SRANK_TAU):
    """Return Kumar et al.'s singular-value threshold rank (srank).

    feature matrix의 rank를 계산
    batch 데이터를 통과시킨 feature matrix의 singular value를 계산
    singular value의 누적합이 전체 singular value 합의 0.99 (1-tau) 이상이 되기 위해 필요한 개수 return
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    if feature.ndim < 2:
        raise ValueError("feature must have shape (n, d) or (n, ...)")
    if feature.shape[0] == 0:
        return 0

    threshold = 1 - tau
    feature = feature.detach().float()

    if len(feature.shape) > 2:
        feature = feature.reshape(-1, feature.shape[-1])

    svals = torch.linalg.svdvals(feature)
    sval_sum = torch.sum(svals)
    cumsum = torch.cumsum(svals, dim=0)
    threshold_crossed = cumsum >= threshold * sval_sum
    sranks = (~threshold_crossed).sum() + 1

    return int(sranks.item())
