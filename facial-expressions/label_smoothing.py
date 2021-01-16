import torch.nn.functional as F


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


def label_smoothing_cross_entropy(
    preds, target, epsilon: float = 0.1, reduction="mean", weight=None
):
    n = preds.size()[-1]
    log_preds = F.log_softmax(preds, dim=-1)
    loss = reduce_loss(-log_preds.sum(dim=-1), reduction)
    nll = F.nll_loss(log_preds, target, reduction=reduction, weight=weight)
    return linear_combination(loss / n, nll, epsilon)
