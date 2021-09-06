import torch
from params import pos_weight

def macro_double_soft_f1(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y_hat = torch.sigmoid(y_hat)
    tp = torch.sum(y_hat * y, axis=1)
    fp = torch.sum(y_hat * (1 - y), axis=1)
    fn = torch.sum((1 - y_hat) * y, axis=1)
    tn = torch.sum((1 - y_hat) * (1 - y), axis=1)
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    # soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    cost_class1 = (1 - soft_f1_class1)
    # cost_class0 = (1 - soft_f1_class0)
    # cost = 0.5 * (cost_class1 + cost_class0)
    # macro_cost = torch.mean(cost)
    
    return torch.mean(cost_class1)


def custom_loss():
    """
    Loss function that maximises both f1-score and BCE. This helps
    in improving model performance while keeping an eye on the f1-score.
    Intuition: BCE  maximises bAcc, while macro_double_soft_f1 maximises f1-score.

    Hopefully this will balance both metrics.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): network output (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch

    """
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(pos_weight))
    soft_f1 = macro_double_soft_f1
    return lambda y_hat, y : 1*bce_loss(y_hat, y.float()) + 0*soft_f1(y_hat, y)