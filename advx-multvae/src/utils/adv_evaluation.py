import torch
from sklearn.metrics import balanced_accuracy_score


def eval_adversary(logits, targets, loss_fn):
    """
    Calculates the loss as well as the balanced accuracy for an adversarial network
    (works basically for any classifier network)
    """
    adv_loss = loss_fn(logits, targets)

    # calculate the mean balanced accuracy score
    pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    bal_acc = balanced_accuracy_score(y_true=targets.cpu().numpy(),
                                      y_pred=pred)
    return adv_loss, bal_acc


def eval_adversaries(logits, targets, loss_fn):
    """
    Calculates the average loss and balanced accuracy over multiple adversarial networks for the same targets.
    """
    device = targets.device if isinstance(targets, torch.Tensor) else None
    adv_loss = torch.tensor(0, dtype=torch.float64, device=device)
    bal_acc = torch.tensor(0, dtype=torch.float64, device=device)

    n_adversaries = len(logits)
    for k in range(n_adversaries):
        al, ba = eval_adversary(logits[k], targets, loss_fn)
        adv_loss += al / n_adversaries
        bal_acc += ba / n_adversaries

    return adv_loss, bal_acc


def eval_adversaries_multi(logits, targets, loss_fn):
    adv_loss = loss_fn(logits, targets)  # N_loss * batch_size * 1
    return adv_loss, 0
