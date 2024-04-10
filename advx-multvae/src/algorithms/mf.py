import torch
from scipy.sparse import vstack
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam

from src.data.data_loading import sparse_scipy_to_tensor
from src.logging.logger import Logger
from src.modules.dmf import DMFAdv
from src.algorithms.utils import process_results
from src.utils.adv_evaluation import eval_adversary
from rmet import calculate


def setup_model_and_optimizers_mf(config, n_users, n_items, device):
    model = DMFAdv.from_config(config["model"], config["adv"], n_users, n_items)
    model.to(device)

    loss_fn = BCEWithLogitsLoss()
    loss_fn_adv = CrossEntropyLoss()

    optim = Adam([
        {"params": model.user_encoder.parameters()},
        {"params": model.item_encoder.parameters()}
    ], **config["optim"])

    adv_opt_modules = [model.adv]
    if config["adv"]["backprop_adv"]:
        adv_opt_modules.append(model.user_encoder)
    optim_adv = Adam([{"params": m.parameters()} for m in adv_opt_modules], **config["optim_adv"])
    return model, optim, optim_adv, loss_fn, loss_fn_adv


def preprocess_data_mf(model_input, device):
    inp_user, inp_item = model_input
    inp_user = torch.sparse_coo_tensor(*inp_user).to(device)
    inp_item = torch.sparse_coo_tensor(*inp_item).to(device)
    return inp_user, inp_item


def validate_epoch_mf(model, device, data_loader, preprocess_fn, loss_fn, logger: Logger, epoch,
                      result_path, best_validation_scores, early_stopping_criteria, conf,
                      perform_adv_training=False, loss_fn_adv=None, is_verbose=False, tr_set=None, vd_set=None):
    model.eval()
    with torch.no_grad():

        item_encodings = []
        # Collect item encodings, as they are the same for all users
        for inp_item in tr_set.iter_items():
            inp_item = sparse_scipy_to_tensor(inp_item).to(device)
            item_encodings.append(model.item_encoder(inp_item))
        item_encodings = torch.cat(item_encodings)

        user_traits = []
        user_targets = []
        user_encodings = []
        # Collect user encodings as they do not change during validation
        for inp_user, inp_targets, inp_traits in vd_set.iter_users():
            user_traits.append(torch.tensor(inp_traits.flatten()).to(device, dtype=torch.long))
            user_targets.append(inp_targets)
            inp_user = sparse_scipy_to_tensor(inp_user).to(device)
            user_encodings.append(model.user_encoder(inp_user))

        user_traits = torch.cat(user_traits)
        user_targets = vstack(user_targets)
        user_encodings = torch.cat(user_encodings)

        predictions = model.similarity(user_encodings, item_encodings).detach().cpu()
        loss = loss_fn(predictions, sparse_scipy_to_tensor(user_targets).to_dense())
        logger.log_value(f"val/loss", loss, epoch)

        # TODO: allow multiple "top_k" to be evaluated
        # TODO: also evaluate for different user groups
        metric_results, _ = calculate(conf["metrics"], predictions, user_targets, conf["top_k"][0],
                                      aggregate_results=True)

        if perform_adv_training:
            trait_logits = model.adv(user_encodings)
            adv_loss, adv_bacc = eval_adversary(trait_logits, user_traits, loss_fn_adv)
            metric_results["adv_bacc"] = adv_bacc
            logger.log_value(f"val/adv_loss", adv_loss, epoch)
            logger.log_value(f"val/adv_bacc", adv_bacc, epoch)

        return process_results(model, logger, epoch, early_stopping_criteria, metric_results,
                               best_validation_scores, result_path, is_verbose)
