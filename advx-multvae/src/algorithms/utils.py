from numbers import Number
from collections import defaultdict
from typing import Union, Tuple, Dict

import torch
from rmet import calculate
from tqdm import tqdm

from src.logging.logger import Logger
from src.utils.adv_evaluation import eval_adversaries
from src.utils.helper import save_model


def postprocess_loss_results(result: Union[Number, Tuple[Number, Dict[str, Number]]]):
    """
    Utility function that ensures that the loss results have the format of
    (total_loss: float-like, sub_losses: dict[float-like])
    so that we can do easier processing (and logging) of results.
    """
    # we also want to allow regular loss functions that do not return additional partial losses
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    elif len(result) == 1:
        return result, {}
    else:
        raise SystemError("This type of loss result is not yet supported.")


def add_to_dict(original: dict, new: dict, multiplier: float = 1.) -> dict:
    for k, l in new.items():
        original[k] += l * multiplier
    return original


def scale_dict_items(d: dict, scale: float) -> dict:
    return {k: v * scale for k, v in d.items()}


def train_epoch(model, optimizer, device, data_loader, preprocess_fn, loss_fn, logger: Logger, epoch,
                perform_adv_training=False, optimizer_adv=None, loss_fn_adv=None, adv_loss_weight=1,
                log_every_n_batches=100, is_verbose=False):
    raise NotImplementedError("changes are required so that it works similar to 'train_epoch_parallel'")

    model.train()
    batch_count = len(data_loader) * epoch
    train_loss_dict = defaultdict(lambda: 0)
    sample_count, train_loss, train_adv_loss, train_adv_bacc = 0, 0, 0, 0
    for _, *model_input, targets, adv_targets in tqdm(data_loader, desc="Training steps", position=2, leave=True,
                                                      disable=True):
        n_samples = len(targets)
        sample_count += n_samples
        targets = targets.to(device)

        model_input = preprocess_fn(model_input, device)
        logits, loss_dict, _ = model(*model_input)

        # custom loss function, may be used to weight different loss terms
        result = loss_fn(logits, *loss_dict.values(), targets)
        # we also want to allow regular loss functions that do not return additional partial losses
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            loss, loss_dict = result
        else:
            loss, loss_dict = result, {}
        train_loss += loss * n_samples

        # Models may provide other losses, which may otherwise be difficult to calculate.
        # Returning a loss dict seems to be a rather generic way to offer this possibility
        for k, l in loss_dict.items():
            train_loss_dict[k] += l * n_samples

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if perform_adv_training:
            adv_targets = adv_targets.to(device, dtype=torch.long)
            *_, adv_logits = model(*model_input)
            # TODO: for now we only support adversarial training on one feature
            #       in case we want to train on more, remove line below and adjust 'eval_adversaries()'
            adv_targets = adv_targets.flatten()
            adv_loss, adv_bacc = eval_adversaries(adv_logits, adv_targets, loss_fn=loss_fn_adv)
            train_adv_loss += adv_loss * n_samples
            train_adv_bacc += adv_bacc * n_samples

            # Update adversary, need to step with both optimizers to change encoder
            optimizer_adv.zero_grad()
            optimizer.zero_grad()
            adv_loss.backward()
            optimizer_adv.step()
            optimizer.step()

        # Log results inside batch to get a better idea whether the model works
        if batch_count % log_every_n_batches == 0:
            logger.log_value(f"train/batch_loss", loss, batch_count)
            for k, l in loss_dict.items():
                logger.log_value(f"train/batch_loss_{k}", l, batch_count)

            if perform_adv_training:
                logger.log_value(f"train/batch_adv_loss", adv_loss, batch_count)
                logger.log_value(f"train/batch_adv_bacc", adv_bacc, batch_count)
        batch_count += 1

    logger.log_value(f"train/loss", train_loss / sample_count, epoch)
    for k, l in train_loss_dict.items():
        logger.log_value(f"train/loss_{k}", l / sample_count, epoch)

    if perform_adv_training:
        logger.log_value(f"train/adv_loss", train_adv_loss / sample_count, epoch)
        logger.log_value(f"train/adv_bacc", train_adv_bacc / sample_count, epoch)


def train_epoch_parallel(model, optimizer, device, data_loader, preprocess_fn, loss_fn, logger: Logger, epoch,
                         perform_adv_training=False, optimizer_adv=None, loss_fn_adv=None,
                         log_every_n_batches=100, is_verbose=False):
    model.train()
    batch_count = len(data_loader) * epoch
    train_loss_dict = defaultdict(lambda: 0)
    train_adv_loss_dict = defaultdict(lambda: 0)

    # initialize variables for result accumulation
    sample_count, train_loss, train_adv_loss = 0, 0., 0.
    for indices, *model_input, targets, adv_targets in tqdm(data_loader, desc="Training steps", position=2, leave=True,
                                                      disable=True):
        n_samples = len(indices)
        sample_count += n_samples
        targets = targets.to(device)

        # push data through model
        model_input = preprocess_fn(model_input, device)
        logits, loss_dict, adv_logits = model(*model_input)

        # custom loss function, may be used to weight different loss terms
        result = loss_fn(logits, *loss_dict.values(), targets)
        loss, loss_dict = postprocess_loss_results(result)

        # store results over batches
        train_loss += loss * n_samples
        train_loss_dict = add_to_dict(train_loss_dict, loss_dict, multiplier=n_samples)

        adv_loss, adv_loss_dict = 0., {}
        if perform_adv_training:
            # gather adversarial results
            adv_targets = [t.to(device) for t in adv_targets]
            adv_result = loss_fn_adv(adv_logits, adv_targets)
            adv_loss, adv_loss_dict = postprocess_loss_results(adv_result)

            # store results over batches
            train_adv_loss += adv_loss * n_samples
            train_adv_loss_dict = add_to_dict(train_adv_loss_dict, adv_loss_dict, multiplier=n_samples)

            # Regular model and adversary are optimized at the same time (that's why we have the GRU layer!)
            # To support multiple losses, the scaling of the individual terms must happen in the loss function itself
            loss = loss + adv_loss

        # Update model (+ optionally its adversarial modules)
        optimizer.zero_grad()
        if perform_adv_training:
            optimizer_adv.zero_grad()
        loss.backward()
        optimizer.step()
        if perform_adv_training:
            optimizer_adv.step()

        # Log results inside batch to get a better idea whether the model works
        if batch_count % log_every_n_batches == 0:
            logger.log_value(f"train/batch_loss", loss, batch_count)
            logger.log_value_dict(f"train/batch_loss", loss_dict, batch_count)

            if perform_adv_training:
                logger.log_value(f"train/batch_adv_loss", adv_loss, batch_count)
                logger.log_value_dict(f"train/batch_adv_loss", adv_loss_dict, batch_count)

        batch_count += 1

    logger.log_value(f"train/loss", train_loss / sample_count, epoch)
    logger.log_value_dict(f"train/loss", scale_dict_items(train_loss_dict, 1 / sample_count), epoch)

    if perform_adv_training:
        logger.log_value(f"train/adv_loss", train_adv_loss / sample_count, epoch)
        logger.log_value_dict(f"train/adv_loss", scale_dict_items(train_adv_loss_dict, 1 / sample_count), epoch)


def validate_epoch(model, device, data_loader, preprocess_fn, loss_fn, logger: Logger, epoch,
                   result_path, best_validation_scores, early_stopping_criteria, conf,
                   perform_adv_training=False, loss_fn_adv=None, eval_adv_fn=None, is_verbose=False, **kwargs):
    if perform_adv_training:
        if loss_fn_adv is None or eval_adv_fn is None:
            raise AttributeError("'perform_adv_training' is set, but no loss or eval function was provided.")

    model.eval()

    # no need to keep gradients, as we won't do any backpropagation
    with torch.no_grad():
        val_loss_dict = defaultdict(lambda: 0)
        val_adv_loss_dict = defaultdict(lambda: 0)

        # initialize variables for result accumulation
        sample_count, val_loss, val_adv_loss, val_adv_score = 0, 0., 0., 0.

        adversary_logits, adversary_targets = [], []
        model_logits, model_targets = [], []
        for indices, *model_input, targets, adv_targets in data_loader:
            n_samples = len(indices)
            sample_count += n_samples
            targets = targets.to(device)

            # push data through model
            model_input = preprocess_fn(model_input, device)
            logits, loss_dict, adv_logits = model(*model_input)

            # remove known interactions from logits
            logits[model_input[0].nonzero(as_tuple=True)] = .0  # TODO: check whether this works as intended

            # calculate loss(es)
            result = loss_fn(logits, *loss_dict.values(), targets)
            loss, loss_dict = postprocess_loss_results(result)

            # store loss(es)
            val_loss += loss * n_samples
            val_loss_dict = add_to_dict(val_loss_dict, loss_dict, multiplier=n_samples)

            # gather all predictions and ground_truth labels
            model_logits.append(logits.detach().cpu())
            model_targets.append(targets.detach().cpu())

            if perform_adv_training:
                adv_targets = [tar.to(device) for tar in adv_targets]
                adv_result = loss_fn_adv(adv_logits, adv_targets)
                adv_loss, adv_loss_dict = postprocess_loss_results(adv_result)

                # store results over batches
                val_adv_loss += adv_loss * n_samples
                val_adv_loss_dict = add_to_dict(val_adv_loss_dict, adv_loss_dict, multiplier=n_samples)

                # logits and targets for each adversary group, and each adversary in each group
                adversary_logits.append([[log.detach().cpu() for log in log_grp] for log_grp in adv_logits])
                adversary_targets.append([tar.detach().cpu() for tar in adv_targets])

        val_loss = val_loss / sample_count
        if scheduler := kwargs.get("scheduler"):
            scheduler.step(val_loss)

        val_adv_loss = val_adv_loss / sample_count
        if scheduler := kwargs.get("scheduler_adv"):
            scheduler.step(val_adv_loss)

        # log validation results
        logger.log_value("val/loss", val_loss, epoch)
        logger.log_value_dict(f"val/loss", scale_dict_items(val_loss_dict, 1 / sample_count), epoch)

        if perform_adv_training:
            logger.log_value(f"val/adv_loss", val_adv_loss, epoch)
            logger.log_value_dict(f"val/adv_loss", scale_dict_items(val_adv_loss_dict, 1 / sample_count), epoch)

        # combine all logits and targets for overall model evaluation
        model_logits = torch.cat(model_logits)
        model_targets = torch.cat(model_targets)
        results_dict = calculate(conf["metrics"], model_logits, model_targets, conf["top_k"],
                                 return_individual=False)

        # log all results for analyzing purposes
        logger.log_value_dict(f"val/metrics", results_dict, epoch)

        if perform_adv_training:
            # concat output of individual adversaries in the different adversary groups
            adversary_group_logits = [[torch.cat(log) for log in zip(*grp)] for grp in zip(*adversary_logits)]
            adversary_group_targets = [torch.cat(grp) for grp in zip(*adversary_targets)]
            adv_eval_results_dict = eval_adv_fn(adversary_group_logits, adversary_group_targets)
            logger.log_value_dict(f"val/adv_scores", adv_eval_results_dict, epoch)
            results_dict.update(adv_eval_results_dict)

        if is_verbose:
            # params blow up the log files, so let's use it only for verbose (e.g. debug) settings
            logger.log_params(model, tag_prefix="val", epoch=epoch)

        return process_results(model, epoch, early_stopping_criteria, results_dict,
                               best_validation_scores, result_path, is_verbose)


def process_results(model, epoch, early_stopping_criteria, results_dict, best_validation_scores,
                    result_path, is_verbose=False):
    for label, criterion in early_stopping_criteria.items():
        if (wu := criterion.get("warmup")) and epoch <= wu:
            break

        # Allow early stopping on non rank-based metrics
        validation_score = results_dict[criterion["metric"]]
        if "top_k" in criterion:
            validation_score = validation_score[criterion["top_k"]]

        better_model_found = False
        if "highest_is_best" in criterion:
            better_model_found = validation_score >= best_validation_scores[label] and criterion[
                "highest_is_best"] or validation_score <= best_validation_scores[label] and not criterion[
                "highest_is_best"]
        elif "closest_is_best" in criterion:
            old_diff = abs(criterion["value"] - best_validation_scores[label])
            new_diff = abs(criterion["value"] - validation_score)
            better_model_found = new_diff <= old_diff

        if better_model_found:
            if is_verbose:
                print("Better model found!")
                if top_k := criterion.get("top_k"):
                    print(f'{criterion["metric"]}@{top_k}={validation_score:.4f}\n')
                else:
                    print(f'{criterion["metric"]}={validation_score:.4f}\n')
            save_model(model, result_path, "best_model_" + label)
            best_validation_scores[label] = validation_score

    return best_validation_scores

def validate_epoch_test(model, device, data_loader, preprocess_fn, loss_fn, conf: dict, perform_adv_training=False,
                   loss_fn_adv=None, eval_adv_fn=None, is_verbose=False, **kwargs):
    if perform_adv_training:
        if loss_fn_adv is None or eval_adv_fn is None:
            raise AttributeError("'perform_adv_training' is set, but no loss or eval function was provided.")

    model.eval()

    # no need to keep gradients, as we won't do any backpropagation
    with torch.no_grad():
        val_loss_dict = defaultdict(lambda: 0)
        val_adv_loss_dict = defaultdict(lambda: 0)

        # initialize variables for result accumulation
        sample_count, val_loss, val_adv_loss, val_adv_score = 0, 0., 0., 0.

        adversary_logits, adversary_targets = [], []
        model_logits, model_targets = [], []
        for indices, *model_input, targets, adv_targets in data_loader:
            n_samples = len(indices)
            sample_count += n_samples
            targets = targets.to(device)

            # push data through model
            model_input = preprocess_fn(model_input, device)
            logits, loss_dict, adv_logits = model(*model_input)

            # remove known interactions from logits
            logits[model_input[0].nonzero(as_tuple=True)] = .0  # TODO: check whether this works as intended

            # calculate loss(es)
            result = loss_fn(logits, *loss_dict.values(), targets)
            loss, loss_dict = postprocess_loss_results(result)

            # store loss(es)
            val_loss += loss * n_samples
            val_loss_dict = add_to_dict(val_loss_dict, loss_dict, multiplier=n_samples)

            # gather all predictions and ground_truth labels
            model_logits.append(logits.detach().cpu())
            model_targets.append(targets.detach().cpu())

            if perform_adv_training:
                adv_targets = [tar.to(device) for tar in adv_targets]
                adv_result = loss_fn_adv(adv_logits, adv_targets)
                adv_loss, adv_loss_dict = postprocess_loss_results(adv_result)

                # store results over batches
                val_adv_loss += adv_loss * n_samples
                val_adv_loss_dict = add_to_dict(val_adv_loss_dict, adv_loss_dict, multiplier=n_samples)

                # logits and targets for each adversary group, and each adversary in each group
                adversary_logits.append([[log.detach().cpu() for log in log_grp] for log_grp in adv_logits])
                adversary_targets.append([tar.detach().cpu() for tar in adv_targets])

        # collect all results
        results = {
            "loss": val_loss / sample_count,
            "loss_items": scale_dict_items(val_loss_dict, 1 / sample_count)
        }

        if perform_adv_training:
            results["adv_loss"] = val_adv_loss / sample_count
            results["adv_loss_items"] = scale_dict_items(val_adv_loss_dict, 1 / sample_count)

        # combine all logits and targets for overall model evaluation
        model_logits = torch.cat(model_logits)
        model_targets = torch.cat(model_targets)
        results["metrics"] = calculate(conf["metrics"], model_logits, model_targets, conf["top_k"],
                                       return_individual=False)
        results["metrics_individual"] = calculate(conf["metrics"], model_logits, model_targets, conf["top_k"],
                                       return_individual=True)
        if perform_adv_training:
            # concat output of individual adversaries in the different adversary groups
            adversary_group_logits = [[torch.cat(log) for log in zip(*grp)] for grp in zip(*adversary_logits)]
            adversary_group_targets = [torch.cat(grp) for grp in zip(*adversary_targets)]
            results["adv_scores"] = eval_adv_fn(adversary_group_logits, adversary_group_targets)

        return results