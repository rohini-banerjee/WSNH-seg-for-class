# -----------------------------------------------------------------------------
# Evaluation metrics and loss functions to evaluate model performance during
# training, testing, and validation phases.
# -----------------------------------------------------------------------------
import imblearn.metrics
import json
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from sklearn.metrics import jaccard_score, roc_auc_score, average_precision_score, recall_score
from torcheval.metrics.functional import multiclass_auroc, multiclass_accuracy, multiclass_recall
from torchmetrics.functional.classification import multiclass_specificity, binary_accuracy, binary_calibration_error
from torchmetrics.classification import MulticlassSpecificity, MulticlassRecall, MulticlassAccuracy


METRIC_KEYS = [
    'Accuracy_SK',
    'Sens_SK',
    'Spec_SK',
    'AUROC_SK',
    'Accuracy_M',
    'Sens_M',
    'Spec_M',
    'AUROC_M',
    'Average_Accuracy(micro)',
    'Average_Accuracy(macro)',
    'Average_AUROC(micro)',
    'Average_AUROC(macro)',
    'EU',
]

######################
# LOSS FUNCTIONS
######################
def get_bce(y_pred, y_true, is_logits):
    """
    Compute binary cross entropy (BCE) loss value.
    """
    if is_logits:
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float(), reduction='mean')
    else:
        loss = F.binary_cross_entropy(y_pred, y_true.float(), reduction='mean')
    return loss

def get_ce(y_pred, y_true):
    """
    Compute (multi-class) cross entropy (BCE) loss value.
    """
    ce_loss_fn = nn.CrossEntropyLoss()
    return ce_loss_fn(y_pred, y_true)

def get_dice_bce(y_pred, y_true, is_logits=False, l_bce=1.0, l_d=1.0):
    """
    Compute combined binary cross entropy (BCE) loss + Dice
    score loss value.
    """
    dice_loss_fn = smp.losses.DiceLoss(
        mode='binary',
        log_loss=True,
        from_logits=is_logits,
    )
    dice_loss = dice_loss_fn(y_pred=y_pred, y_true=y_true)
    bce_loss = get_bce(y_pred, y_true, is_logits)
    return l_bce*bce_loss + l_d*dice_loss

def get_dice_fl(y_pred, y_true, is_logits=True, l_fl=1.0, l_d=1.0):
    """
    Compute combined Dice score loss + Focal loss value.
    NOTE:
        1. Expects y_pred is in logits and is NOT converted
        to probabilities.
    """
    dice_loss_fn = smp.losses.DiceLoss(
        mode='binary',
        log_loss=True,
        from_logits=True,
    )
    focal_loss_fn = smp.losses.FocalLoss(
        mode='binary',
    )
    dice_loss = dice_loss_fn(y_pred=y_pred, y_true=y_true)
    fl_loss = focal_loss_fn(y_pred=y_pred, y_true=y_true)
    return l_fl*fl_loss + l_d*dice_loss

def kl_loss(y_pred, y_true, i):
    """
    Compute Kullback-Leibler (KL) divergence loss value.
    """
    if (y_true < 0).any():
        print(f'ytrue (model {i}) has negative values')
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)
    return kl_loss_fn(y_pred, utils.add_small_constant(y_true))

def get_dml_loss(p_s, i, b_y, lambda_kl=0.85):
    """
    Compute combined cross entropy loss and DML loss, defined as the
    KL divergence loss of all other model predictions from the current
    model prediction.
    NOTE:
        1. Expects p_s are logits --> please apply log_softmax
        2. Expects b_y_hat is logits --> please apply softmax
    """
    ce_loss = get_ce(p_s[i], b_y)
    sum_loss = 0.0
    for j in range(utils.N_CLASSIFIERS):
        if j != i:
            sum_loss += kl_loss(F.log_softmax(p_s[i], dim=1), F.softmax(p_s[j], dim=1), i)
    avg_kl_loss = sum_loss / (utils.N_CLASSIFIERS - 1)
    return ce_loss + lambda_kl*avg_kl_loss

######################
# METRICS
######################
### MODEL PERFORMANCE
def get_dice_score(y_pred, y_true, from_logits=False):
    """
    Compute dice score coefficient (DSC) metric.
    """
    dice_loss_fn = smp.losses.DiceLoss(
        mode='binary',
        from_logits=from_logits,
    )
    dice_score = 1.0 - dice_loss_fn(y_pred=y_pred, y_true=y_true)
    return dice_score

def get_acc(y_pred, y_true):
    """
    Compute accuracy given predicted binary segmentation mask and ground truth mask.
    """
    y_pred = utils.convert_to_binary_mask(y_pred, threshold=0.5)
    return torch.mean(y_true.eq(y_pred).float())

def get_iou_score(y_pred, y_true):
    """
    Compute intersection-over-union (IoU) metric.
    """
    tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y_true, mode='binary', threshold=0.5)
    return smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro-imagewise')

def get_binary_accuracy(y_pred, y_true):
    return binary_accuracy(y_pred, y_true).detach().item()

def get_binary_auroc(y_scores, y_true):
    """
    Compute binary Area under the Receiver Operating Characteristic Curve
    (AUROC) score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = utils.convert_to_numpy(y_true)
        y_scores = utils.convert_to_numpy(y_scores)
    return roc_auc_score(y_true=y_true, y_score=y_scores)

def get_binary_ap(y_pred, y_true):
    """
    Compute binary average precision score.
    """
    return average_precision_score(y_true=y_true, y_score=y_pred)

def get_binary_sensitivity(y_pred, y_true):
    """
    Compute binary sensitivity score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = utils.convert_to_numpy(y_true)
        y_pred = utils.convert_to_numpy(y_pred)
    return imblearn.metrics.sensitivity_score(y_true=y_true, y_pred=y_pred, pos_label=1, average='binary')

def get_binary_specificity(y_pred, y_true):
    """
    Commpute binary specificity score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = utils.convert_to_numpy(y_true)
        y_pred = utils.convert_to_numpy(y_pred)
    return imblearn.metrics.specificity_score(y_true=y_true, y_pred=y_pred, pos_label=1, average='binary')

def get_multi_auroc(y_probs, targets, reduction='micro'):
    """
    Compute multi-class AUROC score and apply specified reduction.
    """
    y_true = utils.convert_to_numpy(targets)
    y_scores = utils.convert_to_numpy(y_probs)
    return roc_auc_score(y_true=y_true, y_score=y_scores, multi_class='ovr', average=reduction)

def get_multi_accuracy(y_preds, targets, reduction='micro'):
    """
    Compute multi-class accuracy and apply specified reduction.
    """
    return multiclass_accuracy(y_preds, targets, num_classes=len(utils.LABEL_TO_IND), average=reduction).detach().item()

### MODEL QUALITY
def entropy(tensor, dim=-1, eps=1e-7):
    """
    Compute entropy of a probability distribution.
    """
    return -torch.sum(tensor * torch.log(tensor + eps), dim=dim)

def get_predictive_entropy(logits, reduction=None, dim=1):
    """
    Compute predictive entropy across class predictions. If reduction is
    not None, return mean value across batch dimension.
    """
    if reduction not in [None, 'mean', 'sum']:
        raise ValueError(f"Invalid reduction: {reduction}")
    
    prob_dist = F.softmax(logits, dim=dim)
    ent = entropy(prob_dist, dim=dim)
    if reduction is None:
        return ent
    elif reduction == 'mean':
        return torch.mean(ent)
    elif reduction == 'sum':
        return torch.sum(ent)

def binary_entropy(tensor, eps=1e-7):
    """
    Compute entropy of a binary tensor.
    """
    tensor = torch.clamp(tensor, eps, 1 - eps)
    return -tensor * torch.log(tensor) - (1 - tensor) * torch.log(1 - tensor)

def get_mutual_information_segm(outputs, mc_dim):
    """
    Compute mutual information between ensemble of binary segmentation
    masks and ground truth mask.
    """
    prob_bar = outputs.mean(dim=mc_dim)
    pred_entropy = binary_entropy(prob_bar)
    expected_entropy = binary_entropy(outputs).mean(dim=mc_dim)
    # epistemic uncertainty
    epistemic = pred_entropy - expected_entropy
    return epistemic

def get_mutual_information_classif(outputs, mc_dim):
    """
    Compute mutual information between ensemble of classification
    probabilities and target label.
    """
    prob_bar = outputs.mean(dim=mc_dim)     # (B, C)
    pred_entropy = entropy(prob_bar)       # (B,)
    expected_entropy = entropy(outputs).mean(dim=mc_dim) # (B,)
    # epistemic uncertainty
    epistemic = pred_entropy - expected_entropy
    return epistemic

def get_uncertainties(ensemble, dim=0, is_classif=False, use_mi=False):
    """
    Compute epistemic and aleatoric uncertainties across ensemble of 
    classifiers or predictions.
    """
    if use_mi:
        epistemic = get_mutual_information_classif(ensemble, mc_dim=dim) if is_classif else get_mutual_information_segm(ensemble, mc_dim=dim)
    else:
        epistemic = torch.mean(ensemble**2, dim=dim) - torch.mean(ensemble, dim=dim)**2
    aleatoric = torch.mean(ensemble*(1-ensemble), dim=dim)
    return epistemic, aleatoric

def ece(probs, targets):
    """
    Compute expected calibration error (ECE) given predicted class probabilities
    and target labels.
    """
    return binary_calibration_error(probs, targets)

######################
# UTILITIES
######################
def compute_metrics(metric_dict, model_type, probs, preds, eus, targets, save_results, \
    roc_results_file, model_results_file):
    """
    Compute relevant metrics for model evaluation.
    """
    malignant_vs_rest = (preds == 0).type(torch.FloatTensor)
    malignant_target = (targets == 0).type(torch.FloatTensor)

    sebk_vs_rest = (preds == 1).type(torch.FloatTensor)
    sebk_target = (targets == 1).type(torch.FloatTensor)

    # Average and class-specific AUROC values
    metric_dict['Average_AUROC(micro)'] = get_multi_auroc(probs, targets, reduction='micro')
    metric_dict['Average_AUROC(macro)'] = get_multi_auroc(probs, targets, reduction='macro')
    metric_dict['AUROC_M'] = get_binary_auroc(malignant_vs_rest, malignant_target)
    metric_dict['AUROC_SK'] = get_binary_auroc(sebk_vs_rest, sebk_target)

    # Average and class-specific accuracy values
    metric_dict['Average_Accuracy(micro)'] = get_multi_accuracy(preds, targets, reduction='micro')
    metric_dict['Average_Accuracy(macro)'] = get_multi_accuracy(preds, targets, reduction='macro')
    metric_dict['Accuracy_M'] = get_binary_accuracy(malignant_vs_rest, malignant_target)
    metric_dict['Accuracy_SK'] = get_binary_accuracy(sebk_vs_rest, sebk_target)

    # Average and class-specific specificity values
    metric_dict['Spec_M'] = get_binary_specificity(malignant_vs_rest, malignant_target)
    metric_dict['Spec_SK'] = get_binary_specificity(sebk_vs_rest, sebk_target)

    # Average and class-specific sensitivity values
    metric_dict['Sens_M'] = get_binary_sensitivity(malignant_vs_rest, malignant_target)
    metric_dict['Sens_SK'] = get_binary_sensitivity(sebk_vs_rest, sebk_target)

    # Average epistemic uncertainty value
    metric_dict['EU'] = torch.mean(eus).item()

    # Generate .txt file with results for ROC curve plotting
    print()
    print(f'Saving results? {save_results}')
    if save_results:
        values = [utils.convert_to_numpy(probs[:, 0]), utils.convert_to_numpy(malignant_target), \
            utils.convert_to_numpy(probs[:, 1]), utils.convert_to_numpy(sebk_target)]
        with open(roc_results_file, 'w') as file:
            for row in values:
                file.write(','.join([str(item) for item in row]))
                file.write('\n')

    # Display model results
    print()
    print(model_type)
    print("-" * 50)
    print('***** SEBORRHEIC KERATOSIS *****')
    for key in METRIC_KEYS[:4]:
        print(f"{key} = {metric_dict[key]:.4f}")
    print('********** MALIGNANT ***********')
    for key in METRIC_KEYS[4:8]:
        print(f"{key} = {metric_dict[key]:.4f}")
    print('*********** AVERAGE ************')
    for key in METRIC_KEYS[8:]:
        print(f"{key} = {metric_dict[key]:.4f}")
    print("-" * 50)

    # Save results to .json file
    if save_results:
        with open(model_results_file, 'w') as fp:
            json.dump(metric_dict, fp)
