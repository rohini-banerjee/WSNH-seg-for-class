# -----------------------------------------------------------------------------
# Functions for model inference.
# -----------------------------------------------------------------------------
import argparse
import dataset
import matplotlib.pyplot as plt
import metrics
import numpy as np
import plotting
import torch
import utils

from classifiers import *
from metrics import compute_metrics
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from segmenter import UNetSegmenter
from torch.nn import functional as F
from tqdm import tqdm


def evaluate_segmentation(model, model_type, ensemble, dataloader, device):
    """
    Evaluate single segmenter OR segmentation ensemble.

    Args:
        model (torch.nn.Module): Trained classifier.

        model_type (str): Model type.

        ensemble (List[segmenter.UNetSegmenter]): List of segmentation models.

        dataloader (torch.utils.data.dataloader.DataLoader): Dataloader.

        device (torch.device): Selected device.

    Returns:
        None.
    """
    test_val = 0.0
    with tqdm(dataloader, desc='Testing', ascii=' >=') as pbar:
        with torch.no_grad():
            for b_x, b_y, _ in pbar:
                # Move to device
                b_x = b_x.type(torch.FloatTensor).to(device)
                b_y = b_y.round().type(torch.LongTensor).to(device)

                # Forward pass
                if model_type == 'unet':
                    b_y_hat = model(b_x)
                elif model_type == 'mcunet':
                    _, b_y_logits, _, _ = model(b_x)
                    b_y_hat = F.sigmoid(b_y_logits)
                else:
                    inferences = []
                    for i in range(len(ensemble)):
                        mod = ensemble[i]
                        inferences.append(mod(b_x))
                    # Concatenate network predictions, has shape (B, N, H, W)
                    b_x_prime = torch.cat(inferences, dim=1)
                    b_y_hat = torch.mean(b_x_prime, dim=1)

                image_val = metrics.get_dice_score(y_pred=b_y_hat, y_true=b_y).detach().item()
                test_val += image_val / len(dataloader)
        
                # Update the progress bar
                pbar.set_postfix(test_val=f"{image_val:.4f}")
                pbar.update()
    
    print("-" * 50)
    print(f"Test DSC Value = {test_val:.4f}")
    print("-" * 50)

def generate_grad_cam(model, inp, target_layers, target):
    """
    Overlay gradCAM saliency map on target over input image.

    Args:
        model (torch.nn.Module): Trained classifier.

        inp (torch.Tensor): Input image as preprocessed tensor.

        target_layers (List[torch.nn.Module]): List of target layers
        for gradient computation.

        target (List[int]): List of target classes for GradCAM maps.

    Returns:
        plot_img (np.ndarray): Input image in RGB color space.

        visualization (np.ndarray): GradCAM saliency map overlaid
        over plot_img.
    """
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=inp, targets=target, aug_smooth=True)
        grayscale_cam = grayscale_cam[0]
        plot_img = utils.clean_tensor(inp, is_img=True)
        visualization = show_cam_on_image(np.clip(plot_img, 0, 1), grayscale_cam, use_rgb=True)
    return plot_img, visualization

def extract_grad_cam_repr(test_loaders, all_models, device, save_path, plot_idx=0):
    """
    Visualize GradCAM saliency map on specified test image.

    Args:
        test_loaders (List[torch.utils.data.dataloader.DataLoader]): List
        of dataloaders for classifiers.

        all_models (List[torch.nn.Module]): List of trained classifiers.

        device (torch.device): Selected device.

        save_path (str): List to save GradCAM visualization plot.

        plot_idx (int): Index of image to visualize.
    
    Returns:
        None.
    """
    all_results = []
    plot_img_name = None
    for i, test_loader in enumerate(test_loaders):
        # Choose appropriate target layers for GradCAM. For example, we select the
        # last convolution layer for the Xception architecture.
        target_layers = [all_models[i].model.conv4]
        for idx, (b_x, b_y, id) in enumerate(test_loader):
            # Move to device
            b_x = b_x.type(torch.FloatTensor).to(device)
            b_y = b_y.round().type(torch.LongTensor).to(device)

            # Select target. If set to None, will produce map for the highest scoring class.
            # Otherwise, will produce map for specified class(es).
            target = [ClassifierOutputTarget(b_y.item())]
            plot_img, gradcam_result = generate_grad_cam(all_models[i], b_x, target_layers, target)
            if idx == plot_idx:
                plot_img_name = id[0]
                all_results.append((plot_img, gradcam_result))

            # Forward pass
            b_y_logits = all_models[i](b_x)
            b_y_hat = F.softmax(b_y_logits, dim=-1)
            b_y_pred = torch.argmax(b_y_hat, dim=-1)
        
    plotting.plot_gradCAM_results(plot_img_name, all_results, save_path)
    
def evaluate_single_classifier(test_loader, model, model_type, device, save_results, \
    roc_results_file, model_results_file):
    """
    Evaluate single classifier.

    Args:
        test_loader (torch.utils.data.dataloader.DataLoader): Testing dataloader.

        model (torch.nn.Module): Trained classifier.

        model_type (str): Model type.

        device (torch.device): Selected device.

        save_results (bool): Whether to save evaluation metrics.

        roc_results_file (str): Pathname to store results for ROC curves (.txt).

        model_results_file (str): Pathname to store evaluation results (.json).
    
    Returns:
        None.
    """
    # Create empty dictionary to store results
    metric_dict = dict()

    # Perform single pass of the dataset
    y_probs, targets, pes = [], [], []
    with tqdm(test_loader, desc='Testing', ascii=' >=') as pbar:
        with torch.no_grad():
            for b_x, b_y, id in pbar:
                # Move to device
                b_x = b_x.type(torch.FloatTensor).to(device)
                b_y = b_y.round().type(torch.LongTensor).to(device)

                # Inference
                b_y_logits, predictive_entropy = model(b_x)
                b_y_hat = F.softmax(b_y_logits, dim=-1) 
                b_y_pred = torch.argmax(b_y_hat, dim=-1)

                y_probs.append(b_y_hat)
                targets.append(b_y)
                pes.append(predictive_entropy)

                # Update the progress bar
                pbar.set_postfix()
                pbar.update()

    # Format prediction results for evaluation
    y_probs = torch.cat(y_probs, dim=0)
    y_preds = torch.argmax(y_probs, dim=1)
    targets = torch.cat(targets, dim=0)
    pes = torch.cat(pes, dim=0)

    # Compute evaluation metrics
    compute_metrics(metric_dict, model_type, y_probs, y_preds, pes, targets, save_results, roc_results_file, model_results_file)

def evaluate_independent_ensemble(test_loaders, ensemble, device, save_results, \
    roc_results_file, model_results_file):
    """
    Evaluate expert ensemble.

    Args:
        test_loaders (List[torch.utils.data.dataloader.DataLoader]): List of
        dataloaders for ensemble members.

        ensemble (List[torch.nn.Module]): List of classifiers.

        model_type (str): Model type.

        device (torch.device): Selected device.

        save_results (bool): Whether to save evaluation metrics.

        roc_results_file (str): Pathname to store results for ROC curves (.txt).

        model_results_file (str): Pathname to store evaluation results (.json).
    
    Returns:
        None.
    """
    # Create empty dictionary to store results
    metric_dict = dict()

    # Iterate through each model's respective dataloader
    ensemble_probs, all_logits = [], []
    for i, test_loader in enumerate(test_loaders):
        y_logits, pes, y_probs, targets = [], [], [], []

        # Perform single pass of the dataset
        with tqdm(test_loader, desc='Testing', ascii=' >=') as pbar:
            with torch.no_grad():
                for b_x, b_y, id in pbar:
                    # Move to device
                    b_x = b_x.type(torch.FloatTensor).to(device)
                    b_y = b_y.round().type(torch.LongTensor).to(device)

                    # Inference
                    b_y_logits, predictive_entropy = ensemble[i](b_x)
                    b_y_hat = F.softmax(b_y_logits, dim=-1)
                    b_y_pred = torch.argmax(b_y_hat, dim=-1)

                    pes.append(predictive_entropy)
                    y_logits.append(b_y_logits)
                    y_probs.append(b_y_hat)
                    targets.append(b_y)

                    # Update the progress bar
                    pbar.set_postfix()
                    pbar.update()
        
        # Store results
        pes = torch.cat(pes, dim=0)
        y_logits = torch.cat(y_logits, dim=0)
        y_probs = torch.cat(y_probs, dim=0)
        targets = torch.cat(targets, dim=0)

        all_logits.append(y_logits.unsqueeze(dim=0))
        ensemble_probs.append(y_probs)

    # Aggregate all model predictions together and format results for evaluation
    aggregated_probs = torch.stack(ensemble_probs).mean(dim=0)
    aggregated_preds = torch.argmax(aggregated_probs, dim=1)
    all_logits = torch.cat(all_logits, dim=0)
    aggregated_entropies = metrics.get_predictive_entropy(all_logits, reduction=None, dim=1)

    # Compute evaluation metrics
    metrics.compute_metrics(metric_dict, 'ind-ensemble', aggregated_probs, aggregated_preds, aggregated_entropies, targets, save_results, roc_results_file, model_results_file)


def main(args):
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Correct types
    args.pin_memory = bool(args.pin_memory)
    args.save_results = bool(args.save_results)

    # Set the device
    device = utils.choose_device(args.device, args.device_num)
    if str(device) == 'cuda':
        print('Setting CUDA Device Node')
    print(f"Running on {device}")
    
    ####   SINGLE CLASSIFIER   ####
    # Create dataloader for single model
    test_loader = dataset.get_testing_dataloader(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        batch_size=args.batch_size,
        model_type=args.model_type,
        device=device,
    )

    # Modify 'model_save_path' in commandline with trained classifier pathname
    trained_model = ModXception('legacy_xception', True).to(device)
    utils.load_model(trained_model, args.model_save_path, device)

    # Test classification model
    # evaluate_single_classifier(
    #     test_loader=test_loader,
    #     model=trained_model,
    #     model_type=args.model_type,
    #     device=device,
    #     save_results=args.save_results,
    #     roc_results_file=utils.RESULTS_DIR+f'{args.model_type}_roc_values.txt',
    #     model_results_file=utils.RESULTS_DIR+f'{args.model_type}_metrics.json',
    # )
    ###############################

    ####    EXPERT ENSEMBLE    ####
    # Load in expert ensemble of trained classifiers
    ensemble = utils.load_cohort(device)

    loaders = []
    for mtype in ['roi-cls', 'roi-uq-cls', 'kl-uq-cls', 'ks-uq-cls']:
        test_loader = dataset.get_testing_dataloader(
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            batch_size=args.batch_size,
            model_type=mtype,
            device=device,
        )
        loaders.append(test_loader)

    # Test expert ensemble of classifiers
    # evaluate_independent_ensemble(
    #     test_loaders=loaders,
    #     ensemble=ensemble,
    #     device=device,
    #     save_results=args.save_results,
    #     roc_results_file=utils.RESULTS_DIR+'ind-ensemble_roc_values.txt',
    #     model_results_file=utils.RESULTS_DIR+'ind-ensemble_metrics.json',
    # )
    ###############################
    
    ####      GradCAM VIS      ####
    # # Create dataloader for baseline classifier
    # baseline_loader = dataset.get_testing_dataloader(
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_memory,
    #     batch_size=args.batch_size,
    #     model_type='v-cls',
    #     device=device,
    # )

    # # Load in trained baseline classifier
    # path_to_baseline = utils.WEIGHTS_DIR + 'baseline_xception_model.pth'
    # baseline_model = ModXception('legacy_xception', True).to(device)
    # utils.load_model(baseline_model, path_to_baseline, device)

    # # To properly run gradCAM, modify forward pass of GenericClassifier module to
    # # only output logits.
    # extract_grad_cam_repr(
    #     test_loaders=([baseline_loader]+loaders),
    #     all_models=([baseline_model]+ensemble),
    #     device=device,
    #     save_path=(utils.PLOTS_DIR+f'ex_gradCAM.png'),
    #     plot_idx=0,
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for inference.')

    parser.add_argument('model_type', type=str, help='Model type.')
    parser.add_argument('model_save_path', type=str, help='Path to trained model weights.')

    # Image parameters
    parser.add_argument('--image_size', type=int, default=utils.CLS_SIZE[0], help='The size of the input image.')
    parser.add_argument('--channels', type=int, default=3, help='The number of channels of the input image.')

    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--save_results', type=int, default=0, help='Bool saving evaluation results.')

    # System parameters
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for the data loader.')
    parser.add_argument('--pin_memory', type=int, default=1, help='Pin memory for the data loader.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training.')
    parser.add_argument('--device_num', type=str, default='2', help='Device number.')

    # Seed parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    args = parser.parse_args()
    
    main(args)
