import argparse
import dataset
import metrics
import numpy as np
import os
import plotting
import test
import time
import torch
import torch.nn as nn
import utils

from segmenter import MonteCarloSegmenter, UNetSegmenter, EarlyStopping
from torch.optim import Adam
from test import evaluate_segmentation
from tqdm import tqdm


def train(train_loader, val_loader, model, model_type, criterion, metric, optimizer, num_epochs, \
          device, model_save_path, model_save_freq, boot_seed=None, save_results=True, \
          early_stopping=False):
    """
    Train the U-Net model.
    """
    start_time, es_time = time.time(), None

    # Save training results
    if save_results:
        if model_type == 'unet':
            save_file = utils.RESULTS_DIR + f'bunet{boot_seed}_train_results.txt' if boot_seed else \
                utils.RESULTS_DIR + 'unet_train_results.txt'
        else:
            save_file = utils.RESULTS_DIR + f'mcunet_train_results.txt'
        print(f'Save training data to: {save_file}')
    
    # Invoke early stopping
    if early_stopping:
        es_handler = EarlyStopping(save_pathname=model_save_path)
        print('Performing early stopping...')

    utils.print_line()

    train_running_losses, train_running_dice = [], []
    val_running_losses, val_running_dice = [], []

    for epoch in range(num_epochs):
        # Running losses
        train_loss = 0.0
        val_loss = 0.0
        train_metric = 0.0
        val_metric = 0.0
        phases = ['train', 'val']
        with torch.autograd.set_detect_anomaly(True):
            for phase in phases:
                # Choose the dataloader
                dataloader = train_loader if phase == 'train' else val_loader
                # Set the model mode
                model.train() if phase == 'train' else model.eval()
                # Set if gradients are required to be computed
                torch.set_grad_enabled(phase == 'train')
                # Perform one pass over the dataset
                with tqdm(dataloader, desc=f'Epoch: {epoch + 1}/{num_epochs} - Phase {phase}', ascii=' >=') as pbar:
                    for b_x, b_y, _ in pbar:
                        # Move to device
                        b_x = b_x.type(torch.FloatTensor).to(device)
                        b_y = b_y.round().type(torch.LongTensor).to(device)

                        # Forward pass
                        if model_type == 'unet':
                            b_y_hat = model(b_x)
                        else:
                            _, b_y_logits, _, _ = model(b_x)
                            b_y_hat = torch.nn.functional.sigmoid(b_y_logits)

                        # Compute the loss value
                        loss = criterion(y_pred=b_y_hat, y_true=b_y, is_logits=False)
                        loss_value = loss.detach().item()

                        # Compute the metric score
                        score = metric(y_pred=b_y_hat, y_true=b_y)
                        score_value = score.detach().item()
                        
                        if phase == 'train':
                            # Backward pass
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_loss += loss_value / len(dataloader)
                            train_metric += score_value / len(dataloader)
                        else:
                            val_loss += loss_value / len(dataloader)
                            val_metric += score_value / len(dataloader)

                        # Update the progress bar
                        pbar.set_postfix(train_loss=f"{loss_value:.6f}") if phase == 'train' else pbar.set_postfix(val_loss=f"{loss_value:.6f}")
                        pbar.update()

        print("-" * 50)
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        print(f"Train Loss = {train_loss:.6f}")
        print(f"Val Loss = {val_loss:.6f}")
        print(f"Train DSC = {train_metric:.6f}")
        print(f"Val DSC = {val_metric:.6f}")
        print("-" * 50)

        train_running_losses.append(train_loss)
        train_running_dice.append(train_metric)
        val_running_losses.append(val_loss)
        val_running_dice.append(val_metric)

        if early_stopping:
            es_handler(val_loss, epoch+1, model)
            if es_handler.is_terminated():
                es_handler.save_best_model()
                print(f"Early stopping at epoch: {epoch + 1}")
                es_time = time.time()
                early_stopping = False

    # Save metrics to file
    if save_results:
        values = [train_running_losses, val_running_losses, train_running_dice, val_running_dice]
        with open(save_file, 'w') as file:
            for row in values:
                file.write(','.join([str(item) for item in row]))
                file.write('\n')
    
    # Store the best model
    if early_stopping and es_time is not None:
        print("--- Early stopping took %s seconds ---" % (es_time - start_time))
        return es_handler.es_pathname

    end_time = time.time()
    print("--- All epochs took %s seconds ---" % (end_time - start_time))
    
    # Save the model
    torch.save(model.state_dict(), model_save_path)
    return model_save_path


def main(args):
    print(args)

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Correct types
    args.pin_memory = bool(args.pin_memory)
    args.perform_es = bool(args.perform_es)

    # Set the device
    device = utils.choose_device(args.device, args.device_num)
    if str(device) == 'cuda':
        print('Setting CUDA Device Node')
    print(f"Running on {device}")
    utils.print_line()

    # Load in training and validation dataloaders
    train_loader, val_loader = dataset.get_training_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
        model_type=args.model_type,
        device=device,
        boot_seed=args.boot_seed,
    )

    # Create the model and define save path
    # If training MCU-Net, call "Segmenter", else call "UNetSegmenter"
    if args.model_type == 'mcunet':
        model = MonteCarloSegmenter(add_activation=False).to(device)
        model = nn.DataParallel(model)
        model_save_path = args.save_path + f'{args.model_type}_model.pth'
    else:
        model = UNetSegmenter().to(device)
        model_save_path = args.save_path + f'b{args.model_type}{args.boot_seed}_model.pth'
    print(f'Save model to: {model_save_path}')

    # Display number of additional trainable parameters in current architecture
    total_params = utils.numel(model, only_trainable=True)
    print(f"Number of trainable params: {total_params}")

    # Define metric of evaluation
    metric = metrics.get_dice_score
    # Define the loss
    if args.loss_crit == 'dice_bce':
        criterion = metrics.get_dice_bce
    else:
        criterion = metrics.get_dice_fl
    # Define optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)

    # Train the model
    trained_model_path = train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        model_type=args.model_type, 
        criterion=criterion,
        metric=metric,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        model_save_path=model_save_path,
        model_save_freq=args.save_freq,
        boot_seed=args.boot_seed,
        save_results=args.save_train_results,
        early_stopping=args.perform_es,
    )
    
    ########################################################################
    
    # Load single segmentation model
    if args.model_type == 'unet':
        trained_model_path = args.save_path + f'bunet{args.boot_seed}_model.pth'
        trained_model = UNetSegmenter().to(device)
    else:
        trained_model_path = args.save_path + f'{args.model_type}_model.pth'
        trained_model = MonteCarloSegmenter(add_activation=False).to(device)
    utils.load_model(trained_model, trained_model_path, device, rename_dict=(args.model_type=='mcunet'))
    
    # Load test dataloader
    test_loader = dataset.get_testing_dataloader(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        batch_size=1,
        model_type=args.model_type,
        device=device,
    )
    
    # You can specify the type of the model and pass in `trained_model`
    # as the model to evaluate a single segmentation model instead. Here
    # we demonstrate testing the full ensemble of segmentation models.
    evaluate_segmentation(
        model=None,
        model_type='ensemble',
        ensemble=utils.load_ensemble(device),
        dataloader=test_loader,
        device=device,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for segmentation training.')

    # REQUIRED: specify model type
    parser.add_argument('model_type', type=str, choices=['unet', 'mcunet'], help='Model type for training.')

    # Image parameters
    parser.add_argument('--image_size', type=int, default=512, help='The size of the input image.')
    parser.add_argument('--channels', type=int, default=3, help='The number of channels of the input image.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs.')
    parser.add_argument('--boot_seed', type=int, default=42, help='Random seed for bootstrapping data.')

    # Optimization parameters
    parser.add_argument('--loss_crit', type=str, default='dice_bce', help='Combined loss criterion.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta 1.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta 2.')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon.')

    # System parameters
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for the data loader.')
    parser.add_argument('--pin_memory', type=int, default=1, help='Pin memory for the data loader.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training.')
    parser.add_argument('--device_num', type=str, default='2', help='Device number.')

    # Save parameters
    parser.add_argument('--perform_es', type=int, default=1, help='Early stopping during training.')
    parser.add_argument('--save_train_results', type=int, default=1, help='Whether to save training results.')
    parser.add_argument('--save_path', type=str, default=utils.WEIGHTS_DIR, help='Directory to save the model.')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency of epochs to save the model.')

    # Seed parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    args = parser.parse_args()
    
    main(args)
    