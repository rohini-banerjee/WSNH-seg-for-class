import os
import argparse
import dataset
import metrics
import numpy as np
import statistics
import time
import torch
import utils

from classifiers import ModXception
from segmenter import EarlyStopping
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.optim import Adam
from tqdm import tqdm


def train_single_classifier(train_loader, val_loader, model_type, model, swa_model, swa_scheduler, criterion, metric, \
    optimizer, num_epochs, device, model_save_path, patience, save_results=True):
    """
    Train single classification model with arbitrary architecture.
    """
    
    start_time = time.time()
    swa_start = int(num_epochs * 0.75)

    print(f'Saving model to {model_save_path}')

    if save_results:
        save_filename = utils.RESULTS_DIR + f'{model_type}_{utils.MOD_NAMES[model.get_model_name()]}_train_results.txt'
        print(f'Saving training data to {save_filename}')
    
    # Invoke early stopping
    es_handler = EarlyStopping(save_pathname=model_save_path)
    print('Performing early stopping...')
    print(f'Stochastic Average Weighting starts at epoch {swa_start}')
    
    utils.print_line()

    finished_early_stopping = False
    train_running_losses, train_running_metric = [], []
    val_running_losses, val_running_metric = [], []
    for epoch in range(num_epochs):
        # Running losses
        train_loss = 0
        val_loss = 0
        train_metric = 0
        val_metric = 0
        phases = ['train', 'val']
        with torch.autograd.set_detect_anomaly(True):
            for phase in phases:
                # Choose the dataloader
                dataloader = train_loader if phase == 'train' else val_loader
                # Set the mode
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
                        b_y_hat, _ = model(b_x)
                        b_y_pred = torch.argmax(b_y_hat, dim=1)

                        # Compute the loss value
                        loss = criterion(y_pred=b_y_hat, y_true=b_y)
                        loss_value = loss.detach().item()

                        # Compute the metric score
                        score_value = metric(b_y_pred, b_y)
                        
                        if phase == 'train':
                            # Backward pass
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_loss += loss_value / len(dataloader)
                            train_metric += score_value / len(dataloader)

                            if epoch > swa_start:
                                swa_model.update_parameters(model)
                                swa_scheduler.step()
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
        print(f"Train Avg. Accuracy= {train_metric:.6f}")
        print(f"Val Avg. Accuracy= {val_metric:.6f}")
        print("-" * 50)

        train_running_losses.append(train_loss)
        train_running_metric.append(train_metric)
        val_running_losses.append(val_loss)
        val_running_metric.append(val_metric)
        
        # Check for early stopping
        if not finished_early_stopping:
            es_handler(val_loss, epoch+1, model)
            if es_handler.terminate:
                es_handler.save_best_model(utils.MOD_NAMES[model.get_model_name()])
                finished_early_stopping = True

        # Save model checkpoint
        if (epoch+1) % 50 == 0 or ((epoch + 1) >= patience and (epoch + 1) % 10 == 0):
            if epoch > swa_start:
                torch.save(swa_model.state_dict(), model_save_path.replace('model.pth', f'swa_model_e{epoch + 1}.pth'))
            else:
                torch.save(model.state_dict(), model_save_path.replace('model.pth', f'model_e{epoch + 1}.pth'))

    end_time = time.time()
    print("--- All epochs took %s seconds ---" % (end_time - start_time))

    # Save trained model
    torch.save(model.state_dict(), model_save_path)

    if save_results:
        values = [train_running_losses, val_running_losses, train_running_metric, val_running_metric]
        # Save metrics to file
        with open(save_filename, 'w') as file:
            for row in values:
                file.write(','.join([str(item) for item in row]))
                file.write('\n')

def main(args):
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Correct types
    args.pin_memory = bool(args.pin_memory)
    args.device_num = int(args.device_num)

    # Set the device
    device = utils.choose_device(args.device, args.device_num)
    if str(device) == 'cuda':
        print('Setting CUDA Device Node')
    print(f"Running on {device}")

    # Construct dataloaders for training
    train_loader, val_loader = dataset.get_training_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
        model_type=args.model_type,
        device=device,
    )

    # Create model and save path. Subsitute line below for any desired architecture.
    model = ModXception('legacy_xception', True).to(device)
    model_save_path = utils.WEIGHTS_DIR+f'ind_{args.model_type}_{utils.MOD_NAMES[model.get_model_name()]}_model.pth'

    # Define optimizer and Stochastic Weight Averaging (SWA) model
    optimizer = Adam(params=model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
    swa_model = AveragedModel(model).to(device)
    swa_scheduler = SWALR(optimizer, anneal_strategy='cos', anneal_epochs=5, swa_lr=0.01)

    # Define loss and metric of evaluation
    criterion = metrics.get_ce
    metric = metrics.get_multi_accuracy

    # Train model
    train_single_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        model_type=args.model_type,
        model=model,
        swa_model=swa_model,
        swa_scheduler=swa_scheduler,
        criterion=criterion,
        metric=metric,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        model_save_path=model_save_path,
        patience=args.patience,
        save_results=True,
    )

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for classification training.')

    parser.add_argument('model_type', type=str, help='Model type.')

    # Image parameters
    parser.add_argument('--image_size', type=int, default=utils.CLS_SIZE[0], help='The size of the input image.')
    parser.add_argument('--channels', type=int, default=3, help='The number of channels of the input image.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta 1.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta 2.')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon.')
    parser.add_argument('--patience', type=int, default=150, help='Epochs prior to initiating SWA replacement.')

    # System parameters
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for the data loader.')
    parser.add_argument('--pin_memory', type=int, default=1, help='Pin memory for the data loader.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training.')
    parser.add_argument('--device_num', type=str, default='2', help='Device number.')

    # Seed parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    args = parser.parse_args()
    
    main(args)
