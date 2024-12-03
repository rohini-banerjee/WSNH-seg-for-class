# -----------------------------------------------------------------------------
# Define UNet, MCU-Net, and MSU-Net architectures for lesion image segmentation.
# -----------------------------------------------------------------------------
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import utils


class MonteCarloSegmenter(nn.Module):
    """
    A model that performs ensembled segmentation.
    """
    def __init__(self, in_channels=3, add_activation=True):
        """
        Initialize the model.
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type='scse',
            in_channels=in_channels,
            classes=1,
            activation="sigmoid" if add_activation else None,
        )
        # Add dropout to the model
        for i in range(len(self.model.decoder.blocks)):
            self.model.decoder.blocks[i].conv1.add_module('dropout', nn.Dropout(p=0.4, inplace=False))
            self.model.decoder.blocks[i].conv2.add_module('dropout', nn.Dropout(p=0.5, inplace=False))
    
    def forward(self, x, device='cuda', T=50):
        """
        Forward pass of the model.
        """
        # Turn on dropout for training and inference
        self.turn_on_dropout()
        
        segmentation_shape = list(x.shape)
        if len(segmentation_shape) == 4:
            segmentation_shape[1] = 1
        elif len(segmentation_shape) == 3:
            segmentation_shape[0] = 1
        else:
            raise ValueError(f"Invalid shape. Expected shape of 3 or 4 dimensions. Go {segmentation_shape}")

        # Placeholder to collect the ensemble
        ensemble = torch.zeros([T] + segmentation_shape).to(x.device)

        # Collects all ensembles
        for i in range(T):
            ensemble[i] = self.model(x)
        
        # Finds mean pred across ensemble
        p_bar = torch.mean(ensemble, dim=0)

        # Epistemic uncertainty
        epistemic = torch.mean(ensemble**2, dim=0) - torch.mean(ensemble, dim=0)**2

        # Aleatoric uncertainty
        aleatoric = torch.mean(ensemble*(1-ensemble), dim=0)
        
        return ensemble, p_bar, epistemic, aleatoric
    
    def turn_on_dropout(self, p=0.4):
        """
        Turn on dropout.
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                module.p = p


class UNetSegmenter(nn.Module):
    """
    A U-Net model that performs segmentation.
    """
    def __init__(self):
        """
        Initialize the model.
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type='scse',
            in_channels=3,
            classes=1,
            activation="sigmoid"
        )

    def forward(self, x):
        """
        Forward pass of the model.
        """
        return self.model(x)


class EarlyStopping:
    """
    Performs early stopping during model training using validation loss.
    Inspired by PyTorch Ignite handler for early stopping:
    https://pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html#EarlyStopping.
    """
    def __init__(self, save_pathname, patience=10, delta=0, model_num=None):
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")
        if delta < 0.0:
            raise ValueError("Argument delta should not be a negative number.")

        self.patience = patience
        self.min_delta = delta
        self.best_score = None
        self.terminate = False
        self.best_model_state = None
        self.counter = 0
        self.final_epoch = 0
        self.original_pathname = save_pathname
        self.es_pathname = None
        self.model_num = model_num

    def __call__(self, val_loss, epoch, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.terminate = True
                self.final_epoch = epoch-self.patience
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def save_best_model(self, model_name):
        """
        Save the early-stopped model with the current epoch as
        part of its filename.
        """
        if self.model_num is not None:
            new_fname = utils.WEIGHTS_DIR + f'ES{self.final_epoch}_{model_name}{self.model_num}_model.pth'
        else:
            fname_ext = self.original_pathname.split('/')[-1]
            new_fname = utils.WEIGHTS_DIR + f'ES{self.final_epoch}_' + fname_ext
        self.es_pathname = new_fname
        torch.save(self.best_model_state, self.es_pathname)
