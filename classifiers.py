# -----------------------------------------------------------------------------
# Define architectures for lesion classification.
# -----------------------------------------------------------------------------
import dataset
import metrics
import numpy as np
import timm
import torch
import torch.nn as nn
import utils

from torch.optim.swa_utils import update_bn


class GenericClassifier(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(GenericClassifier, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained)

    def get_model_name(self):
        """
        Return model name.
        """
        return self.model_name

    def forward(self, x):
        """
        Forward pass of the model.
        """
        logits = self.model(x)
        predictive_entropy = metrics.get_predictive_entropy(logits, reduction=None)
        return logits, predictive_entropy
    
    def freeze_all_layers(self):
        """
        Freeze all layers of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_fc_layers(self):
        """
        Unfreeze fully connected layers.
        """
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def check_trainable_params(self):
        """
        Prints only trainable layers in specified model.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer {name} is trainable")


class ModXception(GenericClassifier):
    def __init__(self, model_name, pretrained=True):
        """
        Xception model.
        """
        super().__init__(model_name, pretrained)
        n_features, n_classes = 2048, len(utils.LABEL_TO_IND)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(n_features, n_classes)
        )

    def freeze_early_layers(self, exclude_i):
        """
        Freeze layers in Xception blocks up to specificied block number.
        """
        freeze_til = f'block{exclude_i}'
        for name, param in self.model.named_parameters():
            if freeze_til in name:
                break
            param.requires_grad = False


class ModViT(GenericClassifier):
    def __init__(self, model_name, pretrained=True):
        """
        Vision transformer model.
        """
        super().__init__(model_name, pretrained)
        self.model.head_drop = nn.Dropout(0.4)
        n_features, n_classes = 768, len(utils.LABEL_TO_IND)
        self.model.head = nn.Linear(n_features, n_classes)
    
    def unfreeze_fc_layers(self):
        """
        Unfreeze fully connected layers.
        """
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def unfreeze_attn_ffn_layers(self):
        """
        Unfreeze attention and feed-forward layers.
        """
        for block in self.model.blocks:
            # Unfreeze multi-headed attention
            for param in block.attn.parameters():
                param.requires_grad = True

            # Unfreeze feed-forward layers
            for param in block.mlp.parameters():
                param.requires_grad = True


def update_bn_statistics(train_loader, ensemble, indices, batch_size, device):
    """
    Update BN stats for Xception classifiers only.
    """
    for _, (inputs, labels, _) in enumerate(train_loader):
        for i in indices:
            batch_dataloader = dataset.data_loader_for_bn_update(inputs[:, i, :, :, :].to(device), labels.to(device), batch_size)
            update_bn(batch_dataloader, ensemble[i])
