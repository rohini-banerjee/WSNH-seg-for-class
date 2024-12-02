import cv2
import dataset
import glob
import label_images
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import metrics
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch
import utils

import torch.nn.functional as F

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from segmenter import UNetSegmenter
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve


def plot_augmentations(id, img_path, mask_path, save_path=None):
    fig, axes = plt.subplots(1, len(utils.AUGMENTORS)+1)
    plt.subplots_adjust(wspace=0.0)

    img = Image.open(str(img_path))
    mask = Image.open(str(mask_path))
    img = utils.RESIZE(img)
    mask = utils.RESIZE(mask)

    axes[0].imshow(img)
    axes[0].axis("off")
    for j in range(len(utils.AUGMENTORS)):
        axes[j+1].imshow(utils.AUGMENTORS[j][0](img))
        axes[j+1].axis("off")
    
    if save_path is None:
        save_path = f'{id}_data_augmentations.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_prediction(id, image, gt, p_map, eu_map, au_map, save_path=None):
    fig, axes = plt.subplots(1, 5, figsize=(14,4), sharey=True)

    # Input image
    axes[0].imshow(image)
    axes[0].set_title('Dermascopy image', fontsize=14)

    # Ground truth
    axes[1].imshow(gt)
    axes[1].set_title('Ground truth', fontsize=14)

    # Mask prediction map
    pred = axes[2].imshow(p_map)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pred, cax=cax)
    axes[2].set_title('Prediction', fontsize=14)

    # Epistemic uncertainty map
    eu = axes[3].imshow(eu_map, cmap='plasma')
    axes[3].set_title('Epistemic Uncertainty', fontsize=14)
    divider1 = make_axes_locatable(axes[3])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(eu, cax=cax1)

    # Aleatoric uncertainty map
    au = axes[4].imshow(au_map, cmap='plasma')
    axes[4].set_title('Aleatoric Uncertainty', fontsize=14)
    divider2 = make_axes_locatable(axes[4])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(au, cax=cax2)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save_path is None:
        save_path = f'{id}_segmentation_result.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_loaded_classif_data(id, resized_img, resized_roi, resized_eu, resized_ks, resized_kl, save_path=None):
    fig, ax = plt.subplots(1, 5)
    plt.subplots_adjust(wspace=0.02)

    ax[0].imshow(resized_img)
    ax[0].set_title('Image')
    ax[1].imshow(resized_roi)
    ax[1].set_title('ROI')
    ax[2].imshow(resized_eu)
    ax[2].set_title(r'ROI+$\mathcal{U}_{ES}$')
    ax[3].imshow(resized_ks)
    ax[3].set_title(r'$\mathcal{K}^*_S$+$\mathcal{U}_{ES}$')
    ax[4].imshow(resized_kl)
    ax[4].set_title(r'$\mathcal{K}^*_L$+$\mathcal{U}_{ES}$')

    for a in ax:
        a.axis("off")

    if save_path is None:
        save_path = f'{id}_uq-enhanced.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
