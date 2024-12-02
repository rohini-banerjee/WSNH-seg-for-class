# -----------------------------------------------------------------------------
# Loading, preprocesssing, and augmentation of relevant datasets.
# NOTE: You must have datasets stored as described in `collect_data` method.
# -----------------------------------------------------------------------------
import cv2
import glob
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotting
import random
import shutil
import torch
import torchvision
import urllib
import utils

from selection_agent import SelectionAgent
from label_images import generate_lookup_dict
from natsort import natsorted
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms


def data_loader_for_bn_update(b_x, b_y, B):
    """
    Construct mini-batch dataloaders to update batch norm statistics
    for SWA models.
    """
    dataset = TensorDataset(b_x, b_y)
    return DataLoader(dataset, batch_size=B, shuffle=False)
    
def sort_by_diagnosis(dir_name, lookup_filename, phase="train"):
    """
    Given a directory of all images for a certain phase, re-organize
    images in sub-directories by diagnosis. Parent directory must
    be named 'images' for consistency.

    Args:
        dir_name (str): Directory containing images for specified phase.

        lookup_filename (str): Pathname to dictionary mapping image IDs
        to corresponding gold standard lesion diagnoses.

        phase (str): Specified phase.

    Returns:
        None.
    """
    lookup_dict = generate_lookup_dict(lookup_filename)

    phase_matching = {
        "train": "./training/images/",
        "test": "./testing/images/",
        "vs1": "./validation/vs1/images/",
        "vs2": "./validation/vs2/images/",
    }

    # Create directory to store images by diagnosis if it does not
    # already exist
    labels = ['malignant/', 'seb_keratosis/', 'benign_nevi/']
    label_matching = {
        'M': labels[0],
        'SK': labels[1],
        'BN': labels[2],
    }

    for label in labels:
        parent_dir = phase_matching[phase]
        sub_dir = parent_dir + label
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

    # Annotate images in specified directory
    for filename in os.listdir(dir_name):
        f = os.path.join(dir_name, filename)
        if os.path.isfile(f) and utils.is_valid_image_file(f):
            # Lookup diagnosis label
            filepath = f.split('.')[-2]
            image_name = filepath.split('/')[-1]
            label = lookup_dict[image_name]

            new_f = phase_matching[phase] + label_matching[label] + filename
            shutil.move(f, new_f)

########################################################################

def boostrap_sample(images, labels, seed=10):
    """
    Naive Efron-type bootstrap. Random selection with replacement
    to generate bootstrapped sample of size n from data set of size n.

    Args:
        images (List[np.ndarray]): List of images.

        labels (List[np.ndarray]): List of segmentation masks.

        seed (int): Seed for reproducibility.
    
    Returns:
        new_imgs (List[np.ndarray]): Bootstrapped images.

        new_labs (List[np.ndarray]): Bootstrapped segmentation masks.
    """
    print(f'Seed {seed}.')
    random.seed(seed)

    zipped = list(zip(images, labels))
    bootstrapped = random.choices(zipped, k=len(images))
    new_imgs, new_labs = [], []
    for x, y in bootstrapped:
        new_imgs.append(x)
        new_labs.append(y)
    return new_imgs, new_labs

def sorted_data(dir_path, is_gt=False):
    """
    Get paths to sorted images.

    Args:
        dir_path (str): Directory containing input objects.

        is_gt (bool): Whether directory contains ground truth
        masks/labels.
    
    Returns:
        sortedPatients (List[pathlib.PosixPath]): List of Posix
        pathnames to sorted objects.
    """
    p = Path(dir_path)
    lstOfPatients = []
    sortedPatients = []

    # Ground truth masks are not stored in diagnosis-specific directories
    if is_gt:
        for i in p.iterdir():
            if utils.is_valid_image_file(i):
                    lstOfPatients.append(i)
    # Images are stored in sub-folders based on gold standard diagnoses
    else:
        for i in p.iterdir():
            if os.path.isdir(i):
                sub_dir = Path(i)
                # Only extract image files (.jpg) or (.png) file extensions
                for file in sub_dir.iterdir():
                    if utils.is_valid_image_file(file):
                        lstOfPatients.append(file)

    # Sort by basename, not absolute path
    sortedPatients = natsorted(lstOfPatients, key=lambda x: os.path.basename(x))
    return sortedPatients

def collect_data(type='train', bootstrapped=False, seed=42):
    """
    Return set of images and corresponding ground truth labels given data set
    type ('train', 'val', 'val1', 'val2', 'aug-val2', 'test'). Should only be
    internally called by corresponding dataloader.

    Args:
        type (str): Phase.

        bootstrapped (bool): Whether to bootstrap data.

        seed (int): Seed for reproducibility.
    
    Returns:
        images (List[np.ndarray]): List of sorted images for current phase.

        labels (List[np.ndarray]): List of sorted masks/labels for current phase.
    """
    images, labels = None, None

    # Check if valid data type is provided
    valid_data_types = ['train', 'vc-train', 'val', 'val1', 'val2', 'aug-val2', 'test']
    if type not in valid_data_types:
        raise ValueError(f"Data type {type} invalid. Valid options include {', '.join(valid_data_types)}.")

    if type == 'train':
        images = sorted_data('./training/images', is_gt=False)
        labels = sorted_data('./training/masks_gt', is_gt=True)
        
        # Bootstrap training samples if training U-Net models
        if bootstrapped:
            images, labels = boostrap_sample(images, labels, seed=seed)

    elif type == 'val1':
        images = sorted_data('./validation/vs1/images', is_gt=False)
        labels = sorted_data('./validation/vs1/masks_gt', is_gt=True)

    elif type == 'val2':
        images = sorted_data('./validation/vs2/images', is_gt=False)
        labels = sorted_data('./validation/vs2/masks_gt', is_gt=True)
    
    elif type == 'aug-val2':
        images = sorted_data('./validation/aug_vs2/images', is_gt=False)
        labels = sorted_data('./validation/aug_vs2/masks_gt', is_gt=True)

    else:
        images = sorted_data('./testing/images', is_gt=False)
        labels = sorted_data('./testing/masks_gt', is_gt=True)

    return images, labels

def get_dataset(type='train', bootstrapped=False, seed=42, model_type='unet', device='cuda'):
    """
    Construct dataset.

    Args:
        type (str): Phase.

        bootstrapped (bool): Whether to bootstrap data.
        
        seed (int): Seed for reproducibility.

        model_type (str): Model type.

        device (torch.device): Selected device.

    Returns:
        dataset (torch.utils.data.Dataset): Corresponding dataset.
    """
    images, labels = collect_data(type=type, bootstrapped=bootstrapped, seed=seed)
    if model_type in ['unet', 'msunet', 'mcunet']:
        return SegmentationDataset(images, labels, preprocessing=utils.PREPROCESSING_FN, augmentation=None)
    elif model_type == 'v-cls':
        return VanillaClassificationDataset(images, type, preprocessing=utils.PREPROCESSING_FN)
    elif model_type == 'dml-ensemble':
        return UQEnhancedClassificationDataset(images, type, device, preprocessing=utils.PREPROCESSING_UQ_CLS)
    return IndModelDataset(images, model_type, type, device, preprocessing=utils.PREPROCESSING_FN)

def get_training_dataloaders(batch_size, num_workers, pin_memory, shuffle, model_type='unet', device='cuda', boot_seed=42):
    """
    Generate dataloaders for each type of model supported (UNet, MCU-Net, MSU-Net, classif)
    for training.
        - U-Net: TR, VS1
        - MCU-Net, MSU-Net, classif: TR U VS2, VS1
    NOTE:
        - For MSU-Net invoke U-Net dataloaders to train candidate models, then invoke
        MSU-Net dataloaders to train Lvl-1 combiner model.
        - VS1 should ALWAYS be used as the held-out validation set for fair comparison
        across model types when performing hyperparameter optimization.

    Args:
        batch_size (int): Batch size.

        num_workers (int): Number of workers.

        pin_memory (bool): Whether to pin memory.

        shuffle (bool): Whether to shuffle the data.

        model_type (str): Type of model.

        device (torch.device): Selected device.

        boot_seed (int): Random seed for bootstrapping.

    Returns:
        train_loader, val_loader (List[torch.utils.data.dataloader.DataLoader]):
        Corresponding dataloaders for training and intermediate validation phases.
    """
    # Training
    bootstrapped = model_type=='unet'
    print(f'Boostrapping set to {bootstrapped}.')
    train_dataset = get_dataset(bootstrapped=bootstrapped, type='train', seed=boot_seed, model_type=model_type, device=device)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, \
        num_workers=num_workers, pin_memory=pin_memory)
    
    # Validation
    val1_dataset = get_dataset(bootstrapped=False, type='val1', model_type=model_type, device=device)
    val1_dl = DataLoader(val1_dataset, batch_size=batch_size, shuffle=shuffle, \
        num_workers=num_workers, pin_memory=pin_memory)

    # Combine training (TR) and validation 2 (VS2) datasets
    aug_val2_dataset = get_dataset(bootstrapped=False, type='aug-val2', model_type=model_type, device=device)
    union_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_val2_dataset])
    trainval2_dl = DataLoader(union_dataset, batch_size=batch_size, shuffle=shuffle, \
        num_workers=num_workers, pin_memory=pin_memory)

    if model_type == 'unet':
        return train_dl, val1_dl
    return trainval2_dl, val1_dl

def get_testing_dataloader(num_workers, pin_memory, batch_size=1, model_type='unet', device='cuda'):
    """
    Generate testing dataloaders for test-time model evaluation.

    Args:
        num_workers (int): Number of workers.

        pin_memory (bool): Whether to pin memory.

        batch_size (int): Batch size.

        model_type (str): Type of model.

        device (torch.device): Selected device.

    Returns:
        test_loader (torch.utils.data.dataloader.DataLoader): Corresponding testing dataloader.
    """
    test_dataset = get_dataset(bootstrapped=False, type='test', model_type=model_type, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, \
        num_workers=num_workers, pin_memory=pin_memory)
    return test_loader

def perform_augmentations(init_dir, augmented_dir, augmented_label_dict):
    """
    Perform and save deterministic augmentations for segmentation and
    classification datasets. Invokes horizontal/vertical flipping and various
    clockwise image rotations.

    Args:
        init_dir (str): Initial directory storing images and corresponding
        ground truth masks for current phase.

        augmented_dir (str): New directory storing augmented images and
        corresponding augmented ground truth masks for current phase.

        augmented_label_dict (str): Pathname to store dictionary mapping
        new augmented image names to corresponding gold standard diagnoses
        in (.csv) format.

    Returns:
        None.
    """
    images_dir = os.path.join(init_dir, 'images')
    masks_dir = os.path.join(init_dir, 'masks_gt')

    # Get (image, mask) pairs from input training directory
    images, masks = sorted_data(images_dir, is_gt=False), sorted_data(masks_dir, is_gt=True)

    # Create new images and masks_gt directories
    augmented_images = os.path.join(augmented_dir, 'images')
    augmented_masks = os.path.join(augmented_dir, 'masks_gt')
    for new_dir in [augmented_images, augmented_masks]:
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

    labeling_data = []
    for i, (f_img, f_mask) in enumerate(zip(images, masks)):
        # Check if there exists a subdirectory with corresponding diagnosis
        # If not, create it under aug_images parent directory
        diagnosis_dir_name = utils.get_image_diagnosis_from_path(f_img)
        full_parent_dir = os.path.join(augmented_images, diagnosis_dir_name)
        if not os.path.exists(full_parent_dir):
            os.mkdir(full_parent_dir)

        img = Image.open(str(f_img))
        mask = Image.open(str(f_mask))
        img = utils.RESIZE(img)
        mask = utils.RESIZE(mask)

        # Save original, non-augmented images to new directories
        new_img_path = os.path.join(full_parent_dir, utils.get_image_id(f_img) + '.jpg')
        new_mask_path = os.path.join(augmented_masks, utils.get_image_id(f_mask) + '.png')
        img.save(new_img_path)
        mask.save(new_mask_path)
        
        for augmentor, ext in utils.AUGMENTORS:
            # Perform augmentation
            aug_img = augmentor(img)
            aug_mask = augmentor(mask)

            # Save new filename
            new_img_path = os.path.join(full_parent_dir, utils.get_image_id(f_img) + ext + '.jpg')
            new_mask_path = os.path.join(augmented_masks, utils.get_image_id(f_mask) + ext + '.png')

            # Save augmented image and paired mask to new directory
            aug_img.save(new_img_path)
            aug_mask.save(new_mask_path)

            # Add entry to labels_dict file
            entry = [
                utils.get_image_id(new_img_path),
                utils.DIR_NAME_TO_LABEL[diagnosis_dir_name],
            ]
            labeling_data.append(entry)

    # Add entries for new augmentated images to labels_dict
    df = pd.DataFrame(labeling_data, columns=['image_id', 'value'])
    df.to_csv(augmented_label_dict, index=False)

########################################################################
class VanillaClassificationDataset(BaseDataset):
    """
    Dataset for vanilla (baseline) classification. Loads original images
    and ground truth labels with no enhancements.

    Args:
        images (RGB Images): Dataset images.

        type (str): Phase name.

        preprocessing (function): Data preprocessing transforms.
    """
    def __init__(self, images, type, preprocessing=None):
        self.images_fps = images
        self.preprocessing = preprocessing
        if type == 'test':
            self.lookup_dict = generate_lookup_dict(utils.RESULTS_DIR + '/aux/test_labels_dict.csv')
        else:
            self.lookup_dict = generate_lookup_dict(utils.RESULTS_DIR + '/aux/all_aug_train_val_labels_dict.csv')
    
    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, i):
        # Read in data
        pathname = str(self.images_fps[i])
        image_id = utils.get_image_id(pathname)

        img = cv2.imread(pathname)
        if img is None:
            raise ValueError(f'Missing image: {image_id}')
        if img.shape[:2] != utils.CLS_SIZE:
            img = utils.resize_img(img, down_size=utils.CLS_SIZE)
 
        # Apply preprocessing
        if self.preprocessing:
            img = self.preprocessing(img)

        # Extract corresponding label
        int_label = utils.get_int_label(self.lookup_dict, image_id)

        return img, int_label, image_id


class UQEnhancedClassificationDataset(BaseDataset):
    """
    Dataset for classification with an expert ensemble. Loads images
    and ground truth labels, and performs UQ-driven enhancements to
    generate appropriate data for ensemble members.

    Args:
        images (RGB Images): Dataset images.

        type (str): Phase name.

        device (torch.device): Selected device.

        preprocessing (function): Data preprocessing transforms.
    """
    def __init__(self, images, type, device, preprocessing=None):
        self.device = device
        self.images_fps = images
        self.instructor = SelectionAgent(gamma=0.75)
        self.preprocessing = preprocessing
        self.ids = []
        self.img_rois = []
        self.uq_rois = []
        self.uq_ks = []
        self.uq_kl = []
        if type == 'test':
            self.lookup_dict = generate_lookup_dict(utils.RESULTS_DIR + '/aux/test_labels_dict.csv')
        else:
            self.lookup_dict = generate_lookup_dict(utils.RESULTS_DIR + '/aux/all_aug_train_val_labels_dict.csv')

        # Load in segmentation ensemble
        self.ensemble = utils.load_ensemble(self.device)

        for i in range(len(self.images_fps)):
            image_id = utils.get_image_id(str(self.images_fps[i]))
            self.ids.append(image_id)

            # Read in data
            img = cv2.imread(str(self.images_fps[i]))
            if img is None:
                raise ValueError(f'Missing image: {image_id}')
            
            # Perform inference
            inferences = []
            for j in range(len(self.ensemble)):
                mod = self.ensemble[j]
                tensor_img = utils.load_tensor(img, device)
                inferences.append(mod(tensor_img))
            ensemble_preds = torch.cat(inferences, dim=1)
            _, prob_map, epistemic, aleatoric = utils.ensemble_infer(ensemble_preds)

            # Clean results
            img = utils.clean_tensor(tensor_img, is_img=True)
            p_map = utils.clean_tensor(prob_map)
            eu_map = utils.clean_tensor(epistemic)
            au_map = utils.clean_tensor(aleatoric)

            # Run Selection Agent
            roi_img, eu_img, best_ks_kernel, best_kl_kernel = self.instructor.run_pipeline(img, p_map, eu_map, au_map)

            # Store ROI image
            resized_roi = utils.resize_img(roi_img, down_size=utils.CLS_SIZE)
            self.img_rois.append(resized_roi)

            # Store UQ-enhanced ROI image
            resized_eu = utils.resize_img(eu_img, down_size=utils.CLS_SIZE)
            self.uq_rois.append(resized_eu)

            # Store UQ-enhanced small + large kernel images
            resized_ks = utils.resize_img(best_ks_kernel, down_size=utils.CLS_SIZE)
            resized_kl = utils.resize_img(best_kl_kernel, down_size=utils.CLS_SIZE)
            self.uq_ks.append(resized_ks)
            self.uq_kl.append(resized_kl)
    
    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, i):
        # Construct inputs
        inputs = torch.stack([
            self.preprocessing(self.img_rois[i]),
            self.preprocessing(self.uq_rois[i]),
            self.preprocessing(self.uq_ks[i]),
            self.preprocessing(self.uq_kl[i]),
        ], dim=0)

        # Extract corresponding labels
        int_label = utils.get_int_label(self.lookup_dict, image_id)

        return inputs, int_label, image_id

class IndModelDataset(BaseDataset):
    """
    Dataset for classification with individual members of the expert
    ensemble. Loads images and ground truth labels, and performs necessary
    UQ-driven enhancement to generate appropriate data for the
    individual ensemble member.

    Args:
        images (RGB Images): Dataset images.

        model_type (str): Individual model name.

        type (str): Phase name.

        device (torch.device): Selected device.

        preprocessing (function): Data preprocessing transforms.
    """
    def __init__(self, images, model_type, type, device, preprocessing=None):
        self.device = device
        self.images_fps = images
        self.instructor = SelectionAgent(gamma=0.75)
        self.preprocessing = preprocessing
        self.ids = []
        self.images = []
        if type == 'test':
            self.lookup_dict = generate_lookup_dict(utils.RESULTS_DIR + '/aux/test_labels_dict.csv')
        else:
            self.lookup_dict = generate_lookup_dict(utils.RESULTS_DIR + '/aux/all_aug_train_val_labels_dict.csv')

        # Load in segmentation ensemble
        self.ensemble = utils.load_ensemble(self.device)

        for i in range(len(self.images_fps)):
            image_id = utils.get_image_id(str(self.images_fps[i]))
            self.ids.append(image_id)

            # Read in data
            img = cv2.imread(str(self.images_fps[i]))
            if img is None:
                raise ValueError(f'Missing image: {image_id}')
            
            # Perform inference
            inferences = []
            for j in range(len(self.ensemble)):
                mod = self.ensemble[j]
                tensor_img = utils.load_tensor(img, device)
                inferences.append(mod(tensor_img))
            ensemble_preds = torch.cat(inferences, dim=1)
            _, prob_map, epistemic, aleatoric = utils.ensemble_infer(ensemble_preds)

            # Clean results
            img = utils.clean_tensor(tensor_img, is_img=True)
            p_map = utils.clean_tensor(prob_map)
            eu_map = utils.clean_tensor(epistemic)
            au_map = utils.clean_tensor(aleatoric)

            # Run Selection Agent
            roi_img, eu_img, best_ks_kernel, best_kl_kernel = self.instructor.run_pipeline(img, p_map, eu_map, au_map)

            # Select enhancement to apply based on model type
            if model_type == 'roi-uq-cls':
                resized_img = utils.resize_img(eu_img, down_size=utils.CLS_SIZE)
            elif model_type == 'ks-uq-cls':
                resized_img = utils.resize_img(best_ks_kernel, down_size=utils.CLS_SIZE)
            elif model_type == 'kl-uq-cls':
                resized_img = utils.resize_img(best_kl_kernel, down_size=utils.CLS_SIZE)
            else:
                resized_img = utils.resize_img(roi_img, down_size=utils.CLS_SIZE)
            self.images.append(resized_img)
                
    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, i):
        img = self.images[i]
        image_id = self.ids[i]

        # Apply preprocessing
        if self.preprocessing:
            img = self.preprocessing(img)

        # Extract corresponding labels
        int_label = utils.get_int_label(self.lookup_dict, image_id)

        return img, int_label, image_id


class SegmentationDataset(BaseDataset):
    """
    Dataset for image segmentation.

    Args:
        images (RGB Images): Dataset images.

        masks (RGB Images): Ground truth binary masks for the corresponding images.

        preprocessing (function): Data preprocessing transforms.

        augmentation (function): (Stochastic) data augmentation transforms.
    """
    def __init__(self, images, masks, preprocessing=None, augmentation=None):
        self.images_fps = images
        self.masks_fps = masks
        self.preprocessing = preprocessing
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, i):
        # Read in data
        img = cv2.imread(str(self.images_fps[i]))
        if img is None:
            print(self.images_fps[i])
        if img.shape[:2] != utils.SIZE:
            img = utils.resize_img(img)

        mskImg = cv2.imread(str(self.masks_fps[i]))
        if mskImg.shape[:2] != utils.SIZE:
            mskImg = utils.resize_img(mskImg)

        # After resizing, BGR values of binary mask are modified; We fix
        # this by converting the resized mask back to a 2d binary mask
        zerothChannel = np.zeros([utils.SIZE[0], utils.SIZE[1]])
        mask_2d = mskImg.copy()
        mask_2d[:, :, 0] = zerothChannel
        mask = mask_2d.argmax(2) + 2 * (mskImg.argmin(2))
        mask = mask.astype('float32')
        mask[mask >= 1.0] = 1.0
        mask[mask < 1.0] = 0.0
 
        # Apply preprocessing
        if self.preprocessing:
            img = self.preprocessing(img)
            mask = self.preprocessing(mask)

        # Apply augmentation
        if self.augmentation:
            img = self.augmentation(img)
            mask = self.augmentation(mask)

        return img, mask, utils.get_image_id(str(self.images_fps[i]))
