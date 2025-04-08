# -----------------------------------------------------------------------------
# Global variables (constants) and utility functions.
# -----------------------------------------------------------------------------
import classifiers
import cv2
import json
import metrics
import numpy as np
import os
import pandas as pd
import torch

from collections import OrderedDict
from segmenter import UNetSegmenter
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel
from torchvision import transforms
from torchvision.transforms import functional as H

######################
# GLOBALS
######################
# Size of images for segmentation models
SIZE = (512, 512)
# Size of images for classification models
CLS_SIZE = (224, 224)
N_CLASSIFIERS = 4
GENERATOR_SEED = 10742

# Directories below store relevant objects
WEIGHTS_DIR = './weights/'
RESULTS_DIR = './results/'
PLOTS_DIR = './plots/'

# Candidate model paths
UNET_MODEL_PATHS = [
    './weights/bunet111_model.pth',
    './weights/bunet232_model.pth',
    './weights/bunet334_model.pth',
    './weights/bunet494_model.pth',
    './weights/bunet551_model.pth',
    './weights/bunet676_model.pth',
    './weights/bunet727_model.pth',
    './weights/bunet838_model.pth',
    './weights/bunet949_model.pth',
    './weights/bunet1001_model.pth',
    './weights/bunet1181_model.pth',
    './weights/bunet1223_model.pth',
    './weights/bunet1373_model.pth',
    './weights/bunet1454_model.pth',
    './weights/bunet1555_model.pth',
]

CLASSIF_PATHS = [
    './weights/ind_roi-cls_Xception_model_e60.pth',
    './weights/ind_roi-uq-cls_Xception_model_e60.pth',
    './weights/ind_kl-uq-cls_Xception_model_e60.pth',
    './weights/ind_ks-uq-cls_Xception_model_e60.pth'
]

MOD_NAMES = {'vit_base_patch16_224': 'ViT', 'legacy_xception': 'Xception'}

######################
# GLOBALS (LABELING)
######################
# Path to file storing .csv file mapping images to corresponding classes
TRAIN_VAL_DICT_FILE = './results/aux/train_val_labels_dict.csv'
TEST_DICT_FILE = './results/aux/test_labels_dict.csv'
DICTS = (TRAIN_VAL_DICT_FILE, TEST_DICT_FILE)

# M: malignant, SK: seborrheic keratosis, BN: benign nevi
LABEL_TO_IND = {'M': 0, 'SK': 1, 'BN': 2}
IND_TO_LABEL = {0: 'M', 1: 'SK', 2: 'BN'}

DIR_NAME_TO_LABEL = {'benign_nevi': 'BN', 'malignant': 'M', 'seb_keratosis': 'SK'}
LABEL_TO_DIR_NAME = {'BN': 'benign_nevi', 'M':'malignant', 'SK': 'seb_keratosis'}

# Number of 'malignant', 'seborrheic keratosis', or 'benign nevi' classified
# images in (training, validation, testing) datasets
NUM_M = (374, 30, 117, 521)
NUM_SK = (254, 42, 90, 386)
NUM_BN = (1372, 78, 393, 1843)

# Colors (B, G, R) codes
COLOR_M = (56, 85, 248)
COLOR_SK = (89, 236, 118)
COLOR_BN = (211, 63, 93)
LABEL_COLORS = (COLOR_M, COLOR_SK, COLOR_BN)

######################
# PREPROCESSING &
# AUGMENTATION
######################
hflipper = transforms.RandomHorizontalFlip(p=1.0)
zflipper = transforms.RandomVerticalFlip(p=1.0)
RESIZE = transforms.Resize(size=(SIZE[0],SIZE[1]))

def rotate90(image):
    """
    Deterministically rotate image left by 90 degrees.
    """
    return H.rotate(image, angle=90)
    
def rotate180(image):
    """
    Deterministically rotate image left by 180 degrees.
    """
    return H.rotate(image, angle=180)

def rotate270(image):
    """
    Deterministically rotate image left by 270 degrees.
    """
    return H.rotate(image, angle=270)

AUGMENTORS = [
    (hflipper, '_HF'),      # HF
    (zflipper, '_VF'),      # VF
    (rotate90, '_R90'),     # R90
    (rotate180, '_R180'),   # R180
    (rotate270, '_R270'),   # R270
]

class PermuteChannels:
    def __init__(self, order):
        self.order = order

    def __call__(self, tensor):
        return tensor[self.order, :, :]

PREPROCESSING_FN = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

PREPROCESSING_PRETRAINED = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
        
PREPROCESSING_UQ_CLS = transforms.Compose(
    [
        transforms.ToTensor(),
        PermuteChannels(order=[2, 0, 1]),
    ]
)

######################
# TRAINING
######################
def choose_device(device, device_num=None):
    """
    Choose the device to run model on.
    """
    if not ((torch.cuda.is_available() and 'cuda' in device) or (torch.backends.mps.is_available() and device == 'mps')):
        device = 'cpu'
    elif torch.backends.mps.is_available() and device == 'mps':
        device = 'mps'
    elif device_num is not None:
        device = f'cuda:{device_num}'
    else:
        device = 'cuda'
    return torch.device(device)

def validate_trainable_params(m):
    """
    Asserts functionality of `numel` function when counting only the
    the number of trained parameters.
    """
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == numel(m, only_trainable=True)

def numel(m, only_trainable=False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

def print_params(model):
    """
    Prints all layers in specified model.
    """
    for name, param in model.named_parameters():
        print(f"{name}")

def check_layer_in_model(model, layer):
    """
    Check if model includes any batch normalization layers.
    """
    # i.e. Layed: torch.nn.BatchNorm2d, torch.nn.Linear
    print(any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules()))

def random_init(m):
    """
    Set random initialization of weights in CNN.
    """
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)

######################
# INFERENCE
######################
def convert_to_binary_mask(x, threshold=0.5):
    """
    Converts input tensor or numpy array to binary mask.
    """
    if isinstance(x, np.ndarray):
        return (x >= threshold).astype(np.float32)
    return (x >= threshold).to(torch.float32)

def convert_to_numpy(cuda_tensor):
    """
    Detach cuda tensor from GPU and attach to CPU as numpy array.
    """
    return cuda_tensor.detach().cpu().numpy()

def clean_tensor(tensor, is_img=False):
    """
    Clean and convert tensor into numpy array for plotting purpose.
    image: --> returns [H, W, C] numpy array with [0..1] values
    probability map: --> returns [H, W] numpy array with [0..1] values
    uncertainty map: --> returns [H, W] numpy array with [0..1] values
    """
    tensor = torch.squeeze(tensor)
    if is_img:
        return convert_to_numpy(tensor.permute(1, 2, 0))
    else:
        return convert_to_numpy(tensor)

def get_rgb(bgr_img):
    """
    Wrapper function converting image from BGR to RGB color space.
    """
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

def resize_img(img, down_size=None):
    """
    Resize image using OpenCV's `resize` function.
    """
    if down_size:
        resized_img = cv2.resize(img, down_size, interpolation= cv2.INTER_CUBIC)
    else:
        resized_img = cv2.resize(img, SIZE, interpolation= cv2.INTER_CUBIC)
    return resized_img
    
def lesion_percentage(p_map):
    """
    Compute percentage of total pixels in segmentation prediction
    map that are lesion pixels, given probability map.
    """
    mask = convert_to_binary_mask(p_map, threshold=0.5)
    if isinstance(p_map, np.ndarray):
        return np.mean(mask)
    return torch.mean(mask)

def sigmoid(arr):
    """
    Apply sigmoid activation.
    """
    tensor = torch.tensor(arr)
    return convert_to_numpy(F.sigmoid(tensor))

def add_small_constant(tensor, eps=1e-7):
    """
    Add negligible constant to tensor for numerical stability.
    """
    tensor = tensor + eps
    return tensor / tensor.sum(dim=1, keepdim=True)

def rescale(arr, apply_activation=False):
    """
    Rescale tensor to [0, 1] values.
    """
    if apply_activation:
        tensor = torch.tensor(arr)
        s_tensor = F.sigmoid(tensor)
        scaled_tensor = (s_tensor - torch.min(s_tensor)) / (torch.max(s_tensor) - torch.min(s_tensor))
        return convert_to_numpy(scaled_tensor)
    else:
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def load_tensor(img, device):
    """
    Prepare image as GPU/CPU tensor for model inference.
    """
    if img.shape[:2] != SIZE:
        img = resize_img(img)
    img = PREPROCESSING_FN(img)
    return img.unsqueeze(0).type(torch.FloatTensor).to(device)

def load_model(model, model_path, device):
    """
    Load model to specified device.
    """
    try:
        # Assume we are loading model onto same device it was trained using
        model.load_state_dict(rename_state_dict(torch.load(model_path, weights_only=True)))
    except:
        # If not, force load onto available device
        model.load_state_dict(rename_state_dict(torch.load(model_path, map_location=torch.device(device))))
    # Set model to eval mode
    model.eval()

def load_ensemble(device):
    """
    Load U-Net segmentation ensemble to specified device.
    """
    ensemble = []
    for i in range(len(UNET_MODEL_PATHS)):
        mod = UNetSegmenter().to(device)
        load_model(mod, UNET_MODEL_PATHS[i], device)
        ensemble.append(mod)
    return ensemble

def load_cohort(device):
    """
    Load expert ensemble of classification models to specified device.
    """
    cohort = []
    for i in range(len(CLASSIF_PATHS)):
        trained_model = classifiers.ModXception('legacy_xception', True).to(device)
        load_model(trained_model, CLASSIF_PATHS[i], device)
        cohort.append(trained_model)
    return cohort

def infer(image, model, device, add_activation, set_eval=True, task='seg'):
    """
    Perform inference of a single image by running forward pass of the model.
    Sets model to eval mode in 'val' phase.
    """
    if set_eval:
        model.eval()
    loaded_img = image.to(device)
    # Segmentation
    if task == 'seg':
        # MCU-Net or Deep Ensemble
        if add_activation:
            ensemble, inference, epistemic, aleatoric = model(loaded_img)
            inference = F.sigmoid(inference)
            return ensemble, inference, epistemic, aleatoric
        # U-Net
        else:
            inference = model(loaded_img)
            return inference, None, None, None
    # Classification
    else:
        inference, entropy = model(loaded_img)
        return inference, entropy

def ensemble_infer(predictions):
    """
    Perform ensemble inference by aggregating ensemble member predictions 
    into a single prediction.
    """
    # Ensemble prediction
    prob_map = torch.mean(predictions, dim=1)

    # Epistemic uncertainty
    epistemic = torch.mean(predictions**2, dim=1) - torch.mean(predictions, dim=1)**2

    # Aleatoric uncertainty
    aleatoric = torch.mean(predictions*(1-predictions), dim=1)

    return predictions, prob_map, epistemic, aleatoric

def rename_state_dict(state_dict):
    """
    Adjust naming convention of model state dict items.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

def get_results_dict(filename):
    """
    Return model results dictionary.
    """
    with open(filename) as json_file:
        model_eval_results = json.load(json_file)
    return model_eval_results

######################
# MISC. FUNCTIONS
######################
def get_int_label(lookup_dict, image_id):
    """
    Extract gold standard lesion diagnosis of image from
    specified dictionary.
    """
    return LABEL_TO_IND[lookup_dict[image_id]]

def get_image_id(pathname):
    """
    Extract image ID from relative or absolute pathname.
    """
    path_no_ext = str(pathname).split('.')[-2]
    filename = path_no_ext.split('/')[-1]
    return filename

def get_image_diagnosis_from_path(pathname):
    """
    Extract diagnosis type from relative or absolute pathname.
    """
    return str(pathname).split('/')[-2]

def concat_dfs(file_list, res_savefile):
    """
    Efficiently concatenate multiple dataframes and save to file.
    """
    dfs = []
    for file in file_list:
        dfs.append(pd.read_csv(file))
    res = pd.concat(dfs)
    res.to_csv(res_savefile, index=False)

def is_valid_image_file(filename):
    """
    Check whether current file stores image (.jpg) or mask (.png).
    """
    return str(filename).lower().endswith('.jpg') or \
        str(filename).lower().endswith('.png')

def check_masks(phase="training"):
    """
    Confirm that there exists an exact one-to-one matching between
    images and masks in given phase directory. Choose from ["training",
    "validation/vs1", "validation/vs2", "testing"].
    """
    dirs = [
        f'./{phase}/images/benign_nevi', 
        f'./{phase}/images/malignant',
        f'./{phase}/images/seb_keratosis',
    ]

    count_f = 0
    for dir in dirs:
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)
            full_name = get_image_id(f)
            lst = full_name.split('_')
            if len(lst) == 3:
                maskname = lst[0] + '_' + lst[1] + '_segmentation_' + lst[2] + '.png'
            else:
                maskname = lst[0] + '_' + lst[1] + '_segmentation.png'
            mask = os.path.join(f'./{phase}/masks_gt', maskname)
            if os.path.isfile(f) and is_valid_image_file(f) and not os.path.isfile(mask):
                print(maskname)
                print(f'File {f} is missing mask.')
                return
            count_f += 1
    
    count_m = 0
    mask_dir = f'./{phase}/masks_gt'
    for filename in os.listdir(mask_dir):
        f = os.path.join(mask_dir, filename)
        if os.path.isfile(f) and is_valid_image_file(f):
            count_m += 1

    if count_f == count_m:
        print('No missing masks.')
    elif count_f < count_m:
        print(f'Fewer images than masks in phase {phase}.')
    else:
        print(f'Fewer masks than images in phase {phase}.')

def check_cuda_devices():
    """
    Check available GPU devices.
    """
    print(f"{torch.cuda.device_count()} devices detected")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

def print_line():
    """
    Print line.
    """
    print('+---------------------------------+')
