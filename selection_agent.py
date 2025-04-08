# -----------------------------------------------------------------------------
# Define selection agent.
# -----------------------------------------------------------------------------
import os
import argparse
import cv2
import dataset
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotting
import torch
import utils


class SelectionAgent:
    """
    The Selection Agent lies between segmentation and classification in
    our proposed pipeline:

    1) Performs inference using trained U-Net(s) from the Segmentation Agent
    to define the appropriate region-of-interest (ROI) centered around the
    skin lesion.
        a) AutoCropper finds ROI and minimum-bounding-square around lesion
        to preserve aspect ratio of lesion.
        b) SmartSelector generates duo-scale image kernels and selects kernels
        with lowest segmentation uncertainty.
    2) Outputs uncertainty quantification (UQ)-enhanced images for future
    classification.

    Args:
        gamma (float): Accepted percentage of pixels classified as lesion.

        alpha (float): Percentage of original luminance retained.

        beta (float): Percentage of luminance from uncertainty map applied.
    """
    def __init__(self, gamma, alpha=0.95, beta=0.40):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _add_zero_padding(self, img, t, b, l, r):
        """
        Add zero-padding to image given border width in number of pixels in
        top (t), bottom (b), left (l), and right (r) directions.
        """
        new_img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return new_img
    
    def _crop_mbs(self, obj, x, y, d):
        """
        Crop and return minimum bounding square (MBS).
        """
        return obj[y:y+d, x:x+d]

    def AutoCropper(self, img, p_map, eu_map, au_map):
        """
        Given an image and its corresponding predicted probability and
        uncertainty maps from segmentation, locate minimum bounding square
        (MBS) with lesion centered, and crop accordingly. MBS over MBR
        to preserve aspect ratio of skin lesion.

        Args:
            img (BGR image): Image in BGR color space.

            p_map (torch.Tensor): Output probability map from image segmentation.

            eu_map (torch.Tensor): Output epistemic uncertainty map from
            image segmentation.

            au_map (torch.Tensor): Output aleatoric uncertainty map from
            image segmentation.

        Returns:
            img (RGB image): Image in RGB color space.

            contours_img (RGB image): Image with lesion contours overlaid.

            p_map (torch.Tensor): Original output probability map from
            image segmentation.
            
            contours_p_map (torch.Tensor): Original probability map from
            image segmentation with lesion contours overlaid.
            
            cropped_padded_img (torch.Tensor): RGB image after ROI selection
            and cropping.
            
            cropped_p_map (torch.Tensor): Output probability map from
            image segmentation after ROI selection and cropping.

            cropped_padded_eu_map (torch.Tensor): Epistemic uncertainty map
            from image segmentation after ROI selection and cropping.

            cropped_padded_au_map (torch.Tensor): Aleatoric uncertainty map
            from image segmentation after ROI selection and cropping.
        """
        is_failed_seg = False
        delta = 20

        # Convert to RGB space
        img = utils.get_rgb(img)
        bad_img = img.copy()
        
        # Convert probability map to binary mask and clip values to RGB range
        mask = utils.convert_to_binary_mask(p_map)
        mask = (mask * 255).astype(np.uint8)

        # Utilize epistemic uncertainty map to locate approximate lesion location
        # as fail-safe when segmentation fails
        lesion_pixels = np.argwhere(mask == 255)
        if len(lesion_pixels) == 0:
            is_failed_seg = True
            normalized_eu = utils.rescale(eu_map, apply_activation=False)
            eu_mask = utils.convert_to_binary_mask(normalized_eu)
            mask = (eu_mask * 255).astype(np.uint8)

        # Extract contours from segmentation mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find best contour by selecting contour with highest average
        # probability provided by segmentation prediction
        best_contour_i, curr = None, float('-inf')
        for i in range(len(contours)):
            cimg = np.zeros_like(img)
            cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
            mask = cimg == 255
            mask_2d = np.max(mask, axis=2)
            p_mean = np.mean(p_map[mask_2d])
            if p_mean > curr:
                best_contour_i = i
                curr = p_mean

        # Draw contours on copy of image
        contours_img = img.copy()
        cv2.drawContours(contours_img, contours, -1, (255, 255, 255), thickness=2)

        # Draw contours on copy of probability heatmap
        contours_p_map = p_map.copy()
        contours_p_map = (255*contours_p_map).astype(np.uint8)
        contours_p_map = cv2.applyColorMap(contours_p_map, cv2.COLORMAP_VIRIDIS)
        cv2.drawContours(contours_p_map, contours, -1, (255, 255, 255), thickness=2)

        if is_failed_seg:
            x, y, w, h = cv2.boundingRect(contours[best_contour_i])
            if (x-delta >= 0) and (x+(w+delta) < img.shape[1]) and (y-delta >= 0) and (y+(h+delta) < img.shape[0]):
                x -= delta
                w += delta
                y -= delta
                h += delta
        elif contours is not None and len(contours) > 0:
            # Find minimum bounding square (MBS) given contours
            x, y, w, h = cv2.boundingRect(contours[best_contour_i])
        else:
            # If no contours can be extracted, use pixel values from mask directly
            lesion_pixels = np.argwhere(mask == 1)
            y_min, x_min = lesion_pixels.min(axis=0)
            y_max, x_max = lesion_pixels.max(axis=0)
            d = max(x_max - x_min, y_max - y_min)
            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
            
        # Crop image to MBS
        d = max(w, h)
        l = r = t = b = 0
        new_x, new_y = x, y
        
        # Left padding
        left_pos = (x + w//2 - d//2)
        if left_pos < 0:
            l = abs(left_pos)
            new_x = 0
        else:
            new_x = left_pos
        # Right padding
        if (x + w//2 + d//2) > img.shape[1]:
            r = abs(x + w//2 - d//2)
        # Top padding
        top_pos = y + h//2 - d//2
        if top_pos < 0:
            t = abs(top_pos)
            new_y = 0
        else:
            new_y = top_pos
        # Bottom padding
        if (y + h//2 + d//2) > img.shape[0]:
            b = abs(y + h//2 - d//2)

        # Add visualization of MBS
        cv2.rectangle(contours_p_map, (new_x, new_y), (new_x + d, new_y + d), (0, 255, 0), 2)

        padded_img = self._add_zero_padding(img, t, b, l, r)
        padded_p_map = self._add_zero_padding(p_map, t, b, l, r)
        padded_eu_map = self._add_zero_padding(eu_map, t, b, l, r)
        padded_au_map = self._add_zero_padding(au_map, t, b, l, r)

        # Crop around lesion
        cropped_padded_img = self._crop_mbs(padded_img, new_x, new_y, d)
        cropped_p_map = self._crop_mbs(padded_p_map, new_x, new_y, d)
        cropped_padded_eu_map = self._crop_mbs(padded_eu_map, new_x, new_y, d)
        cropped_padded_au_map = self._crop_mbs(padded_au_map, new_x, new_y, d)
        
        return img, contours_img, p_map, contours_p_map, cropped_padded_img, cropped_p_map, cropped_padded_eu_map, cropped_padded_au_map

    def LuminanceEffector(self, cropped_img, cropped_u_map):
        """
        Given an ROI image and its corresponding epistemic uncertainty map from
        segmentation, perform UQ-determined visual enhancement in the luminance
        channel in YCrCb color space.

        Args:
            cropped_img (RGB image): ROI image in RGB color space from AutoCropper.

            cropped_u_map (torch.Tensor): Output epistemic uncertainty map from
            image segmentation cropped to lesion ROI by AutoCropper.

        Returns:
            new_img (RGB image): UQ-enhanced RGB image.
        """
        # Copy image and rescale uncertainties to [0,1] probability range
        rgb_img = cropped_img.copy()
        u_map = utils.rescale(cropped_u_map, apply_activation=True)

        # Change luminance channel in dermascopic image in YCrCb color space
        img_YCrCb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
        img_YCrCb[:, :, 0] = (img_YCrCb[:, :, 0] * self.alpha) + (img_YCrCb[:, :, 0] * u_map * self.beta)
        
        # Convert back to RGB color space
        new_img = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)
        new_img = utils.rescale(new_img, apply_activation=False)
        return new_img

    def SmartSelector(self, cropped_img, cropped_u_map, cropped_bin_map):
        """
        Given an ROI image, its corresponding epistemic uncertainty map, and its
        corresponding cropped segmentation mask, generate duo-scale image kernels
        and return small- and large-scale kernel with the lowest model uncertainty.

        Args:
            cropped_img (RGB image): ROI image in RGB color space from AutoCropper.

            cropped_eu_map (torch.Tensor): Output epistemic uncertainty map from
            image segmentation cropped to lesion ROI by AutoCropper.

            cropped_bin_map (torch.Tensor): Output binary segmentation mask cropped
            to lesion ROI by AutoCropper.

        Returns:
            best_ks_kernel (torch.Tensor): Small kernel with lowest epistemic uncertainty.
            
            best_kl_kernel (torch.Tensor): Large kernel with lowest epistemic uncertainty.
        """
        h = w = min(cropped_img.shape[0], cropped_img.shape[1])
        cropped_u_map = utils.rescale(cropped_u_map)

        def generate_small_kernels():
            """
            Generate 9 kernels of size (m // 3) x (m // 3).
            """
            k_len = (h // 3)
            all_kernels, uncertainties, means = [], [], []
            for row in range(3):
                for col in range(3):
                    init_y = row * k_len
                    end_y = init_y + k_len
                    init_x = col * k_len
                    end_x = init_x + k_len
                    
                    kernel = cropped_img[init_y:end_y, init_x:end_x]
                    bin_kernel = cropped_bin_map[init_y:end_y, init_x:end_x]
                    u_kernel = cropped_u_map[init_y:end_y, init_x:end_x]

                    all_kernels.append(kernel)
                    k_mean = np.mean(u_kernel)
                    perc_lesion = utils.lesion_percentage(bin_kernel)
                    
                    uncertainties.append(k_mean)
                    means.append(perc_lesion)

            # Convert list to np.ndarray
            all_kernels = np.array(all_kernels)
            uncertainties = np.array(uncertainties)
            means = np.array(means)
            return all_kernels, uncertainties, means
    
        def generate_large_kernels():
            """
            Generate 4 kernels of size (2m // 3) x (2m // 3).
            """
            k_len = math.floor((2 / 3) * w)
            # Top-left
            x1, y1 = 0, 0
            # Top-right
            x2, y2 = w - k_len, 0
            # Bottom-left
            x3, y3 = 0, w - k_len
            # Bottom-right
            x4, y4 = w - k_len, w - k_len

            coords = [
                [(x1, y1),
                (x2, y2)],
                [(x3, y3),
                (x4, y4)],
            ]

            all_kernels, uncertainties, means = [], [], []
            for r in range(2):
                for c in range(2):
                    x, y = coords[r][c]
                    kernel = cropped_img[y:y + k_len, x:x + k_len]
                    bin_kernel = cropped_bin_map[y:y + k_len, x:x + k_len]
                    u_kernel = cropped_u_map[y:y + k_len, x:x + k_len]

                    all_kernels.append(kernel)
                    k_mean = np.mean(u_kernel)
                    perc_lesion = utils.lesion_percentage(bin_kernel)
                    
                    uncertainties.append(k_mean)
                    means.append(perc_lesion)

            all_kernels = np.array(all_kernels)
            uncertainties = np.array(uncertainties)
            means = np.array(means)
            return all_kernels, uncertainties, means
        
        def select_kernels(all_kernels, uncertainties, means):
            """
            Select best small- and large-scale kernel.
            """
            assert(all_kernels.shape[0] == uncertainties.shape[0] == means.shape[0])

            # Verify whether segmentation is sufficient
            indices = np.where(means >= self.gamma)
            segmentation_succeeded = len(indices[0]) > 0
            if segmentation_succeeded:
                remaining_kernels = all_kernels[indices]
                remaining_uncertainties = uncertainties[indices]
                remaining_means = means[indices]
            else:
                remaining_kernels = all_kernels
                remaining_uncertainties = uncertainties
                remaining_means = means
            
            # Rank remaining kernels by uncertainty
            ranked_kernels = remaining_kernels[np.argsort(remaining_uncertainties)]
            ranked_uncertainties = np.sort(remaining_uncertainties)

            # If segmentation is sufficient, select and return kernel with SMALLEST
            # uncertainty. Otherwise, select and return kernel with LARGEST uncertainty
            # since regions with highest uncertainty usually coincide with areas of
            # difficulty for lesion identification.
            if segmentation_succeeded:
                return ranked_kernels[0]
            return ranked_kernels[-1]
        
        sk_kernels, sk_uncertainties, sk_means = generate_small_kernels()
        bk_kernels, bk_uncertainties, bk_means = generate_large_kernels()

        best_ks_kernel = select_kernels(sk_kernels, sk_uncertainties, sk_means)
        best_kl_kernel = select_kernels(bk_kernels, bk_uncertainties, bk_means)
        return best_ks_kernel, best_kl_kernel
    
    def run_pipeline(self, img, p_map, eu_map, au_map):
        """
        Run entire Selection Agent module.

        Args:
            img (BGR image): Image in BGR color space.

            p_map (torch.Tensor): Output probability map from image segmentation.

            eu_map (torch.Tensor): Output epistemic uncertainty map from
            image segmentation.

            au_map (torch.Tensor): Output aleatoric uncertainty map from
            image segmentation.

        Returns:
            roi_img (RGB Image): RGB image after ROI selection and cropping.
            
            eu_img (torch.Tensor): EU-enhanced RGB image after ROI selection
            and cropping.

            best_ks_kernel (torch.Tensor): Small-scale image kernel with lowest
            epistemic uncertainty.

            best_kl_kernel (torch.Tensor): Large-scale image kernel with lowest
            epistemic uncertainty.
        """
        # Perform auto-cropping using predicted segmentation mask
        rgb_img, contours_img, p_map, contours_p_map, roi_img, new_p_map, new_eu_map, new_au_map = self.AutoCropper(img, p_map, eu_map, au_map)
        
        # Generate epistemic uncertainty-enhanced image
        eu_img = self.LuminanceEffector(roi_img, new_eu_map)

        # Select optimal small- and large-scale image kernels
        binary_mask = utils.convert_to_binary_mask(new_p_map, threshold=0.5)
        best_ks_kernel, best_kl_kernel = self.SmartSelector(eu_img, new_eu_map, binary_mask)
        
        return roi_img, eu_img, best_ks_kernel, best_kl_kernel

def main(args):
    ### Example usage of Selection Agent ###
    device = utils.choose_device(args.device)
    if str(device) == 'cuda':
        print('Setting CUDA Device Node')
    print(f"Running on {device}")

    # Instantiate Selection Agent
    ScA = SelectionAgent(gamma=0.75)
    
    # Load segmentation models
    ensemble = utils.load_ensemble(device)

    # Load test image, mask, and corresponding gold standard lesion diagnosis
    img = cv2.imread(str(args.img_pathname))
    mask = cv2.imread(str(args.mask_pathname))
    image_id = utils.get_image_id(str(args.img_pathname))
    f = open(args.label_pathname, 'r')
    label = f.readline()

    # Perform segmentation inference on test image
    inferences = []
    for j in range(len(ensemble)):
        mod = ensemble[j]
        tensor_img = utils.load_tensor(img, device)
        inferences.append(mod(tensor_img))
    ensemble_preds = torch.cat(inferences, dim=1)
    _, prob_map, epistemic, aleatoric = utils.ensemble_infer(ensemble_preds)

    # Clean results
    img = utils.clean_tensor(tensor_img, is_img=True)
    p_map = utils.clean_tensor(prob_map)
    eu_map = utils.clean_tensor(epistemic)
    au_map = utils.clean_tensor(aleatoric)

    # Plot image, gt, predictions, and uncertainty maps together
    plotting.plot_prediction(image_id, utils.get_rgb(img), mask, p_map, eu_map, au_map, save_path=(utils.PLOTS_DIR+f'ex_segmentation_result.png'))

    roi_img, eu_img, best_ks_kernel, best_kl_kernel = ScA.run_pipeline(img, p_map, eu_map, au_map)
    resized_img = utils.get_rgb(utils.resize_img(img))
    resized_roi = utils.resize_img(roi_img)
    resized_eu = utils.resize_img(eu_img)
    resized_ks = utils.resize_img(best_ks_kernel)
    resized_kl = utils.resize_img(best_kl_kernel)
    
    # Plot UQ-enhanced images
    plotting.plot_loaded_classif_data(image_id, resized_img, resized_roi, resized_eu, resized_ks, resized_kl, save_path=(utils.PLOTS_DIR+f'ex_uq-enhanced-data.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for Selection Agent.')
    
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training.')
    parser.add_argument('--img_pathname', type=str, default='./examples/ISIC_0000002_R270.jpg', help='Pathname to test image.')
    parser.add_argument('--mask_pathname', type=str, default='./examples/ISIC_0000002_segmentation_R270.png', help='Pathname to mask for test image.')
    parser.add_argument('--label_pathname', type=str, default='./examples/ISIC_0000002_R270_label.txt', help='Path to gold standard diagnosis for test image.')

    args = parser.parse_args()
    main(args)
    