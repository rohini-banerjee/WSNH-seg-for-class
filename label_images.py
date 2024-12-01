# -----------------------------------------------------------------------------
# Creates hashtable to lookup corresponding labels for images.
# 
# Given an individual (.png) image or directory containing (.png) images, and
# corresponding gold standard ground truth file of classes, return new image or
# new directory of images with corresponding labels.
# -----------------------------------------------------------------------------
import argparse
import cv2
import numpy as np
import os
import pandas as pd
import utils


def save_lookup_csv(gt_train_file, gt_val_file, gt_test_file):
    """
    Given the input gold standard ground truth (.csv) files, saves new (.csv)
    file with image_ids and corresponding class label where
    M: malignant, SK: seborrheic keratosis, BN: benign nevi.
    """
    df1 = pd.read_csv(gt_train_file)
    df2 = pd.read_csv(gt_val_file)
    train_val_labels_df = pd.concat([df1, df2], ignore_index=True)
    test_labels_df = pd.read_csv(gt_test_file)

    def get_value(row):
        if row['melanoma'] == 1.0:
            return 'M'
        elif row['seborrheic_keratosis'] == 1.0:
            return 'SK'
        return 'BN'

    # Non-vectorized approach
    train_val_labels_df['value'] = train_val_labels_df.apply(get_value, axis=1)
    test_labels_df['value'] = test_labels_df.apply(get_value, axis=1)
    
    assert len(train_val_labels_df[train_val_labels_df['value']=='M']) == utils.NUM_M[3] - utils.NUM_M[2]
    assert len(train_val_labels_df[train_val_labels_df['value']=='SK']) == utils.NUM_SK[3] - utils.NUM_SK[2]
    assert len(train_val_labels_df[train_val_labels_df['value']=='BN']) == utils.NUM_BN[3] - utils.NUM_BN[2]

    assert len(test_labels_df[test_labels_df['value']=='M']) == utils.NUM_M[2]
    assert len(test_labels_df[test_labels_df['value']=='SK']) == utils.NUM_SK[2]
    assert len(test_labels_df[test_labels_df['value']=='BN']) == utils.NUM_BN[2]

    # Save to output csv file
    for i, df in enumerate([train_val_labels_df, test_labels_df]):
        sub_df = df[['image_id', 'value']]
        sub_df.to_csv(utils.DICTS[i], index=False)

def generate_lookup_dict(filename):
    """
    Returns lookup dictionary mapping image IDs to corresponding gold
    standard class label.
    """
    class_lookup_dict = pd.read_csv(filename, index_col=0, header=0).to_dict()['value']
    return class_lookup_dict

def annotate_image(image_filename, lookup_dict):
    """
    Annotate single image with corresponding label and image ID.
    """
    filepath = image_filename.split('.')[-2]
    image_name = filepath.split('/')[-1]
    
    label = lookup_dict[image_name]
    img = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    img_name_dim = (7, 20)
    label_dim = (13, 50)

    (text_width, text_height), baseline = cv2.getTextSize(image_name, font, \
        img_name_dim[0], img_name_dim[1])
    textX = 25
    textY = 200

    # Define the top-left and bottom-right corners of the rectangle
    top_left_corner = (textX, textY - text_height - baseline)
    bottom_right_corner = (textX + text_width, textY + baseline)

    # Draw the black rectangle behind the text
    cv2.rectangle(img, top_left_corner, bottom_right_corner, (0, 0, 0), thickness=-1)
    # Draw image ID text
    cv2.putText(img, image_name, (textX, textY), font, img_name_dim[0], (255, 255, 255), img_name_dim[1])
    color = utils.LABEL_COLORS[utils.LABEL_TO_IND[label]]
    # Draw label text
    cv2.putText(img, label, (textX, textY + 425), font, label_dim[0], color, label_dim[1])

    return img

def annotate_image_dir(dir_name, lookup_dict):
    """
    Annotate all images in specified directory with corresponding label and
    image ID using lookup dictionary.
    """
    # Create directory to store annotated training data for clustering if it
    # does not already exist
    annotated_clustering_data_dir = utils.ANNOTATED_CLUSTER_DATA
    if not os.path.exists(annotated_clustering_data_dir):
        os.mkdir(annotated_clustering_data_dir)

    # Annotate images in specified directory
    for filename in os.listdir(dir_name):
        f = os.path.join(dir_name, filename)
        if os.path.isfile(f):
            new_f = annotated_clustering_data_dir + filename
            cv2.imwrite(new_f, annotate_image(f, lookup_dict))
