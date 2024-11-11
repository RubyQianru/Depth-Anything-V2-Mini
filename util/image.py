import numpy as np
import matplotlib.pyplot as plt
import cv2

cmap = plt.cm.viridis

def split_data(image_set, depth_set, train_size=0.9, val_size=0.1):
    # Calculate the size of training and validation sets given the percentages
    train_size = int(len(image_set) * train_size)
    val_size = int(len(image_set) * val_size)

    image_train, image_val = image_set[:train_size], image_set[-val_size:]
    depth_train, depth_val = depth_set[:train_size], depth_set[-val_size:]

    return image_train, image_val, depth_train, depth_val


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def resize_image(input):
    input = cv2.resize(input, 
        (input.shape[0]//2, 
        input.shape[1]//2), 
        interpolation=cv2.INTER_AREA)
    return input


def merge_into_row(input, depth_target):
    # Transpose input from (3, H, W) to (H, W, 3)
    input = np.transpose(input, (1, 2, 0))
    input = resize_image(input)
    depth_target = np.squeeze(np.array(depth_target))

    depth_target = resize_image(depth_target)
    d_min = np.min(depth_target)
    d_max = np.max(depth_target)
    depth_target_col = colored_depthmap(depth_target, d_min, d_max)

    img_merge = np.hstack([input, depth_target_col])
    return img_merge

