import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from torchvision.transforms.functional import rotate
from PIL import Image


def padded_square_crop(im, x, y, r):
    c, h, w = im.shape
    output = torch.full((c, r, r), 0, dtype=torch.float32, device=im.device)

    left_edge = max(x, 0)
    right_edge = min(x + r, w)
    top_edge = max(y, 0)
    bottom_edge = min(y + r, h)

    x_out = 0 if x >= 0 else abs(x)
    y_out = 0 if y >= 0 else abs(y)

    cropped = im[:, top_edge:bottom_edge, left_edge:right_edge]

    output[:, y_out:y_out + cropped.shape[1], x_out:x_out + cropped.shape[2]] = cropped
    return output


def make_square(x, y, w, h):
    if w > h:
        r = w
        y = y + (h / 2) - (w / 2)
    else:
        r = h
        x = x + (w / 2) - (h / 2)

    return int(x), int(y), int(r)


def square_pad(tensor, fill=0.0):
    c, height, width = tensor.shape

    if width == height:
        return tensor

    long_edge = max(width, height)
    output = torch.full((c, long_edge, long_edge), fill, dtype=torch.float32, device=tensor.device)

    if width < height:
        pad = int((long_edge - width) / 2)
        output[:, :, pad:width + pad] = tensor.clone()
    else:
        pad = int((long_edge - height) / 2)
        output[:, pad:height + pad, :] = tensor.clone()

    return output


to_tensor = torchvision.transforms.ToTensor()


def read_image(path, device):
    tensor = to_tensor(Image.open(path))
    tensor = tensor.to(device)
    return tensor


def similarity_align(im, keypoints, desired_eye_ratio=0.3, mid_eye_pos_ratio=0.45):
    # TODO This weekend
    pass


def eye_based_align(im, l_eye_center, r_eye_center, desired_eye_ratio=0.3, mid_eye_pos_ratio=0.45):
    """Align input image based on the size and center of the eyes"""

    im = to_tensor(im.copy())

    # kps = list(map(lambda i: np.array([kps.part(i).x, kps.part(i).y]), range(5)))
    mid_eye = (l_eye_center + r_eye_center) / 2  # half way between eye centers
    eye_dist = np.sqrt(np.sum(np.square(l_eye_center - r_eye_center)))

    angle = np.arctan2(*(l_eye_center - r_eye_center))  # angle between eyes on y axis
    angle = 90 - np.degrees(angle)  # convert to degrees from x axis

    # rotated = im
    rotated = rotate(im, angle, center=mid_eye.tolist())

    new_w = int(eye_dist / desired_eye_ratio)  # scale eye dist to 0.5 the image width

    crop_x = int(mid_eye[0] - (new_w / 2))
    crop_y = int(mid_eye[1] - new_w * mid_eye_pos_ratio)
    try:
        return padded_square_crop(rotated, crop_x, crop_y, new_w)
    except Exception as e:
        print('error with image', im.shape, rotated.shape, crop_x, crop_y, new_w)
        print(e)
        return


def box_align(img, detection):
    x, y, r = make_square(*tuple(detection))
    return padded_square_crop(to_tensor(img.copy()), x, y, r)
