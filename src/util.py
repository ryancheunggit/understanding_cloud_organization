import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_log(message, log_file):
    with open(log_file, 'a+') as f:
        if '\n' not in message:
            message += '\n'
        f.write(message)


def get_masks(rles, height=1400, width=2100, num_masks=4):
    masks = np.zeros((height, width, num_masks), dtype=np.float32)
    for idx, rle in enumerate(rles):
        if rle == 'NAN':
            continue
        rle = rle.split(" ")
        positions = map(int, rle[0::2])
        length = map(int, rle[1::2])
        mask = np.zeros(height * width, dtype=np.uint8)
        for pos, le in zip(positions, length):
            mask[pos: (pos + le)] = 1
        masks[:, :, idx] = mask.reshape(height, width, order='F')
    return masks


def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
