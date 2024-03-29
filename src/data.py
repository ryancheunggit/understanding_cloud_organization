import cv2
import os
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import Dataset
from util import get_masks


TRAIN_IMAGE_PATH = '../data/train_images'
TEST_IMAGE_PATH = '../data/test_images'
CLASS_TARGETS = ['Fish', 'Flower', 'Gravel', 'Sugar']
RLE_TARGETS = ['{}_EncodedPixels'.format(c) for c in CLASS_TARGETS]


def get_transform(mode='train', H=384, W=576):
    if mode == 'train':
        transform = A.Compose([
            A.Flip(p=.5),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=.5),
                A.IAASharpen(p=.25),
            ], p=.35),
            A.OneOf([
                A.RandomBrightness(),
                A.RandomContrast(),
                A.RandomGamma()
            ], p=.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=.3),
            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, shift_limit=0, border_mode=0, p=.5),
            A.CropNonEmptyMaskIfExists(height=int(1400 * .9), width=int(2100 * .9), p=.5),
            A.Resize(H, W),
            ToTensor()
        ])
    else:
        transform = A.Compose([
            A.Resize(352, 544),
            ToTensor()
        ])
    return transform


def get_train_metadata():
    train = pd.read_csv('../data/train.csv')
    train['image'] = train['Image_Label'].map(lambda x: x.split('_')[0])
    train['label'] = train['Image_Label'].map(lambda x: x.split('_')[1])
    train_wide = pd.pivot(train, index='image', columns='label', values='EncodedPixels').reset_index()
    train_wide.columns = ['image'] + ['{}_EncodedPixels'.format(l) for l in CLASS_TARGETS]
    for l in CLASS_TARGETS:
        train_wide['{}'.format(l)] = 1 - train_wide['{}_EncodedPixels'.format(l)].isnull().astype(int)
        train_wide['{}_EncodedPixels'.format(l)].fillna('NAN', inplace=True)
    train_wide['image_path'] = train_wide['image'].map(lambda x: os.path.join(TRAIN_IMAGE_PATH, x))
    return train_wide


def get_test_metadata():
    test = pd.read_csv('../data/sample_submission.csv')
    test['image'] = test['Image_Label'].map(lambda x: x.split('_')[0])
    test.drop_duplicates('image', inplace=True)
    test['image_path'] = test['image'].map(lambda x: os.path.join(TEST_IMAGE_PATH, x))
    return test


def get_train_test_split(metadata, num_folds=5, random_state=42):
    mskf = MultilabelStratifiedKFold(n_splits=num_folds, random_state=random_state)
    folds = list(mskf.split(metadata, metadata[CLASS_TARGETS]))
    return folds


def blackout(blackout_p, image, classes, masks, version=2):
    if classes.sum() > 0 and np.random.random() <= blackout_p:
        idx = np.random.choice(np.where(classes > .5)[0])
        classes[idx] = 0
        if version == 1:
            image[masks[:, :, idx] == 1] = 0
            masks[:, :, idx] == 0
        else:
            mask = masks[:, :, idx] == 1
            image[mask] = 0
            masks[mask] = 0
    return image, classes, masks


class CloudDataset(Dataset):
    def __init__(self, metadata, mode='train', blackout_p=0, H=384, W=576):
        self.image_path = metadata['image_path'].values
        self.labels = metadata[CLASS_TARGETS].values if mode in ['train', 'valid'] else None
        self.rles = metadata[RLE_TARGETS].values if mode in ['train', 'valid'] else None
        self.blackout_p = blackout_p
        self.transform = get_transform(mode=mode, H=H, W=W)
        self.mode = mode

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mode in ['train', 'valid']:
            classes = self.labels[idx, :].copy()
            masks = get_masks(self.rles[idx, :].copy())
            if self.blackout_p:
                image, classes, masks = blackout(self.blackout_p, image, classes, masks)
            if self.transform:
                transformed = self.transform(image=image, mask=masks)
                image = transformed['image']
                masks = transformed['mask']
                masks = masks[0].permute(2, 0, 1)
            return image, classes, masks
        else:
            image_name = image_path.split('/')[-1]
            if self.transform:
                image = self.transform(image=image)['image']
            return image_name, image
