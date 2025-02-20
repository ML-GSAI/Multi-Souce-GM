# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
from typing import Any

import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from torchvision.datasets import ImageNet

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        use_labels  = True,     # Enable conditioning labels? False = label dimension is zero.
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self._raw_shape[1:]
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self): # [CHW]
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = anything goes.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        ext = self._file_ext(fname)
        with self._open_file(fname) as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

class ImageNetSpecficClassDataset(ImageNet):
    def __init__(self, root: str, split: str = "train",level_name:str=None, class_labels: str = None, nums: int=-1, **kwargs: Any):
        super().__init__(root, split, **kwargs)

        if "level" in class_labels:
            classes_info = hiera_infos[class_labels]
            classes_serial_str = [i for i in classes_info.values()][0]
            with open("imagenet_class_info.json") as jsf:
                inet_class_info_js = json.load(jsf)
            class_labels = []
            for k, v in inet_class_info_js.items():
                if v[0] in classes_serial_str:
                    print(k,v[1])
                    class_labels.append(int(k))
        else:
            class_labels = [int(i) for i in class_labels.split(",")]

        # get data for specfic classes
        class_labels = torch.tensor(class_labels)
        targets_tensor = torch.tensor(self.targets)
        print(targets_tensor.shape)
        idxs = []
        for class0 in class_labels:

            specific_idxs = torch.where(targets_tensor == class0)[0]
            if nums > 0:
                a = torch.randint(0, specific_idxs.shape[0] - 1, size=[nums])
                idxs.append(specific_idxs[a])
            else:
                idxs.append(specific_idxs)
        idxs = torch.concat(idxs, 0)


        self.imgs = [self.imgs[i] for i in idxs]
        self.samples = [self.samples[i] for i in idxs]
        self.targets = targets_tensor[idxs].tolist()

        self.targets, self.samples = trans_label_to_new(level_name, self.targets, self.samples)


hiera_infos = {
    "level1": {
        "dog": [
            'n02091032',
            'n02093754',
            'n02097209',
            'n02099712',
            'n02101388',
            'n02106030',
            'n02107142',
            'n02108551',
            'n02108915',
            'n02111889',
        ]
    },
    "level2": {
        "mammal": [
            'n02091032',
            'n02091032',
            'n02123394',
            'n02114367',
            'n02120079',
            'n02132136',
            'n02391049',
            'n02480855',
            'n02484975',
            'n02325366',
            'n02504458',

        ]
    },
    "level3": {
        "imagenet": [
            'n02091032',
            'n03594945',
            'n03201208',
            'n03452741',
            'n09472597',
            'n04398044',
            'n01644373',
            'n01582220',
            'n01443537',
            'n01693334',

        ]
    },
}

level1 = {
    171: 0,
    182: 1,
    198: 2,
    208: 3,
    215: 4,
    231: 5,
    236: 6,
    244: 7,
    245: 8,
    258: 9
}


level2 = {
    171: 0,
    269: 1,
    279: 2,
    283: 3,
    294: 4,
    330: 5,
    340: 6,
    366: 7,
    370: 8,
    386: 9
}

#
## level 3
level3 = {
    171: 0,
    1: 1,
    18: 2,
    31: 3,
    46: 4,
    532: 5,
    579: 6,
    609: 7,
    849: 8,
    980: 9
}

def trans_label_to_new(level_name: str, ori_targets: list, ori_samples:list):
    all_labels = list(set(ori_targets))
    if level_name == "level1":
        level_map = level1
    elif level_name == "level2":
        level_map = level2
    elif level_name == "level3":
        level_map = level3
    else:
        raise ValueError("Wrong level name!")

    for i in range(len(ori_targets)):
        ori_targets[i] = level_map[ori_targets[i]]
    for i in range(len(ori_samples)):
        ori_samples[i] = list(ori_samples[i])
        ori_samples[i][-1] = level_map[ori_samples[i][-1]]
        ori_samples[i] = tuple(ori_samples[i])
    return ori_targets, ori_samples

def trans_classid_to_ori(level_name: str, fake_id):
    if level_name == "level1":
        level_map = level1
    elif level_name == "level2":
        level_map = level2
    elif level_name == "level3":
        level_map = level3
    else:
        raise ValueError("Wrong level name!")
    for item in level_map.items():
        if item[1] == fake_id:
            return item[0]

def trans_classid_to_fake(level_name: str, real_id):
    if level_name == "level1":
        level_map = level1
    elif level_name == "level2":
        level_map = level2
    elif level_name == "level3":
        level_map = level3
    else:
        raise ValueError("Wrong level name!")
    for item in level_map.items():
        if item[0] == real_id:
            return item[1]



#----------------------------------------------------------------------------