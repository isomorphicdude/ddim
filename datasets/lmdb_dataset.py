"""Implements base class for lmdb datasets."""

import os
import io
import ast
import os.path as osp
from typing import Union

import torch
import torch.utils
import torch.utils.data
import torchvision
from torchvision.transforms.functional import to_pil_image
import numpy as np
import lmdb
import pickle
from PIL import Image

__LMDB_DATASETS__ = {}


def register_dataset(name: str):
    def wrapper(cls):
        if __LMDB_DATASETS__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __LMDB_DATASETS__[name] = cls
        return cls

    return wrapper


def get_dataset(
    name: str,
    db_path=None,
    transform=None,
    target_transform=None,
    max_len=None,
    image_shape=None,
    binarize=False,
    train=True,
    **kwargs,
):
    if name in ['cifar10', 'svhn', 'mnist']:
        # use torchvision datasets
        if transform is None:
            transform_list = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(image_shape[1:]),
                torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ]
            if name == "mnist":
                transform_list[-1] = torchvision.transforms.Normalize((0.5,), (0.5,))
            transform = torchvision.transforms.Compose(transform_list)
        if name != 'svhn':
            ret_dataset = __LMDB_DATASETS__[name](
                root=db_path,
                train=train,
                download=True,
                transform=transform,
            )
        else:
            ret_dataset = __LMDB_DATASETS__[name](
                root=db_path,
                split="train" if train else "test",
                download=True,
                transform=transform,
            )
    else:
        # loading from provided path
        # e.g. data/celeba64/
        if train:
            db_path = osp.join(db_path, "train.lmdb")
        else:
            db_path = osp.join(db_path, "valid.lmdb")
        if __LMDB_DATASETS__.get(name, None) is None:
            raise NameError(f"Name {name} is not defined!")

        # db path join for train and valid
        print(f"Loading lmdb dataset {name} from {db_path}")

        ret_dataset = __LMDB_DATASETS__[name](
            db_path, transform, target_transform, image_shape, binarize, train, **kwargs
        )

    if max_len is not None:
        print(f"Using subset of length {max_len}")
        ret_dataset = torch.utils.data.Subset(ret_dataset, range(max_len))
    else:
        print(f"Using full dataset of length {len(ret_dataset)}")
    return ret_dataset


def get_loader(
    dataset_name: str,
    sample_shape: Union[str, tuple],
    batch_size: int = 100,
    shuffle_train: bool = True,
    binarize: bool = False,
    max_len: int = 10_000,
    use_distributed: bool = False,
    train: bool = True,
    return_dset: bool = False,
    db_path: str = None,
    transforms=None,
):
    image_shape = (
        ast.literal_eval(sample_shape)
        if isinstance(sample_shape, str)
        else sample_shape
    )

    try:
        if db_path is None:
            db_path = (
                f"data/{dataset_name}/train.lmdb"
                if train
                else f"data/{dataset_name}/valid.lmdb"
            )
        train_dataset = get_dataset(
            name=dataset_name,
            db_path=db_path,
            max_len=max_len,
            binarize=binarize,
            train=train,
            transform=transforms,
            target_transform=None,
            image_shape=image_shape,
        )
    except:
        train_dataset = get_dataset(
            name=dataset_name,
            db_path=None,
            max_len=max_len,
            binarize=binarize,
            train=train,
            transform=transforms,
            target_transform=None,
            image_shape=image_shape,
        )

    if use_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    num_workers = min(8, os.cpu_count() // 2)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None) and shuffle_train,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=num_workers,
    )

    if return_dset:
        return train_loader, train_dataset
    else:
        return train_loader


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


class LMDBDataset(torch.utils.data.Dataset):
    """
    Implements base class for lmdb datasets.
    Data is store as triplets of (image, label, index).

    :param db_path: path to lmdb database
    :param transform: image transformation
    :param target_transform: label transformation
    """

    def __init__(
        self,
        db_path,
        transform=None,
        target_transform=None,
        image_shape=None,
        binarize=False,
        train=True,
    ):
        self.db_path = db_path
        self.transform = transform
        self.target_transform = target_transform
        self.image_shape = image_shape
        self.binarize = binarize

        # build generic transform
        if self.transform is None:
            self.transform = self.init_transform()

        env = lmdb.open(
            self.db_path,
            subdir=osp.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))

    def init_transform(self):
        transform_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.image_shape[1:]),
            torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            # torchvision.transforms.RandomHorizontalFlip(),  # TODO: necessary? remove? used in NVIDIA code
        ]

        img_transform = torchvision.transforms.Compose(transform_list)
        return img_transform

    def open_lmdb(self):
        self.env = lmdb.open(
            self.db_path,
            subdir=osp.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False, buffers=True)
        self.length = pickle.loads(self.txn.get(b"__len__"))
        self.keys = pickle.loads(self.txn.get(b"__keys__"))

    def __getitem__(self, index):
        if not hasattr(self, "txn"):
            self.open_lmdb()

        img, target = None, None
        byteflow = self.txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = io.BytesIO()
        # first element is image and second is path
        # see ImageFolderWithPaths
        buf.write(imgbuf[0])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"


# celebrity face 256x256
@register_dataset(name="celeba64")
class CelebA64Dataset(LMDBDataset):
    def __init__(self, *args, **kwargs):
        super(CelebA64Dataset, self).__init__(*args, **kwargs)
        assert self.image_shape == (3, 64, 64), "Image shape must be (3, 64, 64)"


@register_dataset(name="celeba256")
class CelebA256Dataset(LMDBDataset):
    def __init__(self, *args, **kwargs):
        super(CelebA256Dataset, self).__init__(*args, **kwargs)
        assert self.image_shape == (3, 256, 256), "Image shape must be (3, 256, 256)"


@register_dataset(name="ffhq256")
class FFHQ256Dataset(LMDBDataset):
    def __init__(self, *args, **kwargs):
        super(FFHQ256Dataset, self).__init__(*args, **kwargs)


# cats only
@register_dataset(name="afhq")
class AFHQDataset(LMDBDataset):
    def __init__(self, *args, **kwargs):
        super(AFHQDataset, self).__init__(*args, **kwargs)


#### torchvision datasets (we do not use MNIST)
@register_dataset(name="cifar10")
class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super(CIFAR10Dataset, self).__getitem__(index)
        return img, target, index


@register_dataset(name="svhn")
class SVHNDataset(torchvision.datasets.SVHN):
    def __getitem__(self, index):
        img, target = super(SVHNDataset, self).__getitem__(index)
        return img, target, index


@register_dataset(name="mnist")
class MNISTDataset(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        img, target = super(MNISTDataset, self).__getitem__(index)
        return img, target, index


def num_samples(dataset, train):
    if dataset == "celeba64" or dataset == "celeba256":
        return 27000 if train else 3000
    elif dataset == "imagenet-oord":
        return 1281147 if train else 50000
    elif dataset == "ffhq":
        return 63000 if train else 7000
    elif dataset == "mnist":
        return 60000 if train else 10000
    elif dataset == "cifar10":
        return 50000 if train else 10000
    elif dataset == "svhn":
        return 73257 if train else 26032
    else:
        raise NotImplementedError("dataset %s is unknown" % dataset)


def num_classes(dataset):
    if dataset == "celeba64" or dataset == "celeba256":
        return 40
    elif dataset == "imagenet-oord":
        return 1000
    elif dataset == "ffhq":
        return 18
    elif dataset == "mnist":
        return 10
    elif dataset == "cifar10":
        return 10
    elif dataset == "svhn":
        return 10
    else:
        raise NotImplementedError("dataset %s is unknown" % dataset)


def mean_std(dataset):
    if dataset == "cifar10":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise NotImplementedError("dataset %s is unknown" % dataset)
    return mean, std


class Binarize(object):
    """This class introduces a binarization transformation"""

    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToZeros(object):
    """
    For debugging purposes, this class sets all pixels to zero.
    """

    def __call__(self, pic):
        return torch.zeros_like(pic)

    def __repr__(self):
        return self.__class__.__name__ + "()"


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale from [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def raw_reader(path):
    with open(path, "rb") as f:
        bin_data = f.read()
    return bin_data


def dump_pickle(obj):
    """
    Serialize an object.

    Returns :
        The pickled representation of the object obj as a bytes object
    """
    return pickle.dumps(obj)


def torch2lmdb(torch_dataset, lmdb_path=None, write_frequency=5000) -> str:
    env = lmdb.open(lmdb_path, map_size=1024**3)
    txn = env.begin(write=True)
    idx = 0

    for idx, (img, label) in enumerate(torch_dataset):
        img_bytes = io.BytesIO()

        # Convert image to PIL if it's a tensor
        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)

        img.save(img_bytes, format="PNG")

        extra_data = None
        imgbuf = (img_bytes.getvalue(), extra_data)

        if isinstance(label, torch.Tensor):
            label = label.item()
        elif isinstance(label, np.ndarray):
            label = label.tolist()

        data = (imgbuf, label)
        txn.put(f"{idx}".encode(), pickle.dumps(data))

        if (idx + 1) % write_frequency == 0:
            txn.commit()
            print(f"Processed {idx + 1} images.")
            txn = env.begin(write=True)

    txn.commit()
    print(f"Processed {idx + 1} images in total.")

    with env.begin(write=True) as txn:
        keys = [f"{k}".encode() for k in range(idx + 1)]
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__len__", pickle.dumps(len(keys)))

    env.sync()
    env.close()
    return lmdb_path


def folder2lmdb(
    dpath, name="train_images", write_frequency=5000, num_workers=0, torch_dataset=None
) -> str:
    """
    Converts a folder of images to lmdb dataset and returns the path to the lmdb dataset.
    """
    if torch_dataset is None:
        directory = osp.expanduser(osp.join(dpath, name))
        print("Loading dataset from %s" % directory)
        dataset = torchvision.datasets.ImageFolder(directory, loader=raw_reader)
        data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers)
    # else:
    #     print("Using torch dataset")
    #     dataset = torch_dataset
    #     # conver
    #     data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers)
    #     name = dataset.__class__.__name__

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generating LMDB to %s" % lmdb_path)
    map_size = 30737418240  # this should be adjusted based on OS/db size
    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=map_size,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    print(
        f"Length of dataset: {len(dataset)}; Length of dataloader: {len(data_loader)}"
    )

    txn = db.begin(write=True)
    for idx, (data, label) in enumerate(data_loader):
        image = data
        label = label.numpy()
        txn.put("{}".format(idx).encode("ascii"), dump_pickle((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dump_pickle(keys))
        txn.put(b"__len__", dump_pickle(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

    return lmdb_path
