from torchvision import datasets
from torchvision import transforms

import numpy as np

class QuarterCrop(object):
    """Crop the image in a sample. Retain the right-top corner

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size

    def __call__(self, image):

        w, h = image.size  # read as a PIL image, return width x height, note Pytorch tensor is height x width
        c_h, c_w = self.crop_size

        top = 0
        left = c_w
        right = left + (w - c_w)
        bottom = top + (h - c_h)

        image = image.crop((left, top, right, bottom))

        return image


def createDataset(dataset, data_rootpath, nc, img_cropsize, img_size, isTrain=True):
    if dataset == 'folder':
        # folder dataset
        dataset = datasets.ImageFolder(root=data_rootpath,
                                       transform=transforms.Compose([
                                           # transforms.CenterCrop(img_cropsize),  # crop extra white borders
                                           QuarterCrop(img_cropsize),
                                           transforms.Grayscale(num_output_channels=1), # transform to greyscale image with one channel
                                           # transforms.Resize((img_size, img_size)),   # resize input image to be consistent with the output of generator
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5] * nc, [0.5] * nc)])  # PIL image used by PyTorch is in (0, 1), bring images to (-1,1)
                                       )

    elif dataset == 'lsun':
        dataset = datasets.LSUN(root=data_rootpath, classes=['bedroom_train'],
                                transform=transforms.Compose([
                                    transforms.CenterCrop(256),
                                    transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

    elif dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=data_rootpath, train=isTrain, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif dataset == 'mnist':
        dataset = datasets.MNIST(root=data_rootpath, train=isTrain, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(img_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5]),
                                 ]))
    elif dataset == 'fashion-mnist':
        dataset = datasets.FashionMNIST(root=data_rootpath, train=isTrain, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5]),
                                        ]))
    elif dataset == 'stl10':
        dataset = datasets.STL10(root=data_rootpath, train=isTrain, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

    return dataset