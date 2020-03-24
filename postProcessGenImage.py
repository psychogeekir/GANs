import argparse
import math
import os
import numpy as np
import time

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from PIL import Image
import matplotlib.pyplot as plt

import scipy.ndimage

##

from dcgan.dcgan_architecture import weights_init_normal, Generator, Generator_NoLL, Generator_NoLL_3, Generator_MLP, Generator_PS, Generator_Res,\
                                     Discriminator, Discriminator_NoLL, Discriminator_NoLL_3, Discriminator_Res, Discriminator_MLP, Discriminator_SN, Discriminator_MLP_SN, Discriminator_Res_SN


from dcgan.utils import fixSeed, generateNoise


##
def restoreFullImage(gen_imgs):
    # a right-top quarter 32 X 32 -> full 64 x 64
    half_size = gen_imgs.shape[2]
    full_size = 2 * gen_imgs.shape[2]

    gen_imgs_full = torch.zeros((gen_imgs.shape[0], gen_imgs.shape[1], full_size, full_size), dtype=torch.float)
    for i in range(gen_imgs.shape[0]):
        gen_imgs_full.data[i, 0, 0:half_size, half_size:] = gen_imgs.data[i][0]

        gen_imgs_lt = np.flip(gen_imgs.data[i][0].cpu().numpy(), 1).copy()
        gen_imgs_ld = np.flip(gen_imgs_lt, 0).copy()
        gen_imgs_rd = np.flip(gen_imgs_ld, 1).copy()
        gen_imgs_lt = torch.from_numpy(gen_imgs_lt)
        gen_imgs_ld = torch.from_numpy(gen_imgs_ld)
        gen_imgs_rd = torch.from_numpy(gen_imgs_rd)

        gen_imgs_full.data[i, 0, 0:half_size, 0:half_size] = gen_imgs_lt.data
        gen_imgs_full.data[i, 0, half_size:, 0:half_size] = gen_imgs_ld.data
        gen_imgs_full.data[i, 0, half_size:, half_size:] = gen_imgs_rd.data
    return gen_imgs_full

def filterGenImgConv(gen_imgs, thres=100 / 255, mu=(0.5, 0.5, 0.5), sig=(0.5, 0.5, 0.5), iFig=None):
    # mu and sig are the transform.Normalize() parameters
    if isinstance(mu, tuple):
        nc = len(mu)
    else:
        nc = 1

    if nc == 1:
        imgs_rt = gen_imgs * torch.tensor(sig).view(nc, 1, 1).expand(gen_imgs.size(0), nc, 1, 1) + torch.tensor(mu).view(nc, 1, 1).expand(gen_imgs.size(0), nc, 1, 1)
    else:
        imgs_rt = gen_imgs * torch.tensor(sig).view(nc, 1, 1).expand(gen_imgs.size(0), nc, 1, 1) + torch.tensor(mu).view(nc, 1, 1).expand(gen_imgs.size(0), nc, 1, 1)

    if iFig is not None:
        fig = plt.figure(figsize=(45, 15))
        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.title("Original")
        plt.imshow(np.transpose(imgs_rt[iFig, :, :, :].numpy(), (1, 2, 0)).squeeze(), cmap='gray', vmin=0, vmax=1)
        # Image.fromarray(imgs_rt[iFig, :, :, :], 'L').show()

    # matlab imfilter() equivalent
    # N-D filtering of multidimensional images
    windowWidth = 3
    weight = -1 * np.ones((windowWidth, windowWidth))
    index = math.ceil(windowWidth / 2)
    weight[index - 1, index - 1] = -sum(weight.flatten('F')) - 1 + windowWidth ** 2
    weight = weight / sum(weight.flatten('F'))

    weight = torch.from_numpy(weight).type(torch.FloatTensor)
    weight = weight[None, None, :, :].expand(imgs_rt.shape[1], imgs_rt.shape[1], windowWidth, windowWidth)

    # since nn.Conv2d does not have 'replicate' padding_mode == 'nearest' mode in scipy.ndimage.convolve()
    # pad=(1, 1, 1, 1) from last dimension, backward add padding pair
    # convolution in Machine Learning is actually cross-correlation
    # since the kernel is symmetric, conv and correlation are the same
    imgs_rt_filtered0 = torch.nn.functional.conv2d(
        torch.nn.functional.pad(imgs_rt, pad=(1, 1, 1, 1), mode='replicate'), weight, bias=None, stride=1, padding=0)

    if iFig is not None:
        plt.subplot(1, 3, 2)
        plt.axis("off")
        plt.title("Convolved")
        plt.imshow(np.transpose(imgs_rt_filtered0[iFig, :, :, :].numpy(), (1, 2, 0)).squeeze(), cmap='gray', vmin=0, vmax=1)
        # Image.fromarray(imgs_rt_filtered0[iFig, :, :, :], 'L').show()

    # threshold filter
    imgs_rt_filtered0[imgs_rt_filtered0 < thres] = 0  # black
    imgs_rt_filtered0[imgs_rt_filtered0 >= thres] = 255 / 255  # white

    if iFig is not None:
        plt.subplot(1, 3, 3)
        plt.axis("off")
        plt.title("Thresold")
        plt.imshow(np.transpose(imgs_rt_filtered0[iFig, :, :, :].numpy(), (1, 2, 0)).squeeze(), cmap='gray', vmin=0, vmax=1)
        # Image.fromarray(imgs_rt_filtered0[iFig, :, :, :], 'L').show()
        fig.show()
        fig.savefig('convfilter.png')

    imgs_rt_filtered = imgs_rt_filtered0

    return imgs_rt_filtered

def filterGenImg(gen_imgs, img_size, thres=100 / 255, mu=(0.5, 0.5, 0.5), sig=(0.5, 0.5, 0.5), iFig=None):
    # mu and sig are the transform.Normalize() parameters
    if isinstance(mu, tuple):
        nc = len(mu)
    else:
        nc = 1

    # the gen_imgs is actually a right top part, batch_size x nc x 32 x 32
    imgs_rt_filtered = torch.zeros((gen_imgs.shape[0], gen_imgs.shape[1], img_size, img_size), dtype=torch.float)

    for i in range(gen_imgs.shape[0]):

        # gen_imgs.data[i][0] is between -1 and 1, transform to [0 1] first
        # to_pil_image() directly transform tensor to PIL.image object (uint8 [0 255] array with mode='L')  32 x 32 x nc
        if nc == 1:
            img_rt = np.asarray(transforms.ToPILImage(mode='L')(
                gen_imgs.data[i] * torch.tensor(sig).view(nc, 1, 1) + torch.tensor(mu).view(nc, 1, 1)))
        else:
            img_rt = np.asarray(transforms.ToPILImage(mode='RGB')(
                gen_imgs.data[i] * torch.tensor(sig).view(nc, 1, 1) + torch.tensor(mu).view(nc, 1, 1)))

        if i == iFig:
            fig = plt.figure(figsize=(60, 15))
            plt.subplot(1, 4, 1)
            plt.axis("off")
            plt.title("Original")
            plt.imshow(img_rt, cmap='gray', vmin=0, vmax=255)
            # Image.fromarray(img_rt, 'L').show()

        # matlab imfilter() equivalent
        # N-D filtering of multidimensional images
        windowWidth = 3
        kernel = -1 * np.ones((windowWidth, windowWidth))
        index = math.ceil(windowWidth / 2)
        kernel[index - 1, index - 1] = -sum(kernel.flatten('F')) - 1 + windowWidth ** 2
        kernel = kernel / sum(kernel.flatten('F'))
        imgs_rt_filtered0 = scipy.ndimage.convolve(img_rt / 255, kernel, mode='nearest')

        if i == iFig:
            plt.subplot(1, 4, 2)
            plt.axis("off")
            plt.title("Convolved")
            plt.imshow(imgs_rt_filtered0, cmap='gray', vmin=0, vmax=1)

        # threshold filter
        imgs_rt_filtered0[imgs_rt_filtered0 < thres] = 0  # black
        imgs_rt_filtered0[imgs_rt_filtered0 >= thres] = 255 / 255  # white

        if i == iFig:
            plt.subplot(1, 4, 3)
            plt.axis("off")
            plt.title("Thresold")
            plt.imshow(imgs_rt_filtered0, cmap='gray', vmin=0, vmax=1)

        # convert PIL.Image to be converted to tensor [0 1]
        imgs_rt_filtered[i, :, :, :] = transforms.ToTensor()(imgs_rt_filtered0)

        if i == iFig:
            plt.subplot(1, 4, 4)
            plt.axis("off")
            plt.title("Thresold")
            tensor = imgs_rt_filtered[i, :, :, :]
            if tensor.dim() == 2:  # single image H x W
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == 3:  # single image
                if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
                    tensor = torch.cat((tensor, tensor, tensor), 0)
            plt.imshow(np.transpose(tensor.numpy(), (1, 2, 0)))   # the default input of imshow() is either hxw or hxwx3
            fig.show()
            fig.savefig('slowfilter.png')

    return imgs_rt_filtered


# removing single isolated pixels using convolution
def accumulateImgConv(imgs_filtered, iFig=None):

    imgs_final = imgs_filtered.clone().detach()

    ## single multi-channel conv

    kernel = torch.nn.Conv2d(imgs_filtered.shape[1], 5,
                               kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros')
    weight = torch.FloatTensor([
                                [[1, 1, 1], [1, 0, 1], [1, 1, 1]],   # neighbor sum
                                [[1, 1, 1], [1, 0, 1], [0, 0, 0]],   # upper sum
                                [[0, 0, 0], [1, 0, 1], [1, 1, 1]],   # bottom sum
                                [[1, 1, 0], [1, 0, 0], [1, 1, 0]],   # left sum
                                [[0, 1, 1], [0, 0, 1], [0, 1, 1]],   # right sum
                               ])
    weight = weight.unsqueeze(1)   # unsqueeze to match the kernel.weight.size(): 5 x 1 x 3 x 3
    kernel.weight = torch.nn.Parameter(weight, requires_grad=False)

    delta = kernel(imgs_final)
    delta_s = delta[:, 0:1, :, :]  # 0:1 retain the same dimension with delta
    delta_su = delta[:, 1:2, :, :]
    delta_sd = delta[:, 2:3, :, :]
    delta_sl = delta[:, 3:4, :, :]
    delta_sr = delta[:, 4:, :, :]

    ## multiple single-channel conv
    # kernel_s = torch.nn.Conv2d(imgs_filtered.shape[1], imgs_filtered.shape[1],
    #                            kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros')
    # weight = torch.FloatTensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # weight = weight[None, None, :, :].expand(kernel_s.weight.size())
    # kernel_s.weight = torch.nn.Parameter(weight, requires_grad=False)
    #
    # kernel_su = torch.nn.Conv2d(imgs_filtered.shape[1], imgs_filtered.shape[1],
    #                             kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros')
    # weight = torch.FloatTensor([[1, 1, 1], [1, 0, 1], [0, 0, 0]])
    # weight = weight[None, None, :, :].expand(kernel_su.weight.size())
    # kernel_su.weight = torch.nn.Parameter(weight, requires_grad=False)
    #
    # kernel_sd = torch.nn.Conv2d(imgs_filtered.shape[1], imgs_filtered.shape[1],
    #                             kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros')
    # weight = torch.FloatTensor([[0, 0, 0], [1, 0, 1], [1, 1, 1]])
    # weight = weight[None, None, :, :].expand(kernel_sd.weight.size())
    # kernel_sd.weight = torch.nn.Parameter(weight, requires_grad=False)
    #
    # kernel_sl = torch.nn.Conv2d(imgs_filtered.shape[1], imgs_filtered.shape[1],
    #                             kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros')
    # weight = torch.FloatTensor([[1, 1, 0], [1, 0, 0], [1, 1, 0]])
    # weight = weight[None, None, :, :].expand(kernel_sl.weight.size())
    # kernel_sl.weight = torch.nn.Parameter(weight, requires_grad=False)
    #
    # kernel_sr = torch.nn.Conv2d(imgs_filtered.shape[1], imgs_filtered.shape[1],
    #                             kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros')
    # weight = torch.FloatTensor([[0, 1, 1], [0, 0, 1], [0, 1, 1]])
    # weight = weight[None, None, :, :].expand(kernel_sr.weight.size())
    # kernel_sr.weight = torch.nn.Parameter(weight, requires_grad=False)
    #
    # delta_s = kernel_s(imgs_final)
    # delta_su = kernel_su(imgs_final)
    # delta_sd = kernel_sd(imgs_final)
    # delta_sl = kernel_sl(imgs_final)
    # delta_sr = kernel_sr(imgs_final)

    ## remove isolated 1, white
    criteria = (imgs_final == 1) * ((delta_s <= 1) + (delta_su * delta_sd * delta_sl * delta_sr == 0))
    imgs_final[criteria] = 0  # black

    ##
    if iFig is not None:
        fig = plt.figure(figsize=(30, 15))

        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Original")
        plt.imshow(np.transpose(imgs_filtered[iFig, :, :, :].numpy(), (1, 2, 0)).squeeze(), cmap='gray', vmin=0, vmax=1)

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("After removing isolated 1")
        plt.imshow(np.transpose(imgs_final[iFig, :, :, :].numpy(), (1, 2, 0)).squeeze(), cmap='gray', vmin=0, vmax=1)

        fig.show()
        fig.savefig('convaccum1.png')

    ## remove isolated 0, black, reverse the image
    imgs_final_flip = 1 - imgs_final

    ## single multi-channel conv
    delta = kernel(imgs_final_flip)
    delta_s = delta[:, 0:1, :, :]
    delta_su = delta[:, 1:2, :, :]
    delta_sd = delta[:, 2:3, :, :]
    delta_sl = delta[:, 3:4, :, :]
    delta_sr = delta[:, 4:, :, :]

    ## multiple single-channel conv
    # delta_s = kernel_s(imgs_final_flip)
    # delta_su = kernel_su(imgs_final_flip)
    # delta_sd = kernel_sd(imgs_final_flip)
    # delta_sl = kernel_sl(imgs_final_flip)
    # delta_sr = kernel_sr(imgs_final_flip)

    ##
    criteria = (imgs_final_flip == 1) * ((delta_s <= 1) + (delta_su * delta_sd * delta_sl * delta_sr == 0))
    imgs_final_flip[criteria] = 0

    imgs_final = 1 - imgs_final_flip

    ##
    if iFig is not None:
        fig = plt.figure(figsize=(30, 15))

        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Original")
        plt.imshow(np.transpose(imgs_filtered[iFig, :, :, :].numpy(), (1, 2, 0)).squeeze(), cmap='gray', vmin=0, vmax=1)

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("After removing isolated 1 and 0")
        plt.imshow(np.transpose(imgs_final[iFig, :, :, :].numpy(), (1, 2, 0)).squeeze(), cmap='gray', vmin=0, vmax=1)

        fig.show()
        fig.savefig('convaccum2.png')

    return imgs_final


# removing isolated pixels one by one, slow
def accumulateImg(imgs_filtered, fullImgSize, iFig=None):
    imgs_final = torch.zeros((imgs_filtered.shape[0], imgs_filtered.shape[1], fullImgSize, fullImgSize),
                             dtype=torch.float)
    for k in range(imgs_filtered.shape[0]):
        x_matrix = imgs_filtered[k, 0, :, :].clone().numpy()  # grayscale image
        n_px = x_matrix.shape[0]
        n_iso = 0

        # remove isolated 1
        x_augmatrix = np.zeros((n_px + 2, n_px + 2))
        x_augmatrix[1:n_px + 1, 1:n_px + 1] = x_matrix

        for i in range(1, n_px + 1, 1):
            for j in range(1, n_px + 1, 1):
                local_matrix = x_augmatrix[i - 1: i + 2, j - 1: j + 2]
                delta_s = sum(sum(local_matrix)) - local_matrix[1, 1]
                delta_su = sum(sum(local_matrix[0:2, :])) - local_matrix[1, 1]
                delta_sd = sum(sum(local_matrix[1:3, :])) - local_matrix[1, 1]
                delta_sl = sum(sum(local_matrix[:, 0:2])) - local_matrix[1, 1]
                delta_sr = sum(sum(local_matrix[:, 1:3])) - local_matrix[1, 1]
                if local_matrix[1, 1] == 1:
                    if delta_s <= 1:
                        x_matrix[i - 1, j - 1] = 0
                        n_iso += 1
                    elif delta_su * delta_sd * delta_sl * delta_sr == 0:
                        x_matrix[i - 1, j - 1] = 0
                        n_iso += 1

        if k == iFig:
            fig = plt.figure(figsize=(30, 15))

            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Original")
            plt.imshow(np.transpose(imgs_filtered[iFig, :, :, :].numpy(), (1, 2, 0)).squeeze(), cmap='gray', vmin=0,
                       vmax=1)

            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("After removing isolated 1")
            plt.imshow(x_matrix, cmap='gray', vmin=0, vmax=1)

            fig.show()
            fig.savefig('slowaccum1.png')

        # remove isolated 0
        x_matrix_flip = 1 - x_matrix
        x_augmatrix = np.zeros((n_px + 2, n_px + 2))
        x_augmatrix[1:n_px + 1, 1:n_px + 1] = x_matrix_flip

        for i in range(1, n_px + 1, 1):
            for j in range(1, n_px + 1, 1):
                local_matrix = x_augmatrix[i - 1:i + 2, j - 1:j + 2]
                delta_s = sum(sum(local_matrix)) - local_matrix[1, 1]
                delta_su = sum(sum(local_matrix[0:2, :])) - local_matrix[1, 1]
                delta_sd = sum(sum(local_matrix[1:3, :])) - local_matrix[1, 1]
                delta_sl = sum(sum(local_matrix[:, 0:2])) - local_matrix[1, 1]
                delta_sr = sum(sum(local_matrix[:, 1:3])) - local_matrix[1, 1]
                if local_matrix[1, 1] == 1:
                    if delta_s <= 1:
                        x_matrix[i - 1, j - 1] = 1
                        n_iso += 1
                    elif delta_su * delta_sd * delta_sl * delta_sr == 0:
                        x_matrix[i - 1, j - 1] = 1
                        n_iso += 1

        if k == iFig:
            fig = plt.figure(figsize=(30, 15))

            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Original")
            plt.imshow(np.transpose(imgs_filtered[iFig, :, :, :].numpy(), (1, 2, 0)).squeeze(), cmap='gray', vmin=0,
                       vmax=1)

            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("After removing isolated 1 and 0")
            plt.imshow(x_matrix, cmap='gray', vmin=0, vmax=1)

            fig.show()
            fig.savefig('slowaccum2.png')

        imgs_final[k, 0, :, :] = torch.from_numpy(x_matrix)
        # print('figure {} accumulated'.format(k))

    return imgs_final


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--outModelDir", type=str, default='./model', help="directory of saved model")
    parser.add_argument("--outGenImgsDir", type=str, default='./gen_images',
                        help="directory of output single images after training")

    parser.add_argument("--img_cropsize", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")

    parser.add_argument("--manualSeed", type=int, default=999, help="seed for reproducibility")
    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument('--n_extra_layers_g', type=int, default=0, help='number of extra conv layers in G')
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=0, help="Number of GPUs available. Use 0 for CPU mode.")

    parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--nc", type=int, default=1, help="number of image channels")
    parser.add_argument('--ngf', type=int, default=32, help="size of feature maps in generator")
    parser.add_argument('--ndf', type=int, default=32, help="size of feature maps in discriminator")

    parser.add_argument("--binaryNoise", type=bool, default=False,
                        help="whether latent variable z from bernoulli distribution, with prob=0.5")

    parser.add_argument("--nsamples", type=int, default=5000, help="number of images generated for each epoch")

    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument("--modelsave_interval", type=int, default=10, help="interval between model saved")
    parser.add_argument("--netG", type=str, default='True',
                        help="previously trained G to be loaded")  # ./model/netG_epoch_100.pth
    parser.add_argument("--netD", type=str, default='',
                        help="previously trained D to be loaded")  # ./model/netD_epoch_100.pth

    opt = parser.parse_args()

    img_shape = (opt.nc, opt.img_size, opt.img_size)
    cuda = True if torch.cuda.is_available() else False

    fixSeed(opt.manualSeed, cuda)

    print(opt)

    ##
    os.makedirs(opt.outGenImgsDir, exist_ok=True)

    ## Decide which device we want to run on
    device = torch.device("cuda:0" if (cuda and opt.n_gpu > 0) else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # for moving tensor from cpu to gpu

    ##

    # Initialize generator and discriminator
    # generator = Generator(nz=opt.nz, nc=opt.nc, ngf=opt.ngf, n_extra_layers_g=opt.n_extra_layers_g, n_gpu=opt.n_gpu).to(device)
    # generator = Generator_PS(nz=opt.nz, nc=opt.nc, ngf=opt.ngf, n_extra_layers_g=opt.n_extra_layers_g, n_gpu=opt.n_gpu).to(device)
    generator = Generator_NoLL(nz=opt.nz, nc=opt.nc, ngf=opt.ngf, n_extra_layers_g=opt.n_extra_layers_g, n_gpu=opt.n_gpu).to(device)
    # generator = Generator_MLP(nz=opt.nz, ngf=opt.ngf, img_shape=img_shape, n_gpu=opt.n_gpu).to(device)

    # discriminator = Discriminator(nc=opt.nc, ndf=opt.ndf, n_gpu=opt.n_gpu).to(device)
    # discriminator = Discriminator_NoLL(nc=opt.nc, ndf=opt.ndf, n_gpu=opt.n_gpu).to(device)
    # discriminator = Discriminator_SN(nc=opt.nc, ndf=opt.ndf, n_gpu=opt.n_gpu).to(device)
    # discriminator = Discriminator_MLP(ndf=opt.ndf, img_shape=img_shape, n_gpu=opt.n_gpu).to(device)
    # discriminator = Discriminator_MLP_SN(ndf=opt.ndf, img_shape=img_shape, n_gpu=opt.n_gpu).to(device)

    ##

    with torch.no_grad():
        for epoch in list(range(0, opt.n_epochs, opt.modelsave_interval)) + [opt.n_epochs - 1]:
            os.makedirs(opt.outGenImgsDir + "/final_gen_images_{}".format(epoch), exist_ok=True)
            os.makedirs(opt.outGenImgsDir + "/final_gen_images_{}/figures".format(epoch), exist_ok=True)
            generator.load_state_dict(torch.load(opt.outModelDir + '/netG_epoch_{}.pth'.format(epoch)))

            z = generateNoise(opt.nsamples, opt.nz, opt.binaryNoise, Tensor)

            # Generate a batch of quarter images
            candidate_imgs = generator(z).detach().cpu()

            # plt.figure(figsize=(8, 8))
            # plt.axis("off")
            # plt.title("Training Images")
            # # since the real_batch has been transformed to [-1, 1], to correctly show it, transformed it back to [0, 1] by make_grid(..., normalize=True)
            # plt.imshow(np.transpose(make_grid(candidate_imgs[:64], padding=2, normalize=True).numpy(), (1, 2, 0)))
            # plt.show()

            # filter the quarter images
            # start = time.time()
            # candidate_imgs_filtered = filterGenImg(candidate_imgs, opt.img_size, thres=100 / 255, mu=0.5, sig=0.5, iFig=23)
            # end = time.time()
            # print('orignal filter: ', end-start)

            # start = time.time()
            candidate_imgs_filtered = filterGenImgConv(candidate_imgs, thres=100 / 255, mu=0.5, sig=0.5, iFig=None)
            # end = time.time()
            # print('conv filter: ', end - start)

            # restore to the full images
            candidate_imgs_full = restoreFullImage(candidate_imgs_filtered)

            # remove isolated pixels
            # start = time.time()
            # candidate_imgs_final = accumulateImg(candidate_imgs_full, candidate_imgs_full.shape[2], iFig=23)
            # end = time.time()
            # print('original accum: ', end - start)

            # start = time.time()
            candidate_imgs_final = accumulateImgConv(candidate_imgs_full, iFig=None)
            # end = time.time()
            # print('conv accum: ', end - start)

            # save figures one by one, note they are already binary, do not use normalize
            for i in range(opt.nsamples):
                save_image(candidate_imgs_final.data[i], opt.outGenImgsDir + "/final_gen_images_{}/figures/{}.png".format(epoch, i),
                           nrow=1, normalize=False)

            print('epoch [{}/{}] finished'.format(epoch, opt.n_epochs))

