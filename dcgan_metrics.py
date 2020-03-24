import argparse
import math
import os
import numpy as np

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from PIL import Image
import matplotlib.pyplot as plt

import pickle

##

from dcgan.dcgan_architecture import weights_init_normal, Generator, Generator_NoLL, Generator_NoLL_3, Generator_MLP, Generator_PS, Generator_Res,\
                                     Discriminator, Discriminator_NoLL, Discriminator_NoLL_3, Discriminator_Res, Discriminator_MLP, Discriminator_SN, Discriminator_MLP_SN, Discriminator_Res_SN


from dcgan.utils import fixSeed, generateNoise

from dcgan.getInceptionScore import get_inception_score
from dcgan.getFIDScore import calculate_fid_given_paths


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_rootpath", type=str, default='../train_data_m', help="root path of training images")
    parser.add_argument("--outModelDir", type=str, default='./model', help="directory of saved model")
    parser.add_argument("--outGenImgsDir", type=str, default='./gen_images',
                        help="directory of output single images after training")
    parser.add_argument("--resultPlotDir", type=str, default='./result/',
                        help="directory of output images after training")


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

    parser.add_argument("--nsamples", type=int, default=100, help="number of images generated for each epoch")

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
    os.makedirs(opt.resultPlotDir, exist_ok=True)

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
        is_value_list = []
        fid_value_list = []
        for epoch in list(range(0, opt.n_epochs, opt.modelsave_interval)) + [opt.n_epochs - 1]:

            ## calculate inception score
            # image passed in inceptionV3 is normalized by mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] and in range of [-1, 1]
            # the mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]: required by inception_v3 since it is a pretrained on Imagenet
            # image passed in is normalized by mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] => transform_input=True
            # image passed in is normalized by mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500] => transform_input=False

            topo = datasets.ImageFolder(root=opt.outGenImgsDir + "/final_gen_images_{}".format(epoch),
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)   # in [-1, 1], so transform_input=False
                                        ]))
            is_value = get_inception_score(topo, transform_input=False, cuda=False, batch_size=opt.batch_size, resize=True, splits=10)
            is_value_list.append(is_value[0])
            print("epoch[{}/{}] IS: ".format(epoch, opt.n_epochs), is_value)

            ##
            # calculate FID score
            path = (opt.data_rootpath + '/refimgs', opt.outGenImgsDir + "/final_gen_images_{}/figures".format(epoch))
            # no transforms, so normalize_input=True
            # calculate_fid_given_paths() has its own procedure of defining dataset and corresponding transforms
            fid_value = calculate_fid_given_paths(path, batch_size=opt.batch_size, cuda=cuda, dims=2048,
                                                  resize_input=True, normalize_input=True, requires_grad=False)
            fid_value_list.append(fid_value)
            print("epoch[{}/{}] FID: ".format(epoch, opt.n_epochs), fid_value)

        ## save results
        pickle.dump([fid_value_list, is_value_list, opt], open(opt.resultPlotDir + '/gan_metrics_results.p', 'wb'), -1)

        # gan_metrics_results = pickle.load(open(opt.resultPlotDir + '/gan_metrics_results.p', "rb"))
        # fid_value_list, is_value_list, opt = tuple(gan_metrics_results)

        ##
        # plot IS
        fig = plt.figure(figsize=(10, 5))
        plt.title("Inception Scores During Training")
        plt.plot(list(range(0, opt.n_epochs, opt.modelsave_interval)) + [opt.n_epochs - 1], is_value_list, label="IS")
        plt.xlabel("Epoch")
        plt.ylabel("IS")
        plt.legend()
        plt.show()
        fig.savefig(opt.resultPlotDir + '/Inception_score.png')

        # plot IS
        fig = plt.figure(figsize=(10, 5))
        plt.title("FID Score During Training")
        plt.plot(list(range(0, opt.n_epochs, opt.modelsave_interval)) + [opt.n_epochs - 1], fid_value_list, label="FID")
        plt.xlabel("Epoch")
        plt.ylabel("FID")
        plt.legend()
        plt.show()
        fig.savefig(opt.resultPlotDir + '/FID_score.png')

