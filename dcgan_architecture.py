import torch
import torch.nn as nn
import numpy as np

from dcgan.stabilization_techniques import SpectralNorm


def weights_init_normal(m):
    # use string to locate conv and batchnorm layer, so do not use "Conv" or "BatchNorm" in other places, like class name
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight_bar"):
            # weight_bar is defined in 'spectral_normalization'
            torch.nn.init.normal_(m.weight_bar.data, 0.0, 0.02)
        else:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# MLP
class Generator_MLP(nn.Module):
    def __init__(self, nz, ngf, img_shape, n_gpu):
        super(Generator_MLP, self).__init__()
        self.n_gpu = n_gpu
        self.img_shape = img_shape  # (1 x height x width)

        def generator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *generator_block(nz, ngf * 2**2, normalize=False),
            *generator_block(ngf * 2**2, ngf * 2**3),
            *generator_block(ngf * 2**3, ngf * 2**4),
            *generator_block(ngf * 2**4, ngf * 2**5),
            nn.Linear(ngf * 2**5, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        if len(z.shape) == 4:
            z = z.squeeze()
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)  # reshape a vector to a image tensor
        return img


# pure convolution
class Generator_NoLL(nn.Module):
    def __init__(self, nz, nc, ngf, n_extra_layers_g, n_gpu):
        super(Generator_NoLL, self).__init__()
        self.n_gpu = n_gpu
        def generator_block(in_filters, out_filters, kernel_size, stride, padding):
            # transpose convolution upsamples (decompresses) the latent vector
            # w_out = (w_in - 1) * stride - 2 * padding + kernel
            # kernel_size=4, stride=2, padding=1 -> upscale to twice size
            # kernel_size=3, stride=1, padding=1 -> stay at the same size
            # when using transpose conv on a vector (z here), the length of z is the number of in_channels,
            # each channel has a size of 1x1 tensor, state size. nz x 1 x 1
            # w_in=1, stride=1, padding=0 -> output size = kernel_size
            block = [
                nn.ConvTranspose2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_filters),
                # nn.Dropout2d(0.5),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            return block


        self.model = nn.Sequential(
            # input is Z, state size. nz x 1 x 1, going into a convolution
            *generator_block(nz, ngf * 8, 2, 1, 0),
            # state size. (ngf * 8) x 2 x 2
            *generator_block(ngf * 8, ngf * 4, 4, 2, 1),
            # state size. (ngf * 4) x 4 x 4
            *generator_block(ngf * 4, ngf * 2, 4, 2, 1),
            # state size. (ngf * 2) x 8 x 8
            *generator_block(ngf * 2, ngf, 4, 2, 1),
            # state size. (ngf) x 16 x 16
        )

        for i in range(n_extra_layers_g):
            self.model.add_module('extra_layers-{0}-{1}-conv'.format(i, ngf),
                                  nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False))
            self.model.add_module('extra_layers-{0}-{1}-batchnorm'.format(i, ngf),
                                  nn.BatchNorm2d(ngf))
            self.model.add_module('extra_layers-{0}-{1}-relu'.format(i, ngf),
                                  nn.LeakyReLU(0.2, inplace=True))
        # these extra layers does not modify state size. (ngf) x 16 x 16


        # the last layer of generator should yield the image with the same size with the database imgs, use output to decide the dimension
        # w_out = (w_in - 1) * stride - 2 * padding + kernel
        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # state size. (nc) x 32 x 32
            # nn.ConvTranspose2d(ngf, nc, 8, 1, 0, bias=False),
            # state size. (nc) x 23 x 23
            nn.Tanh()  # transform value to (-1, 1)
        )


    def forward(self, z):
        out = self.model(z)
        # print(out.shape)
        img = self.last_layer(out)
        return img


# pure convolution
class Generator_NoLL_3(nn.Module):
    def __init__(self, nz, nc, ngf, n_extra_layers_g, n_gpu):
        super(Generator_NoLL_3, self).__init__()
        self.n_gpu = n_gpu
        def generator_block(in_filters, out_filters, kernel_size, stride, padding):
            # transpose convolution upsamples (decompresses) the latent vector
            # w_out = (w_in - 1) * stride - 2 * padding + kernel
            # kernel_size=4, stride=2, padding=1 -> upscale to twice size
            # kernel_size=3, stride=1, padding=1 -> stay at the same size
            # when using transpose conv on a vector (z here), the length of z is the number of in_channels,
            # each channel has a size of 1x1 tensor, state size. nz x 1 x 1
            # w_in=1, stride=1, padding=0 -> output size = kernel_size
            block = [
                nn.ConvTranspose2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_filters),
                nn.ReLU(True),
            ]
            return block


        self.model = nn.Sequential(
            # input is Z, state size. nz x 1 x 1, going into a convolution
            *generator_block(nz, ngf * 8, 4, 1, 0),
            # state size. (ngf * 8) x 4 x 4
            *generator_block(ngf * 8, ngf * 4, 4, 2, 1),
            # state size. (ngf * 4) x 8 x 8
            *generator_block(ngf * 4, ngf * 2, 4, 2, 1),
            # state size. (ngf * 2) x 16 x 16
        )

        for i in range(n_extra_layers_g):
            self.model.add_module('extra_layers-{0}-{1}-conv'.format(i, ngf),
                                  nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1))
            self.model.add_module('extra_layers-{0}-{1}-batchnorm'.format(i, ngf),
                                  nn.BatchNorm2d(ngf * 2))
            self.model.add_module('extra_layers-{0}-{1}-relu'.format(i, ngf),
                                  nn.ReLU(inplace=True))
        # these extra layers does not modify state size. (ngf) x 16 x 16


        # the last layer of generator should yield the image with the same size with the database imgs, use output to decide the dimension
        # w_out = (w_in - 1) * stride - 2 * padding + kernel
        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1),
            # state size. (nc) x 32 x 32
            nn.Tanh()  # transform value to (-1, 1)
        )


    def forward(self, z):
        out = self.model(z)
        # print(out.shape)
        img = self.last_layer(out)
        return img


# linear layer
class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, n_extra_layers_g, n_gpu):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.ngf = ngf
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(nz, (ngf * 2) * self.init_size ** 2))
        # state size. 8 x 8

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 16 x 16

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 32 x 32
        )

        for i in range(n_extra_layers_g):
            self.conv_blocks.add_module('extra_layers-{0}-{1}-conv'.format(i, ngf),
                                  nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False))
            self.conv_blocks.add_module('extra_layers-{0}-{1}-batchnorm'.format(i, ngf),
                                  nn.BatchNorm2d(ngf))
            self.conv_blocks.add_module('extra_layers-{0}-{1}-relu'.format(i, ngf),
                                  nn.LeakyReLU(0.2, inplace=True))
        # these extra layers does not modify state size. (ngf) x 32 x 32

        # the last layer of generator should yield the image with the same size with the database imgs, use output to decide the dimension
        # w_out = (w_in - 1) * stride - 2 * padding + kernel
        self.last_layer = nn.Sequential(
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            # state size. (nc) x 32 x 32
            nn.Tanh()  # transform value to (-1, 1)
        )

    def forward(self, z):
        if len(z.shape) == 4:
            z = z.squeeze()
        out = self.l1(z)
        out = out.view(out.shape[0], self.ngf * 2, self.init_size, self.init_size)
        # print(out.shape)
        out = self.conv_blocks(out)
        # print(out.shape)
        img = self.last_layer(out)
        return img


# linear layer, PixelShuffle
class Generator_PS(nn.Module):
    def __init__(self, nz, nc, ngf, n_extra_layers_g, n_gpu):
        super(Generator_PS, self).__init__()
        self.n_gpu = n_gpu
        self.ngf = ngf
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(nz, (ngf * 2**4) * self.init_size ** 2))
        # state size. 8 x 8

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2**4),
            # state size. (ngf * 2**4) x 8 x 8
            nn.PixelShuffle(upscale_factor=2),
            # state size. (ngf * 2**2) x 16 x 16
            nn.Conv2d(ngf * 2**2, ngf * 2**2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2**2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf * 2**2) x 16 x 16

            nn.PixelShuffle(upscale_factor=2),
            # state size. (ngf) x 32 x 32
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 32 x 32
        )

        for i in range(n_extra_layers_g):
            self.conv_blocks.add_module('extra_layers-{0}-{1}-conv'.format(i, ngf),
                                  nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False))
            self.conv_blocks.add_module('extra_layers-{0}-{1}-batchnorm'.format(i, ngf),
                                  nn.BatchNorm2d(ngf))
            self.conv_blocks.add_module('extra_layers-{0}-{1}-relu'.format(i, ngf),
                                  nn.LeakyReLU(0.2, inplace=True))
        # these extra layers does not modify state size. (ngf) x 32 x 32

        # the last layer of generator should yield the image with the same size with the database imgs, use output to decide the dimension
        # w_out = (w_in - 1) * stride - 2 * padding + kernel
        self.last_layer = nn.Sequential(
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            # state size. (nc) x 32 x 32
            nn.Tanh()  # transform value to (-1, 1)
        )

    def forward(self, z):
        if len(z.shape) == 4:
            z = z.squeeze()
        out = self.l1(z)
        out = out.view(out.shape[0], self.ngf * 2**4, self.init_size, self.init_size)
        # print(out.shape)
        out = self.conv_blocks(out)
        # print(out.shape)
        img = self.last_layer(out)
        return img


## ResNet
class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # state size x 2
        return self.model(x) + self.bypass(x)


class Generator_Res(nn.Module):
    def __init__(self, nz, nc, ngf, n_extra_layers_g, n_gpu):
        super(Generator_Res, self).__init__()
        self.n_gpu = n_gpu
        self.in_channels = ngf * 2**2

        self.dense = nn.Linear(nz, 4 * 4 * self.in_channels)
        # state size. in_channels x 4 x 4
        self.final = nn.Conv2d(self.in_channels, nc, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(self.in_channels, self.in_channels, stride=2),
            # state size. in_channels x 8 x 8
            ResBlockGenerator(self.in_channels, self.in_channels, stride=2),
            # state size. in_channels x 16 x 16
            ResBlockGenerator(self.in_channels, self.in_channels, stride=2),
            # state size. in_channels x 32 x 32
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            self.final,
            # state size. nc x 32 x 32
            nn.Tanh())

    def forward(self, z):
        if len(z.shape) == 4:
            z = z.squeeze()
        out = self.dense(z)
        out = out.view(-1, self.in_channels, 4, 4)
        img = self.model(out)
        return img

## ResNet with SN

class ResBlockDiscriminator_SN(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator_SN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )

        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator_SN(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator_SN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Discriminator_Res_SN(nn.Module):
    def __init__(self, nc, ndf, n_gpu):
        super(Discriminator_Res_SN, self).__init__()
        self.n_gpu = n_gpu
        self.out_channels = ndf * 2**2

        self.model = nn.Sequential(
                FirstResBlockDiscriminator_SN(nc, self.out_channels, stride=2),
                # state size. out_channels x 16 x 16
                ResBlockDiscriminator_SN(self.out_channels, self.out_channels, stride=2),
                # state size. out_channels x 8 x 8
                ResBlockDiscriminator_SN(self.out_channels, self.out_channels),
                # state size. out_channels x 8 x 8
                ResBlockDiscriminator_SN(self.out_channels, self.out_channels),
                # state size. out_channels x 8 x 8
                nn.ReLU(),
                nn.AvgPool2d(8),
                # state size. out_channels x 1 x 1
            )
        self.fc = nn.Linear(self.out_channels, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, img):
        out = self.model(img)
        out = out.view(-1, self.out_channels)   # equivalent to out = out.view(out.shape[0], -1), note there is a batch_size dimension
        validity = self.fc(out)
        validity = validity.view(img.shape[0], -1)
        return validity


## ResNet

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2,
                nn.AvgPool2d(2, stride=stride, padding=0)
                )

        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                self.bypass_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         nn.Conv2d(in_channels,out_channels, 1, 1, padding=0),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Discriminator_Res(nn.Module):
    def __init__(self, nc, ndf, n_gpu):
        super(Discriminator_Res, self).__init__()
        self.n_gpu = n_gpu
        self.out_channels = ndf * 2**2

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(nc, self.out_channels, stride=2),
                # state size. out_channels x 16 x 16
                ResBlockDiscriminator(self.out_channels, self.out_channels, stride=2),
                # state size. out_channels x 8 x 8
                ResBlockDiscriminator(self.out_channels, self.out_channels),
                # state size. out_channels x 8 x 8
                ResBlockDiscriminator(self.out_channels, self.out_channels),
                # state size. out_channels x 8 x 8
                nn.ReLU(),
                nn.AvgPool2d(8),
                # state size. out_channels x 1 x 1
            )
        self.fc = nn.Linear(self.out_channels, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)

    def forward(self, img):
        out = self.model(img)
        out = out.view(-1, self.out_channels)   # equivalent to out = out.view(out.shape[0], -1), note there is a batch_size dimension
        validity = self.fc(out)
        validity = validity.view(img.shape[0], -1)
        return validity


##
class Discriminator_NoLL(nn.Module):
    def __init__(self, nc, ndf, n_gpu):
        super(Discriminator_NoLL, self).__init__()
        self.n_gpu = n_gpu
        def discriminator_block(in_filters, out_filters, bn=True):
            # w_out = floor( (w_in - kernel_size + 2 * padding) / stride + 1 )
            # kernel_size=4, stride=2, padding=1 -> shrink to half of the size
            # kernel_size=3, stride=1, padding=1 -> stay at the same size
            # kernel_size=1, stride=1, padding=0 -> stay at the same size
            # kernel_size=img_size, stride=1, padding=0 -> reduce to scalar 1, used for the last layer of discriminator
            if bn:
                block = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(out_filters, 0.8),
                         nn.LeakyReLU(0.2, inplace=True)]
            else:
                block = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.LeakyReLU(0.2, inplace=True)]
            return block

        # shrink 2**3=8 times, floor(img_size/8) = 23/8 = 2
        # shrink 2**4=16 times, floor(img_size/16) = 32/16 = 2
        self.model = nn.Sequential(
            # input state size. nc x img_size x img_size
            *discriminator_block(nc, ndf, bn=False),  # the input layer does not need batchnorm for DCGAN
            # state size. ndf x (img_size/2) x (img_size/2)
            *discriminator_block(ndf, ndf * 2),
            # state size. (ndf*2) x (img_size/2**2) x (img_size/2**2)
            *discriminator_block(ndf * 2, ndf * 4),
            # state size. (ndf*4) x (img_size/2**3) x (img_size/2**3)
            *discriminator_block(ndf * 4, ndf * 8),
            # state size. (ndf*8) x (img_size/2**4) x (img_size/2**4)
        )
        # state size. (ndf*4) x 2 x 2

        # the last layer of discriminator should return a scalar value, but the input features depends on the previous conv layers,
        # use print to decide the dimensions, it varies from input to input
        self.last_layer = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),    # for 32 x 32
            # nn.Conv2d(ndf * 4, 1, 2, 1, 0, bias=False),  # for 23 x 23
            # state size. 1 x 1 x 1

            # originally, DCGAN output the value in (0, 1)
            # but if we choose loss function during training as BCEWithLogitsLoss, we do not need to define sigmoid() here
            # To better compare with other GANs, we do not use sigmoid() in D
            # nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        # print(out.shape)
        validity = self.last_layer(out)
        validity = validity.view(img.shape[0], -1)
        return validity


class Discriminator_NoLL_3(nn.Module):
    def __init__(self, nc, ndf, n_gpu):
        super(Discriminator_NoLL_3, self).__init__()
        self.n_gpu = n_gpu
        def discriminator_block(in_filters, out_filters, bn=True):
            # w_out = floor( (w_in - kernel_size + 2 * padding) / stride + 1 )
            # kernel_size=4, stride=2, padding=1 -> shrink to half of the size
            # kernel_size=3, stride=1, padding=1 -> stay at the same size
            # kernel_size=1, stride=1, padding=0 -> stay at the same size
            # kernel_size=x, stride=1, padding=0 -> w_out = w_in - (kernel_size - 1)
            # kernel_size=img_size, stride=1, padding=0 -> reduce to scalar 1, used for the last layer of discriminator
            if bn:
                block = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False),
                         # nn.BatchNorm2d(out_filters, 0.8),
                         nn.InstanceNorm2d(out_filters),
                         nn.LeakyReLU(0.2, inplace=True)]
            else:
                block = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.LeakyReLU(0.2, inplace=True)]
            return block


        self.model = nn.Sequential(
            # input state size. nc x img_size x img_size
            *discriminator_block(nc, ndf, bn=False),  # the input layer does not need batchnorm for DCGAN
            # state size. ndf x (img_size/2) x (img_size/2)
            *discriminator_block(ndf, ndf * 2),
            # state size. (ndf*2) x (img_size/2**2) x (img_size/2**2)
            *discriminator_block(ndf * 2, ndf * 4),
            # state size. (ndf*4) x (img_size/2**3) x (img_size/2**3)
        )
        # state size. (ndf*4) x 4 x 4

        # the last layer of discriminator should return a scalar value, but the input features depends on the previous conv layers,
        # use print to decide the dimensions, it varies from input to input
        self.last_layer = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),    # for 32 x 32
            # state size. 1 x 1 x 1

            # originally, DCGAN output the value in (0, 1)
            # but if we choose loss function during training as BCEWithLogitsLoss, we do not need to define sigmoid() here
            # To better compare with other GANs, we do not use sigmoid() in D
            # nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        # print(out.shape)
        validity = self.last_layer(out)
        validity = validity.view(img.shape[0], -1)
        return validity


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, n_gpu):
        super(Discriminator, self).__init__()
        self.n_gpu = n_gpu
        def discriminator_block(in_filters, out_filters, bn=True):
            # w_out = floor( (w_in - kernel_size + 2 * padding) / stride + 1 )
            # kernel_size=4, stride=2, padding=1 -> shrink to half of the size
            # kernel_size=3, stride=1, padding=1 -> stay at the same size
            # kernel_size=img_size, stride=1, padding=0 -> reduce to scalar 1, used for the last layer of discriminator
            if bn:
                block = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(out_filters, 0.8),
                         nn.LeakyReLU(0.2, inplace=True)]
            else:
                block = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.LeakyReLU(0.2, inplace=True)]
            return block

        # shrink 2**3=8 times, floor(img_size/8) = 23/8 = 2
        # shrink 2**4=16 times, floor(img_size/16) = 32/16 = 2
        self.model = nn.Sequential(
            # input state size. nc x img_size x img_size
            *discriminator_block(nc, ndf, bn=False),  # the input layer does not need batchnorm for DCGAN
            # state size. ndf x (img_size/2) x (img_size/2)
            *discriminator_block(ndf, ndf * 2),
            # state size. (ndf*2) x (img_size/2**2) x (img_size/2**2)
            *discriminator_block(ndf * 2, ndf * 4),
            # state size. (ndf*4) x (img_size/2**3) x (img_size/2**3)
            *discriminator_block(ndf * 4, ndf * 8),
            # state size. (ndf*8) x (img_size/2**4) x (img_size/2**4)
        )
        # state size. (ndf*8) x 2 x 2

        # the last layer of discriminator should return a scalar value, but the input features depends on the previous conv layers,
        # use print to decide the dimensions, it varies from input to input
        self.last_layer = nn.Sequential(
            nn.Linear((ndf * 8) * 2 * 2, 1),
            # state size. 1 x 1 x 1

            # originally, DCGAN output the value in (0, 1)
            # but if we choose loss function during training as BCEWithLogitsLoss, we do not need to define sigmoid() here
            # To better compare with other GANs, we do not use sigmoid() in D
            # nn.Sigmoid()

        )

    def forward(self, img):
        out = self.model(img)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        validity = self.last_layer(out)
        validity = validity.view(img.shape[0], -1)
        return validity


class Discriminator_SN(nn.Module):
    def __init__(self, nc, ndf, n_gpu):
        super(Discriminator_SN, self).__init__()
        self.n_gpu = n_gpu
        def discriminator_block(in_filters, out_filters, kernel_size, stride, padding):
            # w_out = floor( (w_in - kernel_size + 2 * padding) / stride + 1 )
            # kernel_size=4, stride=2, padding=1 -> shrink to half of the size
            # kernel_size=3, stride=1, padding=1 -> stay at the same size
            # kernel_size=img_size, stride=1, padding=0 -> reduce to scalar 1, used for the last layer of discriminator

            block = [SpectralNorm(nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)),
                     nn.LeakyReLU(0.2, inplace=True)]
            return block

        # shrink 2**3=8 times, floor(img_size/8) = 23/8 = 2
        # shrink 2**4=16 times, floor(img_size/16) = 32/16 = 2
        self.model = nn.Sequential(
            # input state size. nc x img_size x img_size
            *discriminator_block(nc, ndf, kernel_size=4, stride=2, padding=1),
            # state size. ndf x (img_size/2) x (img_size/2)
            *discriminator_block(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            # state size. (ndf*2) x (img_size/2**2) x (img_size/2**2)
            *discriminator_block(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            # state size. (ndf*4) x (img_size/2**3) x (img_size/2**3)
            *discriminator_block(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            # state size. (ndf*8) x (img_size/2**4) x (img_size/2**4)
        )
        # state size. (ndf*4) x 2 x 2

        # the last layer of discriminator should return a scalar value, but the input features depends on the previous conv layers,
        # use print to decide the dimensions, it varies from input to input
        self.last_layer = nn.Sequential(
            SpectralNorm(nn.Linear((ndf * 8) * 2 * 2, 1)),
            # state size. 1 x 1 x 1

            # originally, DCGAN output the value in (0, 1)
            # but if we choose loss function during training as BCEWithLogitsLoss, we do not need to define sigmoid() here
            # To better compare with other GANs, we do not use sigmoid() in D
            # nn.Sigmoid()

        )

    def forward(self, img):
        out = self.model(img)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        validity = self.last_layer(out)
        validity = validity.view(img.shape[0], -1)
        return validity


class Discriminator_MLP(nn.Module):
    def __init__(self, ndf, img_shape, n_gpu):
        super(Discriminator_MLP, self).__init__()
        self.gpu = n_gpu
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), ndf * 2**4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf * 2**4, ndf * 2**3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf * 2**3, 1),   # return a scalar score
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        validity = validity.view(img.shape[0], -1)
        return validity


class Discriminator_MLP_SN(nn.Module):
    def __init__(self, ndf, img_shape, n_gpu):
        super(Discriminator_MLP_SN, self).__init__()
        self.gpu = n_gpu
        self.img_shape = img_shape

        self.model = nn.Sequential(
            SpectralNorm(nn.Linear(int(np.prod(img_shape)), ndf * 2**4)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(ndf * 2**4, ndf * 2**3)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(ndf * 2**3, 1)),   # return a scalar score
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        validity = validity.view(img.shape[0], -1)
        return validity

