import argparse
import os
import random
import numpy as np
import math

from torchvision import transforms
from torchvision.utils import save_image, make_grid

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

##
from dcgan.prepareDataset import createDataset

from dcgan.dcgan_architecture import weights_init_normal, Generator, Generator_NoLL, Generator_NoLL_3, Generator_MLP, Generator_PS, Generator_Res,\
                                     Discriminator, Discriminator_NoLL, Discriminator_NoLL_3, Discriminator_Res, Discriminator_MLP, Discriminator_SN, Discriminator_MLP_SN, Discriminator_Res_SN

from dcgan.stabilization_techniques import smooth_positive_labels, smooth_negative_labels, noisy_labels, \
                                           compute_gradient_penalty

from dcgan.utils import fixSeed, plotDAccuracy,  \
                        plotRealBatch, plotDScore, plotLoss, plotGEvolution, PlotRealFake

##

def generateNoise(batch_size, nz, binaryNoise):
    if binaryNoise:
        bernoulli_prob = Tensor(batch_size, nz, 1, 1).fill_(0.5)
        z = torch.bernoulli(bernoulli_prob)
    else:
        z = Tensor(np.random.normal(0, 1, (batch_size, nz, 1, 1)))
    return z



##
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist', help='folder | cifar10 | mnist | lsun')
    parser.add_argument("--data_rootpath", type=str, default='./mnist', help="root path of training images")
    parser.add_argument("--outModelDir", type=str, default='./model', help="directory of saved model")
    parser.add_argument("--outTrainImgsDir", type=str, default='./images', help="directory of output images during training")
    parser.add_argument("--outGenImgsDir", type=str, default='./candidates', help="directory of output images after training")
    parser.add_argument("--resultPlotDir", type=str, default='./result/', help="directory of output images after training")

    parser.add_argument("--img_cropsize", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")

    parser.add_argument("--manualSeed", type=int, default=999, help="seed for reproducibility")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument('--n_extra_layers_g', type=int, default=0, help='number of extra conv layers in G')
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=0, help="Number of GPUs available. Use 0 for CPU mode.")

    parser.add_argument("--use_ResNet", type=bool, default=False, help="whether to use ResNet to build D and G")

    parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--nc", type=int, default=1, help="number of image channels")
    parser.add_argument('--ngf', type=int, default=32, help="size of feature maps in generator")
    parser.add_argument('--ndf', type=int, default=32, help="size of feature maps in discriminator")

    # Adam or RMSprop optimizer
    parser.add_argument("--optimizer", type=str, default='Adam', help="use Adam or RMSprop optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_critic", type=int, default=5, help="interval of training generator per discriminator")
    # common values
    #     lr     |   beta1  |  beta2  |    n_critic    |
    # --------------------------------------------------
    #    1e-4    |    0.5   |  0.900  |        5       |
    # --------------------------------------------------
    #    1e-4    |    0.5   |  0.999  |        1       |
    # --------------------------------------------------
    #    2e-4    |    0.5   |  0.999  |        1       |   DCGAN, LAMBDA=10
    # --------------------------------------------------
    #    1e-4    |    0.0   |  0.900  |        5       |   WGAN_GP, LAMBDA=10
    # --------------------------------------------------
    #    1e-4    |    0.5   |  0.999  |        5       |   WGAN_GP, LAMBDA=10
    # --------------------------------------------------
    #    1e-3    |    0.0   |  0.900  |        1       |
    # --------------------------------------------------
    #    5e-5    |     -    |    -    |        5       |   WGAN with weight clipping


    parser.add_argument("--loss_func", type=str, default='wasserstein',
                        help="which loss funciton to be used: standard, wasserstein, hinge, softplus")

    parser.add_argument("--use_GP", type=bool, default=True, help="whether to use gradient penalty")
    parser.add_argument("--use_dragan", type=bool, default=False, help="whether to use DRAGAN gradient penalty")
    parser.add_argument("--center", type=float, default=1.0, help="center (Lipschitz constant) to calculate of gradient penalty: 0 or 1")
    parser.add_argument("--LAMBDA", type=float, default=100, help="coefficient of gradient penalty")

    parser.add_argument("--use_clip", type=bool, default=False, help="whether to clip weights value of D in WGAN")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for D weights in WGAN")

    parser.add_argument("--binaryNoise", type=bool, default=False, help="latent variable z from bernoulli distribution, with prob=0.5")
    parser.add_argument('--use_smoothlabel', type=bool, default=False, help='use soft label for training D')
    parser.add_argument('--use_noisylabel', type=bool, default=False, help='flip label for training D')

    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument("--modelsave_interval", type=int, default=10, help="interval between model saved")
    parser.add_argument("--netG", type=str, default='', help="previously trained G to be loaded")  # ./model/netG_epoch_100.pth
    parser.add_argument("--netD", type=str, default='', help="previously trained D to be loaded")  # ./model/netD_epoch_100.pth

    opt = parser.parse_args()

    cuda = True if torch.cuda.is_available() else False

    # opt.manualSeed = random.randint(1, 10000)  # fix seed, a scalar
    # fixSeed(opt.manualSeed, cuda)

    print(opt)

    # the actual picture should be put in a subfolder under ./data, e.g., ./data/refimgs/
    os.makedirs(opt.data_rootpath, exist_ok=True)
    os.makedirs(opt.outModelDir, exist_ok=True)
    os.makedirs(opt.outTrainImgsDir, exist_ok=True)
    os.makedirs(opt.outGenImgsDir, exist_ok=True)
    os.makedirs(opt.resultPlotDir, exist_ok=True)

    ## Configure data loader
    img_shape = (opt.nc, opt.img_size, opt.img_size)
    dataSet = createDataset(opt.dataset, opt.data_rootpath, opt.nc, opt.img_cropsize, opt.img_size)
    dataloader = torch.utils.data.DataLoader(
        dataSet,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # plot some images after transforms
    plotRealBatch(dataloader, opt.batch_size)

    ## Decide which device we want to run on
    device = torch.device("cuda:0" if (cuda and opt.n_gpu > 0) else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # for moving tensor from cpu to gpu

    ## Configure GAN models
    if opt.loss_func == 'standard':
        # BCEWithLogitsLoss will apply sigmoid automatically
        BCELogits_loss = torch.nn.BCEWithLogitsLoss()

    # Initialize generator and discriminator
    # generator = Generator(nz=opt.nz, nc=opt.nc, ngf=opt.ngf, n_extra_layers_g=opt.n_extra_layers_g, n_gpu=opt.n_gpu)
    # generator = Generator_PS(nz=opt.nz, nc=opt.nc, ngf=opt.ngf, n_extra_layers_g=opt.n_extra_layers_g, n_gpu=opt.n_gpu)
    # generator = Generator_NoLL(nz=opt.nz, nc=opt.nc, ngf=opt.ngf, n_extra_layers_g=opt.n_extra_layers_g, n_gpu=opt.n_gpu)
    # generator = Generator_NoLL_3(nz=opt.nz, nc=opt.nc, ngf=opt.ngf, n_extra_layers_g=opt.n_extra_layers_g, n_gpu=opt.n_gpu)
    generator = Generator_MLP(nz=opt.nz, ngf=opt.ngf, img_shape=img_shape, n_gpu=opt.n_gpu)
    # generator = Generator_Res(nz=opt.nz, nc=opt.nc, ngf=opt.ngf, n_extra_layers_g=opt.n_extra_layers_g, n_gpu=opt.n_gpu)

    # discriminator = Discriminator(nc=opt.nc, ndf=opt.ndf, n_gpu=opt.n_gpu)
    # discriminator = Discriminator_NoLL(nc=opt.nc, ndf=opt.ndf, n_gpu=opt.n_gpu)
    # discriminator = Discriminator_NoLL_3(nc=opt.nc, ndf=opt.ndf, n_gpu=opt.n_gpu)
    # discriminator = Discriminator_SN(nc=opt.nc, ndf=opt.ndf, n_gpu=opt.n_gpu)
    discriminator = Discriminator_MLP(ndf=opt.ndf, img_shape=img_shape, n_gpu=opt.n_gpu)
    # discriminator = Discriminator_MLP_SN(ndf=opt.ndf, img_shape=img_shape, n_gpu=opt.n_gpu)
    # discriminator = Discriminator_Res(nc=opt.nc, ndf=opt.ndf, n_gpu=opt.n_gpu)
    # discriminator = Discriminator_Res_SN(nc=opt.nc, ndf=opt.ndf, n_gpu=opt.n_gpu)

    # Initialize weights and whether to load previous trained models
    if ~opt.use_ResNet:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    if opt.netG != '':
        generator.load_state_dict(torch.load(opt.netG))
    # print(generator)

    if opt.netD != '':
        discriminator.load_state_dict(torch.load(opt.netD))
    # print(discriminator)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        if opt.loss_func == 'standard':
            BCELogits_loss.cuda()

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (opt.n_gpu > 1):
        generator = torch.nn.DataParallel(generator, list(range(opt.n_gpu)))
        discriminator = torch.nn.DataParallel(discriminator, list(range(opt.n_gpu)))

    # Optimizers
    # because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to
    # optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
    if opt.optimizer == 'Adam':
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    elif opt.optimizer == 'RMSprop':
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
        optimizer_D = torch.optim.RMSprop(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=opt.lr)

    # use an exponentially decaying learning rate
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=1)  # 0.99
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=1)  # 0.99

    ##
    # ----------
    #  Training
    # ----------
    real_label = 1
    fake_label = 0

    fixed_noise = generateNoise(opt.batch_size, opt.nz, opt.binaryNoise)

    ##
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    GP = []
    D_real = []
    D_fake1 = []
    D_fake2 = []
    D_acc = []
    img_list = []
    batches_done = 0


    # D(x) - the average output (across the batch) of the discriminator for the all real batch.
    #        This should start close to 1 then theoretically converge to 0.5 when G gets better.
    # D(G(z)) - average discriminator outputs for the all fake batch.
    #           The first number is before D is updated and the second number is after D is updated.
    #           These numbers should start near 0 and converge to 0.5 as G gets better.
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.type(Tensor)

            # Adversarial ground truths
            valid = Tensor(imgs.shape[0], 1).fill_(real_label)
            valid.requires_grad = False
            fake = Tensor(imgs.shape[0], 1).fill_(fake_label)
            fake.requires_grad = False

            # use smooth labels for D
            if opt.use_smoothlabel:
                valid = smooth_positive_labels(valid)
                fake = smooth_negative_labels(fake)

            # use noisy labels for D, flip labels
            if opt.use_noisylabel:
                valid = noisy_labels(valid, 0.05)
                fake = noisy_labels(fake, 0.05)


            # ---------------------
            #  DCGAN Train Discriminator: maximize E[log(D(x)) + log(1 - D(G(z)))]
            # ---------------------
            # recall the last layer of D(x) is sigmoid which transforms real number to (0, 1)
            # so the D(x) outputs the probability of the image being real (0, 1)
            # Train D to make it give high probability to real image and low probability to fake image
            # since the neural network uses the form of min optimization problem
            # maximize 1/N * (log(D(x)) + log(1 - D(G(z))))  <=> minimize loss_D = - 1/N * (log(D(x)) + log(1 - D(G(z))))
            # the latter is also the binary cross entropy
            # D(x) in (0, 1) => -log(D(x)) > 0 => loss_D > 0
            # at first, D() is not trained and weak and cannot distinguish from fake from real, D(x) and D(G(z)) both around 0.5
            # as time progresses, G() is still weak, so D(G(z)) around 0 < D(x) around 1
            # ideally, with training, G() catches up, D(G(z)) should be closed to D(x), D(x) and D(G(z)) both around 0.5

            # ---------------------
            #  WGAN Train Discriminator: maximize E[D(x) - D(G(z))]
            # ---------------------
            # recall the last layer of D in WGAN does not have sigmoid()
            # so D(x) output the real number (-inf, +inf), the larger, the more real the image
            # Train D to make it give high score to real image and low score to fake image
            # since the neural network uses the form of min optimization problem
            # maximize 1/N * (D(x) - D(G(z)))  <=> minimize loss_D = 1/N * (D(G(z)) - D(x))
            # at first, D() is not trained and weak and cannot distinguish from fake from real
            # as time progresses, G() is still weak, so D(G(z)) < D(x) -> loss_D < 0
            # ideally, with training, G() catches up, D(G(z)) should be closed to D(x), loss_D -> 0

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            ###### train with real
            out_real = discriminator(real_imgs)

            # Loss function
            if opt.loss_func == 'standard':
                real_loss = BCELogits_loss(out_real, valid)
                D_x = torch.nn.Sigmoid()(out_real).mean().item()
            elif opt.loss_func == 'wasserstein':
                real_loss = - torch.mean(out_real)
                D_x = out_real.mean().item()
            elif opt.loss_func == 'hinge':
                real_loss = torch.nn.ReLU()(1.0 - out_real).mean()
                D_x = out_real.mean().item()
            elif opt.loss_func == 'softplus':
                # SoftPlus is a smooth approximation to the ReLU function and
                # can be used to constrain the output of a machine to always be positive
                real_loss = torch.nn.Softplus()(- out_real).mean()
                D_x = out_real.mean().item()

            # real_loss.backward()

            ###### train with fake
            # Sample noise as generator input
            z = generateNoise(imgs.shape[0], opt.nz, opt.binaryNoise)

            # Generate a batch of images, fix Generator, only update D, so use .detach()
            fake_imgs = generator(z)

            # Measure discriminator's ability to classify real from generated samples
            out_fake = discriminator(fake_imgs.detach())

            # Loss function
            if opt.loss_func == 'standard':
                fake_loss = BCELogits_loss(out_fake, fake)
                D_G_z1 = torch.nn.Sigmoid()(out_fake).mean().item()
            elif opt.loss_func == 'wasserstein':
                fake_loss = torch.mean(out_fake)
                D_G_z1 = out_fake.mean().item()
            elif opt.loss_func == 'hinge':
                fake_loss = torch.nn.ReLU()(1.0 + out_fake).mean()
                D_G_z1 = out_fake.mean().item()
            elif opt.loss_func == 'softplus':
                # SoftPlus is a smooth approximation to the ReLU function and
                # can be used to constrain the output of a machine to always be positive
                fake_loss = torch.nn.Softplus()(out_fake).mean()
                D_G_z1 = out_fake.mean().item()


            # fake_loss.backward()  # gradients for fake/real will be accumulated

            if opt.use_GP:
                gradpen = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), use_dragan=opt.use_dragan,
                                                   center=opt.center, alpha=None, LAMBDA=opt.LAMBDA, device=device)
                # gradpen.backward()
            else:
                gradpen = Tensor([0])

            # whether to add binary penalty to facilitate the generation of binary value
            # bp = 0.1 * torch.mean(torch.sum(torch.sum(torch.abs(fake_imgs) * (2 - torch.abs(fake_imgs)), axis=2), axis=2) / opt.img_size ** 2)
            # bp = bp.backward()

            d_loss = fake_loss + real_loss + gradpen

            d_loss.backward()

            optimizer_D.step()  # .step() can be called once the gradients are computed, only update D params

            # Clip weights of discriminator as WGAN
            if opt.use_clip:
                for p in discriminator.parameters():
                    if p.requires_grad:   # for spectral normalization, u, v
                        p.data.clamp_(-opt.clip_value, opt.clip_value)

            D_real.append(D_x)
            D_fake1.append(D_G_z1)
            D_losses.append(real_loss.item() + fake_loss.item())
            GP.append(gradpen.item())

            # the classification accuracy of D
            D_acc.append(accuracy_score(np.concatenate((torch.nn.Sigmoid()(out_real.detach().squeeze()).numpy(),
                                                        torch.nn.Sigmoid()(out_fake.detach().squeeze()).numpy())) >= 0.5,
                                        np.concatenate((np.ones((imgs.shape[0])), np.zeros((imgs.shape[0]))))
                                        )
                         )

            # Train the generator
            if i % opt.n_critic == 0:
                # -----------------
                #  DCGAN Train Generator: maximize E[log(D(G(z)))]
                # -----------------
                # recall the last layer of D() is sigmoid which transforms real number to (0, 1)
                # so the D() outputs the probability of the image being real (0, 1)
                # Train G to make it yield fake image to mislead D to give high probability to generated fake images
                # since the neural network uses the form of min optimization problem
                # maximize 1/N * log(D(G(z)))  <=> minimize loss_G = 1/N * (log(1 - D(G(z))))
                # the latter is also the binary cross entropy
                # D(G(z)) in (0, 1) => -log(1-D(G(z))) > 0 => loss_G > 0

                # -----------------
                #  WGAN Train Generator: maximize E[D(G(z))]
                # -----------------
                # recall the last layer of D() in WGAN does not have sigmoid()
                # so D() outputs the real number (-inf, +inf), the larger, the more real the image
                # Train G to make it yield fake image to mislead D to give high score to generated fake images
                # since the neural network uses the form of min optimization problem
                # maximize 1/N * D(G(z))  <=> minimize loss_G = - 1/N * D(G(z))
                # at first, G() is bad -> D(G(z)) give low score (negative) -> loss_G is positive and large
                # ideally, with training, G is enhanced, D(G(z)) increase the score (maybe postive) -> loss_G might cross zero and becomes negative

                optimizer_G.zero_grad()
                optimizer_D.zero_grad()

                # provide real labels for updating G to mislead D
                valid.data.fill_(real_label)

                # since we need to backprop from D to G, no .detach() is needed for fake_imgs, since fake_imgs is generated by G
                # the gradients of params of D and G will both be computed
                # Loss measures generator's ability to fool the discriminator

                # use a different set of noise
                # z = generateNoise(imgs.shape[0], opt.nz, opt.binaryNoise)
                # fake_imgs = generator(z)

                out = discriminator(fake_imgs)

                if opt.loss_func == 'hinge' or opt.loss_func == 'wasserstein':
                    g_loss = - out.mean()
                    D_G_z2 = out.mean().item()
                elif opt.loss_func == 'standard':
                    g_loss = BCELogits_loss(out, valid)
                    D_G_z2 = torch.nn.Sigmoid()(out).mean().item()
                elif opt.loss_func == 'softplus':
                    # SoftPlus is a smooth approximation to the ReLU function and
                    # can be used to constrain the output of a machine to always be positive
                    g_loss = torch.nn.Softplus()(- out).mean()
                    D_G_z2 = out.mean().item()

                g_loss.backward()
                optimizer_G.step()  # only updating G, although we calculate the gradient of params of D

                D_fake2.append(D_G_z2)
                G_losses.append(g_loss.item())

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [GP: %f] [D(x): %f] [D(G(z)): %f/%f] [D_acc: %f]"
                    % (
                    epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), D_losses[-1], G_losses[-1], gradpen, D_x,
                    D_G_z1, D_G_z2, D_acc[-1])
                )

            # save a sample of fake images every interval or the last one
            if batches_done % opt.sample_interval == 0 or ((epoch == opt.n_epochs - 1) and (i == len(dataloader)-1)):
                # Generate a batch of images
                gen_imgs = generator(z).detach().cpu()
                save_image(gen_imgs.data[:opt.batch_size], "%s/fake_%d.png" % (opt.outTrainImgsDir, batches_done),
                           nrow=int(math.sqrt(opt.batch_size)), normalize=True)

                # Generate a batch of images for fixed noise
                gen_imgs = generator(fixed_noise).detach().cpu()
                save_image(gen_imgs.data[:opt.batch_size], "%s/fake_fixednoise_%d.png" % (opt.outTrainImgsDir, batches_done),
                           nrow=int(math.sqrt(opt.batch_size)), normalize=True)
                img_list.append(make_grid(gen_imgs, padding=2, normalize=True))

            # save models every interval or the last one
            if (epoch % opt.modelsave_interval == 0 or epoch == opt.n_epochs - 1) and (i == len(dataloader) - 1):
                # do checkpointing
                print('************************** models saved of epoch {} ************************'.format(epoch))
                torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outModelDir, epoch))
                torch.save(optimizer_G.state_dict(), '{0}/optiG_epoch_{1}.pth'.format(opt.outModelDir, epoch))
                torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outModelDir, epoch))
                torch.save(optimizer_D.state_dict(), '{0}/optiD_epoch_{1}.pth'.format(opt.outModelDir, epoch))

            # print('batches_done =', batches_done)
            batches_done += 1

        scheduler_D.step()
        scheduler_G.step()

    ## plot D(x) and D(G(z)) over time
    plotDAccuracy(D_acc, batches_done, dataloader, opt.n_epochs, interval_epochs=50,
                  resultPlotDir=opt.resultPlotDir + 'D_accuracy_train.png')

    ## plot D(x) and D(G(z)) over time

    plotDScore(D_real, D_fake1, D_fake2, batches_done, dataloader, opt.n_critic, opt.n_epochs, interval_epochs=50,
               resultPlotDir=opt.resultPlotDir + 'D_score_train.png')

    ## plot losses over time
    # Generator loss on fake images is expected to sit between 0.5 and perhaps 2.0
    # Variance of generator and discriminator loss is expected to remain modest
    # Training stability may degenerate into periods of high-variance loss and corresponding lower quality generated images
    # A loss of 0.0 in the discriminator is a failure mode
    # If loss of the generator steadily decreases, it is likely fooling the discriminator with garbage images.

    plotLoss(G_losses, D_losses, GP, batches_done, dataloader, opt.n_critic, opt.n_epochs, interval_epochs=50,
               resultPlotDir=opt.resultPlotDir + 'loss_train.png')

    ## Visualization of G’s progression

    plotGEvolution(img_list, opt.resultPlotDir + 'G_evolution.gif')

    ## Real Images vs. Fake Images
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    real_batch = make_grid(real_batch, padding=5, normalize=True).cpu()  # make a grid from a batch

    fake_batch = img_list[-1].detach().numpy()  # element in img_list is already a grid saved in training

    PlotRealFake(real_batch, fake_batch, opt.resultPlotDir + 'Real_Fake.png')

    ##
    # ----------------------------------------
    #  Generating after training and filtering
    # ----------------------------------------

    with torch.no_grad():
        from scipy.ndimage import gaussian_filter
        from PIL import Image
        from PIL import ImageFilter

        nsamples = 100

        z = generateNoise(nsamples, opt.nz, opt.binaryNoise)

        # Generate a batch of images
        candidate_imgs = generator(z).detach().cpu()
        save_image(candidate_imgs.data[:], "%s/candidate_images_nofilter.png" % opt.outGenImgsDir, nrow=int(math.sqrt(nsamples)), normalize=True)

##

