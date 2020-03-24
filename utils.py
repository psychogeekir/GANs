import os
import random
import numpy as np
from numpy.random import choice
import math
import imageio

from torchvision.utils import save_image, make_grid

import torch


import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML


def epoch2iteration(epoch, datasetLen):
    return epoch * datasetLen


def iteration2epoch(iteration, datasetLen):
    return iteration // datasetLen


def fixSeed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
    return None


def generateNoise(batch_size, nz, binaryNoise, Tensor):
    if binaryNoise:
        bernoulli_prob = Tensor(batch_size, nz, 1, 1).fill_(0.5)
        z = torch.bernoulli(bernoulli_prob)
    else:
        z = Tensor(np.random.normal(0, 1, (batch_size, nz, 1, 1)))
    return z


def plotRealBatch(dataloader, batch_size):
    # plot some images after transforms
    real_batch = next(iter(dataloader))
    print('the tensor size of one image is:', real_batch[0][0].shape)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    # since the real_batch has been transformed to [-1, 1], to correctly show it, transformed it back to [0, 1] by make_grid(..., normalize=True)
    plt.imshow(np.transpose(make_grid(real_batch[0][:batch_size], padding=2, normalize=True), (1, 2, 0)))
    plt.show()
    return None


def plotDAccuracy(D_acc, batches_done, dataloader, n_epochs, interval_epochs, resultPlotDir):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Discriminator Accuracy During Training")
    ax1.plot([i for i in range(batches_done)], D_acc, label="D_acc")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # add second x ticks of epochs
    ax2 = ax1.secondary_xaxis('top', functions=(lambda x: iteration2epoch(x, len(dataloader)),
                                                lambda x: epoch2iteration(x, len(dataloader))))
    ax2.set_xlabel("epochs")

    fig.show()
    fig.savefig(resultPlotDir, bbox_inches='tight')
    return None


def plotDScore(D_real, D_fake1, D_fake2, batches_done, dataloader, n_critic, n_epochs, interval_epochs, resultPlotDir):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Discriminator Score During Training")
    # D_real sits around 0.5
    ax1.plot([i for i in range(batches_done)], D_real, label="D_real")
    # D_fake sits around 0.5
    ax1.plot([i for i in range(batches_done)], D_fake1, label="D_fake")
    # D_fake_updated sits around 0.5
    ax1.plot([i for i in range(batches_done) if (i % len(dataloader)) % n_critic == 0], D_fake2,
             label="D_fake_updated")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Score")
    ax1.legend()

    # add second x ticks of epochs
    ax2 = ax1.secondary_xaxis('top', functions=(lambda x: iteration2epoch(x, len(dataloader)),
                                                lambda x: epoch2iteration(x, len(dataloader))))
    ax2.set_xlabel("epochs")

    fig.show()
    fig.savefig(resultPlotDir, bbox_inches='tight')
    return None


def plotLoss(G_losses, D_losses, GP, batches_done, dataloader, n_critic, n_epochs, interval_epochs, resultPlotDir):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Generator and Discriminator Loss During Training")
    ax1.plot([i for i in range(batches_done) if (i % len(dataloader)) % n_critic == 0], G_losses, label="G")
    ax1.plot([i for i in range(batches_done)], D_losses, label="D")
    ax1.plot([i for i in range(batches_done)], GP, label="GP")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # add second x ticks of epochs
    ax2 = ax1.secondary_xaxis('top', functions=(lambda x: iteration2epoch(x, len(dataloader)),
                                                lambda x: epoch2iteration(x, len(dataloader))))
    ax2.set_xlabel("epochs")

    fig.show()
    fig.savefig(resultPlotDir, bbox_inches='tight')
    return None


def plotGEvolution(img_list, resultPlotDir):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i.detach().numpy(), (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1e6, repeat_delay=5e6, blit=True)
    # plt.show()
    ani.save(resultPlotDir, writer=animation.PillowWriter(), dpi=300)
    HTML(ani.to_jshtml())
    return None


def getEvolutionGIF(n, interval, outTrainImgsDir, resultPlotDir, gifName='G_Progress.gif', fps=60, loop=1):
    # read all generated images during training and combine them into one gif

    imgs = []

    # for file_name in os.listdir(resultPlotDir):
    #     if file_name.endswith('.png'):
    #         file_path = os.path.join(resultPlotDir, file_name)
    #         imgs.append(imageio.imread(file_path))

    for i in range(0, n, interval):
        file_path = os.path.join(outTrainImgsDir, 'fake_fixednoise_{}.png'.format(i))
        imgs.append(imageio.imread(file_path))

    imageio.mimsave(resultPlotDir + '/' + gifName, imgs, fps=fps, loop=loop)
    return None



def PlotRealFake(real_batch, fake_batch, resultPlotDir):
    # Grab a batch of real images from the dataloader

    # Plot the real images
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(real_batch, (1, 2, 0)))

    # Plot the fake images
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(fake_batch, (1, 2, 0)))
    plt.show()

    fig.savefig(resultPlotDir, bbox_inches='tight')
    return None