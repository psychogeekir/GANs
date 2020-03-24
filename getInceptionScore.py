import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy

def get_inception_score(imgs, transform_input=False, cuda=True, batch_size=32, resize=True, splits=1):
    """
        Computes the inception score of the generated images imgs
        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits

        for inception_v3:
        All pre-trained models expect input images normalized in the same way,
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 299.
        The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=transform_input).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, (batch, _) in enumerate(dataloader, 0):
        batchv = batch.type(dtype)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []
    with torch.no_grad():
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)