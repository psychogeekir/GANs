import torch
from torch import nn
from torch.nn import Parameter
import torch.autograd as autograd

from numpy.random import choice


# example of smoothing class=1 to [0.7, 1.2]
def smooth_positive_labels(valid):
    return valid - 0.3 + (torch.rand_like(valid) * 0.5)


# example of smoothing class=0 to [0, 0.3]
def smooth_negative_labels(fake):
    return fake + (torch.rand_like(fake) * 0.3)


# randomly flip some labels
def noisy_labels(label, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * label.shape[0])
    # choose labels to flip
    flip_ix = choice([i for i in range(label.shape[0])], size=n_select)
    # invert the labels in place
    if len(flip_ix) > 0:
        if any(label[flip_ix] > 0.3):
            # positive -> negative
            label[flip_ix] = smooth_negative_labels(torch.zeros_like(label[flip_ix], dtype=torch.float))
        else:
            # negative -> positive
            label[flip_ix] = smooth_positive_labels(torch.ones_like(label[flip_ix], dtype=torch.float))
    return label


def genInstanceNoise(data, sigma, device):
    s = sigma * torch.randn(data.size(0), 1, device=device)
    s = s[:, :, None, None].expand(data.size())
    return s


def compute_gradient_penalty(netD, real_data, fake_data, use_dragan=False, center=0, alpha=None, LAMBDA=10, device=None):
    """Calculates the gradient penalty loss for 0-GP"""
    # https: // github.com / htt210 / GeneralizationAndStabilityInGANs / blob / master / GradientPenaltiesGAN.py
    if alpha is not None:
        alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
    else:
        alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha[:, :, None, None].expand(real_data.size())

    if use_dragan:  # only operates on real_data
        interpolates = alpha * real_data + ((1 - alpha) * (real_data + 0.5 * real_data.std() * torch.rand(real_data.size())))
    else:
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = LAMBDA * ((gradients.norm(2, dim=1) - center) ** 2).mean()  # WGAN two-sided penalty
    # gradient_penalty = LAMBDA * ((torch.nn.ReLU()(gradients.norm(2, dim=1) - 1)) ** 2).mean()  # WGAN one-sided penalty
    return gradient_penalty


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))  # torch.mv(): matrix-vector multiplication
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)