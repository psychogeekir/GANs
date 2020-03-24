# GAN

- dcgan_train: several GANs (loss functions, gradient penalty, spectral normalization, network architecture)
- evaluation: FID and inception scores
- some helper function: to visualize training progress and generated images and prepare data
- post filtering: remove isolated pixels

## Training progress

WGAN + spetral normalization + gradient penalty 10

![loss][loss]

![score][score]

![evolution][evolution]


[loss]: https://github.com/psychogeekir/GANs/blob/master/result/loss_train.png "Loss during training"
[score]: https://github.com/psychogeekir/GANs/blob/master/result/D_score_train.png "Score during training"
[evolution]: https://github.com/psychogeekir/GANs/blob/master/result/G_evolution.gif "Evolution of generator"

## Result

![read_fake][read_fake]

![no_filter][no_filter]


[read_fake]: https://github.com/psychogeekir/GANs/blob/master/result/Real_Fake.png "Real v.s. Fake"
[no_filter]: https://github.com/psychogeekir/GANs/blob/master/candidates/candidate_images_nofilter.png "Raw image given by generator"

