data_dir = 'datasets'
result_dir = 'results'

random_seed = 1000
dataset = None

train = dict(                               # Training parameters:
    func                    = 'train_gan',  # Main training func.
    images_dir1 = 'datasets/CelebA',
    images_dir2 = 'datasets/dogs',
    batch_size = 2,
    D_training_repeats      = 1,            # n_{critic}
    G_learning_rate_max     = 0.001,        # \alpha
    D_learning_rate_max     = 0.001,        # \alpha
    G_smoothing             = 0.999,        # Exponential running average of generator weights.
    adam_beta1              = 0.0,          # \beta_1
    adam_beta2              = 0.99,         # \beta_2
    adam_epsilon            = 1e-8,         # \epsilon
    minibatch_default       = 16,           # Minibatch size for low resolutions.
    rampup_kimg             = 40,           # Learning rate rampup.
    rampdown_kimg           = 0,            # Learning rate rampdown.
    lod_initial_resolution  = 4,            # Network resolution at the beginning.
    lod_training_kimg       = 600,          # Thousands of real images to show before doubling network resolution.
    lod_transition_kimg     = 600,          # Thousands of real images to show when fading in new layers.
    total_kimg              = 15000,        # Thousands of real images to show in total.
    gdrop_coef              = 0.0,          # Do not inject multiplicative Gaussian noise in the discriminator.
)

G = dict(                                   # Generator architecture:
    func                    = 'G_paper',    # Configurable network template.
    fmap_base               = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay              = 1.0,          # log2 of feature map reduction when doubling the resolution.
    fmap_max                = 512,          # Maximum number of feature maps on any resolution.
    use_wscale              = True,         # Use equalized learning rate?
    use_pixelnorm           = True,         # Use pixelwise normalization?
    use_leakyrelu           = True,         # Use leaky ReLU?
    use_batchnorm           = True,        # Use batch normalization?
    tanh_at_end             = None,         # Use tanh activation for the last layer? If so, how much to scale the output?
)

D = dict(                                   # Discriminator architecture:
    func                    = 'D_paper',    # Configurable network template.
    fmap_base               = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay              = 1.0,          # log2 of feature map reduction when doubling the resolution.
    fmap_max                = 512,          # Maximum number of feature maps on any resolution.
    mbstat_avg              = 'all',        # Which dimensions to average the statistic over?
    use_wscale              = True,         # Use equalized learning rate?

    use_layernorm           = False,         # Use layer normalization? => Will need to set to True later !!!!!!!!!

    use_batchnorm=True,                     # Use batch normalization?
)

loss = dict(                                # Loss function:
    type                    = 'iwass',       #  Wasserstein (WGAN).
    iwass_lambda            = 10.0,         # \lambda
    iwass_epsilon           = 0.001,        # \epsilon_{drift}
    iwass_target            = 1.0,          # \alpha
    cond_type               = 'acgan',      # AC-GAN
    cond_weight             = 1.0,          # Weight of the conditioning terms.
)

if 1:
    run_desc = 'celeba'

    dataset = dict(h5_path='celeba-128x128.h5', resolution=32, max_labels=0, mirror_augment=True)

    train.update(lod_training_kimg=800, lod_transition_kimg=800, rampup_kimg=0, total_kimg=10000)
    G.update(fmap_base=2048)
    D.update(fmap_base=2048)
