import os
import sys
import time
import glob
from model import *
from data_generator import *
from config import *
from keras.models import load_model,save_model
from keras.layers import Input
from keras import optimizers

import dataset
import config


import misc

def load_GD(path, compile = False):
    G_path = os.path.join(path,'Generator.h5')
    D_path = os.path.join(path,'Discriminator.h5')
    G = load_model(G_path, compile = compile)
    D = load_model(D_path, compile = compile)
    return G,D

def save_GD(G,D,path,overwrite = False):

        os.makedirs(path);
        G_path = os.path.join(path,'Generator.h5')
        D_path = os.path.join(path,'Discriminator.h5')
        save_model(G,G_path,overwrite = overwrite)
        save_model(D,D_path,overwrite = overwrite)
        print("Save model to %s"%path)


def load_GD_weights(G,D,path, by_name = True):
    G_path = os.path.join(path,'Generator.h5')
    D_path = os.path.join(path,'Discriminator.h5')
    G.load_weights(G_path, by_name = by_name)
    D.load_weights(D_path, by_name = by_name)
    return G,D

def save_GD_weights(G,D,path):
    try:
        os.makedirs(path);
        G_path = os.path.join(path,'Generator.h5')
        D_path = os.path.join(path,'Discriminator.h5')
        G.save_weights(G_path)
        D.save_weights(D_path)
        print("Save weights to %s:"%path)
    except:
        print("Save model snapshot failed!")


def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def format_time(seconds):
    s = int(np.round(seconds))
    if s < 60:         return '%ds'                % (s)
    elif s < 60*60:    return '%dm %02ds'          % (s / 60, s % 60)
    elif s < 24*60*60: return '%dh %02dm %02ds'    % (s / (60*60), (s / 60) % 60, s % 60)
    else:              return '%dd %dh %02dm'      % (s / (24*60*60), (s / (60*60)) % 24, (s / 60) % 60)

def rampdown_linear(epoch, num_epochs, rampdown_length):
    if epoch >= num_epochs - rampdown_length:
        return float(num_epochs - epoch) / rampdown_length
    else:
        return 1.0

def create_result_subdir(result_dir, run_desc):

    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase[:fbase.find('-')])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d-%s' % (run_id, run_desc))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    print ("Saving results to", result_subdir)
    return result_subdir

def random_latents(num_latents, G_input_shape):
    return np.random.randn(num_latents, *G_input_shape[1:]).astype(np.float32)

def random_labels(num_labels, training_set):
    return training_set.labels[np.random.randint(training_set.labels.shape[0], size=num_labels)]

def wasserstein_loss( y_true, y_pred):
        return K.mean(y_true * y_pred)

def multiple_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def mean_loss(y_true, y_pred):
    return K.mean(y_pred)


def adversarial_loss(y_true, y_pred):
    return wasserstein_loss( y_true, y_pred)

def cycle_consistency_loss(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

def semantic_consistency_loss(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))


def load_dataset(dataset_spec=None, verbose=True, **spec_overrides):
    if verbose: print('Loading dataset...')
    if dataset_spec is None: dataset_spec = config.dataset
    dataset_spec = dict(dataset_spec) # take a copy of the dict before modifying it
    dataset_spec.update(spec_overrides)
    dataset_spec['h5_path'] = os.path.join(config.data_dir, dataset_spec['h5_path'])
    if 'label_path' in dataset_spec: dataset_spec['label_path'] = os.path.join(config.data_dir, dataset_spec['label_path'])
    training_set = dataset.Dataset(**dataset_spec)
    if verbose: print('Dataset shape =', np.int32(training_set.shape).tolist())
    drange_orig = training_set.get_dynamic_range()
    if verbose: print('Dynamic range =', drange_orig)
    return training_set, drange_orig



speed_factor = 20

def train_gan(
    images_dir1,
    images_dir2,
    batch_size,
    D_training_repeats      = 1,
    G_learning_rate_max     = 0.0010,
    D_learning_rate_max     = 0.0010,
    G_smoothing             = 0.999,
    adam_beta1              = 0.0,
    adam_beta2              = 0.99,
    adam_epsilon            = 1e-8,
    minibatch_default       = 16,
    rampup_kimg             = 40/speed_factor,
    rampdown_kimg           = 0,
    lod_initial_resolution  = 4,
    lod_training_kimg       = 400/speed_factor,
    lod_transition_kimg     = 400/speed_factor,
    total_kimg              = 10000/speed_factor,
    dequantize_reals        = False,
    gdrop_beta              = 0.9,
    gdrop_lim               = 0.5,
    gdrop_coef              = 0.2,
    gdrop_exp               = 2.0,
    drange_net              = [-1,1],
    drange_viz              = [-1,1],
    image_grid_size         = None,
    tick_kimg_default       = 50/speed_factor,
    tick_kimg_overrides     = {32:20, 64:10, 128:10, 256:5, 512:2, 1024:1},
    image_snapshot_ticks    = 1,
    network_snapshot_ticks  = 4,
    image_grid_type         = 'default',
    #resume_network          = '000-celeba/network-snapshot-000488',
    resume_network          = None,
    resume_kimg             = 0.0,
    resume_time             = 0.0):

    training_set, drange_orig = load_dataset()

    # if resume_network:
    #     print("Resuming weight from:"+resume_network)
    #     G = Generator(num_channels=training_set.shape[3], resolution=training_set.shape[1], label_size=training_set.labels.shape[1], **config.G)
    #     D = Discriminator(num_channels=training_set.shape[3], resolution=training_set.shape[1], label_size=training_set.labels.shape[1], **config.D)
    #     G,D = load_GD_weights(G,D,os.path.join(config.result_dir,resume_network),True)
    # else:


    E_G = Encoder_Generator(num_channels=training_set.shape[3], resolution=training_set.shape[1], **config.G)
    D = Discriminator(num_channels=training_set.shape[3], resolution=training_set.shape[1], **config.D)

    E_twin_G_twin = new_batch_norm(E_G)
    D_twin = new_batch_norm(D)

    E = extract_encoder(E_G)
    E_twin = extract_encoder(E_twin_G_twin)


    E_twin_G = replace_batch_norm(E_G, E_twin_G_twin, apply='encoder')
    E_G_twin = replace_batch_norm(E_G, E_twin_G_twin, apply='generator')

    E_G_twin_E_twin = Sequential([E_G_twin, E_twin])


    E_G_D = Sequential([E_G, D])
    E_G_twin_D_twin = Sequential([E_G_twin, D_twin])

    E_twin_G_E = Sequential([E_twin_G, E])
    E_twin_G_D = Sequential([E_twin_G, D])
    E_twin_G_twin_D_twin = Sequential([E_twin_G_twin, D_twin])

    # Misc init.
    resolution_log2 = int(np.round(np.log2(E_G.output_shape[2])))
    initial_lod = max(resolution_log2 - int(np.round(np.log2(lod_initial_resolution))), 0)
    cur_lod = 0.0
    min_lod, max_lod = -1.0, -2.0
    fake_score_avg = 0.0

    D.trainable = False
    D_twin.trainable = False
    E_G_D.compile(optimizers.Adam(lr=0.0, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon),
                  loss=adversarial_loss)
    E_twin_G_D.compile(optimizers.Adam(lr=0.0, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon),
                  loss=adversarial_loss)

    E_G_twin_D_twin.compile(optimizers.Adam(lr=0.0, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon),
                  loss=adversarial_loss)
    E_twin_G_twin_D_twin.compile(optimizers.Adam(lr=0.0, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon),
                  loss=adversarial_loss)


    D.trainable = True
    D_twin.trainable = True
    D.compile(optimizers.Adam(lr=0.0, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon),
              loss=adversarial_loss)
    D_twin.compile(optimizers.Adam(lr=0.0, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon),
              loss=adversarial_loss)

    E_G.compile(optimizers.Adam(lr=0.0, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon),
              loss=cycle_consistency_loss)
    E_twin_G_twin.compile(optimizers.Adam(lr=0.0, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon),
              loss=cycle_consistency_loss)

    E_twin_G_E.compile(optimizers.Adam(lr=0.0, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon),
              loss=semantic_consistency_loss)
    E_G_twin_E_twin.compile(optimizers.Adam(lr=0.0, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon),
              loss=semantic_consistency_loss)


    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    tick_train_out = []

    # Set up generators

    data_generator1 = DataGenerator(images_dir=images_dir1)
    data_generator2 = DataGenerator(images_dir=images_dir2)

    generator1 = data_generator1.generate(batch_size=2, img_size=2**resolution_log2)
    generator2 = data_generator2.generate(batch_size=2, img_size=2**resolution_log2)

    real_1 = next(generator1)
    real_2 = next(generator2)

    result_subdir = misc.create_result_subdir(config.result_dir, config.run_desc)

    print("real_1.shape:", real_1.shape)
    print("real_2.shape:", real_2.shape)

    misc.save_image_grid_twin(real_1, real_2, os.path.join(result_subdir, 'reals.png'))

    nimg_h = 0

    valid = np.ones((batch_size, 1, 1, 1))
    fake = np.zeros((batch_size, 1, 1, 1))

    while cur_nimg < total_kimg * 1000:
        
        # Calculate current LOD.
        cur_lod = initial_lod
        if lod_training_kimg or lod_transition_kimg:
            tlod = (cur_nimg / (1000.0/speed_factor)) / (lod_training_kimg + lod_transition_kimg)
            cur_lod -= np.floor(tlod)
            if lod_transition_kimg:
                cur_lod -= max(1.0 + (np.fmod(tlod, 1.0) - 1.0) * (lod_training_kimg + lod_transition_kimg) / lod_transition_kimg, 0.0)
            cur_lod = max(cur_lod, 0.0)

        # Look up resolution-dependent parameters.
        cur_res = 2 ** (resolution_log2 - int(np.floor(cur_lod)))
        tick_duration_kimg = tick_kimg_overrides.get(cur_res, tick_kimg_default)

        generator1 = data_generator1.generate(batch_size=batch_size, img_size=cur_res)
        generator2 = data_generator2.generate(batch_size=batch_size, img_size=cur_res)

        # Update network config.
        lrate_coef = rampup(cur_nimg / 1000.0, rampup_kimg)
        lrate_coef *= rampdown_linear(cur_nimg / 1000.0, total_kimg, rampdown_kimg)

        models = [E_G_D, E_twin_G_D, E_G_twin_D_twin, E_twin_G_twin_D_twin, D, D_twin, E_G, E_twin_G_twin,
                  E_twin_G_E, E_G_twin_E_twin]

        learning_rate_max = 0.001

        for model in models:

            K.set_value(model.optimizer.lr, np.float32(lrate_coef * learning_rate_max))
            if hasattr(model, 'cur_lod'): K.set_value(model.cur_lod,np.float32(cur_lod))

        new_min_lod, new_max_lod = int(np.floor(cur_lod)), int(np.ceil(cur_lod))
        if min_lod != new_min_lod or max_lod != new_max_lod:
            min_lod, max_lod = new_min_lod, new_max_lod

        fake_2 = E_G_twin.predict_on_batch(real_1)
        fake_1 = E_twin_G.predict_on_batch(real_2)
        misc.save_image_grid_twin(real_1, fake_2, os.path.join(result_subdir, 'fakes_dog%06d.png' % (cur_nimg / 1000)))
        misc.save_image_grid_twin(real_2, fake_1, os.path.join(result_subdir, 'fakes_celeb%06d.png' % (cur_nimg / 1000)))

        1 / 0




        ################################################################################################################
        # train D
        #mb_reals, mb_labels = training_set.get_random_minibatch_channel_last(minibatch_size, lod=cur_lod, shrink_based_on_lod=True, labels=True)

        images1 = next(generator1)

        # if min_lod > 0: # compensate for shrink_based_on_lod
        #      mb_reals = np.repeat(mb_reals, 2**min_lod, axis=1)
        #      mb_reals = np.repeat(mb_reals, 2**min_lod, axis=2)

        img_fakes = E_G.predict_on_batch([images1])

        d_true = D.train_on_batch(images1, valid)
        d_fake = D.train_on_batch(img_fakes, fake)
        cur_nimg += batch_size

        #train E_G_D
        g_loss = E_G_D.train_on_batch(images1, valid)
        print ("%d [D loss: %f] [G loss: %f]" % (cur_nimg, np.mean(d_true, d_fake), g_loss))

        ################################################################################################################
        # train D_twin
        images2 = next(generator2)
        img_fakes = E_twin_G_twin.predict_on_batch([images2])

        d_true = D_twin.train_on_batch(images1, valid)
        d_fake = D_twin.train_on_batch(img_fakes, fake)

        #train E_twin_G_twin_D_twin
        g_loss = E_twin_G_twin_D_twin.train_on_batch(images2, valid)
        print ("%d [D loss: %f] [G loss: %f]" % (cur_nimg, np.mean(d_true, d_fake), g_loss))




        fake_score_cur = np.clip(np.mean(d_loss), 0.0, 1.0)
        fake_score_avg = fake_score_avg * gdrop_beta + fake_score_cur * (1.0 - gdrop_beta)

        if cur_nimg >= tick_start_nimg + tick_duration_kimg * 1000 or cur_nimg >= total_kimg * 1000:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            tick_start_time = cur_time
            tick_train_avg = tuple(np.mean(np.concatenate([np.asarray(v).flatten() for v in vals])) for vals in zip(*tick_train_out))
            tick_train_out = []



            # Visualize generated images.
            if cur_tick % image_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                snapshot_fake_images = G.predict_on_batch(snapshot_fake_latents)
                misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg / 1000)), drange=drange_viz, grid_size=image_grid_size)

            if cur_tick % network_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                save_GD_weights(G,D,os.path.join(result_subdir, 'network-snapshot-%06d' % (cur_nimg / 1000)))


    save_GD(G,D,os.path.join(result_subdir, 'network-final'))
    training_set.close()
    print('Done.')

if __name__ == '__main__':

    np.random.seed(config.random_seed)
    func_params = config.train

    func_name = func_params['func']
    del func_params['func']
    globals()[func_name](**func_params)