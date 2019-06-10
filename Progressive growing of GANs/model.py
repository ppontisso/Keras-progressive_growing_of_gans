import tensorflow as tf
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.layers.merge import _Merge
from keras.layers import *
from keras import activations
from keras import initializers
from keras.models import Model, Sequential
import numpy as np
from layers import *

linear, linear_init = activations.linear, initializers.he_normal()
relu, relu_init = activations.relu, initializers.he_normal()
lrelu, lrelu_init = lambda x: K.relu(x, 0.2), initializers.he_normal()


def NINBlock(
        net,
        num_channels,
        actv,
        init,
        use_wscale=True,
        name=None):
    if use_wscale:

        NINlayer = Conv2D(num_channels, 1, padding='same',
                          activation=None, use_bias=False, kernel_initializer=init, name=name + 'NIN')
        net = NINlayer(net)
        Wslayer = WScaleLayer(NINlayer, name=name + 'NINWS')
        net = Wslayer(net)
        net = AddBiasLayer()(net)
        net = Activation(actv)(net)
    else:
        NINlayer = Conv2D(num_channels, 1, padding='same',
                          activation=actv, kernel_initializer=init, name=name + 'NIN')
        net = NINlayer(net)
    return net


def Downscale2DLayer(incoming, scale_factor, name, **kwargs):
    return AveragePooling2D(pool_size=scale_factor, name=name, **kwargs)(incoming)


def D_ConvBlock(net,
                num_filter,
                filter_size,
                actv,
                init,
                use_wscale,
                use_layernorm,
                epsilon,
                padding = 'same',
                use_batchnorm=True,
                name=None):
    if use_wscale:
        Conv = Conv2D(num_filter, filter_size, padding=padding,
                      activation=None, kernel_initializer=init, use_bias=False, name=name)
    else:
        Conv = Conv2D(num_filter, filter_size, padding=padding,
                      activation=actv, kernel_initializer=init, name=name)
    net = Conv(net)
    if use_wscale:
        layer = WScaleLayer(Conv, name=name + 'ws')
        net = layer(net)
        Addbias = AddBiasLayer()
        net = Addbias(net)
        net = Activation(actv)(net)
    if use_batchnorm:
        Bslayer = BatchNormalization(name=name + 'BN')
        net = Bslayer(net)
    if use_layernorm:
        layer = LayerNormLayer(layer, epsilon, name=name + 'ln')
        net = layer(net)
    return net


def DenseBlock(
        net,
        size,
        act,
        init,
        use_wscale,
        name=None):

    if use_wscale:
        layer = Dense(size, activation=None, use_bias=False, kernel_initializer=init, name=name)
        net = layer(net)
        layer = WScaleLayer(layer, name=name + 'ws')
        net = layer(net)
        Addbias = AddBiasLayer()
        net = Addbias(net)
        net = Activation(act)(net)
    else:
        layer = Dense(size, activation=act, kernel_initializer=init, name=name)
        net = layer(net)
    return net


def vlrelu(x): return K.relu(x, 0.3)


def G_convblock(
        net,
        num_filter,
        filter_size,
        actv,
        init,
        use_wscale=True,
        use_pixelnorm=True,
        use_batchnorm=True,
        name=None, ):
    if use_wscale:
        Conv = Conv2D(num_filter, filter_size, padding='same',
                      activation=None, kernel_initializer=init, use_bias=False, name=name)
    else:
        Conv = Conv2D(num_filter, filter_size, padding='same',
                      activation=actv, kernel_initializer=init, name=name)
    net = Conv(net)
    if use_wscale:
        Wslayer = WScaleLayer(Conv, name=name + 'WS')
        net = Wslayer(net)
        Addbias = AddBiasLayer()
        net = Addbias(net)
        net = Activation(actv)(net)
    if use_batchnorm:
        Bslayer = BatchNormalization(name=name + 'BN')
        net = Bslayer(net)
    if use_pixelnorm:
        Pixnorm = PixelNormLayer(name=name + 'PN')
        net = Pixnorm(net)
    return net


def Encoder_Generator(
        num_channels=1,
        resolution=32,
        fmap_base=4096,
        fmap_decay=1.0,
        fmap_max=256,
        use_wscale=True,
        use_pixelnorm=True,
        use_leakyrelu=True,
        use_batchnorm=True,
        tanh_at_end=None,
        use_layernorm=False,
        **kwargs):

    #ENCODER

    def numf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    epsilon = 0.01
    R = int(np.log2(resolution))
    assert resolution == 2 ** R and resolution >= 4
    cur_lod = K.variable(np.float(0.0), dtype='float32', name='cur_lod')

    input = Input(shape=[2 ** R, 2 ** R, num_channels], name='Eimages')
    net = NINBlock(input, numf(R - 1), lrelu, lrelu_init, use_wscale, name='E%dx' % (R - 1))

    concat_layers = []

    for i in range(R - 1, 1, -1):
        net = D_ConvBlock(net, numf(i), 3, lrelu, lrelu_init, use_wscale, use_layernorm,
                          epsilon, use_batchnorm=use_batchnorm, name='E%db' % i)
        net = D_ConvBlock(net, numf(i - 1), 3, lrelu, lrelu_init, use_wscale, use_layernorm,
                          epsilon, use_batchnorm=use_batchnorm, name='E%da' % i)

        concat_layers.append(net)

        net = Downscale2DLayer(net, name='E%ddn' % i, scale_factor=2)

        lod = Downscale2DLayer(input, name='E%dxs' % (i - 1), scale_factor=2 ** (R - i))
        lod = NINBlock(lod, numf(i - 1), lrelu, relu_init, use_wscale, name='E%dx' % (i - 1))
        net = LODSelectLayer(cur_lod, name='E%dlod' % (i - 1), first_incoming_lod=R - i - 1)([net, lod])

    net = D_ConvBlock(net, numf(1), 3, lrelu, lrelu_init, use_wscale, use_layernorm,
                      epsilon, use_batchnorm=use_batchnorm, name='E1b')
    net = D_ConvBlock(net, numf(0), 4, lrelu, lrelu_init, use_wscale, use_layernorm,
                      epsilon, use_batchnorm=use_batchnorm, name='E1a')

    # GENERATOR

    (act, act_init) = (lrelu, lrelu_init) if use_leakyrelu else (relu, relu_init)

    net = G_convblock(net, numf(1), 3, act, act_init, use_wscale=use_wscale,
                      use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm, name='G1a')

    lods = [net]
    for I in range(2, R):
        net = UpSampling2D(2, name='G%dup' % I)(net)
        # concat U-Net
        encoder_layer = concat_layers[-I+1]

        net = concatenate(inputs=[net, encoder_layer], axis=-1)
        net = G_convblock(net, numf(I), 3, act, act_init, use_wscale=use_wscale,
                          use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm, name='G%da' % I)
        net = G_convblock(net, numf(I), 3, act, act_init, use_wscale=use_wscale,
                          use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm, name='G%db' % I)
        lods += [net]

    lods = [NINBlock(l, num_channels, linear, linear_init, use_wscale=use_wscale,
                     name='Glod%d' % i) for i, l in enumerate(reversed(lods))]
    output = LODSelectLayer(cur_lod, name='Glod')(lods)

    if tanh_at_end is not None:
        output = Activation('tanh', name='Gtanh')(output)
        if tanh_at_end != 1.0:
            output = Lambda(lambda x: x * tanh_at_end, name='Gtanhs')

    encoder_generator = Model(inputs=input, outputs=[output])
    encoder_generator.cur_lod = cur_lod

    return encoder_generator


def Discriminator(
        num_channels=1,  # Overridden based on dataset.
        resolution=32,  # Overridden based on dataset.
        fmap_base=4096,
        fmap_decay=1.0,
        fmap_max=256,
        mbstat_avg='all',
        use_wscale=True,
        use_layernorm=False,
        use_batchnorm=True,
        **kwargs):
    def numf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    epsilon = 0.01
    R = int(np.log2(resolution))
    assert resolution == 2 ** R and resolution >= 4
    cur_lod = K.variable(np.float(0.0), dtype='float32', name='cur_lod')

    inputs = Input(shape=[2 ** R, 2 ** R, num_channels], name='Dimages')
    net = NINBlock(inputs, numf(R - 1), lrelu, lrelu_init, use_wscale, name='D%dx' % (R - 1))
    for i in range(R - 1, 1, -1):
        net = D_ConvBlock(net, numf(i), 3, lrelu, lrelu_init, use_wscale, use_layernorm,
                          epsilon, use_batchnorm=use_batchnorm, name='D%db' % i)
        net = D_ConvBlock(net, numf(i - 1), 3, lrelu, lrelu_init, use_wscale, use_layernorm,
                          epsilon, use_batchnorm=use_batchnorm, name='D%da' % i)
        net = Downscale2DLayer(net, name='D%ddn' % i, scale_factor=2)
        lod = Downscale2DLayer(inputs, name='D%dxs' % (i - 1), scale_factor=2 ** (R - i))
        lod = NINBlock(lod, numf(i - 1), lrelu, relu_init, use_wscale, name='D%dx' % (i - 1))
        net = LODSelectLayer(cur_lod, name='D%dlod' % (i - 1), first_incoming_lod=R - i - 1)([net, lod])

    if mbstat_avg is not None:
        net = MinibatchStatConcatLayer(averaging=mbstat_avg, name='Dstat')(net)

    net = D_ConvBlock(net, numf(1), 3, lrelu, lrelu_init, use_wscale, use_layernorm,
                      epsilon, use_batchnorm=use_batchnorm, name='D1b')
    net = D_ConvBlock(net, numf(0), 4, lrelu, lrelu_init, use_wscale, use_layernorm,
                      epsilon, padding = 'valid', use_batchnorm=use_batchnorm, name='D1a')

    net = DenseBlock(net, 1, linear, linear_init, use_wscale, name='Dscores')
    output_layers = [net]

    model = Model(inputs=[inputs], outputs=output_layers)
    model.cur_lod = cur_lod
    return model


def new_batch_norm(model):
    # source : https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
    # replace all the batch normalization layers in the model with new ones and return the twin model
    # batch norm layers must have 'BN' in their names

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if 'BN' in layer.name:
            # replace layer
            x = layer_input

            new_layer = BatchNormalization(name=layer.name + '_twin')

            x = new_layer(x)

        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    twin = Model(inputs=model.inputs, outputs=x)

    return twin


def replace_batch_norm(model, model_bn, apply='encoder'):
    # source : https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
    # replace all the batch normalization layers in the model with new ones and return the twin model
    # batch norm layers must have 'BN' in their names

    if apply !='encoder' and apply != 'generator':
        raise ValueError('This functiun must be applied to either encoder or generator ')

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            elif layer.name not in network_dict['input_layers_of'][layer_name]:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # replace layer if name matches the regular expression
        # and it is one to be applied to
        if ('BN' in layer.name) and ((apply == 'encoder' and layer.name[0] =='E') or (apply == 'generator' and layer.name[0] =='G')):

            # replace layer
            x = layer_input
            layer_bn_name = layer.name +'_twin' if not layer.name.endswith('twin') else layer.name[:-5]
            new_layer = model_bn.get_layer(layer_bn_name)
            x = new_layer(x)

        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    twin = Model(inputs=model.inputs, outputs=x)

    return twin

def extract_encoder(model, key ='E1aBN'):
    try:
        encoder = Model(inputs=model.input, outputs=model.get_layer(key).output)
    except ValueError as e:
        encoder = Model(inputs=model.input, outputs=model.get_layer(key + '_twin').output)

    return encoder

def PG_GAN(G, D, label_size, resolution, num_channels):
    print("Label size:")
    print(label_size)

    G_train = Sequential([G, D])
    G_train.cur_lod = G.cur_lod

    shape = D.get_input_shape_at(0)[1:]
    gen_input, real_input, interpolation = Input(shape), Input(shape), Input(shape)

    sub = Subtract()([D(gen_input), D(real_input)])
    norm = GradNorm()([D(interpolation), interpolation])
    D_train = Model([gen_input, real_input, interpolation], [sub, norm])
    D_train.cur_lod = D.cur_lod

    return G_train, D_train


def twin_gan(G, D):
    pass


if __name__ == '__main__':
    model = Generator(
        num_channels=3,  # Overridden based on dataset.
        resolution=64,  # Overridden based on dataset.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 of feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps on any resolution.
        use_wscale=True,  # Use equalized learning rate?
        use_pixelnorm=True,  # Use pixelwise normalization?
        use_leakyrelu=True,  # Use leaky ReLU?
        use_batchnorm=False,  # Use batch normalization?
        tanh_at_end=None,  # Use tanh activation for the last layer? If so, how much to scale the
    )
    print(model.summary())
    print(model.cur_lod)

    model = Discriminator(
        num_channels=3,  # Overridden based on dataset.
        resolution=64,  # Overridden based on dataset.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 of feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps on any resolution.
        mbstat_avg='all',  # Which dimensions to average the statistic over?
        use_wscale=True,  # Use equalized learning rate?
        use_layernorm=False,  # Use layer normalization?
    )
    print(model.summary())
    print(model.cur_lod)
