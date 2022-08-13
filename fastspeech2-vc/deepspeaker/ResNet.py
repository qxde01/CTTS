import  tensorflow as tf
from tensorflow import keras

def conv1d_bn(inputs, filters, kernel_size,strides=1, dilation_rate=1,activation=None,name=None):
    x=keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='causal', activation=None,
                        dilation_rate=dilation_rate,kernel_regularizer=keras.regularizers.l2(l=0.0001),name=name+"_con1d")(inputs)
    x=keras.layers.BatchNormalization(name=name + '_1_bn')(x)
    if activation is not None:
        if activation=='relu':
            x=keras.layers.ReLU(max_value=20.,name=name+'_relu20')(x)
        else:
            x=keras.layers.Activation(activation)(x)
    return x

def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    #bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    preact = keras.layers.BatchNormalization(epsilon=1.001e-5,name=name + '_preact_bn')(x)
    preact = keras.layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = keras.layers.Conv1D(4 * filters, 1, strides=stride,padding='causal', name=name + '_0_conv')(preact)
    else:
        shortcut = keras.layers.MaxPooling1D(1, strides=stride)(x) if stride > 1 else x

    #x = keras.layers.Conv2D(filters, 1, strides=1, use_bias=False,name=name + '_1_conv')(preact)
    #x = keras.layers.BatchNormalization( epsilon=1.001e-5,name=name + '_1_bn')(x)
    #x = keras.layers.Activation('relu', name=name + '_1_relu')(x)
    x=conv1d_bn(preact, filters, kernel_size, strides=1, dilation_rate=1, activation='relu', name=name+'_1')
    #print(x)
    x = conv1d_bn(x, filters, kernel_size, strides=1, dilation_rate=1, activation='relu', name=name+'_2')

    #x = keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = keras.layers.Conv1D(4 * filters, 1, strides=stride, padding='causal', name=name + '_3_conv')(x)
    x = keras.layers.Add(name=name + '_out')([shortcut, x])
    return x

def stack2(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def ResNet(stack_fn,
           preact=True,use_bias=True,
           model_name='resnet',include_top=True,
           input_shape=(None,80), classes=1000,class_type='softmax'):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.

        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    img_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(64, 7, strides=1, padding='causal', use_bias=use_bias, name='conv1_conv')(img_input)

    if preact is False:
        x = keras.layers.BatchNormalization(epsilon=1.001e-5,name='conv1_bn')(x)
        x = keras.layers.ReLU(max_value=20., name='conv1_relu20')(x)
    #print(x)

    #x = keras.layers.ZeroPadding1D(padding=(1, 2 ), name='pool1_pad')(x)
    x = keras.layers.MaxPooling1D(3, strides=2, name='pool1_pool')(x)
    #print(x)
    x = stack_fn(x)

    if preact is True:
        x = keras.layers.BatchNormalization( epsilon=1.001e-5,name='post_bn')(x)
        x = keras.layers.ReLU(max_value=20., name='post_relu20')(x)
        #x = keras.layers.Activation('relu', name='post_relu')(x)

    x = keras.layers.Bidirectional(keras.layers.GRU(units=512, return_sequences=True), merge_mode='concat')(x)
    x = keras.layers.GlobalMaxPooling1D(name='max_pool')(x)


    x=keras.layers.Dense(256, activation=None, name='spk_emb')(x)

    if include_top:
        x=keras.layers.Dropout(0.5)(x)
        #x = keras.layers.Dense(classes, activation='softmax', name='probs')(x)
        if class_type=='amsoftmax':
            x = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)
            x = keras.layers.Dense(classes, activation=None, name='classify',kernel_constraint=keras.constraints.unit_norm())(x)
        else:
            x = keras.layers.Dense(classes, activation='softmax', name='classify')(x)

    model = keras.models.Model(img_input, x, name=model_name)
    return model

def ResNet50V2(include_top=True,input_shape=(None,80), classes=1000,class_type='softmax'):
    def stack_fn(x):
        x = stack2(x, 64, 3,stride1=1, name='conv2')
        x = stack2(x, 128, 4, name='conv3')
        x = stack2(x, 256, 6, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')
        return x
    return ResNet(stack_fn, True, True, 'resnet50v2',include_top, input_shape, classes,class_type)

def SpeakerResNetV2(input_shape=(None,80),classes=10,class_type='softmax'):
    speark_emb_model=ResNet50V2(include_top=False,input_shape=input_shape, classes=classes,class_type=class_type)
    anchor_input = keras.layers.Input(input_shape, name='anchor_input')
    positive_input = keras.layers.Input(input_shape, name='positive_input')
    negative_input = keras.layers.Input(input_shape, name='negative_input')
    encoded_anchor = speark_emb_model(anchor_input)
    encoded_positive = speark_emb_model(positive_input)
    encoded_negative = speark_emb_model(negative_input)
    x = keras.layers.Dropout(0.5)(encoded_anchor)
    softmax = keras.layers.Dense(classes, activation='softmax', name='softmax')(x)

    merged_vector = keras.layers.concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='triplet')
    model = keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=[merged_vector,softmax],name='SpeakerResNet50V2')
    #inputs={"anchor_input":anchor_input, "positive_input":positive_input, "negative_input":negative_input}
    #outputs={"triplet":merged_vector,"softmax":softmax}
    #model = keras.Model(inputs=inputs, outputs=outputs,name='SpeakerResNetV2')
    return model,speark_emb_model


if __name__ == '__main__':
    #model=ResNet50V2(input_shape=(31,80))
    model,speark_emb_model=SpeakerResNetV2(input_shape=(None, 80), classes=100)
    model.summary()
