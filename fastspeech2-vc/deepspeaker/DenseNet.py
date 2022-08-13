
from tensorflow import keras

def conv1d_bn(inputs, filters, kernel_size,strides=1, dilation_rate=1,activation=None):
    #print("conv1d_bn:",inputs)
    x=keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='causal', activation=None,
                        dilation_rate=dilation_rate,kernel_regularizer=keras.regularizers.l2(l=0.0001))(inputs)
    x=keras.layers.BatchNormalization(epsilon=1.001e-5)(x)
    if activation is not None:
        if activation=='relu':
            x=keras.layers.ReLU(max_value=20.)(x)
        else:
            x=keras.layers.Activation(activation)(x)
    return x

def dense_block(x, blocks, name,do_norm=True):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    #print('dense_block:',x)
    x = conv_block(x, 32 ,do_norm=do_norm)
    if blocks>1:
        for i in range(1,blocks):
            x = conv_block(x, 32)
    return x

def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    #bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    x = keras.layers.BatchNormalization( momentum=0.9, epsilon=1.001e-5,name=name + '_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_relu')(x)

    x = keras.layers.Conv1D(int(keras.backend.int_shape(x)[-1] * reduction), 1,  use_bias=False, kernel_initializer='Orthogonal',  name=name + '_conv')(x)
    x = keras.layers.AveragePooling1D(2, strides=2, name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, do_norm=True):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = -1 #3 if keras.backend.image_data_format() == 'channels_last' else 1
    if do_norm == True:
        x1 = keras.layers.BatchNormalization(axis=bn_axis,momentum=0.9, epsilon=1.001e-5)(x)
        x1 = keras.layers.Activation('relu')(x1)
        #x1 = conv1d_bn(x1, 4 * growth_rate, kernel_size=1, strides=1, dilation_rate=1, activation='relu' name=name)
    else:
        x1=x

    #print( "conv_block:",x1)
    x1=conv1d_bn(x1, 4 * growth_rate, kernel_size=1, strides=1, dilation_rate=1, activation=None)

    x1 = keras.layers.Conv1D(filters=growth_rate, kernel_size=3, strides=1, padding='causal', activation=None,
                            dilation_rate=1, kernel_regularizer=keras.regularizers.l2(l=0.0001))(x1)

    x = keras.layers.Concatenate(axis=bn_axis)([x, x1])
    return x

def DenseNet(blocks=[6,12,24,16],input_shape=(None,80),classes=1000,dropout=0.5):
    #bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    img_input = keras.layers.Input(shape=input_shape, name='input')
    x = keras.layers.ZeroPadding1D(padding=(3, 3),name='zeroPad1')(img_input)

    x = conv1d_bn(x, 64, kernel_size=7, strides=1, dilation_rate=1, activation='relu')

    #x = keras.layers.Conv2D(64, 7, strides=1, use_bias=False, name=name+'/conv1/conv',kernel_initializer='Orthogonal')(x)
    #x = keras.layers.BatchNormalization(axis=bn_axis,momentum=0.9, epsilon=1.001e-5, name=name+'/conv1/bn')(x)
    #x = keras.layers.Activation('relu', name=name+'/conv1/relu')(x)

    x = keras.layers.ZeroPadding1D(padding=(1, 1),name='zeroPad2')(x)
    x = keras.layers.MaxPooling1D(3, strides=2, name='pool1')(x)
    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')

    x = dense_block(x, blocks[3], name='conv5')
    x = keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='bn')(x)
    x=keras.layers.ReLU(max_value=20.)(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(units=512, return_sequences=True), merge_mode='concat')(x)
    x = keras.layers.GlobalMaxPooling1D(name='max_pool')(x)
    spk_emb = keras.layers.Dense(256, activation=None, name='spk_emb')(x)
    output = keras.layers.Dropout(dropout)(spk_emb)
    output = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(output)
    output = keras.layers.Dense(classes, activation=None, name='classify_amsoftmax',
                           kernel_constraint=keras.constraints.unit_norm())(output)

    softmax = keras.layers.Softmax(name='classify')(output)
    speark_emb_model = keras.models.Model(img_input, spk_emb, name='DenseNet_emb')
    model_amsoftmax = keras.Model(inputs=img_input, outputs=output, name='DenseNet_amsoftmax')
    # inputs={"anchor_input":anchor_input, "positive_input":positive_input, "negative_input":negative_input}
    # outputs={"triplet":merged_vector,"softmax":softmax}
    model_softmax = keras.Model(inputs=img_input, outputs=softmax, name='DenseNet_softmax')
    return model_softmax,model_amsoftmax,speark_emb_model

def SpeakerDenseNet(input_shape=(None,80),classes=10,dropout=0.5):
    _,_,speark_emb_model=DenseNet(blocks=[6, 12, 24, 16], input_shape=input_shape, classes=classes, dropout=dropout)
    anchor_input = keras.layers.Input(input_shape, name='anchor_input')
    positive_input = keras.layers.Input(input_shape, name='positive_input')
    negative_input = keras.layers.Input(input_shape, name='negative_input')
    encoded_anchor = speark_emb_model(anchor_input)
    encoded_positive = speark_emb_model(positive_input)
    encoded_negative = speark_emb_model(negative_input)
    x = keras.layers.Dropout(0.5)(encoded_anchor)
    softmax = keras.layers.Dense(classes, activation='softmax', name='softmax')(x)

    merged_vector = keras.layers.concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1,
                                             name='triplet')
    model = keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=[merged_vector, softmax],
                        name='SpeakerDenseNet')
    # inputs={"anchor_input":anchor_input, "positive_input":positive_input, "negative_input":negative_input}
    # outputs={"triplet":merged_vector,"softmax":softmax}
    # model = keras.Model(inputs=inputs, outputs=outputs,name='SpeakerResNetV2')
    return model, speark_emb_model


if __name__ == '__main__':
    #model=ResNet50V2(input_shape=(31,80))
    #[6, 12, 24, 16] 121
    #[6, 12, 32, 32] 169
    model,_,_=DenseNet(input_shape=(None, 80),blocks=[6,12,24,16], classes=100)
    model.summary()
