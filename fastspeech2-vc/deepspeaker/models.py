import tensorflow as tf
from tensorflow import  keras
from ResNet import ResNet50V2
#https://github.com/philipperemy/deep-speaker/blob/master/triplet_loss.py


def SpeakerLSTM(input_shape=(None,80),classes=6,dropout_rate=0.35,rate=1.):
    img_input = keras.layers.Input(shape=input_shape, name='input')
    gru =keras.layers.LSTM(units=int(128 * rate),return_sequences=True,recurrent_dropout=dropout_rate)(img_input)
    gru = keras.layers.LSTM(units=int(128 * rate),dropout=dropout_rate,return_sequences=True)(gru)
    gru = keras.layers.LSTM(units=int(128 * rate))(gru)
    fc = keras.layers.Dropout(dropout_rate)(gru)
    #conv9 = conv2d_bn(maxpool4, filters=256, kernel_size=5, strides=(1, 1), dilation_rate=1, activation='relu')
    fc1 = keras.layers.Dense(int(1024*rate), activation='relu', name='fc1')(fc)
    spk_emb = keras.layers.Dense(256, activation=None, name='fc2')(fc1)
    #fc = keras.layers.GlobalMaxPool1D()(fc)
    fc3 = keras.layers.Dropout(dropout_rate)(spk_emb)
    output = keras.layers.Dense(classes, activation=None, name='fc3')(fc3)
    if classes > 1:
        output = keras.layers.Activation('softmax',name='classify')(output)
    else:
        output = keras.layers.Activation('sigmoid', name='classify')(output)
    # Create model.
    model = keras.models.Model(img_input, output, name='ClassifyModel')
    speark_emb_model = keras.models.Model(img_input, spk_emb, name='SpeakerEmb')
    return model,speark_emb_model

def conv1d_bn(inputs, filters, kernel_size,strides=1, dilation_rate=1,activation=None):
    x=keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='causal', activation=None,
                        dilation_rate=dilation_rate,kernel_regularizer=keras.regularizers.l2(l=0.0001))(inputs)
    x=keras.layers.BatchNormalization()(x)
    if activation is not None:
        if activation=='relu':
            x=keras.layers.ReLU(max_value=20.)(x)
        else:
            x=keras.layers.Activation(activation)(x)
    return x

def SpeakerCNN(input_shape=(None,80),classes=6,kernel_size=5,dropout_rate=0.5,rate=1.0):
    img_input = keras.layers.Input(shape=input_shape, name='vad_input')
    conv1 = conv1d_bn(img_input, filters=int(64*rate), kernel_size=kernel_size, dilation_rate=1, activation='relu')
    conv2 = conv1d_bn(conv1, filters=int(64*rate), kernel_size=kernel_size,  dilation_rate=1, activation='relu')
    maxpool1 =keras.layers.MaxPool1D(pool_size=2,strides=2,padding='same',name='maxpool1')(conv2)
    maxpool1=keras.layers.Dropout(dropout_rate)(maxpool1)

    conv3 = conv1d_bn(maxpool1, filters=int(128*rate), kernel_size=kernel_size,  dilation_rate=1, activation='relu')
    conv4 = conv1d_bn(conv3, filters=int(128*rate), kernel_size=kernel_size,dilation_rate=1, activation='relu')
    maxpool2 = keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same', data_format='channels_last',  name='maxpool2')(conv4)
    maxpool2 = keras.layers.Dropout(dropout_rate)(maxpool2)

    conv5 = conv1d_bn(maxpool2, filters=int(256*rate), kernel_size=kernel_size,  dilation_rate=1, activation='relu')
    conv6 = conv1d_bn(conv5, filters=int(256*rate), kernel_size=kernel_size, dilation_rate=1, activation='relu')

    maxpool4 = keras.layers.GlobalMaxPool1D()(conv6)
    maxpool4 = keras.layers.Dropout(dropout_rate)(maxpool4)
    #conv9 = conv2d_bn(maxpool4, filters=256, kernel_size=5, strides=(1, 1), dilation_rate=1, activation='relu')
    fc = keras.layers.Dense(int(512*rate), activation='relu', name='fc1')(maxpool4)
    spk_emb = keras.layers.Dense(256, activation=None, name='fc2')(fc)
    fc = keras.layers.Dropout(dropout_rate)(spk_emb)
    # output = keras.layers.Dense(classes, activation='softmax', name='predictions')(fc)
    if classes > 1:
        output = keras.layers.Dense(classes, activation='softmax', name='classify')(fc)
    else:
        output = keras.layers.Dense(1, activation='sigmoid', name='classify')(fc)
    # Create model.
    model = keras.models.Model(img_input, output, name='ClassifyModel')
    speark_emb_model = keras.models.Model(img_input, spk_emb, name='SpeakerEmb')
    return model,speark_emb_model


def identity_block(input_tensor, kernel_size, filters):
        #conv_name_base = f'res{stage}_{block}_branch'
        x=conv1d_bn(input_tensor,filters, kernel_size=kernel_size, dilation_rate=1,activation='relu')
        x = conv1d_bn(x, filters, kernel_size=kernel_size, dilation_rate=1, activation='relu')
        x= keras.layers.add([input_tensor,x])
        x = keras.layers.ReLU(max_value=20.)(x)
        return x


def conv_and_res_block( inp, filters,kernel_size=5):
    #conv_name = 'conv{}-s'.format(filters)
    # TODO: why kernel_regularizer?
    o = conv1d_bn(inp, filters, kernel_size=kernel_size,strides=2, dilation_rate=1, activation='relu')
    for i in range(3):
        o=identity_block(o, kernel_size=3, filters=filters)
        #o = self.identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
    return o

def ResNetV1(input_shape=(None,80),classes=6,dropout_rate=0.5):
    img_input = keras.layers.Input(shape=input_shape, name='input')
    #x = conv1d_bn(img_input, filters=32, kernel_size=7, dilation_rate=1, activation='relu')
    x = conv_and_res_block(img_input, 64)
    x = conv_and_res_block(x, 128 )
    x = conv_and_res_block(x, 256)
    x = conv_and_res_block(x, 512)
    x = keras.layers.GlobalMaxPool1D()(x)
    if dropout_rate>0:
        x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(512, activation=None, name='fc1')(x)
    x = keras.layers.ReLU(max_value=20.,name='fc1_relu')(x)
    spk_emb = keras.layers.Dense(256, activation=None, name='spk_emb')(x)
    #spk_emb = keras.layers.ReLU(max_value=20., name='spk_emb')(x)

    output_amsoftmax = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(spk_emb)
    output_amsoftmax = keras.layers.Dense(classes, activation=None, name='classify',kernel_constraint=keras.constraints.unit_norm())(output_amsoftmax)

    output_softmax = keras.layers.Dense(classes, activation='softmax', name='softmax')(spk_emb)

    # Create model.
    #model = keras.models.Model(img_input, output, name='SpeakerModel')
    speark_emb_model=keras.models.Model(img_input, spk_emb, name='SpeakerEmb')
    model_amsoftmax = keras.Model(inputs=img_input, outputs=output_amsoftmax, name='ResNetV1')
    # inputs={"anchor_input":anchor_input, "positive_input":positive_input, "negative_input":negative_input}
    # outputs={"triplet":merged_vector,"softmax":softmax}
    model_softmax = keras.Model(inputs=img_input, outputs=output_softmax, name='ResNetV1')

    return model_softmax,model_amsoftmax,speark_emb_model

def SpeakerResNet_triple(input_shape=(None,80),dropout_rate=0.5):
    _,speark_emb_model=ResNetV1(input_shape=(None, 80), classes=6, dropout_rate=dropout_rate)
    anchor_input = tf.keras.layers.Input(input_shape, name='anchor_input')
    positive_input = tf.keras.layers.Input(input_shape, name='positive_input')
    negative_input = tf.keras.layers.Input(input_shape, name='negative_input')
    encoded_anchor = speark_emb_model(anchor_input)
    encoded_positive = speark_emb_model(positive_input)
    encoded_negative = speark_emb_model(negative_input)
    merged_vector = tf.keras.layers.concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
    model = tf.keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector,name='SpeakerTriple')
    return model,speark_emb_model

def SpeakerResNetV1(input_shape=(None,80),classes=10,dropout_rate=0.5):
    _,speark_emb_model=ResNetV1(input_shape=(None, 80), classes=6, dropout_rate=dropout_rate)
    anchor_input = tf.keras.layers.Input(input_shape, name='anchor_input')
    positive_input = tf.keras.layers.Input(input_shape, name='positive_input')
    negative_input = tf.keras.layers.Input(input_shape, name='negative_input')
    encoded_anchor = speark_emb_model(anchor_input)
    encoded_positive = speark_emb_model(positive_input)
    encoded_negative = speark_emb_model(negative_input)
    x = keras.layers.Dropout(0.5)(encoded_anchor)
    softmax = keras.layers.Dense(classes, activation='softmax', name='softmax')(x)

    merged_vector = keras.layers.concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='triplet')
    model = keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=[merged_vector,softmax],name='ResNetV1')
    return model,speark_emb_model

def ResNetV2(input_shape=(None,80),classes=10):
    speark_emb_model=ResNet50V2(include_top=False,input_shape=input_shape, classes=classes,class_type='softmax')
    x = keras.layers.Dropout(0.5)(speark_emb_model.outputs[0])

    output_amsoftmax = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)
    output_amsoftmax = keras.layers.Dense(classes, activation=None, name='amsoftmax',
                                kernel_constraint=keras.constraints.unit_norm())(output_amsoftmax)

    output_softmax = keras.layers.Dense(classes, activation='softmax', name='softmax')(x)

    model_amsoftmax = keras.Model(inputs=speark_emb_model.inputs, outputs=output_amsoftmax,name='ResNet50V2')
    #inputs={"anchor_input":anchor_input, "positive_input":positive_input, "negative_input":negative_input}
    #outputs={"triplet":merged_vector,"softmax":softmax}
    model_softmax = keras.Model(inputs=speark_emb_model.inputs, outputs=output_softmax,name='ResNet50V2')
    return model_softmax,model_amsoftmax,speark_emb_model
if __name__ == '__main__':
    model,_,_=ResNetV1(input_shape=(None,80),classes=6,dropout_rate=0.5)
    model.summary()

