import tensorflow as tf
from tensorflow import keras
import numpy as np

x=np.random.random(2*80*100).reshape([2, 100, 80])


def SE(x,se_bottleneck_dim=128,channels=512):
    out=tf.reduce_mean(x,axis=1)
    #channels = tf.shape(out)[-1]
    out=keras.layers.Dense(se_bottleneck_dim)(out)
    out=keras.layers.ReLU()(out)
    out = keras.layers.Dense(channels)(out)
    out = keras.layers.Activation('sigmoid')(out)
    out=tf.reshape(out,shape=(tf.shape(out)[0],1,tf.shape(out)[1]))
    return x*out


def conv1d_relu_bn(inputs, filters, kernel_size,stride=1, dilation=1):
    x=keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, padding='causal', activation=None,
                        dilation_rate=dilation,kernel_regularizer=keras.regularizers.l2(l=0.0001))(inputs)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.ReLU(max_value=20.)(x)
    return x


def Res2Conv1dReluBn(x,channels,scale=8,kernel_size=3,stride=1,dilation=1):
    assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
    width = channels // scale
    nums = scale if scale == 1 else scale - 1
    spx=tf.split(x,scale,axis=-1)
    sp = spx[0]
    out=[]
    for i in range(nums):
        if i >= 1:
            sp = sp + spx[i]
            sp=keras.layers.Conv1D(width,kernel_size=kernel_size,strides=stride,dilation_rate=dilation,padding='same')(sp)
            sp=keras.layers.BatchNormalization()(sp)
            sp=keras.layers.ReLU()(sp)
            out.append(sp)
    if scale != 1:
        out.append(spx[nums])
    out=tf.concat(out,axis=-1)
    return out


def SE_Res2Block(x,channels, kernel_size, stride, dilation,scale):
    out=conv1d_relu_bn(x, filters=channels, kernel_size=1, stride=1, dilation=1)
    out=Res2Conv1dReluBn(out,channels, scale=scale, kernel_size=kernel_size, stride=stride, dilation=dilation)
    out = conv1d_relu_bn(out, filters=channels, kernel_size=1, stride=1, dilation=1)
    out=SE(out, se_bottleneck_dim=channels,channels=channels)
    return x+out

def ASTP(x,bottleneck_dim=128):
    if len(x.shape) == 4:
        c1,c2,c3,c4=tf.shape(x)
        x = tf.reshape(x, shape=(c1,c2*c3,c4))
    assert len(x.shape) == 3
    context_mean=tf.reduce_mean(x,axis=1,keepdims=True)
    context_mean=tf.repeat(context_mean,repeats=tf.shape(x)[1],axis=1)
    context_std =tf.sqrt( tf.math.reduce_std(x, axis=1, keepdims=True)+ 1e-10)
    context_std = tf.repeat(context_std, repeats=tf.shape(x)[1], axis=1)
    x_in=tf.concat([x,context_mean, context_std],axis=-1)
    alpha=tf.keras.layers.Conv1D(bottleneck_dim,kernel_size=1)(x_in)
    alpha=tf.tanh(alpha)
    alpha = tf.keras.layers.Conv1D(bottleneck_dim, kernel_size=1)(alpha)
    alpha=tf.math.softmax(alpha,axis=1)
    mean=tf.reduce_sum(alpha*x,axis=1)
    var = tf.reduce_sum(alpha * (x ** 2),axis=1)-mean ** 2
    std=tf.sqrt(tf.clip_by_value(var,clip_value_min=1e-10,clip_value_max=2147483647))
    out=tf.concat([mean,std],axis=-1)
    return out

def ECAPA_TDNN(input_shape=(None,80),classes=30000, channels=512, embed_dim=192):
    img_input = keras.layers.Input(shape=input_shape, name='input')
    out1 = conv1d_relu_bn(img_input, filters=channels, kernel_size=5, stride=1, dilation=1)
    out2 = SE_Res2Block(out1,channels=channels, kernel_size=3, stride=1, dilation=2, scale=8)
    out3 = SE_Res2Block(out2,channels,kernel_size=3,stride=1, dilation=3,scale=8)
    out4 = SE_Res2Block(out3,channels, kernel_size=3, stride=1,  dilation=4,scale=8)
    out=tf.keras.layers.concatenate([out2,out3,out4],axis=-1)
    out = conv1d_relu_bn(out, filters=512*3, kernel_size=1, stride=1, dilation=1)
    pool=ASTP(out,bottleneck_dim=512*3)
    pool=tf.keras.layers.BatchNormalization()(pool)
    emb=tf.keras.layers.Dense(embed_dim,activation=None, name='spk_emb')(pool)
    speark_emb_model=tf.keras.models.Model(img_input, emb,name='ECAPA_TDNN_SpeakerEmb')
    c=tf.keras.layers.Dropout(0.5)(emb)
    c=tf.keras.layers.Dense(classes)(c)
    softmax=tf.keras.layers.Activation('softmax')(c)
    model_softmax=tf.keras.models.Model(img_input, softmax,name='ECAPA_TDNN_softmax')
    model_amsoftmax = tf.keras.models.Model(img_input, c, name='ECAPA_TDNN')
    return model_softmax,model_amsoftmax,speark_emb_model
if __name__ == '__main__':

    model_softmax,model_amsoftmax,speark_emb_model=ECAPA_TDNN(input_shape=(None,80),classes=30000, channels=512, embed_dim=256)