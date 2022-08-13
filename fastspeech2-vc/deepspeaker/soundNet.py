import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import  keras
import pandas as pd
#https://github.com/camila-ud/SoundNet-keras/blob/master/soundnet_keras-master/soundnet.py
filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                      'kernel_size': 64, 'conv_strides': 2,
                      'pool_size': 8, 'pool_strides': 8},

                     {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                      'kernel_size': 32, 'conv_strides': 2,
                      'pool_size': 8, 'pool_strides': 8},

                     {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                      'kernel_size': 16, 'conv_strides': 2},

                     {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                      'kernel_size': 8, 'conv_strides': 2},

                     {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                      'kernel_size': 4, 'conv_strides': 2,
                      'pool_size': 4, 'pool_strides': 4},

                     {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                      'kernel_size': 4, 'conv_strides': 2},

                     {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                      'kernel_size': 4, 'conv_strides': 2},

                     {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                      'kernel_size': 8, 'conv_strides': 2},
                     ]

def conv1d_bn(inputs, filters, kernel_size,strides=1, padding=1,activation=None):
    x=keras.layers.ZeroPadding1D(padding=padding)(inputs)
    x=keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='causal', activation=None
                        ,kernel_regularizer=keras.regularizers.l2(l=0.0001))(x)
    x=keras.layers.BatchNormalization()(x)
    if activation is not None:
        if activation=='relu':
            x=keras.layers.ReLU(max_value=20.)(x)
        else:
            x=keras.layers.Activation(activation)(x)
    return x

def SoundNet(input_shape=(None,1),classes=100):
    av_input = keras.layers.Input(shape=input_shape, name='input')
    conv1=conv1d_bn(av_input, filters=16, kernel_size=64, strides=2, padding=32, activation='relu')
    conv1=keras.layers.MaxPool1D(pool_size=8,strides=8)(conv1)
    conv2 = conv1d_bn(conv1, filters=32, kernel_size=32, strides=2, padding=16, activation='relu')
    conv2 = keras.layers.MaxPool1D(pool_size=8, strides=8)(conv2)
    conv3 = conv1d_bn(conv2, filters=64, kernel_size=16, strides=2, padding=8, activation='relu')
    conv4 = conv1d_bn(conv3, filters=128, kernel_size=8, strides=2, padding=4, activation='relu')
    conv5 = conv1d_bn(conv4, filters=256, kernel_size=4, strides=2, padding=2, activation='relu')
    conv5 = keras.layers.MaxPool1D(pool_size=4, strides=4)(conv5)
    conv6 = conv1d_bn(conv5, filters=512, kernel_size=4, strides=2, padding=2, activation='relu')
    conv7 = conv1d_bn(conv6, filters=1024, kernel_size=4, strides=2, padding=2, activation='relu')
    conv8 = conv1d_bn(conv7, filters=401, kernel_size=4, strides=2, padding=0, activation='relu')
    pool=keras.layers.GlobalAvgPool1D()(conv8)
    #fc1=keras.layers.Dense(512)(conv8)
    #fc2 = keras.layers.Dense(256)(fc1)
    if classes==1:
        softmax = keras.layers.Dense(classes, activation='sigmoid')(pool)
    else:
        softmax=keras.layers.Dense(classes,activation='softmax')(pool)
    model=keras.models.Model(av_input,softmax)
    return model


def parse_fun(item,label):
    sample_rate=16000
    a_raw_audio = tf.io.read_file(item)
    a_wave, a_sr = tf.audio.decode_wav(a_raw_audio, desired_channels=1, desired_samples=-1)
    if a_sr != sample_rate:
        a_wave = tfio.audio.resample(a_wave, rate_in=tf.cast(a_sr, dtype=tf.int64),rate_out=tf.cast(sample_rate, dtype=tf.int64))
    #a_mel = TFFeaturizer.tf_extract2(a_wave[:, 0])
    return  a_wave*256,label


class ModelCheck(keras.callbacks.Callback):
    """Cosine annealing scheduler.
    """
    def __init__(self, filepath,steps=1):
        super(ModelCheck, self).__init__()
        self.filepath =str(filepath)
        self.steps=steps
        #self.best = np.Inf
    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch
    def set_model(self, model):
        self.model = model
    def _save_model(self, epoch, logs):
        logs = logs or {}
        self.model.save(self.filepath, overwrite=True, options=self._options)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get('val_loss')
        mname = '_%.4f-%03d.h5' % (current, epoch)
        if epoch%self.steps==0  :
            print(epoch, current, self.filepath+mname)
            self.model.save(self.filepath+mname,include_optimizer=False)

if __name__ == '__main__':
    data = pd.read_csv('data/train_ring.csv')
    data = data[data.duration < 0.7]
    data = data[data.duration > 0.02]
    data['label'] = 0
    data['label'][data.emotion == 'ring'] = 1
    data = data[['filepath', 'label', 'duration']]
    data = data.sample(frac=1)
    classes = 1
    ns = int(0.97 * (data.shape[0]))
    train = data.iloc[:ns]
    valid = data.iloc[ns:]
    valid.to_csv('data/valid_ring.csv', index=False)
    print(train.groupby('label')['filepath'].count(), valid.groupby('label')['filepath'].count())
    batch_size = 8
    epochs = 601
    sample_rate = 8000
    train_data = train.values.tolist()
    filenames = [x[0] for x in train_data]
    labels = [x[1] for x in train_data]
    val_filenames = valid.filepath.values.tolist()
    val_label = valid.label.values.tolist()

    with tf.device("/cpu:0"):
        train_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        train_dataset = train_dataset.shuffle(len(filenames))
        train_dataset = train_dataset.map(parse_fun, num_parallel_calls=16)
        train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=([None, None], []))
        valid_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_label))
        # train_dataset = train_dataset.shuffle(len(filenames))
        valid_dataset = valid_dataset.map(parse_fun, num_parallel_calls=12)
        valid_dataset = valid_dataset.padded_batch(batch_size, padded_shapes=([None, None], []))
    model=SoundNet(classes=classes)
    model.summary()
    lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.001, decay_steps=400000,
                                                       end_learning_rate=0.00001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_fn, clipvalue=10.0)
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['acc'])
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='saved/Net33_ring_{val_loss:.4f}-{epoch:04d}.h5' , verbose=1,save_weights_only=False,include_optimizer=False)
    checkpoint = ModelCheck('saved/SoundNet_ring')
    model.fit(train_dataset, batch_size=batch_size, validation_data=valid_dataset, epochs=epochs, callbacks=[checkpoint], workers=4,use_multiprocessing=False,max_queue_size=30)



